// Minimal reader for single-volume NIfTI-1 files (.nii and .nii.gz)

#include "read_mri.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <zlib.h>

namespace
{

    // NIfTI-1 header layout (byte offsets)
    constexpr int HDR_SIZE = 348;
    constexpr int OFF_SIZEOF_HDR = 0;   // int32, must be 348
    constexpr int OFF_DIM = 40;         // int16[8]
    constexpr int OFF_DATATYPE = 70;    // int16
    constexpr int OFF_BITPIX = 72;      // int16
    constexpr int OFF_VOX_OFFSET = 108; // float32
    constexpr int OFF_SCL_SLOPE = 112;  // float32
    constexpr int OFF_SCL_INTER = 116;  // float32
    constexpr int OFF_MAGIC = 344;      // char[4], "n+1\0"

    // NIfTI datatype codes
    constexpr int DT_INT16 = 4;
    constexpr int DT_FLOAT32 = 16;
    constexpr int DT_FLOAT64 = 64;

    // Copies value into raw header bytes (the inverse of get_field)
    template <typename T>
    void set_field(unsigned char *hdr, std::size_t offset, T value)
    {
        std::memcpy(hdr + offset, &value, sizeof(T));
    }

    // Copies a field of type T out of raw bytes (memcpy avoids misaligned reads).
    // Assumes a little-endian host, which matches the files we accept.
    template <typename T>
    T get_field(const unsigned char *hdr, std::size_t offset)
    {
        T value;
        std::memcpy(&value, hdr + offset, sizeof(T));
        return value;
    }

    // Owns a gzFile and closes it on every exit path, including exceptions.
    struct GzClose
    {
        void operator()(gzFile file) const { gzclose(file); }
    };
    using GzHandle = std::unique_ptr<gzFile_s, GzClose>;

    // Throws zlib's error. The message is owned by the gzFile, so it is copied
    // before the handle is closed during unwinding. zlib already prefixes the path.
    [[noreturn]] void throw_gz_error(gzFile file)
    {
        int errnum = Z_OK;
        throw std::runtime_error(std::string("failed to read ") + gzerror(file, &errnum));
    }

    // Reads up to n bytes into dst; returns how many were read (< n only at end of file).
    // gzread also handles uncompressed files.
    std::size_t read_bytes(gzFile file, unsigned char *dst, std::size_t n)
    {
        constexpr std::size_t MAX_CHUNK = 1u << 30; // gzread takes an unsigned int length
        std::size_t total = 0;
        while (total < n)
        {
            const unsigned chunk = static_cast<unsigned>(std::min(n - total, MAX_CHUNK));
            const int bytes_read = gzread(file, dst + total, chunk);
            if (bytes_read < 0)
                throw_gz_error(file);
            if (bytes_read == 0)
                break;
            total += static_cast<std::size_t>(bytes_read);
        }

        // A gzip stream that ends early is not a gzread error: it returns the
        // partial data and reports Z_BUF_ERROR here instead.
        int errnum = Z_OK;
        gzerror(file, &errnum);
        if (errnum != Z_OK)
            throw_gz_error(file);

        return total;
    }

    // Like read_bytes, but grows the buffer as data arrives, so a corrupt header
    // claiming a huge size can't trigger a huge allocation up front.
    std::vector<unsigned char> read_up_to(gzFile file, std::size_t n)
    {
        constexpr std::size_t CHUNK = 16u << 20;
        std::vector<unsigned char> out;
        while (out.size() < n)
        {
            const std::size_t old_size = out.size();
            const std::size_t want = std::min(n - old_size, CHUNK);
            out.resize(old_size + want);
            const std::size_t got = read_bytes(file, out.data() + old_size, want);
            out.resize(old_size + got);
            if (got < want)
                break;
        }
        return out;
    }

} // namespace

NiftiVolume read_nifti(const std::string &path)
{
    GzHandle file(gzopen(path.c_str(), "rb"));
    if (!file)
        throw std::runtime_error(path + ": failed to open file");

    // Read and validate only the header first, so a wrong file (e.g. a multi-GB 4D
    // series) is rejected without decompressing its voxel data.
    unsigned char hdr[HDR_SIZE];
    if (read_bytes(file.get(), hdr, HDR_SIZE) < HDR_SIZE)
        throw std::runtime_error(path + ": file too small for NIfTI header");

    // sizeof_hdr doubles as an endianness check: a big-endian file reads as a huge number
    if (get_field<int32_t>(hdr, OFF_SIZEOF_HDR) != HDR_SIZE)
        throw std::runtime_error(path + ": not a little-endian NIfTI-1 file (sizeof_hdr != 348)");

    // "n+1" = header and data in one file; "ni1" (separate .img) is not supported
    if (std::memcmp(hdr + OFF_MAGIC, "n+1\0", 4) != 0)
        throw std::runtime_error(path + ": unsupported NIfTI magic (expected single-file \"n+1\")");

    int16_t dim[8];
    for (int i = 0; i < 8; ++i)
        dim[i] = get_field<int16_t>(hdr, OFF_DIM + 2 * i);

    // Accept trailing singleton dimensions (e.g. 4D with one volume) but nothing larger
    if (dim[0] < 3 || dim[0] > 7)
        throw std::runtime_error(path + ": expected a 3D volume, got dim[0] = " + std::to_string(dim[0]));
    for (int i = 1; i <= 3; ++i)
        if (dim[i] < 1)
            throw std::runtime_error(path + ": invalid dim[" + std::to_string(i) + "] = " + std::to_string(dim[i]));
    for (int i = 4; i <= dim[0]; ++i)
        if (dim[i] != 1)
            throw std::runtime_error(path + ": expected a single 3D volume, but dim[" + std::to_string(i) +
                                     "] = " + std::to_string(dim[i]));

    const int16_t datatype = get_field<int16_t>(hdr, OFF_DATATYPE);
    const int16_t bitpix = get_field<int16_t>(hdr, OFF_BITPIX);
    if (!((datatype == DT_INT16 && bitpix == 16) || (datatype == DT_FLOAT32 && bitpix == 32)))
        throw std::runtime_error(path + ": unsupported datatype " + std::to_string(datatype) +
                                 " (bitpix " + std::to_string(bitpix) + "); only int16 and float32 are handled");

    // vox_offset is stored as a float for historical reasons; !(x >= y) also rejects NaN
    const float vox_offset_f = get_field<float>(hdr, OFF_VOX_OFFSET);
    if (!(vox_offset_f >= HDR_SIZE))
        throw std::runtime_error(path + ": invalid vox_offset " + std::to_string(vox_offset_f));
    const std::size_t vox_offset = static_cast<std::size_t>(vox_offset_f);

    // slope == 0 (or non-finite) means "no scaling" per the NIfTI-1 spec
    const float slope = get_field<float>(hdr, OFF_SCL_SLOPE);
    const float inter = get_field<float>(hdr, OFF_SCL_INTER);
    const bool apply_scaling = std::isfinite(slope) && slope != 0.0f;

    NiftiVolume vol;
    std::memcpy(vol.header.data(), hdr, HDR_SIZE);
    vol.X = static_cast<std::size_t>(dim[1]);
    vol.Y = static_cast<std::size_t>(dim[2]);
    vol.L = static_cast<std::size_t>(dim[3]);

    const std::size_t nvox = vol.size();
    const std::size_t bytes_per_voxel = static_cast<std::size_t>(bitpix) / 8;
    const std::size_t needed = vox_offset + nvox * bytes_per_voxel;

    // Everything after the header up to the end of the voxel data (including any
    // header extension between byte 348 and vox_offset)
    const std::vector<unsigned char> bytes = read_up_to(file.get(), needed - HDR_SIZE);
    const std::size_t have = HDR_SIZE + bytes.size();
    if (have < needed)
        throw std::runtime_error(path + ": file truncated (have " + std::to_string(have) +
                                 " bytes, need " + std::to_string(needed) + ")");

    // Read one byte past the data so zlib reaches the end of the gzip stream and
    // verifies its CRC. In testing zlib already did this within the final read,
    // but its docs don't guarantee that, and the extra read is free.
    unsigned char probe;
    read_bytes(file.get(), &probe, 1);

    const unsigned char *src = bytes.data() + (vox_offset - HDR_SIZE);
    vol.data.resize(nvox);
    for (std::size_t i = 0; i < nvox; ++i)
    {
        const std::size_t off = i * bytes_per_voxel;
        double v = (datatype == DT_INT16) ? static_cast<double>(get_field<int16_t>(src, off))
                                          : static_cast<double>(get_field<float>(src, off));
        if (apply_scaling)
            v = static_cast<double>(slope) * v + static_cast<double>(inter);
        vol.data[i] = v;
    }

    return vol;
}

void write_nifti(const std::string &path, const NiftiVolume &like, const std::vector<double> &data)
{
    if (data.size() != like.size())
        throw std::runtime_error(path + ": data has " + std::to_string(data.size()) + " voxels, expected " +
                                 std::to_string(like.size()));

    // Same geometry as the input; only the shape, type, scaling and data offset change
    std::array<unsigned char, HDR_SIZE> hdr = like.header;
    set_field<int16_t>(hdr.data(), OFF_DIM, 3);
    for (int i = 4; i < 8; ++i)
        set_field<int16_t>(hdr.data(), OFF_DIM + 2 * i, 1);
    set_field<int16_t>(hdr.data(), OFF_DATATYPE, DT_FLOAT64);
    set_field<int16_t>(hdr.data(), OFF_BITPIX, 64);
    set_field<float>(hdr.data(), OFF_VOX_OFFSET, 352.0f);
    set_field<float>(hdr.data(), OFF_SCL_SLOPE, 1.0f);
    set_field<float>(hdr.data(), OFF_SCL_INTER, 0.0f);
    std::memcpy(hdr.data() + OFF_MAGIC, "n+1\0", 4);

    std::ofstream file(path, std::ios::binary);
    const char no_extension[4] = {0, 0, 0, 0}; // bytes 348-351: no header extensions
    file.write(reinterpret_cast<const char *>(hdr.data()), HDR_SIZE);
    file.write(no_extension, 4);
    file.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(double)));
    if (!file)
        throw std::runtime_error(path + ": could not write file");
}
