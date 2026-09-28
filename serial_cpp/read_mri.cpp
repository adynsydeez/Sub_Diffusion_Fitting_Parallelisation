// Minimal reader for single-volume NIfTI-1 files (.nii and .nii.gz)

#include "read_mri.h"

#include <cstdint>
#include <cstring>
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

    // Copies a field of type T out of the raw header bytes.
    template <typename T>
    T get_field(const unsigned char *hdr, int offset)
    {
        T value;
        std::memcpy(&value, hdr + offset, sizeof(T));
        return value;
    }

    // Reads the whole file into memory; gzread also handles uncompressed files.
    std::vector<unsigned char> read_file_bytes(const std::string &path)
    {
        std::vector<unsigned char> bytes;
        // TODO: gzopen, gzread in chunks until EOF, gzclose
        gzFile file = gzopen(path.c_str(), "rb");
        if (!file)
        {
            throw std::runtime_error(path + ": failed to open file");
            return {};
        }

        char buffer[1024];

        

        (void)path;
        return bytes;
    }

} // namespace

NiftiVolume read_nifti(const std::string &path)
{
    std::vector<unsigned char> bytes = read_file_bytes(path);
    if (bytes.size() < HDR_SIZE)
        throw std::runtime_error(path + ": file too small for NIfTI header");

    const unsigned char *hdr = bytes.data();

    // TODO: validate sizeof_hdr and magic
    // TODO: read dim, check dim[0] == 3
    // TODO: read datatype, vox_offset, scl_slope, scl_inter
    // TODO: check file is large enough for vox_offset + nvox * bitpix/8

    NiftiVolume vol;

    // TODO: convert voxels to double (DT_INT16 / DT_FLOAT32), apply scaling if slope != 0

    (void)hdr;
    return vol;
}
