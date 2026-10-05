// Saves one axial slice of a parameter map as a PNG, for checking fits by eye.

#include "save_map.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <zlib.h>

namespace
{

    // MATLAB's 'hot' colormap: black -> red -> yellow -> white, for t in [0, 1]
    std::array<std::uint8_t, 3> hot(double t)
    {
        double r = std::clamp(t / 0.375, 0.0, 1.0);
        double g = std::clamp((t - 0.375) / 0.375, 0.0, 1.0);
        double b = std::clamp((t - 0.75) / 0.25, 0.0, 1.0);
        return {static_cast<std::uint8_t>(255 * r + 0.5),
                static_cast<std::uint8_t>(255 * g + 0.5),
                static_cast<std::uint8_t>(255 * b + 0.5)};
    }

    void put_u32_be(std::vector<unsigned char> &out, std::uint32_t v)
    {
        out.push_back(static_cast<unsigned char>(v >> 24));
        out.push_back(static_cast<unsigned char>(v >> 16));
        out.push_back(static_cast<unsigned char>(v >> 8));
        out.push_back(static_cast<unsigned char>(v));
    }

    // PNG chunk: length, type, data, CRC of type + data
    void put_chunk(std::vector<unsigned char> &out, const char *type, const std::vector<unsigned char> &data)
    {
        put_u32_be(out, static_cast<std::uint32_t>(data.size()));
        std::size_t type_start = out.size();
        out.insert(out.end(), type, type + 4);
        out.insert(out.end(), data.begin(), data.end());
        uLong crc = crc32(0L, out.data() + type_start, static_cast<uInt>(4 + data.size()));
        put_u32_be(out, static_cast<std::uint32_t>(crc));
    }

} // namespace

void save_map_png(const std::string &path, const std::vector<double> &vol,
                  std::size_t X, std::size_t Y, std::size_t l,
                  double cmin, double cmax, int scale)
{
    if (l >= vol.size() / (X * Y))
        throw std::runtime_error("save_map_png: slice index out of range");

    const std::size_t W = X * scale, H = Y * scale;

    // Raw scanlines: each row is a filter byte (0 = none) followed by RGB pixels
    std::vector<unsigned char> raw;
    raw.reserve(H * (1 + 3 * W));
    for (std::size_t row = 0; row < H; ++row)
    {
        std::size_t y = Y - 1 - row / scale; // y increases upwards
        raw.push_back(0);
        for (std::size_t col = 0; col < W; ++col)
        {
            std::size_t x = col / scale;
            double v = vol[x + X * (y + Y * l)];
            std::array<std::uint8_t, 3> rgb = std::isnan(v) ? std::array<std::uint8_t, 3>{0, 0, 0}
                                                            : hot((v - cmin) / (cmax - cmin));
            raw.insert(raw.end(), rgb.begin(), rgb.end());
        }
    }

    uLongf zlen = compressBound(static_cast<uLong>(raw.size()));
    std::vector<unsigned char> idat(zlen);
    if (compress(idat.data(), &zlen, raw.data(), static_cast<uLong>(raw.size())) != Z_OK)
        throw std::runtime_error("save_map_png: compression failed");
    idat.resize(zlen);

    // IHDR: width, height, bit depth 8, colour type 2 (RGB), default compression/filter/interlace
    std::vector<unsigned char> ihdr;
    put_u32_be(ihdr, static_cast<std::uint32_t>(W));
    put_u32_be(ihdr, static_cast<std::uint32_t>(H));
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});

    std::vector<unsigned char> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    put_chunk(png, "IHDR", ihdr);
    put_chunk(png, "IDAT", idat);
    put_chunk(png, "IEND", {});

    std::ofstream file(path, std::ios::binary);
    if (!file.write(reinterpret_cast<const char *>(png.data()), static_cast<std::streamsize>(png.size())))
        throw std::runtime_error("save_map_png: could not write " + path);
}
