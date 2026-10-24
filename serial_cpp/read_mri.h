// Minimal reader for single-volume NIfTI-1 files (.nii and .nii.gz)

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

struct NiftiVolume {
    std::size_t X = 0, Y = 0, L = 0;
    std::vector<double> data; // x fastest: index = x + X*(y + Y*l)
    std::array<unsigned char, 348> header{}; // raw header as read, so outputs keep the geometry

    std::size_t size() const { return X * Y * L; }
};

// Reads a 3D NIfTI-1 volume, applying scl_slope/scl_inter if set.
// Supports uncompressed (.nii) and gzipped (.nii.gz) files.
// Throws std::runtime_error on failure.
NiftiVolume read_nifti(const std::string& path);

// Writes data (same size and layout as `like`) as an uncompressed float64 NIfTI-1 file,
// copying like's header so voxel size and orientation match the input.
// Throws std::runtime_error on failure.
void write_nifti(const std::string& path, const NiftiVolume& like, const std::vector<double>& data);
