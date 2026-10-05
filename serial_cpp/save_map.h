// Saves one axial slice of a parameter map as a PNG, for checking fits by eye.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

// Writes slice l of vol (x fastest: index = x + X*(y + Y*l)) to a PNG using MATLAB's
// 'hot' colormap, clipped to [cmin, cmax] like clim(). NaN voxels are drawn black.
// Oriented like the MATLAB figure (imagesc + view(-90,90)): x to the right, y upwards.
// Each voxel is drawn as a scale x scale block of pixels.
// Throws std::runtime_error on failure.
void save_map_png(const std::string &path, const std::vector<double> &vol,
                  std::size_t X, std::size_t Y, std::size_t l,
                  double cmin, double cmax, int scale = 4);
