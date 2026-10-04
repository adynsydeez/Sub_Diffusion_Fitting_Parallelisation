// Parameter fitting for the sub-diffusion model using the Mittag-Leffler function.

#pragma once

#include <array>

struct FitResult
{
    double D;
    double beta;
    double err;
};

FitResult fit_voxel_sub_ml(const std::array<double, 9> &Svec, const std::array<double, 9> &bvec, double D_guess, double beta_guess, double D_min, double D_max, double beta_min, double beta_max);