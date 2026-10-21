// Calls the serial Ceres fit for compare_serial.cu. It lives in its own file because the
// serial fit_voxel_sub_ml.h and the device fit_voxel_sub_ml.cuh both define FitResult.

#include <array>

#include "fit_voxel_sub_ml.h"

std::array<double, 3> serial_fit(const std::array<double, 9> &svec, const std::array<double, 9> &bvec,
                                 double D_guess, double beta_guess, double D_min, double D_max,
                                 double beta_min, double beta_max)
{
    const FitResult fit = fit_voxel_sub_ml(svec, bvec, D_guess, beta_guess, D_min, D_max, beta_min, beta_max);
    return {fit.D, fit.beta, fit.err};
}
