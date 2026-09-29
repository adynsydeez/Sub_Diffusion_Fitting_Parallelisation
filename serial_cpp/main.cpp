// Serial C++ port of MATLAB/fit_dMRI_signals_19.m
// Sub-diffusion model fitting (Delta = 19 ms)

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Configuration
const std::string SUBJECT  = "sub_001";
const std::string DATATYPE = "_real"; // "_real" or ""
const std::string FOLDER   = "./data/dwi_real/";

constexpr std::array<double, 9> BVEC = {0, 50, 350, 800, 1500, 2400, 3450, 4750, 6000}; // s/mm^2
constexpr std::size_t NB = BVEC.size();

// Fit bounds and initial guesses
constexpr double D_GUESS    = 1e-3, D_MIN    = 0.0, D_MAX    = 3e-3; // mm^2/s
constexpr double BETA_GUESS = 0.5,  BETA_MIN = 0.0, BETA_MAX = 1.0;  // time fractional index

struct FitResult {
    double D;
    double beta;
    double err;
};

// TODO: port MATLAB/ml.m (Mittag-Leffler function, OPC algorithm)
// double ml(double z, double alpha);

// TODO: port MATLAB/fit_voxel_sub_ml.m (bounded least-squares fit)
// FitResult fit_voxel_sub_ml(const std::array<double, NB>& Svec);

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    // --- Load data ---
    // TODO: load brain mask, b0 image and per-b-value images
    
    std::size_t X = 0, Y = 0, L = 0;
    std::vector<double> S0;              // X*Y*L
    std::vector<double> S;               // X*Y*L*NB

    std::size_t num_voxels = 0;
    for (double v : S0)
        if (v > 0) ++num_voxels;
    std::printf("Fitting %zu brain voxels\n", num_voxels);

    // --- Output maps ---
    std::vector<double> D_SUB(X * Y * L, NAN);
    std::vector<double> BETA_SUB(X * Y * L, NAN);
    std::vector<double> ERROR_SUB(X * Y * L, NAN);

    // --- Fit ---
    auto t_start = std::chrono::steady_clock::now();

    for (std::size_t l = 0; l < L; ++l) {
        for (std::size_t x = 0; x < X; ++x) {
            for (std::size_t y = 0; y < Y; ++y) {
                // TODO: skip voxels outside mask, normalise by S0, call fit_voxel_sub_ml
            }
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double fit_time = std::chrono::duration<double>(t_end - t_start).count();
    std::printf("Fit time: %.1f s (%.3f ms/voxel)\n", fit_time,
                num_voxels ? 1e3 * fit_time / num_voxels : 0.0);

    // --- Derived parameters ---
    std::vector<double> K_STAR(X * Y * L, NAN);
    for (std::size_t i = 0; i < K_STAR.size(); ++i) {
        double b = BETA_SUB[i];
        K_STAR[i] = 6.0 * std::pow(std::tgamma(1.0 + b), 2) / std::tgamma(1.0 + 2.0 * b) - 3.0;
    }

    // TODO: write results

    return EXIT_SUCCESS;
}
