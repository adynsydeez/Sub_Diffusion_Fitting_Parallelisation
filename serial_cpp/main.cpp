// Serial C++ port of MATLAB/fit_dMRI_signals_19.m
// Sub-diffusion model fitting (Delta = 19 ms)

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "read_mri.h"
#include "mittag-leffler.h"
#include "fit_voxel_sub_ml.h"
#include "save_map.h"

// Configuration
const std::string SUBJECT = "sub_001";
const std::string DATATYPE = "dwi_real"; // "_real" or ""
const std::string FOLDER = "./data/dwi_real/";

constexpr std::array<double, 9> BVEC = {0, 50, 350, 800, 1500, 2400, 3450, 4750, 6000}; // s/mm^2
constexpr std::size_t NB = BVEC.size();

// Fit bounds and initial guesses
constexpr double D_GUESS = 1e-3, D_MIN = 0.0, D_MAX = 3e-3;        // mm^2/s
constexpr double BETA_GUESS = 0.5, BETA_MIN = 0.0, BETA_MAX = 1.0; // time fractional index

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    // --- Load data ---
    NiftiVolume mask = read_nifti(FOLDER + SUBJECT + "_" + DATATYPE + "_brainmask.nii.gz");

    NiftiVolume S0 = read_nifti(FOLDER + SUBJECT + "_" + DATATYPE + "_b0_delta0_image.nii");
    for (std::size_t i = 0; i < S0.size(); ++i)
    {
        S0.data[i] *= mask.data[i]; // .* brainmask
    }

    std::size_t X = S0.X, Y = S0.Y, L = S0.L;
    std::vector<double> S(X * Y * L * NB);
    std::copy(S0.data.begin(), S0.data.end(), S.begin()); // b=0 first

    for (std::size_t k = 1; k < NB; ++k)
    {
        NiftiVolume S_k = read_nifti(FOLDER + SUBJECT + "_" + DATATYPE + "_b" + std::to_string(static_cast<int>(BVEC[k])) + "_delta19_image.nii");
        for (std::size_t i = 0; i < S_k.size(); ++i)
        {
            S[i + k * X * Y * L] = S_k.data[i] * mask.data[i]; // .* brainmask
        }
    }

    // nnz(S0 > 0)
    std::size_t num_voxels = 0;
    for (double v : S0.data)
    {
        if (v > 0)
        {
            ++num_voxels;
        }
    }
    std::printf("Fitting %zu brain voxels\n", num_voxels);

    // --- Output maps ---
    std::vector<double> D_SUB(X * Y * L, NAN);
    std::vector<double> BETA_SUB(X * Y * L, NAN);
    std::vector<double> ERROR_SUB(X * Y * L, NAN);

    // --- Fit ---
    auto t_start = std::chrono::steady_clock::now();

    for (std::size_t l = 0; l < L; ++l)
    {
        for (std::size_t x = 0; x < X; ++x)
        {
            for (std::size_t y = 0; y < Y; ++y)
            {
                std::size_t idx = x + X * (y + Y * l); // S0(x,y,l)

                if (S0.data[idx] <= 0.0)
                {
                    continue;
                }

                // Svec = squeeze(S(x,y,l,:)) / S0(x,y,l);
                std::array<double, 9> svec;
                for (std::size_t k = 0; k < NB; ++k)
                {
                    svec[k] = S[idx + k * X * Y * L] / S0.data[idx];
                }

                auto [D, beta, err] = fit_voxel_sub_ml(svec, BVEC, D_GUESS, BETA_GUESS, D_MIN, D_MAX, BETA_MIN, BETA_MAX);
                D_SUB[idx] = D;
                BETA_SUB[idx] = beta;
                ERROR_SUB[idx] = err;
            }
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double fit_time = std::chrono::duration<double>(t_end - t_start).count();
    std::printf("Fit time: %.1f s (%.3f ms/voxel)\n", fit_time,
                num_voxels ? 1e3 * fit_time / num_voxels : 0.0);

    // --- Derived parameters ---
    std::vector<double> K_STAR(X * Y * L, NAN);
    for (std::size_t i = 0; i < K_STAR.size(); ++i)
    {
        double b = BETA_SUB[i];
        K_STAR[i] = 6.0 * std::pow(std::tgamma(1.0 + b), 2) / std::tgamma(1.0 + 2.0 * b) - 3.0;
    }

    // --- Parameter maps ---
    const std::size_t l_show = 72; // axial slice to display (l = 73 in MATLAB)
    save_map_png("D_SUB.png", D_SUB, X, Y, l_show, 0.0, 3e-3);
    save_map_png("RMSE.png", ERROR_SUB, X, Y, l_show, 0.0, 0.1);
    save_map_png("BETA.png", BETA_SUB, X, Y, l_show, 0.5, 1.0);
    save_map_png("K_STAR.png", K_STAR, X, Y, l_show, 0.0, 3.0);
    std::printf("Saved parameter maps for slice %zu\n", l_show + 1);

    // TODO: write results

    return EXIT_SUCCESS;
}
