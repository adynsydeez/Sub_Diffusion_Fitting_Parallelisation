// Serial C++ port of MATLAB/fit_dMRI_signals_19.m
// Sub-diffusion model fitting (Delta = 19 ms)

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <string>
#include <vector>

#include "read_mri.h"
#include "save_map.h"

// BEST_SEQUENTIAL (target fit_sub_best): the CUDA version's algorithm on one CPU core,
// i.e. specialised Mittag-Leffler, analytic Jacobian and a small LM solver instead of Ceres
#ifdef BEST_SEQUENTIAL
#include "fit_voxel_sub_ml.cuh"
constexpr const char *VARIANT = "best sequential";
constexpr const char *OUTPUT_DIR = "serial_cpp/outputs_best";
#else
#include "fit_voxel_sub_ml.h"
constexpr const char *VARIANT = "serial, Ceres";
constexpr const char *OUTPUT_DIR = "serial_cpp/outputs";
#endif

// Configuration
const std::string SUBJECT = "sub_001";
const std::string DATATYPE = "dwi_real"; // "_real" or ""
const std::string FOLDER = "./data/dwi_real/";

constexpr std::array<double, 9> BVEC = {0, 50, 350, 800, 1500, 2400, 3450, 4750, 6000}; // s/mm^2
constexpr std::size_t NB = BVEC.size();

// Fit bounds and initial guesses
constexpr double D_GUESS = 1e-3, D_MIN = 0.0, D_MAX = 3e-3;        // mm^2/s
constexpr double BETA_GUESS = 0.5, BETA_MIN = 0.0, BETA_MAX = 1.0; // time fractional index

constexpr std::size_t DEFAULT_SLICE = 73; // slice shown in --all mode (l = 73 in MATLAB)

void print_usage(const char *prog)
{
    std::fprintf(stderr,
                 "Usage: %s [--all | --slice N]\n"
                 "  --all      fit every brain voxel (default; for speed testing)\n"
                 "  --slice N  fit only axial slice N, numbered from 1 as in MATLAB (for visual checks)\n",
                 prog);
}

int main(int argc, char *argv[])
{
    // --- Command line ---
    bool all_slices = true;
    std::size_t slice = DEFAULT_SLICE; // 1-based, as in MATLAB

    const std::string mode = argc > 1 ? argv[1] : "--all";
    if (mode == "--all" && argc <= 2)
    {
        all_slices = true;
    }
    else if (mode == "--slice" && argc == 3)
    {
        const std::string arg = argv[2];
        std::size_t pos = 0;
        try
        {
            slice = std::stoul(arg, &pos);
        }
        catch (const std::exception &)
        {
            pos = 0;
        }
        if (pos == 0 || pos != arg.size() || arg[0] == '-')
        {
            std::fprintf(stderr, "Invalid slice number: %s\n", argv[2]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        all_slices = false;
    }
    else
    {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

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

    if (slice < 1 || slice > L)
    {
        std::fprintf(stderr, "Slice %zu is out of range: the volume has slices 1 to %zu\n", slice, L);
        return EXIT_FAILURE;
    }

    // Slices to fit, as a 0-based range [l_begin, l_end)
    const std::size_t l_begin = all_slices ? 0 : slice - 1;
    const std::size_t l_end = all_slices ? L : slice;

    // nnz(S0 > 0) over the slices being fitted
    std::size_t num_voxels = 0;
    for (std::size_t i = l_begin * X * Y; i < l_end * X * Y; ++i)
    {
        if (S0.data[i] > 0)
        {
            ++num_voxels;
        }
    }
    std::printf("Variant: %s\n", VARIANT);
    if (all_slices)
        std::printf("Fitting %zu brain voxels (all %zu slices)\n", num_voxels, L);
    else
        std::printf("Fitting %zu brain voxels (slice %zu of %zu)\n", num_voxels, slice, L);

    // --- Output maps ---
    std::vector<double> D_SUB(X * Y * L, NAN);
    std::vector<double> BETA_SUB(X * Y * L, NAN);
    std::vector<double> ERROR_SUB(X * Y * L, NAN);

    // --- Fit ---
    auto t_start = std::chrono::steady_clock::now();

    for (std::size_t l = l_begin; l < l_end; ++l)
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

#ifdef BEST_SEQUENTIAL
                auto [D, beta, err] = fit_voxel_sub_ml<NB>(svec.data(), BVEC.data(), D_GUESS, BETA_GUESS, D_MIN, D_MAX, BETA_MIN, BETA_MAX);
#else
                auto [D, beta, err] = fit_voxel_sub_ml(svec, BVEC, D_GUESS, BETA_GUESS, D_MIN, D_MAX, BETA_MIN, BETA_MAX);
#endif
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
    const std::size_t l_show = slice - 1; // axial slice to display, 0-based

    const std::string save_path = std::format("{}/slice_{}/", OUTPUT_DIR, slice);
    std::filesystem::create_directories(save_path); // ofstream won't create missing folders
    save_map_png(save_path + "D_SUB.png", D_SUB, X, Y, l_show, 0.0, 3e-3, 5);
    save_map_png(save_path + "RMSE.png", ERROR_SUB, X, Y, l_show, 0.0, 0.1, 5);
    save_map_png(save_path + "BETA.png", BETA_SUB, X, Y, l_show, 0.5, 1.0, 5);
    save_map_png(save_path + "K_STAR.png", K_STAR, X, Y, l_show, 0.0, 3.0, 5);
    std::printf("Saved parameter maps for slice %zu\n", l_show + 1);

    // TODO: write results

    return EXIT_SUCCESS;
}
