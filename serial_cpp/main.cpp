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

// --threads is only offered when built with OpenMP (fit_sub_best)
#ifdef _OPENMP
constexpr bool HAS_THREADS = true;
#else
constexpr bool HAS_THREADS = false;
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
                 "Usage: %s [--all | --slice N] [--out DIR]%s\n"
                 "  --all        fit every brain voxel (default; for speed testing)\n"
                 "  --slice N    fit only axial slice N, numbered from 1 as in MATLAB (for visual checks)\n"
                 "  --out DIR    results folder (default %s); written to DIR/all or DIR/slice_N\n",
                 prog, HAS_THREADS ? " [--threads N]" : "", OUTPUT_DIR);
    if (HAS_THREADS)
        std::fprintf(stderr, "  --threads N  number of CPU threads (default 1)\n");
}

// Parses a whole positive integer; false for anything else ("abc", "-3", "12x", "0")
bool parse_count(const char *s, std::size_t &out)
{
    const std::string arg = s;
    std::size_t pos = 0;
    try
    {
        out = std::stoul(arg, &pos);
    }
    catch (const std::exception &)
    {
        return false;
    }
    return pos == arg.size() && arg[0] != '-' && out > 0;
}

int main(int argc, char *argv[])
{
    // --- Command line ---
    bool all_slices = true;
    std::size_t slice = DEFAULT_SLICE; // 1-based, as in MATLAB
    std::size_t threads = 1;
    std::string out_dir = OUTPUT_DIR;

    for (int a = 1; a < argc; ++a)
    {
        const std::string opt = argv[a];
        if (opt == "--all")
        {
            all_slices = true;
        }
        else if (opt == "--slice" && a + 1 < argc)
        {
            if (!parse_count(argv[++a], slice))
            {
                std::fprintf(stderr, "Invalid slice number: %s\n", argv[a]);
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            all_slices = false;
        }
        else if (opt == "--out" && a + 1 < argc)
        {
            out_dir = argv[++a];
        }
        else if (opt == "--threads" && a + 1 < argc && HAS_THREADS)
        {
            if (!parse_count(argv[++a], threads))
            {
                std::fprintf(stderr, "Invalid thread count: %s\n", argv[a]);
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
        }
        else
        {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
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
    if (HAS_THREADS)
        std::printf("Threads: %zu\n", threads);
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

    // One flat loop over the voxels of the selected slices; idx = x + X*(y + Y*l) is S0(x,y,l).
    // Voxels are independent, so with OpenMP (fit_sub_best) threads share them out. Dynamic
    // scheduling because voxels need different numbers of solver iterations.
    const std::ptrdiff_t idx_begin = static_cast<std::ptrdiff_t>(l_begin * X * Y);
    const std::ptrdiff_t idx_end = static_cast<std::ptrdiff_t>(l_end * X * Y);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 256) num_threads(static_cast<int>(threads))
#endif
    for (std::ptrdiff_t i = idx_begin; i < idx_end; ++i)
    {
        const std::size_t idx = static_cast<std::size_t>(i);

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

    auto t_end = std::chrono::steady_clock::now();
    double fit_time = std::chrono::duration<double>(t_end - t_start).count();
    std::printf("Fit time: %.3f s (%.4f ms/voxel)\n", fit_time,
                num_voxels ? 1e3 * fit_time / num_voxels : 0.0);

    // --- Derived parameters ---
    std::vector<double> K_STAR(X * Y * L, NAN);
    for (std::size_t i = 0; i < K_STAR.size(); ++i)
    {
        double b = BETA_SUB[i];
        K_STAR[i] = 6.0 * std::pow(std::tgamma(1.0 + b), 2) / std::tgamma(1.0 + 2.0 * b) - 3.0;
    }

    // --- Results: <out>/all/ or <out>/slice_N/ ---
    const std::string save_path = std::format("{}/{}/", out_dir, all_slices ? "all" : std::format("slice_{}", slice));
    std::filesystem::create_directories(save_path); // ofstream won't create missing folders

    // Parameter maps as NIfTI (same geometry as the input), for comparing versions
    write_nifti(save_path + "D_SUB.nii", S0, D_SUB);
    write_nifti(save_path + "BETA_SUB.nii", S0, BETA_SUB);
    write_nifti(save_path + "ERROR_SUB.nii", S0, ERROR_SUB);
    write_nifti(save_path + "K_STAR.nii", S0, K_STAR);

    // Run details, read by the timing and comparison scripts
    if (FILE *f = std::fopen((save_path + "run.txt").c_str(), "w"))
    {
        std::fprintf(f, "variant=%s\nthreads=%zu\nvoxels=%zu\nfit_time_s=%.6f\n", VARIANT, threads, num_voxels,
                     fit_time);
        std::fclose(f);
    }

    // One slice as PNGs, for checking by eye
    const std::size_t l_show = slice - 1; // axial slice to display, 0-based
    save_map_png(save_path + "D_SUB.png", D_SUB, X, Y, l_show, 0.0, 3e-3, 5);
    save_map_png(save_path + "RMSE.png", ERROR_SUB, X, Y, l_show, 0.0, 0.1, 5);
    save_map_png(save_path + "BETA.png", BETA_SUB, X, Y, l_show, 0.5, 1.0, 5);
    save_map_png(save_path + "K_STAR.png", K_STAR, X, Y, l_show, 0.0, 3.0, 5);
    std::printf("Saved results to %s (maps shown for slice %zu)\n", save_path.c_str(), l_show + 1);

    return EXIT_SUCCESS;
}
