// CUDA port of serial_cpp/main.cpp
// Sub-diffusion model fitting (Delta = 19 ms), one GPU thread per voxel

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.cuh"
#include "fit_voxel_sub_ml.cuh"
#include "mittag-leffler.cuh"
#include "read_mri.h"
#include "save_map.h"

// Configuration
const std::string SUBJECT = "sub_001";
const std::string DATATYPE = "dwi_real"; // "_real" or ""
const std::string FOLDER = "./data/dwi_real/";

constexpr std::array<double, 9> BVEC = {0, 50, 350, 800, 1500, 2400, 3450, 4750, 6000}; // s/mm^2
constexpr int NB = BVEC.size();

// Fit bounds and initial guesses
constexpr double D_GUESS = 1e-3, D_MIN = 0.0, D_MAX = 3e-3;        // mm^2/s
constexpr double BETA_GUESS = 0.5, BETA_MIN = 0.0, BETA_MAX = 1.0; // time fractional index

constexpr std::size_t DEFAULT_SLICE = 73; // slice saved as PNG in --all mode, numbered from 1 as in MATLAB

__constant__ double d_bvec[NB];

// One thread fits one voxel. svecs holds n voxels, NB values each (voxel-major).
__global__ void fit_kernel(const double *svecs, int n, double *D, double *beta, double *err)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }

    const double *svec = svecs + i * NB;

    const FitResult fit = fit_voxel_sub_ml<NB>(svec, d_bvec, D_GUESS, BETA_GUESS, D_MIN, D_MAX, BETA_MIN, BETA_MAX);
    D[i] = fit.D;
    beta[i] = fit.beta;
    err[i] = fit.err;
}

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
    // --- Command line (same options as the serial version) ---
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

    // --- Device info ---
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("GPU: %s (compute %d.%d, %zu MiB)\n", prop.name, prop.major, prop.minor,
                prop.totalGlobalMem >> 20);

    // --- Load data (same as serial version) ---
    NiftiVolume mask = read_nifti(FOLDER + SUBJECT + "_" + DATATYPE + "_brainmask.nii.gz");

    NiftiVolume S0 = read_nifti(FOLDER + SUBJECT + "_" + DATATYPE + "_b0_delta0_image.nii");
    for (std::size_t i = 0; i < S0.size(); ++i)
    {
        S0.data[i] *= mask.data[i]; // .* brainmask
    }

    const std::size_t X = S0.X, Y = S0.Y, L = S0.L, V = X * Y * L;
    std::vector<double> S(V * NB);
    std::copy(S0.data.begin(), S0.data.end(), S.begin()); // b=0 first

    for (int k = 1; k < NB; ++k)
    {
        NiftiVolume S_k = read_nifti(FOLDER + SUBJECT + "_" + DATATYPE + "_b" + std::to_string(static_cast<int>(BVEC[k])) + "_delta19_image.nii");
        for (std::size_t i = 0; i < S_k.size(); ++i)
        {
            S[i + k * V] = S_k.data[i] * mask.data[i]; // .* brainmask
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

    // --- Pack brain voxels in those slices into a dense array of normalised signals ---
    std::vector<std::size_t> voxel_idx; // position of each packed voxel in the volume
    std::vector<double> svecs;
    for (std::size_t idx = l_begin * X * Y; idx < l_end * X * Y; ++idx)
    {
        if (S0.data[idx] <= 0.0)
        {
            continue;
        }
        voxel_idx.push_back(idx);
        for (int k = 0; k < NB; ++k)
        {
            svecs.push_back(S[idx + k * V] / S0.data[idx]); // Svec = S(x,y,l,:) / S0(x,y,l)
        }
    }
    const int n = static_cast<int>(voxel_idx.size());
    if (all_slices)
        std::printf("Fitting %d brain voxels (all %zu slices)\n", n, L);
    else
        std::printf("Fitting %d brain voxels (slice %zu of %zu)\n", n, slice, L);

    // --- Copy to device ---
    double *d_svecs, *d_D, *d_beta, *d_err;
    CUDA_CHECK(cudaMalloc(&d_svecs, svecs.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_D, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_beta, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_err, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_svecs, svecs.data(), svecs.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(d_bvec, BVEC.data(), NB * sizeof(double)));
    CUDA_CHECK(ml_device::init_ml_device());

    // --- Fit ---
    auto t_start = std::chrono::steady_clock::now();

    constexpr int BLOCK = 128;
    fit_kernel<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(d_svecs, n, d_D, d_beta, d_err);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t_end = std::chrono::steady_clock::now();
    double fit_time = std::chrono::duration<double>(t_end - t_start).count();
    std::printf("Fit time: %.3f s (%.4f ms/voxel)\n", fit_time, n ? 1e3 * fit_time / n : 0.0);

    // --- Copy back and unpack into full volumes ---
    std::vector<double> D_packed(n), beta_packed(n), err_packed(n);
    CUDA_CHECK(cudaMemcpy(D_packed.data(), d_D, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(beta_packed.data(), d_beta, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(err_packed.data(), d_err, n * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_svecs));
    CUDA_CHECK(cudaFree(d_D));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_err));

    std::vector<double> D_SUB(V, NAN), BETA_SUB(V, NAN), ERROR_SUB(V, NAN);
    for (int i = 0; i < n; ++i)
    {
        D_SUB[voxel_idx[i]] = D_packed[i];
        BETA_SUB[voxel_idx[i]] = beta_packed[i];
        ERROR_SUB[voxel_idx[i]] = err_packed[i];
    }

    // --- Parameter maps ---
    const std::string save_path = "cuda/outputs/slice_" + std::to_string(slice) + "/";
    std::filesystem::create_directories(save_path);
    save_map_png(save_path + "D_SUB.png", D_SUB, X, Y, slice - 1, 0.0, 3e-3, 5);
    save_map_png(save_path + "RMSE.png", ERROR_SUB, X, Y, slice - 1, 0.0, 0.1, 5);
    save_map_png(save_path + "BETA.png", BETA_SUB, X, Y, slice - 1, 0.5, 1.0, 5);
    std::printf("Saved parameter maps for slice %zu\n", slice);

    return EXIT_SUCCESS;
}
