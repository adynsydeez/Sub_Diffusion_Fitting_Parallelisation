// Fits one slice with both the GPU solver and the serial Ceres solver, and compares the
// results voxel by voxel. Run from the repository root, like fit_sub_cuda.
//
// Exits with failure if the GPU fit is worse than Ceres (higher RMSE) by more than
// RMSE_TOLERANCE in any voxel, so it can be rerun as a regression check after changes.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.cuh"
#include "fit_voxel_sub_ml.cuh"
#include "mittag-leffler.cuh"
#include "read_mri.h"

// Serial Ceres fit, returning {D, beta, err} (see serial_fit.cpp)
std::array<double, 3> serial_fit(const std::array<double, 9> &svec, const std::array<double, 9> &bvec,
                                 double D_guess, double beta_guess, double D_min, double D_max,
                                 double beta_min, double beta_max);

// Configuration (same as main.cu)
const std::string SUBJECT = "sub_001";
const std::string DATATYPE = "dwi_real"; // "_real" or ""
const std::string FOLDER = "./data/dwi_real/";

constexpr std::array<double, 9> BVEC = {0, 50, 350, 800, 1500, 2400, 3450, 4750, 6000}; // s/mm^2
constexpr int NB = BVEC.size();

constexpr double D_GUESS = 1e-3, D_MIN = 0.0, D_MAX = 3e-3;        // mm^2/s
constexpr double BETA_GUESS = 0.5, BETA_MIN = 0.0, BETA_MAX = 1.0; // time fractional index

constexpr std::size_t DEFAULT_SLICE = 73; // numbered from 1 as in MATLAB
constexpr double RMSE_TOLERANCE = 1e-4;   // largest acceptable GPU RMSE above Ceres's
constexpr int NUM_WORST = 5;              // voxels listed with the largest beta differences

__constant__ double d_bvec[NB];

__global__ void fit_kernel(const double *svecs, int n, FitResult *out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = fit_voxel_sub_ml<NB>(svecs + i * NB, d_bvec, D_GUESS, BETA_GUESS, D_MIN, D_MAX, BETA_MIN, BETA_MAX);
    }
}

// p-th quantile (0..1) of v, which must be non-empty
double quantile(std::vector<double> v, double p)
{
    std::sort(v.begin(), v.end());
    return v[static_cast<std::size_t>(p * (v.size() - 1))];
}

void print_stats(const char *name, const std::vector<double> &diffs)
{
    std::printf("  %-8s median %.2e   p99 %.2e   max %.2e\n", name, quantile(diffs, 0.5), quantile(diffs, 0.99),
                quantile(diffs, 1.0));
}

int main(int argc, char *argv[])
{
    // --- Command line ---
    std::size_t slice = DEFAULT_SLICE;
    if (argc > 2 || (argc == 2 && (slice = std::strtoul(argv[1], nullptr, 10)) == 0))
    {
        std::fprintf(stderr, "Usage: %s [SLICE]\n  SLICE  axial slice to compare, numbered from 1 (default %zu)\n",
                     argv[0], DEFAULT_SLICE);
        return EXIT_FAILURE;
    }

    // --- Load data (same as main.cu) ---
    NiftiVolume mask = read_nifti(FOLDER + SUBJECT + "_" + DATATYPE + "_brainmask.nii.gz");
    NiftiVolume S0 = read_nifti(FOLDER + SUBJECT + "_" + DATATYPE + "_b0_delta0_image.nii");
    std::vector<NiftiVolume> S_k(NB);
    for (int k = 1; k < NB; ++k)
    {
        S_k[k] = read_nifti(FOLDER + SUBJECT + "_" + DATATYPE + "_b" + std::to_string(static_cast<int>(BVEC[k])) + "_delta19_image.nii");
    }

    const std::size_t X = S0.X, Y = S0.Y, L = S0.L;
    if (slice > L)
    {
        std::fprintf(stderr, "Slice %zu is out of range: the volume has slices 1 to %zu\n", slice, L);
        return EXIT_FAILURE;
    }

    // --- Pack the slice's brain voxels into normalised signals ---
    std::vector<std::size_t> voxel_idx;
    std::vector<std::array<double, NB>> svecs;
    for (std::size_t idx = (slice - 1) * X * Y; idx < slice * X * Y; ++idx)
    {
        const double s0 = S0.data[idx] * mask.data[idx];
        if (s0 <= 0.0)
        {
            continue;
        }
        std::array<double, NB> svec;
        svec[0] = 1.0;
        for (int k = 1; k < NB; ++k)
        {
            svec[k] = S_k[k].data[idx] * mask.data[idx] / s0;
        }
        voxel_idx.push_back(idx);
        svecs.push_back(svec);
    }
    const int n = static_cast<int>(svecs.size());
    if (n == 0)
    {
        std::printf("Slice %zu has no brain voxels, nothing to compare\n", slice);
        return EXIT_SUCCESS;
    }
    std::printf("Comparing %d brain voxels in slice %zu\n", n, slice);

    // --- GPU fit ---
    double *d_svecs;
    FitResult *d_out;
    CUDA_CHECK(cudaMalloc(&d_svecs, n * NB * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(FitResult)));
    CUDA_CHECK(cudaMemcpy(d_svecs, svecs.data(), n * NB * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(d_bvec, BVEC.data(), NB * sizeof(double)));
    CUDA_CHECK(ml_device::init_ml_device());

    auto t_start = std::chrono::steady_clock::now();
    constexpr int BLOCK = 128;
    fit_kernel<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(d_svecs, n, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const double gpu_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

    std::vector<FitResult> gpu(n);
    CUDA_CHECK(cudaMemcpy(gpu.data(), d_out, n * sizeof(FitResult), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_svecs));
    CUDA_CHECK(cudaFree(d_out));

    // --- Serial Ceres fit ---
    t_start = std::chrono::steady_clock::now();
    std::vector<FitResult> cpu(n);
    for (int i = 0; i < n; ++i)
    {
        const auto [D, beta, err] = serial_fit(svecs[i], BVEC, D_GUESS, BETA_GUESS, D_MIN, D_MAX, BETA_MIN, BETA_MAX);
        cpu[i] = {D, beta, err};
    }
    const double cpu_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

    std::printf("Fit time: GPU %.3f s, Ceres %.2f s (%.0fx)\n", gpu_time, cpu_time, cpu_time / gpu_time);

    // --- Compare ---
    std::vector<double> diff_D, diff_beta, diff_err;
    std::vector<int> finite; // voxels where both fits are finite
    int non_finite = 0, gpu_lower = 0, cpu_lower = 0;
    double worst_excess = -INFINITY; // largest GPU RMSE minus Ceres RMSE
    for (int i = 0; i < n; ++i)
    {
        if (!std::isfinite(gpu[i].D + gpu[i].beta + gpu[i].err) || !std::isfinite(cpu[i].D + cpu[i].beta + cpu[i].err))
        {
            ++non_finite;
            continue;
        }
        finite.push_back(i);
        diff_D.push_back(std::abs(gpu[i].D - cpu[i].D));
        diff_beta.push_back(std::abs(gpu[i].beta - cpu[i].beta));
        diff_err.push_back(std::abs(gpu[i].err - cpu[i].err));

        const double excess = gpu[i].err - cpu[i].err;
        worst_excess = std::max(worst_excess, excess);
        gpu_lower += excess < -1e-6;
        cpu_lower += excess > 1e-6;
    }

    if (non_finite > 0)
    {
        std::printf("%d voxels with a NaN or infinite result in either fit (excluded below)\n", non_finite);
    }
    if (finite.empty())
    {
        std::printf("FAIL: no voxels with finite results in both fits\n");
        return EXIT_FAILURE;
    }

    std::printf("Absolute differences, GPU vs Ceres:\n");
    print_stats("D", diff_D);
    print_stats("beta", diff_beta);
    print_stats("RMSE", diff_err);
    std::printf("Lower RMSE by more than 1e-6: GPU in %d voxels, Ceres in %d voxels\n", gpu_lower, cpu_lower);
    std::printf("Largest GPU RMSE above Ceres: %.2e\n", worst_excess);

    // Largest beta differences, usually noisy voxels where the cost surface is flat
    std::vector<int> order(finite.size());
    for (std::size_t j = 0; j < order.size(); ++j)
    {
        order[j] = static_cast<int>(j);
    }
    const int num_worst = std::min<int>(NUM_WORST, static_cast<int>(order.size()));
    std::partial_sort(order.begin(), order.begin() + num_worst, order.end(),
                      [&](int a, int b) { return diff_beta[a] > diff_beta[b]; });
    std::printf("Largest beta differences (x, y numbered from 1):\n");
    for (int j = 0; j < num_worst; ++j)
    {
        const int i = finite[order[j]];
        const std::size_t x = voxel_idx[i] % X, y = voxel_idx[i] / X % Y;
        std::printf("  (%3zu, %3zu)  GPU D=%.4e beta=%.4f RMSE=%.5f | Ceres D=%.4e beta=%.4f RMSE=%.5f\n",
                    x + 1, y + 1, gpu[i].D, gpu[i].beta, gpu[i].err, cpu[i].D, cpu[i].beta, cpu[i].err);
    }

    const bool ok = non_finite == 0 && worst_excess <= RMSE_TOLERANCE;
    std::printf(ok ? "PASS\n" : "FAIL: GPU fit worse than Ceres beyond tolerance %.0e\n", RMSE_TOLERANCE);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
