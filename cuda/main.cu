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

constexpr int BLOCK = 128;   // threads per block
constexpr int MAX_SM_ID = 1024; // upper bound for %smid when sizing the allowed-SM table

// Fits voxel i. svecs holds n voxels, NB values each (voxel-major).
__device__ inline void fit_one(const double *svecs, int i, double *D, double *beta, double *err)
{
    const FitResult fit = fit_voxel_sub_ml<NB>(svecs + i * NB, d_bvec, D_GUESS, BETA_GUESS, D_MIN, D_MAX, BETA_MIN, BETA_MAX);
    D[i] = fit.D;
    beta[i] = fit.beta;
    err[i] = fit.err;
}

// Default: one thread per voxel, using the whole GPU
__global__ void fit_kernel(const double *svecs, int n, double *D, double *beta, double *err)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        fit_one(svecs, i, D, beta, err);
    }
}

// The SM this thread is running on (not guaranteed to be numbered contiguously)
__device__ inline unsigned sm_id()
{
    unsigned id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    return id;
}

// Marks every SM that runs a block, to find the SM ids that exist on this GPU
__global__ void record_sms(int *seen)
{
    if (threadIdx.x == 0)
    {
        seen[sm_id()] = 1;
    }
    const long long start = clock64();
    while (clock64() - start < 100000) // keep the block busy so later blocks spread to other SMs
    {
    }
}

// --sms N: launched with exactly as many blocks as fit on the GPU at once, so every SM holds
// a full set. Blocks on SMs not allowed exit at once; the rest take BLOCK voxels at a time
// from a shared counter until all are done, so only the allowed SMs do any work.
__global__ void fit_kernel_sms(const double *svecs, int n, double *D, double *beta, double *err,
                               const int *sm_allowed, int *next)
{
    if (!sm_allowed[sm_id()])
    {
        return;
    }

    __shared__ int base;
    while (true)
    {
        __syncthreads(); // everyone has read the previous base
        if (threadIdx.x == 0)
        {
            base = atomicAdd(next, BLOCK);
        }
        __syncthreads();

        if (base >= n)
        {
            return;
        }
        const int i = base + threadIdx.x;
        if (i < n)
        {
            fit_one(svecs, i, D, beta, err);
        }
    }
}

void print_usage(const char *prog)
{
    std::fprintf(stderr,
                 "Usage: %s [--all | --slice N] [--sms N] [--out DIR]\n"
                 "  --all      fit every brain voxel (default; for speed testing)\n"
                 "  --slice N  fit only axial slice N, numbered from 1 as in MATLAB (for visual checks)\n"
                 "  --sms N    use only N streaming multiprocessors (default: all; for scaling tests)\n"
                 "  --out DIR  results folder (default cuda/outputs); written to DIR/all or DIR/slice_N\n",
                 prog);
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
    // --- Command line (same options as the serial version, plus --sms) ---
    bool all_slices = true;
    std::size_t slice = DEFAULT_SLICE; // 1-based, as in MATLAB
    std::size_t sms = 0;               // 0 = all SMs, default kernel
    std::string out_dir = "cuda/outputs";

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
        else if (opt == "--sms" && a + 1 < argc)
        {
            if (!parse_count(argv[++a], sms))
            {
                std::fprintf(stderr, "Invalid SM count: %s\n", argv[a]);
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

    // --- Device info ---
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("GPU: %s (compute %d.%d, %d SMs, %zu MiB)\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount, prop.totalGlobalMem >> 20);
    if (sms > static_cast<std::size_t>(prop.multiProcessorCount))
    {
        std::fprintf(stderr, "--sms %zu is more than the %d SMs on this GPU\n", sms, prop.multiProcessorCount);
        return EXIT_FAILURE;
    }

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

    // --- SM selection for --sms N ---
    int *d_sm_allowed = nullptr, *d_next = nullptr;
    int sm_grid = 0;
    if (sms > 0)
    {
        // Find the SM ids that exist, then allow the first N of them
        int *d_seen;
        CUDA_CHECK(cudaMalloc(&d_seen, MAX_SM_ID * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_seen, 0, MAX_SM_ID * sizeof(int)));
        record_sms<<<prop.multiProcessorCount * 32, 32>>>(d_seen);
        CUDA_CHECK(cudaGetLastError());
        std::vector<int> seen(MAX_SM_ID);
        CUDA_CHECK(cudaMemcpy(seen.data(), d_seen, MAX_SM_ID * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_seen));

        std::vector<int> allowed(MAX_SM_ID, 0);
        std::size_t found = 0, used = 0;
        for (int id = 0; id < MAX_SM_ID; ++id)
        {
            if (seen[id])
            {
                ++found;
                if (used < sms)
                {
                    allowed[id] = 1;
                    ++used;
                }
            }
        }
        if (found != static_cast<std::size_t>(prop.multiProcessorCount))
        {
            std::fprintf(stderr, "Warning: found %zu SM ids, expected %d\n", found, prop.multiProcessorCount);
        }

        CUDA_CHECK(cudaMalloc(&d_sm_allowed, MAX_SM_ID * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_sm_allowed, allowed.data(), MAX_SM_ID * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_next, sizeof(int)));

        // Exactly as many blocks as can be resident at once, so every SM gets a full set
        int blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, fit_kernel_sms, BLOCK, 0));
        sm_grid = blocks_per_sm * prop.multiProcessorCount;
        std::printf("Using %zu of %d SMs (%d blocks of %d threads per SM)\n", used, prop.multiProcessorCount,
                    blocks_per_sm, BLOCK);
    }

    // Warm-up with no voxels: CUDA loads kernels lazily on first launch, which shouldn't be timed
    if (sms > 0)
    {
        CUDA_CHECK(cudaMemset(d_next, 0, sizeof(int)));
        fit_kernel_sms<<<sm_grid, BLOCK>>>(d_svecs, 0, d_D, d_beta, d_err, d_sm_allowed, d_next);
    }
    else
    {
        fit_kernel<<<1, BLOCK>>>(d_svecs, 0, d_D, d_beta, d_err);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (sms > 0)
    {
        CUDA_CHECK(cudaMemset(d_next, 0, sizeof(int)));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // --- Fit ---
    auto t_start = std::chrono::steady_clock::now();

    if (sms > 0)
    {
        fit_kernel_sms<<<sm_grid, BLOCK>>>(d_svecs, n, d_D, d_beta, d_err, d_sm_allowed, d_next);
    }
    else
    {
        fit_kernel<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(d_svecs, n, d_D, d_beta, d_err);
    }
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
    CUDA_CHECK(cudaFree(d_sm_allowed)); // no-op for nullptr
    CUDA_CHECK(cudaFree(d_next));

    std::vector<double> D_SUB(V, NAN), BETA_SUB(V, NAN), ERROR_SUB(V, NAN);
    for (int i = 0; i < n; ++i)
    {
        D_SUB[voxel_idx[i]] = D_packed[i];
        BETA_SUB[voxel_idx[i]] = beta_packed[i];
        ERROR_SUB[voxel_idx[i]] = err_packed[i];
    }

    // --- Derived parameters (as in the serial version) ---
    std::vector<double> K_STAR(V, NAN);
    for (std::size_t i = 0; i < V; ++i)
    {
        const double b = BETA_SUB[i];
        K_STAR[i] = 6.0 * std::pow(std::tgamma(1.0 + b), 2) / std::tgamma(1.0 + 2.0 * b) - 3.0;
    }

    // --- Results: <out>/all/ or <out>/slice_N/ ---
    const std::string run_name = all_slices ? std::string("all") : "slice_" + std::to_string(slice);
    const std::string save_path = out_dir + "/" + run_name + "/";
    std::filesystem::create_directories(save_path);

    // Parameter maps as NIfTI (same geometry as the input), for comparing versions
    write_nifti(save_path + "D_SUB.nii", S0, D_SUB);
    write_nifti(save_path + "BETA_SUB.nii", S0, BETA_SUB);
    write_nifti(save_path + "ERROR_SUB.nii", S0, ERROR_SUB);
    write_nifti(save_path + "K_STAR.nii", S0, K_STAR);

    // Run details, read by the timing and comparison scripts
    if (FILE *f = std::fopen((save_path + "run.txt").c_str(), "w"))
    {
        std::fprintf(f, "variant=cuda\nsms=%d\nvoxels=%d\nfit_time_s=%.6f\n",
                     sms ? static_cast<int>(sms) : prop.multiProcessorCount, n, fit_time);
        std::fclose(f);
    }

    // One slice as PNGs, for checking by eye
    save_map_png(save_path + "D_SUB.png", D_SUB, X, Y, slice - 1, 0.0, 3e-3, 5);
    save_map_png(save_path + "RMSE.png", ERROR_SUB, X, Y, slice - 1, 0.0, 0.1, 5);
    save_map_png(save_path + "BETA.png", BETA_SUB, X, Y, slice - 1, 0.5, 1.0, 5);
    std::printf("Saved results to %s (maps shown for slice %zu)\n", save_path.c_str(), slice);

    return EXIT_SUCCESS;
}
