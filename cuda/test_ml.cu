// Checks the device Mittag-Leffler function against the serial ml() over the fit domain:
// values directly, derivatives against central differences of the serial ml().

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.cuh"
#include "mittag-leffler.cuh"
#include "mittag-leffler.h"

__global__ void ml_kernel(const double *z, const double *alpha, int n, ml_device::MlResult *out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = ml_device::ml(z[i], alpha[i]);
    }
}

int main()
{
    // z = -b * D covers [-6000 * 3e-3, 0] = [-18, 0]; alpha covers the beta bounds [1e-6, 1]
    std::vector<double> z, alpha;
    for (int i = 0; i <= 200; ++i)
    {
        for (int j = 0; j <= 100; ++j)
        {
            z.push_back(-18.0 * i / 200);
            alpha.push_back(std::max(1e-6, j / 100.0));
        }
    }
    const int n = static_cast<int>(z.size());

    CUDA_CHECK(ml_device::init_ml_device());
    double *d_z, *d_alpha;
    ml_device::MlResult *d_out;
    CUDA_CHECK(cudaMalloc(&d_z, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_alpha, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(ml_device::MlResult)));
    CUDA_CHECK(cudaMemcpy(d_z, z.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_alpha, alpha.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    ml_kernel<<<(n + 127) / 128, 128>>>(d_z, d_alpha, n, d_out);
    CUDA_CHECK(cudaGetLastError());
    std::vector<ml_device::MlResult> out(n);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(ml_device::MlResult), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_alpha));
    CUDA_CHECK(cudaFree(d_out));

    double max_E = 0, max_dz = 0, max_da = 0;
    for (int i = 0; i < n; ++i)
    {
        const double a = alpha[i];
        max_E = std::max(max_E, std::abs(out[i].E - ml(z[i], a)));

        const double hz = 1e-5;
        const double fd_dz = (ml(z[i] + hz, a) - ml(z[i] - hz, a)) / (2 * hz);
        max_dz = std::max(max_dz, std::abs(out[i].dE_dz - fd_dz));

        // Keep alpha inside (0, 1], one-sided at the ends of the range
        const double a_lo = std::max(a - 1e-6, a / 2), a_hi = std::min(a + 1e-6, 1.0);
        const double fd_da = (ml(z[i], a_hi) - ml(z[i], a_lo)) / (a_hi - a_lo);
        max_da = std::max(max_da, std::abs(out[i].dE_dalp - fd_da));
    }

    std::printf("%d points, max abs difference from serial ml():\n", n);
    std::printf("  E        %.3e\n", max_E);
    std::printf("  dE/dz    %.3e (vs central difference)\n", max_dz);
    std::printf("  dE/dalpha %.3e (vs central difference)\n", max_da);

    const bool ok = max_E < 1e-13 && max_dz < 1e-6 && max_da < 1e-6;
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
