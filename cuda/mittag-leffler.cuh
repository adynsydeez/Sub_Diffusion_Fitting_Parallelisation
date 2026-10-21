// Device Mittag-Leffler function E_alpha(z) for real z <= 0 and 0 < alpha <= 1,
// with its derivatives in z and alpha, for the sub-diffusion model S0 * E_beta(-b * D).
//
// This is serial_cpp/mittag-leffler.cpp specialised to that case. With beta = gama = 1,
// z real and negative and alpha <= 1, the OPC algorithm finds no poles to the right of
// the contour, so it always picks the right-unbounded region from the origin with the
// same parameters (mu, h, N) whatever z and alpha are, and adds no residues:
//   E_alpha(z) = h / (2 pi i) * sum_{k=-N..N} exp(s_k) s_k^(alpha-1) / (s_k^alpha - z) * s'_k
// with s_k = mu * (1 + i u_k)^2 and u_k = h * k. The nodes are fixed, so everything that
// doesn't depend on z or alpha is precomputed into constant memory by init_ml_device().
// Nodes k and -k are complex conjugates, so only k = 0..N are summed.
//
// The same code also compiles as plain C++ (no nvcc), so the CPU "best sequential" version
// (serial_cpp, target fit_sub_best) runs exactly this algorithm. On the host the node
// tables live in an ordinary array that is filled on first use.

#pragma once

#include <cmath>
#include <complex>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define ML_HD __host__ __device__
#else
#define ML_HD
#endif

namespace ml_device
{
    // Contour chosen by optimal_param_ru() for t = 1, phi_s_star = 0, pj = 0, log_epsilon = log(1e-15)
    constexpr double MU = 1.5048769942064695;
    constexpr double H = 0.18125923248023865;
    constexpr int N = 27;
    constexpr int NODES = N + 1; // k = 0..N

    // Per node: w = exp(s) * s' / s (complex), L = log(s) (complex), c = quadrature weight
    struct NodeTables
    {
        double w_re[NODES], w_im[NODES], L_re[NODES], L_im[NODES], c[NODES];
    };

    inline NodeTables make_node_tables()
    {
        using cplx = std::complex<double>;
        constexpr double PI = 3.14159265358979323846;

        NodeTables t;
        for (int k = 0; k < NODES; ++k)
        {
            const double u = H * k;
            const cplx s = MU * cplx(1, u) * cplx(1, u);
            const cplx ds = cplx(-2 * MU * u, 2 * MU);
            const cplx w = std::exp(s) * ds / s;
            const cplx L = std::log(s); // arg(s) = 2 atan(u) stays inside (-pi, pi)
            t.w_re[k] = w.real();
            t.w_im[k] = w.imag();
            t.L_re[k] = L.real();
            t.L_im[k] = L.imag();
            t.c[k] = (k == 0 ? 1.0 : 2.0) * H / (2 * PI); // k > 0 also stands in for node -k
        }
        return t;
    }

    // Host copy of the tables, used when ml() runs on the CPU
    inline const NodeTables host_tables = make_node_tables();

#ifdef __CUDACC__
    // Device copy in constant memory, used when ml() runs on the GPU
    __constant__ double w_re[NODES], w_im[NODES], L_re[NODES], L_im[NODES], c[NODES];

    // Fills the device tables. Call once on the host before launching any kernel that uses ml().
    inline cudaError_t init_ml_device()
    {
        const NodeTables &t = host_tables;
        cudaError_t err;
        if ((err = cudaMemcpyToSymbol(w_re, t.w_re, sizeof t.w_re)) != cudaSuccess) return err;
        if ((err = cudaMemcpyToSymbol(w_im, t.w_im, sizeof t.w_im)) != cudaSuccess) return err;
        if ((err = cudaMemcpyToSymbol(L_re, t.L_re, sizeof t.L_re)) != cudaSuccess) return err;
        if ((err = cudaMemcpyToSymbol(L_im, t.L_im, sizeof t.L_im)) != cudaSuccess) return err;
        return cudaMemcpyToSymbol(c, t.c, sizeof t.c);
    }
#endif

// Table lookup that reads constant memory directly on the device and the host array otherwise
#ifdef __CUDA_ARCH__
#define ML_TABLE(name) name
#else
#define ML_TABLE(name) host_tables.name
#endif

    struct MlResult
    {
        double E;       // E_alpha(z)
        double dE_dz;   // dE/dz
        double dE_dalp; // dE/dalpha
    };

    // E_alpha(z) and its derivatives, for z <= 0 and 0 < alpha <= 1.
    // The derivatives are those of the quadrature itself, so they match E exactly.
    ML_HD inline MlResult ml(double z, double alpha)
    {
        double E = 0, dz = 0, da = 0;
        for (int k = 0; k < NODES; ++k)
        {
            const double Lre = ML_TABLE(L_re)[k], Lim = ML_TABLE(L_im)[k];
            const double wre = ML_TABLE(w_re)[k], wim = ML_TABLE(w_im)[k], ck = ML_TABLE(c)[k];

            // P = s^alpha = exp(alpha * L)
            double sn, cs;
#ifdef __CUDA_ARCH__
            sincos(alpha * Lim, &sn, &cs);
#else
            sn = std::sin(alpha * Lim);
            cs = std::cos(alpha * Lim);
#endif
            const double mag = exp(alpha * Lre);
            const double P_re = mag * cs, P_im = mag * sn;

            // R = 1 / (P - z)
            const double d_re = P_re - z, d_im = P_im;
            const double inv = 1 / (d_re * d_re + d_im * d_im);
            const double R_re = d_re * inv, R_im = -d_im * inv;

            // T = w * P * R      (term of E)
            // U = T * R          (term of dE/dz:     d/dz [P/(P-z)]     = P/(P-z)^2)
            // V = -z * U * L     (term of dE/dalpha: d/dalpha [P/(P-z)] = -z P L/(P-z)^2)
            const double wP_re = wre * P_re - wim * P_im;
            const double wP_im = wre * P_im + wim * P_re;
            const double T_re = wP_re * R_re - wP_im * R_im;
            const double T_im = wP_re * R_im + wP_im * R_re;
            const double U_re = T_re * R_re - T_im * R_im;
            const double U_im = T_re * R_im + T_im * R_re;
            const double V_im = -z * (U_re * Lim + U_im * Lre);

            // Conjugate pairs cancel the real parts; dividing by i leaves the imaginary part
            E += ck * T_im;
            dz += ck * U_im;
            da += ck * V_im;
        }

        if (fabs(z) < 1.0e-15)
        {
            E = 1; // 1 / Gamma(1), as ml() returns for |z| < 1e-15
        }
        return {E, dz, da};
    }

} // namespace ml_device
