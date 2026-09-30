// C++ port of MATLAB/ml.m: Mittag-Leffler function evaluated by the optimal
// parabolic contour (OPC) algorithm, based on inverting its Laplace transform.
//
//   R. Garrappa, Numerical evaluation of two and three parameter Mittag-Leffler
//   functions, SIAM Journal of Numerical Analysis, 2015, 53(3), 1350-1369
//
// Original MATLAB code: Copyright (c) 2015, Roberto Garrappa, University of Bari, Italy.
// Variable names follow the MATLAB so the two can be read side by side.

#include "mittag-leffler.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace
{

    using cplx = std::complex<double>;

    constexpr double PI = 3.14159265358979323846;
    constexpr double INF = std::numeric_limits<double>::infinity();
    const double LOG_EPS = std::log(DBL_EPSILON); // MATLAB log(eps) = -36.0437

    // Parabolic contour z(u) = mu*(1 + i*u)^2, sampled at u = h*k for k = -N..N
    struct ContourParams
    {
        double mu;
        double h;
        double N; // double so it can be +Inf for an unusable region, as in the MATLAB
    };

    // =========================================================================
    // Finding optimal parameters in a right-bounded region (OptimalParam_RB)
    // =========================================================================
    ContourParams optimal_param_rb(double t, double phi_s_star_j, double phi_s_star_j1,
                                   double pj, double qj, double log_epsilon)
    {
        const double fac = 1.01;
        // The MATLAB's conservative_error_analysis option is off, so only that branch is ported

        // Maximum value of fbar as the ratio between tolerance and round-off unit
        const double f_max = std::exp(log_epsilon - LOG_EPS);

        // Starting values for sq_phi_star_j and sq_phi_star_j1
        const double sq_phi_star_j = std::sqrt(phi_s_star_j);
        const double threshold = 2 * std::sqrt((log_epsilon - LOG_EPS) / t);
        const double sq_phi_star_j1 = std::min(std::sqrt(phi_s_star_j1), threshold - sq_phi_star_j);

        double sq_phibar_star_j = 0, sq_phibar_star_j1 = 0;
        // Left at 1 only when pj and qj are both ~0; unreachable because qj = gama > 0 in
        // every right-bounded region (the MATLAB would error on an undefined f_bar there)
        double f_bar = 1;
        bool adm_region = false;

        if (pj < 1.0e-14 && qj < 1.0e-14)
        {
            // Zero or negative values of pj and qj
            sq_phibar_star_j = sq_phi_star_j;
            sq_phibar_star_j1 = sq_phi_star_j1;
            adm_region = true;
        }
        else if (pj < 1.0e-14)
        {
            // Zero or negative values of just pj
            sq_phibar_star_j = sq_phi_star_j;
            const double f_min = sq_phi_star_j > 0
                                     ? fac * std::pow(sq_phi_star_j / (sq_phi_star_j1 - sq_phi_star_j), qj)
                                     : fac;
            if (f_min < f_max)
            {
                f_bar = f_min + f_min / f_max * (f_max - f_min);
                const double fq = std::pow(f_bar, -1 / qj);
                sq_phibar_star_j1 = (2 * sq_phi_star_j1 - fq * sq_phi_star_j) / (2 + fq);
                adm_region = true;
            }
        }
        else if (qj < 1.0e-14)
        {
            // Zero or negative values of just qj
            sq_phibar_star_j1 = sq_phi_star_j1;
            const double f_min = fac * std::pow(sq_phi_star_j1 / (sq_phi_star_j1 - sq_phi_star_j), pj);
            if (f_min < f_max)
            {
                f_bar = f_min + f_min / f_max * (f_max - f_min);
                const double fp = std::pow(f_bar, -1 / pj);
                sq_phibar_star_j = (2 * sq_phi_star_j + fp * sq_phi_star_j1) / (2 - fp);
                adm_region = true;
            }
        }
        else
        {
            // Positive values of both pj and qj
            double f_min = fac * (sq_phi_star_j + sq_phi_star_j1) /
                           std::pow(sq_phi_star_j1 - sq_phi_star_j, std::max(pj, qj));
            if (f_min < f_max)
            {
                f_min = std::max(f_min, 1.5);
                f_bar = f_min + f_min / f_max * (f_max - f_min);
                const double fp = std::pow(f_bar, -1 / pj);
                const double fq = std::pow(f_bar, -1 / qj);
                const double w = -phi_s_star_j1 * t / log_epsilon;
                const double den = 2 + w - (1 + w) * fp + fq;
                sq_phibar_star_j = ((2 + w + fq) * sq_phi_star_j + fp * sq_phi_star_j1) / den;
                sq_phibar_star_j1 = (-(1 + w) * fq * sq_phi_star_j + (2 + w - (1 + w) * fp) * sq_phi_star_j1) / den;
                adm_region = true;
            }
        }

        if (!adm_region)
            return {0, 0, INF};

        log_epsilon = log_epsilon - std::log(f_bar);
        const double w = -sq_phibar_star_j1 * sq_phibar_star_j1 * t / log_epsilon;
        const double base = (1 + w) * sq_phibar_star_j + sq_phibar_star_j1;
        const double muj = (base / (2 + w)) * (base / (2 + w));
        const double hj = -2 * PI / log_epsilon * (sq_phibar_star_j1 - sq_phibar_star_j) / base;
        const double Nj = std::ceil(std::sqrt(1 - log_epsilon / t / muj) / hj);
        return {muj, hj, Nj};
    }

    // =========================================================================
    // Finding optimal parameters in a right-unbounded region (OptimalParam_RU)
    // =========================================================================
    ContourParams optimal_param_ru(double t, double phi_s_star_j, double pj, double log_epsilon)
    {
        // Starting values for sq_phi_star_j
        const double sq_phi_s_star_j = std::sqrt(phi_s_star_j);
        double phibar_star_j = phi_s_star_j > 0 ? phi_s_star_j * 1.01 : 0.01;
        double sq_phibar_star_j = std::sqrt(phibar_star_j);

        const double f_min = 1, f_max = 10, f_tar = 5;

        // Iterative process to look for fbar in [f_min, f_max]
        double Nj, A, sq_muj;
        while (true)
        {
            const double phi_t = phibar_star_j * t;
            const double log_eps_phi_t = log_epsilon / phi_t;
            Nj = std::ceil(phi_t / PI * (1 - 3 * log_eps_phi_t / 2 + std::sqrt(1 - 2 * log_eps_phi_t)));
            A = PI * Nj / phi_t;
            sq_muj = sq_phibar_star_j * std::abs(4 - A) / std::abs(7 - std::sqrt(1 + 12 * A));
            const double fbar = std::pow((sq_phibar_star_j - sq_phi_s_star_j) / sq_muj, -pj);
            if (pj < 1.0e-14 || (f_min < fbar && fbar < f_max))
                break;
            sq_phibar_star_j = std::pow(f_tar, -1 / pj) * sq_muj + sq_phi_s_star_j;
            phibar_star_j = sq_phibar_star_j * sq_phibar_star_j;
        }
        double muj = sq_muj * sq_muj;
        double hj = (-3 * A - 2 + 2 * std::sqrt(1 + 12 * A)) / (4 - A) / Nj;

        // Adjusting integration parameters to keep round-off errors under control
        const double threshold = (log_epsilon - LOG_EPS) / t;
        if (muj > threshold)
        {
            const double Q = std::abs(pj) < 1.0e-14 ? 0 : std::pow(f_tar, -1 / pj) * std::sqrt(muj);
            phibar_star_j = (Q + std::sqrt(phi_s_star_j)) * (Q + std::sqrt(phi_s_star_j));
            if (phibar_star_j < threshold)
            {
                const double w = std::sqrt(LOG_EPS / (LOG_EPS - log_epsilon));
                const double u = std::sqrt(-phibar_star_j * t / LOG_EPS);
                muj = threshold;
                Nj = std::ceil(w * log_epsilon / 2 / PI / (u * w - 1));
                hj = std::sqrt(LOG_EPS / (LOG_EPS - log_epsilon)) / Nj;
            }
            else
            {
                Nj = INF;
                hj = 0;
            }
        }
        return {muj, hj, Nj};
    }

    // =========================================================================
    // Evaluation of the ML function by Laplace transform inversion (LTInversion)
    // =========================================================================
    cplx lt_inversion(double t, cplx lambda, double alpha, double beta, double gama, double log_epsilon)
    {
        // Evaluation of the relevant poles
        const double theta = std::arg(lambda);
        const double kmin = std::ceil(-alpha / 2 - theta / 2 / PI);
        const double kmax = std::floor(alpha / 2 - theta / 2 / PI);
        std::vector<cplx> poles;
        std::vector<double> phi_poles;
        for (double k = kmin; k <= kmax; ++k)
        {
            const cplx s = std::pow(std::abs(lambda), 1 / alpha) * std::exp(cplx(0, (theta + 2 * k * PI) / alpha));
            poles.push_back(s);
            phi_poles.push_back((s.real() + std::abs(s)) / 2); // phi(s_star)
        }

        // Sorting of the poles according to phi(s_star) (stable, like MATLAB's sort),
        // deleting poles with phi(s_star) = 0 and inserting the origin first
        std::vector<std::size_t> order(poles.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](std::size_t a, std::size_t b) { return phi_poles[a] < phi_poles[b]; });

        std::vector<cplx> s_star{0.0};
        std::vector<double> phi_s_star{0.0};
        for (std::size_t i : order)
        {
            if (phi_poles[i] > 1.0e-15)
            {
                s_star.push_back(poles[i]);
                phi_s_star.push_back(phi_poles[i]);
            }
        }
        const std::size_t J1 = s_star.size(); // singularities, including the origin

        // Strength of the singularities
        std::vector<double> p(J1, gama), q(J1, gama);
        p[0] = std::max(0.0, -2 * (alpha * gama - beta + 1));
        q[J1 - 1] = INF;
        phi_s_star.push_back(INF);

        // Looking for the admissible regions with respect to round-off errors.
        // Region 0 (from the origin) always qualifies, so this is never empty.
        std::vector<std::size_t> admissible_regions;
        for (std::size_t j = 0; j < J1; ++j)
            if (phi_s_star[j] < (log_epsilon - LOG_EPS) / t && phi_s_star[j] < phi_s_star[j + 1])
                admissible_regions.push_back(j);

        // Evaluation of parameters for inversion of LT in each admissible region,
        // relaxing the tolerance until some region needs at most 200 nodes
        std::vector<ContourParams> params(J1, ContourParams{INF, INF, INF});
        std::size_t iN = 0;
        while (true)
        {
            for (std::size_t j1 : admissible_regions)
            {
                params[j1] = (j1 + 1 < J1)
                                 ? optimal_param_rb(t, phi_s_star[j1], phi_s_star[j1 + 1], p[j1], q[j1], log_epsilon)
                                 : optimal_param_ru(t, phi_s_star[j1], p[j1], log_epsilon);
            }

            // Region with the minimum number of nodes (first one on ties, like MATLAB's min)
            iN = 0;
            for (std::size_t j = 1; j < J1; ++j)
                if (params[j].N < params[iN].N)
                    iN = j;

            if (params[iN].N > 200)
                log_epsilon = log_epsilon + std::log(10.0);
            else
                break;
        }
        const double mu = params[iN].mu;
        const double h = params[iN].h;
        const long N = static_cast<long>(params[iN].N);

        // Evaluation of the inverse Laplace transform (trapezoidal rule on the contour)
        cplx sum = 0;
        for (long k = -N; k <= N; ++k)
        {
            const double u = h * static_cast<double>(k);
            const cplx w = cplx(1, u);
            const cplx z = mu * (w * w);
            const cplx zd = cplx(-2 * mu * u, 2 * mu);
            const cplx zexp = std::exp(z * t);
            const cplx F = std::pow(z, alpha * gama - beta) / std::pow(std::pow(z, alpha) - lambda, gama) * zd;
            sum += zexp * F;
        }
        const cplx integral = h * sum / (2 * PI) / cplx(0, 1);

        // Evaluation of residues from the poles to the right of the chosen region
        cplx residues = 0;
        for (std::size_t j = iN + 1; j < J1; ++j)
            residues += 1 / alpha * std::pow(s_star[j], 1 - beta) * std::exp(t * s_star[j]);

        return integral + residues;
    }

} // namespace

std::complex<double> ml(std::complex<double> z, double alpha, double beta, double gama)
{
    // Check whether the parameters meet the constraints (!(x > 0) also rejects NaN)
    if (!(alpha > 0) || !(gama > 0) || !std::isfinite(beta))
        throw std::invalid_argument("ml: alpha and gama must be real and positive, beta must be real");

    // Check parameters and arguments for the three parameter case
    if (std::abs(gama - 1) > DBL_EPSILON)
    {
        if (alpha > 1)
            throw std::invalid_argument("ml: with three parameters, alpha must satisfy 0 < alpha < 1");
        if (std::abs(z) > DBL_EPSILON && std::abs(std::arg(z)) <= alpha * PI)
            throw std::invalid_argument("ml: with three parameters, this code works only when |arg(z)| > alpha*pi");
    }

    // Target precision
    const double log_epsilon = std::log(1.0e-15);

    if (std::abs(z) < 1.0e-15)
        return 1 / std::tgamma(beta);
    return lt_inversion(1, z, alpha, beta, gama, log_epsilon);
}

double ml(double z, double alpha, double beta, double gama)
{
    return ml(std::complex<double>(z, 0.0), alpha, beta, gama).real();
}
