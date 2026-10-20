// Device version of serial_cpp/fit_voxel_sub_ml.cpp: fits S0 * E_beta(-b * D) to one voxel.
//
// Ceres doesn't run on the GPU, so this is a small bounded Levenberg-Marquardt solver for
// the two parameters, written to follow Ceres's defaults closely: steps are projected onto
// the bounds, the damping starts at 1 / initial_trust_region_radius, and it stops on the
// same function, gradient and parameter tolerances. The Jacobian is analytic (from
// ml_device::ml) rather than Ceres's central differences, so results agree with the serial
// version to within the fit tolerance, not bit for bit.

#pragma once

#include "mittag-leffler.cuh"

struct FitResult
{
    double D;
    double beta;
    double err;
};

namespace fit_device
{
    // Ceres defaults, with function_tolerance set as in the serial version
    constexpr int MAX_ITERATIONS = 50;
    constexpr double FUNCTION_TOLERANCE = 1e-5;
    constexpr double GRADIENT_TOLERANCE = 1e-10;
    constexpr double PARAMETER_TOLERANCE = 1e-8;
    constexpr double INITIAL_MU = 1e-4;           // 1 / initial_trust_region_radius
    constexpr double MIN_RELATIVE_DECREASE = 1e-3; // smallest actual/predicted reduction accepted
    constexpr double MIN_DIAGONAL = 1e-6, MAX_DIAGONAL = 1e32;

    __device__ inline double clamp(double x, double lo, double hi) { return fmin(fmax(x, lo), hi); }

    // Residuals r_i = S0 * E_beta(-b_i * D) - S_i and their Jacobian, at p = [D, beta].
    // Returns the cost 0.5 * sum(r^2), as Ceres defines it.
    template <int NB>
    __device__ double evaluate(const double p[2], const double *svec, const double *bvec,
                               double r[NB], double J[NB][2])
    {
        double cost = 0;
        for (int i = 0; i < NB; ++i)
        {
            const ml_device::MlResult m = ml_device::ml(-bvec[i] * p[0], p[1]);
            r[i] = svec[0] * m.E - svec[i];
            J[i][0] = -bvec[i] * svec[0] * m.dE_dz;
            J[i][1] = svec[0] * m.dE_dalp;
            cost += 0.5 * r[i] * r[i];
        }
        return cost;
    }
} // namespace fit_device

template <int NB>
__device__ FitResult fit_voxel_sub_ml(const double *svec, const double *bvec, double D_guess, double beta_guess,
                                      double D_min, double D_max, double beta_min, double beta_max)
{
    using namespace fit_device;

    if (svec[0] <= 0)
    {
        return {0, 0, 0};
    }

    const double lo[2] = {D_min, fmax(beta_min, 1e-6)}; // as in the serial version
    const double hi[2] = {D_max, beta_max};
    double p[2] = {clamp(D_guess, lo[0], hi[0]), clamp(beta_guess, lo[1], hi[1])};

    double r[NB], J[NB][2];
    double cost = evaluate<NB>(p, svec, bvec, r, J);
    double mu = INITIAL_MU, nu = 2;

    for (int iter = 0; iter < MAX_ITERATIONS && cost > 0; ++iter)
    {
        // Normal equations: A = J'J, g = J'r
        double A00 = 0, A01 = 0, A11 = 0, g0 = 0, g1 = 0;
        for (int i = 0; i < NB; ++i)
        {
            A00 += J[i][0] * J[i][0];
            A01 += J[i][0] * J[i][1];
            A11 += J[i][1] * J[i][1];
            g0 += J[i][0] * r[i];
            g1 += J[i][1] * r[i];
        }

        // Projected gradient small enough: at a (possibly bounded) minimum
        const double pg0 = p[0] - clamp(p[0] - g0, lo[0], hi[0]);
        const double pg1 = p[1] - clamp(p[1] - g1, lo[1], hi[1]);
        if (fmax(fabs(pg0), fabs(pg1)) <= GRADIENT_TOLERANCE)
        {
            break;
        }

        // A parameter on a bound with the gradient pushing it outwards stays fixed, and the
        // step is solved for the others only. Clamping the full coupled step instead would
        // leave the free parameter with a step meant for a move the bound doesn't allow.
        const bool free0 = !((p[0] <= lo[0] && g0 > 0) || (p[0] >= hi[0] && g0 < 0));
        const bool free1 = !((p[1] <= lo[1] && g1 > 0) || (p[1] >= hi[1] && g1 < 0));

        // Solve (A + mu * diag(A)) delta = -g over the free parameters
        const double a00 = A00 + mu * clamp(A00, MIN_DIAGONAL, MAX_DIAGONAL);
        const double a11 = A11 + mu * clamp(A11, MIN_DIAGONAL, MAX_DIAGONAL);
        double delta0 = 0, delta1 = 0;
        if (free0 && free1)
        {
            const double det = a00 * a11 - A01 * A01;
            delta0 = (A01 * g1 - a11 * g0) / det;
            delta1 = (A01 * g0 - a00 * g1) / det;
        }
        else if (free0)
        {
            delta0 = -g0 / a00;
        }
        else if (free1)
        {
            delta1 = -g1 / a11;
        }

        // Project the step onto the bounds
        const double p_new[2] = {clamp(p[0] + delta0, lo[0], hi[0]), clamp(p[1] + delta1, lo[1], hi[1])};
        const double step0 = p_new[0] - p[0], step1 = p_new[1] - p[1];

        if (sqrt(step0 * step0 + step1 * step1) <=
            PARAMETER_TOLERANCE * (sqrt(p[0] * p[0] + p[1] * p[1]) + PARAMETER_TOLERANCE))
        {
            break;
        }

        // Reduction predicted by the linear model 0.5 * |r + J step|^2
        double model_cost = 0;
        for (int i = 0; i < NB; ++i)
        {
            const double m = r[i] + J[i][0] * step0 + J[i][1] * step1;
            model_cost += 0.5 * m * m;
        }
        const double predicted = cost - model_cost;

        double r_new[NB], J_new[NB][2];
        const double cost_new = evaluate<NB>(p_new, svec, bvec, r_new, J_new);
        const double rho = (cost - cost_new) / predicted;

        if (predicted > 0 && rho > MIN_RELATIVE_DECREASE)
        {
            // Successful step: accept it and trust the model more
            const bool converged = cost - cost_new <= FUNCTION_TOLERANCE * cost;
            p[0] = p_new[0];
            p[1] = p_new[1];
            cost = cost_new;
            for (int i = 0; i < NB; ++i)
            {
                r[i] = r_new[i];
                J[i][0] = J_new[i][0];
                J[i][1] = J_new[i][1];
            }
            const double t = 2 * rho - 1;
            mu *= fmax(1.0 / 3.0, 1 - t * t * t);
            nu = 2;
            if (converged)
            {
                break;
            }
        }
        else
        {
            // Unsuccessful step: damp harder and try again from the same point
            mu *= nu;
            nu *= 2;
        }
    }

    // Ceres cost is 0.5 * sum(r^2), lsqcurvefit's resnorm is sum(r^2)
    return {p[0], p[1], sqrt(2 * cost / NB)};
} // namespace fit_device
