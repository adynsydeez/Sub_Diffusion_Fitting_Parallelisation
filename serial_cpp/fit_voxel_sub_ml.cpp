#include "fit_voxel_sub_ml.h"
#include "mittag-leffler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include <ceres/ceres.h>

namespace
{
    // Residual at one b-value: S0 * E_beta(-b * D) - S, with p = [D, beta]
    struct SubMlResidual
    {
        double b, S, S0;

        bool operator()(const double *p, double *r) const
        {
            try
            {
                r[0] = S0 * ml(-b * p[0], p[1]) - S;
            }
            catch (const std::invalid_argument &)
            {
                return false; // tells Ceres this point is invalid instead of throwing through it
            }
            return true;
        }
    };
}

FitResult fit_voxel_sub_ml(const std::array<double, 9> &Svec, const std::array<double, 9> &bvec, double D_guess, double beta_guess, double D_min, double D_max, double beta_min, double beta_max)
{
    if (Svec[0] <= 0)
    {
        return {0, 0, 0};
    }

    double p[2] = {D_guess, beta_guess};

    ceres::Problem problem; // takes ownership of the cost functions
    for (std::size_t i = 0; i < bvec.size(); ++i)
    {
        problem.AddResidualBlock(
            new ceres::NumericDiffCostFunction<SubMlResidual, ceres::CENTRAL, 1, 2>(
                new SubMlResidual{bvec[i], Svec[i], Svec[0]}),
            nullptr, p);
    }

    problem.SetParameterLowerBound(p, 0, D_min);
    problem.SetParameterUpperBound(p, 0, D_max);
    problem.SetParameterLowerBound(p, 1, std::max(beta_min, 1e-6)); // ml() throws for beta <= 0
    problem.SetParameterUpperBound(p, 1, beta_max);

    ceres::Solver::Options options;
    options.function_tolerance = 1e-5;    // 'FunctionTolerance', 1e-5
    options.logging_type = ceres::SILENT; // 'Display', 'off'

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Ceres cost is 0.5 * sum(r^2), lsqcurvefit's resnorm is sum(r^2)
    double resnorm = 2.0 * summary.final_cost;

    return {p[0], p[1], std::sqrt(resnorm / bvec.size())};
}
