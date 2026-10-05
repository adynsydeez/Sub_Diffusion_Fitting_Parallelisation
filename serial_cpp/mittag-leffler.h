// Mittag-Leffler function with 1, 2 or 3 parameters (C++ port of MATLAB/ml.m)

#pragma once

#include <complex>

// Evaluates E_{alpha,beta}^{gama}(z) = sum_k Gamma(gama+k) z^k / (Gamma(gama) k! Gamma(alpha*k+beta))
// to a relative accuracy of about 1e-15, using the optimal parabolic contour algorithm.
//   ml(z, alpha)             one-parameter  E_alpha(z)        alpha > 0
//   ml(z, alpha, beta)       two-parameter  E_{alpha,beta}(z) beta real
//   ml(z, alpha, beta, gama) three-parameter                  0 < alpha < 1, gama > 0, |arg z| > alpha*pi
// Throws std::invalid_argument if the parameters are out of range.
std::complex<double> ml(std::complex<double> z, double alpha, double beta = 1.0, double gama = 1.0);

// Real-argument version: returns the real part, as ml.m does for real z.
double ml(double z, double alpha, double beta = 1.0, double gama = 1.0);
