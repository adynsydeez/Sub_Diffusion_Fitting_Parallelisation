| Comparison | Param | Voxels | Bias | Median abs diff | 95th pct abs diff | Max abs diff | Within tolerance | Bit-identical |
|---|---|---|---|---|---|---|---|---|
| C++ serial (Ceres) vs MATLAB | D | 4272 | 1.17e-06 | 1.69e-07 | 6.72e-06 | 3.00e-03 | 98.8% | 0.0% |
| C++ serial (Ceres) vs MATLAB | β | 4272 | 2.78e-04 | 2.15e-04 | 1.32e-02 | 1.00e+00 | 93.8% | 0.0% |
| C++ serial (Ceres) vs MATLAB | RMSE | 4272 | 1.99e-04 | 2.68e-08 | 7.73e-05 | 5.83e-01 | 98.4% | 0.0% |
| Best sequential vs MATLAB | D | 4272 | 6.46e-07 | 7.55e-08 | 2.66e-06 | 5.86e-04 | 99.5% | 0.0% |
| Best sequential vs MATLAB | β | 4272 | 1.89e-03 | 8.65e-05 | 1.93e-03 | 1.00e+00 | 99.1% | 0.0% |
| Best sequential vs MATLAB | RMSE | 4272 | -4.34e-07 | 1.15e-08 | 2.25e-06 | 1.09e-04 | 100.0% | 0.0% |
| OpenMP, 16 threads vs MATLAB | D | 4272 | 6.46e-07 | 7.55e-08 | 2.66e-06 | 5.86e-04 | 99.5% | 0.0% |
| OpenMP, 16 threads vs MATLAB | β | 4272 | 1.89e-03 | 8.65e-05 | 1.93e-03 | 1.00e+00 | 99.1% | 0.0% |
| OpenMP, 16 threads vs MATLAB | RMSE | 4272 | -4.34e-07 | 1.15e-08 | 2.25e-06 | 1.09e-04 | 100.0% | 0.0% |
| CUDA, RTX 4070 vs MATLAB | D | 4272 | 6.46e-07 | 7.55e-08 | 2.66e-06 | 5.86e-04 | 99.5% | 0.0% |
| CUDA, RTX 4070 vs MATLAB | β | 4272 | 1.89e-03 | 8.65e-05 | 1.93e-03 | 1.00e+00 | 99.1% | 0.0% |
| CUDA, RTX 4070 vs MATLAB | RMSE | 4272 | -4.34e-07 | 1.15e-08 | 2.25e-06 | 1.09e-04 | 100.0% | 0.0% |
| OpenMP, 16 threads vs Best sequential | D | 4272 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 100.0% | 100.0% |
| OpenMP, 16 threads vs Best sequential | β | 4272 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 100.0% | 100.0% |
| OpenMP, 16 threads vs Best sequential | RMSE | 4272 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 100.0% | 100.0% |
| CUDA, RTX 4070 vs Best sequential | D | 4272 | 4.16e-18 | 2.17e-19 | 1.52e-18 | 1.76e-14 | 100.0% | 26.8% |
| CUDA, RTX 4070 vs Best sequential | β | 4272 | -2.15e-15 | 1.11e-16 | 9.99e-16 | 5.72e-12 | 100.0% | 25.4% |
| CUDA, RTX 4070 vs Best sequential | RMSE | 4272 | 3.05e-18 | 2.78e-17 | 9.02e-17 | 1.78e-15 | 100.0% | 5.6% |

Tolerances: D within 1% (or 1e-5 mm²/s), β within 0.01, RMSE within 1% (or 1e-4).