| Comparison | Param | Voxels | Bias | Median abs diff | 95th pct abs diff | Max abs diff | Within tolerance | Bit-identical |
|---|---|---|---|---|---|---|---|---|
| C++ serial (Ceres) vs MATLAB | D | 180722 | 1.83e-06 | 1.70e-07 | 7.86e-06 | 3.00e-03 | 98.3% | 0.0% |
| C++ serial (Ceres) vs MATLAB | β | 180722 | 9.72e-04 | 2.28e-04 | 1.65e-02 | 1.00e+00 | 93.3% | 0.0% |
| C++ serial (Ceres) vs MATLAB | RMSE | 180722 | 2.08e-04 | 3.01e-08 | 6.16e-05 | 6.27e-01 | 98.6% | 0.0% |
| Best sequential vs MATLAB | D | 180722 | 7.93e-07 | 7.96e-08 | 3.08e-06 | 3.00e-03 | 99.3% | 0.0% |
| Best sequential vs MATLAB | β | 180722 | 1.73e-03 | 1.03e-04 | 2.49e-03 | 1.00e+00 | 98.6% | 0.0% |
| Best sequential vs MATLAB | RMSE | 180722 | 1.36e-05 | 1.18e-08 | 1.60e-06 | 2.67e-01 | 99.9% | 0.0% |
| OpenMP, 16 threads vs MATLAB | D | 180722 | 7.93e-07 | 7.96e-08 | 3.08e-06 | 3.00e-03 | 99.3% | 0.0% |
| OpenMP, 16 threads vs MATLAB | β | 180722 | 1.73e-03 | 1.03e-04 | 2.49e-03 | 1.00e+00 | 98.6% | 0.0% |
| OpenMP, 16 threads vs MATLAB | RMSE | 180722 | 1.36e-05 | 1.18e-08 | 1.60e-06 | 2.67e-01 | 99.9% | 0.0% |
| CUDA, RTX 4070 vs MATLAB | D | 180722 | 7.93e-07 | 7.96e-08 | 3.08e-06 | 3.00e-03 | 99.3% | 0.0% |
| CUDA, RTX 4070 vs MATLAB | β | 180722 | 1.73e-03 | 1.03e-04 | 2.49e-03 | 1.00e+00 | 98.6% | 0.0% |
| CUDA, RTX 4070 vs MATLAB | RMSE | 180722 | 1.36e-05 | 1.18e-08 | 1.60e-06 | 2.67e-01 | 99.9% | 0.0% |
| OpenMP, 16 threads vs Best sequential | D | 180722 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 100.0% | 100.0% |
| OpenMP, 16 threads vs Best sequential | β | 180722 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 100.0% | 100.0% |
| OpenMP, 16 threads vs Best sequential | RMSE | 180722 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 100.0% | 100.0% |
| CUDA, RTX 4070 vs Best sequential | D | 180722 | 1.58e-18 | 2.17e-19 | 1.73e-18 | 5.69e-13 | 100.0% | 26.4% |
| CUDA, RTX 4070 vs Best sequential | β | 180722 | 1.54e-14 | 1.11e-16 | 1.57e-15 | 3.61e-09 | 100.0% | 25.5% |
| CUDA, RTX 4070 vs Best sequential | RMSE | 180722 | 8.24e-14 | 2.78e-17 | 9.41e-17 | 2.98e-08 | 100.0% | 6.6% |

Tolerances: D within 1% (or 1e-5 mm²/s), β within 0.01, RMSE within 1% (or 1e-4).