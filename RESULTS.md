# Results

Sub-diffusion model fit (D, β per voxel; Mittag-Leffler signal model) for `sub_001`, Δ = 19 ms,
180,722 brain voxels (66 × 108 × 108 volume). Measured 23 October 2026.

## Test system

| | |
|---|---|
| CPU | AMD Ryzen 7 7700, 8 cores / 16 threads |
| GPU | NVIDIA GeForce RTX 4070, 46 SMs, compute capability 8.9 (driver 617.42) |
| OS | Windows 11 Home |
| MATLAB | R2024a (Image Processing + Optimization Toolboxes) |
| C++ (serial, best sequential, OpenMP) | g++ 16.2.0 (MSYS2 UCRT64), `-O3`, Ceres 2.2.0, Eigen 5.0.1, OpenMP 5.2 |
| CUDA | CUDA 13.4, MSVC 19.51 host compiler, `sm_75` + `sm_89` |
| Analysis | Python 3.13, numpy 2.2.6, matplotlib 3.10.3 |

## Versions

| Version | What it is | Program |
|---|---|---|
| MATLAB (original) | The original application: `lsqcurvefit` + general `ml.m` (Garrappa) | `MATLAB/run_fit.m` (same code as `fit_dMRI_signals_19.m`) |
| C++ serial (Ceres) | Direct port: Ceres + general `ml()` port, numeric derivatives | `serial_cpp/build/fit_sub.exe` |
| Best sequential | The parallel versions' algorithm on one core: specialised Mittag-Leffler, analytic Jacobian, small bounded LM | `serial_cpp/build/fit_sub_best.exe` |
| OpenMP | Best sequential, voxel loop shared over threads | `fit_sub_best.exe --threads N` |
| CUDA | Best sequential's per-voxel code, one GPU thread per voxel | `cuda/build/fit_sub_cuda.exe` |

## 1. Execution time

Fit time only (excludes loading NIfTI files and writing results). Each long run was done **alone**.

### Whole brain (180,722 voxels)

| Version | Fit time | ms / voxel | vs MATLAB | vs previous row |
|---|---|---|---|---|
| MATLAB (original) | 801.6 s (13.4 min) | 4.44 | 1× | |
| C++ serial (Ceres) | 1,139.5 s (19.0 min) | 6.31 | 0.70× | 0.70× |
| Best sequential | 4.91 s | 0.0272 | 163× | 232× (algorithm) |
| OpenMP, 16 threads | 0.584 s | 0.0032 | 1,373× | 8.4× (parallel, CPU) |
| CUDA, RTX 4070 | 0.233 s | 0.0013 | 3,440× | 21.1× vs best sequential (parallel, GPU) |

Figure: `compare/all/timing.png` (log scale).

### Slice 73 (4,272 voxels)

| Version | Fit time | vs MATLAB |
|---|---|---|
| MATLAB (original) | 21.8 s | 1× |
| C++ serial (Ceres) | 26.6 s | 0.82× |
| Best sequential | 0.118 s | 184× |
| OpenMP, 16 threads | 0.0177 s | 1,233× |
| CUDA, RTX 4070 | 0.0163 s | 1,338× |

One slice does not fill the GPU (4,272 threads, 34 blocks over 46 SMs), so CUDA barely beats
OpenMP here; use whole-brain numbers for speedup. Figure: `compare/slice_73/timing.png`.

### Measurement notes

- **Running long jobs together distorts timings.** MATLAB and Ceres run at the same time took
  991.8 s and 1,292.7 s: 24% and 13% slower than alone, even with 8 cores. MATLAB's
  `lsqcurvefit` uses extra threads internally, so the two competed for more than one core
  each. All numbers above are from runs alone.
- Run-to-run variation is a few percent: an earlier Ceres run (5 Oct, slightly older build)
  took 1,192.5 s; best sequential ranges 4.9–5.4 s.
- **The ~1.5 h originally observed was a bigger job, not a slower fit.** The b-value cutoff
  analysis in the original `fit_dMRI_signals_49.m` (Δ = 49 ms) fits the whole brain **7 times**,
  with the first 3, 4, … 9 b-values. Timed serially on slice 73 it takes 114 s (13.8–22.0 s per
  cutoff), which scales to about **70 min** for the whole brain (× 36.8, the measured
  whole-brain/slice ratio). Its `parfor` on the inner `y` loop starts 7,128 separate parallel
  loops per whole-brain fit (49,896 in total), each over at most 108 voxels, so it gave no
  benefit over serial. One whole-brain fit, as timed above, is 13.4 min. (The Parallel
  Computing Toolbox is not installed on this PC at present, so the `parfor` version could not
  be re-timed.)

## 2. Parallel scaling (strong scaling, whole brain)

Speedup over best sequential on 1 CPU core (5.238 s, median of 5 runs); every point is the
median of 5 runs. Figure: `scaling/speedup.png`, data: `scaling/scaling.csv`.

| CPU threads | 1 | 2 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| Speedup | 1.00 | 1.96 | 3.76 | 5.07 | 5.97 | 8.02 | **8.70** |
| Efficiency | 100% | 98% | 94% | 84% | 75% | — | — |

| GPU SMs | 1 | 2 | 4 | 8 | 16 | 23 | 30 | 38 | 46 |
|---|---|---|---|---|---|---|---|---|---|
| Speedup vs 1 CPU core | 0.51 | 1.04 | 2.04 | 4.01 | 7.96 | 11.4 | 14.9 | 18.6 | **22.3** |
| Speedup vs 1 SM | 1.0 | 2.0 | 4.0 | 7.8 | 15.5 | 22.2 | 29.0 | 36.2 | **43.3** (94% efficiency) |

- CPU: near-linear to 4 threads, 75% efficiency at 8 physical cores (all-core boost clock is
  lower than single-core boost), then +46% from SMT between 8 and 16 threads.
- GPU: near-linear in SMs. One SM is half as fast as one CPU core (FP64 runs at 1/64 of FP32
  rate on this GPU); the GPU passes the whole 8-core CPU at about 17 SMs and is 2.5× faster
  than it at 46.

## 3. Correctness

MATLAB (the original application) is the reference. Whole brain, 180,722 voxels
(`compare/all/summary.md` has every row; slice 73 in `compare/slice_73/summary.md`).
Tolerances: D within 1% (or 1e-5 mm²/s), β within 0.01, RMSE within 1% (or 1e-4).

| Comparison | D within tol. | β within tol. | RMSE within tol. | Bit-identical |
|---|---|---|---|---|
| C++ serial (Ceres) vs MATLAB | 98.3% | 93.3% | 98.6% | 0% |
| Best sequential vs MATLAB | 99.3% | 98.6% | 99.9% | 0% |
| OpenMP (16 threads) vs Best sequential | 100% | 100% | 100% | **100%** |
| CUDA vs Best sequential | 100% | 100% | 100% | 26% / 26% / 7%; max \|Δ\| 5.7e-13 (D), 3.6e-9 (β), 3.0e-8 (RMSE) |

- **The parallel versions reproduce the sequential version they parallelise.** OpenMP is
  bit-identical to best sequential (each voxel's fit is independent and deterministic). CUDA
  differs only by floating-point rounding (the GPU's `sincos` and fused multiply-add) — at
  most 3.6e-9 in β; in a few voxels rounding changes when the solver stops, hence the larger
  maximum than on slice 73 (5.7e-12).
- **Neither C++ version is bit-identical to MATLAB**, as expected: different solvers
  (`lsqcurvefit` trust-region-reflective vs Levenberg-Marquardt) stopping at a tolerance of
  1e-5. Best sequential agrees with MATLAB more closely than the direct Ceres port.
- **Fit quality:** best sequential's RMSE is more than 1% worse than MATLAB's in 134 voxels
  (0.07%) and more than 1% better in 18; Ceres is worse in 2,469 (1.37%) and better in 17.
- **Where they disagree:** the diagonal lines in the β Bland-Altman plots are voxels where one
  method put β on a bound (0 or 1) and the other did not; these are poorly determined voxels
  (very different β fit almost equally well), mostly near tissue boundaries.

Figures:
- `compare/slice_73/maps_slice_73.png`: MATLAB D, β, RMSE maps for slice 73 with difference
  maps (Ceres − MATLAB, Best − MATLAB, OpenMP − Best, CUDA − Best).
- `compare/all/bland_altman.png`: difference vs mean against MATLAB, whole brain.

## 4. Profiling

Details and commands: `profiling/README.md`.

| Version | Tool | Finding |
|---|---|---|
| MATLAB | MATLAB Profiler (`profiling/matlab/top_functions.txt`) | 59% of time in `ml.m` (52.9% in `LTInversion` itself); ~40% in `lsqcurvefit` machinery (`trdog`, `snls`, `aprecon`, `pcgr`, `colamd`, option handling). 146 Mittag-Leffler evaluations per voxel (forward differences). |
| C++ serial (Ceres) | gprof (`profiling/cpp/gprof_fit_sub.txt`) | **945** `ml()` calls per voxel (central differences, 9 residual blocks): why it is slower than MATLAB. Remaining time in `lt_inversion`'s pole sort, `std::vector` set-up and complex `log`/`exp`. (`-pg` overhead is ~60% of samples here; call counts are exact, time splits approximate.) |
| Best sequential | gprof (`profiling/cpp/gprof_fit_sub_best.txt`) | 81% of time in `ml_device::ml`; 63 calls per voxel (about 7 evaluations × 9 b-values). |
| CUDA | Nsight Compute (`profiling/cuda/fit_kernel_all.txt`) | Compute-bound: SM 82.4%, memory 0.76% of peak. **FP64 pipe 82.4% busy: the bottleneck.** 200 registers/thread → 2 blocks/SM, occupancy 16.7% theoretical / 13.9% achieved. Branch efficiency 99.9% (negligible divergence). No register spills. Uncoalesced loads (72-byte stride) but memory is idle, so ~3% at stake. |

## 5. Observations and next steps

- **Redundant FP64 work (profiler-driven):** `evaluate()` calls `ml(z, β)` for the 9 b-values
  with the same β, and each call recomputes `exp` + `sincos` for all 28 nodes although they
  depend only on β. Computing them once per evaluation would cut the transcendental work
  about 9×; estimated 2–4× faster on both CPU and GPU. Changes the best-sequential baseline,
  so the scaling sweep must be re-run.
- **FP32 on the GPU:** Nsight suggests it (64:1 FP32:FP64 throughput), but the Mittag-Leffler
  evaluation is designed for ~1e-15 accuracy; would need an accuracy study with
  `compare/compare_results.py`.
- **Occupancy:** limited by registers, but with the FP64 pipe at 82% the most it could add is
  about 1.2×.

## How to reproduce

From the repo root. Builds: `cmake --build serial_cpp/build`; CUDA from an x64 Visual Studio
shell (`cmake --build cuda/build`).

```
matlab -batch "addpath('MATLAB'); run_fit()"                                   # R2024a
serial_cpp/build/fit_sub.exe --all
serial_cpp/build/fit_sub_best.exe --all
serial_cpp/build/fit_sub_best.exe --all --threads 16 --out serial_cpp/outputs_best_omp16
cuda/build/fit_sub_cuda.exe --all
python compare/compare_results.py --run all          # tables + figures in compare/all/
python scaling/run_scaling.py                        # scaling sweep + scaling/speedup.png
```

Use `--slice 73` / `run_fit(73)` / `--run slice_73` for one slice. Run long jobs one at a time.
