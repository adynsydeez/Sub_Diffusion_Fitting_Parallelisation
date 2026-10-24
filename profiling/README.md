# Profiling

Run everything from the repository root.

## 1. MATLAB (original application): MATLAB Profiler

```
matlab -batch "run('profiling/profile_matlab.m')"
```

Profiles `run_fit(73)` (slice 73). Writes `profiling/matlab/top_functions.txt` (top 20 functions
by self time) and an interactive report at `profiling/matlab/html/file0.html`.
Needs R2024a (Image Processing Toolbox for `niftiread`).

## 2. C++ serial (Ceres) and best sequential: gprof

gprof comes with MSYS2's GCC. Build separate profiling binaries with `-pg`. On Windows
ASLR must be turned off, or gprof prints an empty profile. `-fno-inline` keeps functions
visible as separate entries.

```
g++ -std=gnu++20 -O2 -fno-inline -pg -DNDEBUG -DGFLAGS_IS_A_DLL=0 -DGLOG_NO_ABBREVIATED_SEVERITIES -DGLOG_USE_GFLAGS -DGLOG_USE_GLOG_EXPORT -Iserial_cpp -isystem C:/msys64/ucrt64/include/eigen3 serial_cpp/main.cpp serial_cpp/fit_voxel_sub_ml.cpp serial_cpp/mittag-leffler.cpp serial_cpp/read_mri.cpp serial_cpp/save_map.cpp -pg -no-pie -Wl,--disable-dynamicbase,--disable-high-entropy-va -lceres -lglog -lgflags -lshlwapi -lz -o fit_sub_pg.exe
./fit_sub_pg.exe --slice 73 --out profiling/cpp/out
gprof -b fit_sub_pg.exe gmon.out > profiling/cpp/gprof_fit_sub.txt
```

For best sequential, add `-DBEST_SEQUENTIAL -Icuda`, drop `fit_voxel_sub_ml.cpp`,
`mittag-leffler.cpp` and the Ceres libraries, and run `--all` (one slice is too short to sample).

Caveat: `-pg` instruments every function call. The Ceres version makes over a billion tiny
calls (`std::complex` constructors etc.), so about 60% of its samples are gprof's own
counting (`_mcount_private`, `__fentry__`). Call counts are exact; time splits between
small functions are approximate. A sampling profiler without instrumentation (AMD uProf,
free, suits the Ryzen CPU) gives cleaner time percentages.

## 3. CUDA: Nsight Compute (installed with the CUDA Toolkit)

GPU performance counters are restricted to administrators by default (`ERR_NVGPUCTRPERM`).
Either run the command from an administrator terminal, or allow all users once:
NVIDIA Control Panel → Desktop menu → Enable Developer Settings → Developer → Manage GPU
Performance Counters → "Allow access to the GPU performance counters to all users".

```
ncu --kernel-name fit_kernel --launch-skip 1 --launch-count 1 --set detailed -o profiling/cuda/fit_kernel_all -f cuda/build/fit_sub_cuda.exe --all
ncu --import profiling/cuda/fit_kernel_all.ncu-repz --page details > profiling/cuda/fit_kernel_all.txt
```

`--launch-skip 1` skips the empty warm-up launch. Recent ncu versions save `.ncu-repz` (compressed). Open it in the Nsight Compute
GUI for the full analysis. Sections worth quoting: GPU Speed Of Light (compute vs memory
bound), Occupancy (theoretical vs achieved; registers per thread limit blocks per SM),
Compute Workload Analysis (FP64 pipe utilisation), Launch Statistics.

Nsight Systems (`nsys`, timeline of copies vs kernel) is a separate download if you want a
timeline view.
