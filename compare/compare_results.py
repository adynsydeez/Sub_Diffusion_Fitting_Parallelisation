"""Compares the fitted maps of every version against the original MATLAB application.

Reads <version folder>/<run>/{D_SUB,BETA_SUB,ERROR_SUB}.nii (+ run.txt) written by
MATLAB/run_fit.m, fit_sub, fit_sub_best and fit_sub_cuda, and writes to compare/<run>/:
  summary.md / summary.csv   agreement with MATLAB, and exactness of the parallel versions
  maps_slice_N.png           MATLAB maps and difference maps for one axial slice
  bland_altman.png           Bland-Altman plots (difference vs mean) against MATLAB
  timing.png                 fit time per version (log scale), if run.txt files are present

Run from the repo root after producing the outputs, e.g. for slice 73:
  matlab -batch "addpath('MATLAB'); run_fit(73)"
  serial_cpp/build/fit_sub.exe --slice 73
  serial_cpp/build/fit_sub_best.exe --slice 73
  serial_cpp/build/fit_sub_best.exe --slice 73 --threads 16 --out serial_cpp/outputs_best_omp16
  cuda/build/fit_sub_cuda.exe --slice 73
  python compare/compare_results.py --run slice_73
For the whole brain use --all / run_fit() and --run all (--slice picks the slice to show).
"""

import argparse
import csv
import struct
import sys
from pathlib import Path

import numpy as np

# (key, label, results folder); MATLAB first: it is the reference
VERSIONS = [
    ("matlab", "MATLAB (original)", "MATLAB/outputs"),
    ("ceres", "C++ serial (Ceres)", "serial_cpp/outputs"),
    ("best", "Best sequential", "serial_cpp/outputs_best"),
    ("omp", "OpenMP, 16 threads", "serial_cpp/outputs_best_omp16"),
    ("cuda", "CUDA, RTX 4070", "cuda/outputs"),
]
# (file, label, unit, display range, agreement tolerance: (relative, absolute floor))
PARAMS = [
    ("D_SUB", "D", "mm²/s", (0.0, 3e-3), (0.01, 1e-5)),
    ("BETA_SUB", "β", "", (0.5, 1.0), (0.0, 0.01)),
    ("ERROR_SUB", "RMSE", "", (0.0, 0.1), (0.01, 1e-4)),
]

# Reference palette (light)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # blue 100..700
DIV = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a3a3", "#e34948", "#a8302f"]  # blue - gray - red
SERIES = {"ceres": "#eb6834", "best": "#2a78d6", "cuda": "#1baf7a"}
SHORT = {"matlab": "MATLAB", "ceres": "Ceres", "best": "Best seq.", "omp": "OpenMP", "cuda": "CUDA"}


def read_nifti(path):
    """Reads a 3D NIfTI-1 (.nii) volume as an array indexed [l, y, x]."""
    raw = Path(path).read_bytes()
    dim = struct.unpack("<8h", raw[40:56])
    datatype = struct.unpack("<h", raw[70:72])[0]
    offset = int(struct.unpack("<f", raw[108:112])[0])
    slope, inter = struct.unpack("<2f", raw[112:120])
    dtype = {16: "<f4", 64: "<f8"}[datatype]
    X, Y, L = dim[1], dim[2], dim[3]
    v = np.frombuffer(raw, dtype=dtype, count=X * Y * L, offset=offset).astype(np.float64)
    if slope not in (0.0, 1.0) or inter != 0.0:
        v = slope * v + inter
    return v.reshape(L, Y, X)


def read_run(folder):
    p = Path(folder) / "run.txt"
    if not p.exists():
        return {}
    return dict(line.strip().split("=", 1) for line in p.read_text().splitlines() if "=" in line)


def agreement(v, m, tol):
    """Statistics of v - m over voxels fitted by both."""
    ok = np.isfinite(v) & np.isfinite(m)
    d = v[ok] - m[ok]
    rel, floor = tol
    within = np.abs(d) <= np.maximum(rel * np.abs(m[ok]), floor)
    return {
        "voxels": int(ok.sum()),
        "bias": float(d.mean()),
        "median_abs": float(np.median(np.abs(d))),
        "p95_abs": float(np.percentile(np.abs(d), 95)),
        "max_abs": float(np.abs(d).max()),
        "within_tol_pct": float(100 * within.mean()),
        "identical_pct": float(100 * np.mean(d == 0)),
    }


def show(img):
    """Slice [y, x] -> display orientation of save_map.cpp: x to the right, y upwards."""
    return img[::-1, :]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="slice_73", help="'all' or 'slice_N' (default slice_73)")
    ap.add_argument("--slice", type=int, default=None, help="slice to show in the maps (default: the run's slice, or 73)")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")  # β in the console table on Windows
    run = args.run
    slice_n = args.slice or (int(run.split("_")[1]) if run.startswith("slice_") else 73)
    out = Path("compare") / run
    out.mkdir(parents=True, exist_ok=True)

    # --- Load every version that has results for this run ---
    vols, runs, labels = {}, {}, {}
    for key, label, folder in VERSIONS:
        d = Path(folder) / run
        if (d / "D_SUB.nii").exists():
            vols[key] = {p[0]: read_nifti(d / f"{p[0]}.nii") for p in PARAMS}
            runs[key] = read_run(d)
            labels[key] = label
        else:
            print(f"  (no results for {label} in {d})")
    if "matlab" not in vols:
        raise SystemExit("MATLAB results are needed as the reference: run MATLAB/run_fit.m first")
    print("Versions:", ", ".join(labels[k] for k in vols))

    # --- Agreement with MATLAB, and exactness of the parallel versions vs best sequential ---
    rows = []
    for key in vols:
        if key == "matlab":
            continue
        for f, name, unit, _, tol in PARAMS:
            rows.append({"compare": f"{labels[key]} vs MATLAB", "param": name,
                         **agreement(vols[key][f], vols["matlab"][f], tol)})
    for key in ("omp", "cuda"):
        if key in vols and "best" in vols:
            for f, name, unit, _, tol in PARAMS:
                rows.append({"compare": f"{labels[key]} vs Best sequential", "param": name,
                             **agreement(vols[key][f], vols["best"][f], tol)})

    fields = list(rows[0].keys())
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    md = ["| Comparison | Param | Voxels | Bias | Median abs diff | 95th pct abs diff | Max abs diff | Within tolerance | Bit-identical |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['compare']} | {r['param']} | {r['voxels']} | {r['bias']:.2e} | {r['median_abs']:.2e} | "
                  f"{r['p95_abs']:.2e} | {r['max_abs']:.2e} | {r['within_tol_pct']:.1f}% | {r['identical_pct']:.1f}% |")
    md.append("")
    md.append("Tolerances: D within 1% (or 1e-5 mm²/s), β within 0.01, RMSE within 1% (or 1e-4).")
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.edgecolor": GRID})
    seq = LinearSegmentedColormap.from_list("seq", SEQ)
    seq.set_bad(SURFACE)
    div = LinearSegmentedColormap.from_list("div", DIV)
    div.set_bad(SURFACE)
    l = slice_n - 1

    # --- Figure 1: MATLAB maps and difference maps for one slice ---
    diff_cols = [(k, "matlab") for k in ("ceres", "best") if k in vols] + \
                [(k, "best") for k in ("omp", "cuda") if k in vols and "best" in vols]
    ncol = 1 + len(diff_cols)
    fig, axes = plt.subplots(len(PARAMS), ncol, figsize=(2.15 * ncol + 1.2, 2.9 * len(PARAMS)), squeeze=False,
                             facecolor=SURFACE)
    for r, (f, name, unit, rng, _) in enumerate(PARAMS):
        ref = vols["matlab"][f][l]
        ax = axes[r][0]
        im = ax.imshow(show(ref), cmap=seq, vmin=rng[0], vmax=rng[1], interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02).ax.tick_params(labelsize=7)
        ax.set_title(f"MATLAB {name}" + (f" ({unit})" if unit else ""), fontsize=9, loc="left")
        # One symmetric scale per row, set by the larger of the algorithm differences
        diffs = [vols[a][f][l] - vols[b][f][l] for a, b in diff_cols]
        lim = max(np.nanpercentile(np.abs(d), 99) for d in diffs[:2]) if diffs else 1
        lim = lim if lim > 0 else 1e-12
        for c, ((a, b), d) in enumerate(zip(diff_cols, diffs), start=1):
            ax = axes[r][c]
            im = ax.imshow(show(d), cmap=div, vmin=-lim, vmax=lim, interpolation="nearest")
            ax.set_title(f"{SHORT[a]} − {SHORT[b]}", fontsize=9, loc="left")
            mx = np.nanmax(np.abs(d))
            ax.text(0.02, -0.04, f"max |Δ| = {mx:.1e}" if mx > 0 else "identical (Δ = 0)", transform=ax.transAxes,
                    va="top", fontsize=7.5, color=INK2)
        if diff_cols:
            cb = fig.colorbar(im, ax=axes[r][1:].tolist(), fraction=0.02, pad=0.01,
                              format=matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0e}" if x else "0"))
            cb.ax.tick_params(labelsize=7)
            cb.set_label(f"Δ{name}", fontsize=8)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    fig.suptitle(f"Slice {slice_n}: original MATLAB maps (left) and differences between versions "
                 f"(blue = lower, red = higher, grey = equal)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.savefig(out / f"maps_slice_{slice_n}.png", dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: Bland-Altman against MATLAB; one row per version, columns D and beta (portrait) ---
    ba = [k for k in ("ceres", "best", "cuda") if k in vols]
    # Sized close to its printed size in the report (about 3.4 in wide) so the text stays legible
    fig, axes = plt.subplots(len(ba), 2, figsize=(4.6, 1.75 * len(ba) + 0.45), squeeze=False, sharex="col",
                             facecolor=SURFACE)
    for c, (f, name, unit, rng, _) in enumerate(PARAMS[:2]):
        m = vols["matlab"][f]
        for r, k in enumerate(ba):
            v = vols[k][f]
            ok = np.isfinite(v) & np.isfinite(m)
            mean, diff = (v[ok] + m[ok]) / 2, v[ok] - m[ok]
            bias, sd = diff.mean(), diff.std()
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if ok.sum() > 20000:
                ax.hexbin(mean, diff, gridsize=45, bins="log", mincnt=1,
                          cmap=LinearSegmentedColormap.from_list("h", SEQ[1:]), linewidths=0)
            else:
                ax.scatter(mean, diff, s=5, color=SERIES[k], alpha=0.35, linewidths=0)
            ax.axhline(bias, color=INK, linewidth=0.9)
            for y in (bias - 1.96 * sd, bias + 1.96 * sd):
                ax.axhline(y, color=INK2, linewidth=0.8, linestyle="--")
            ax.text(0.98, 0.97, f"bias {bias:.1e}\n95% limits ±{1.96 * sd:.1e}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=6, color=INK2,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=1.5))
            ax.set_title(f"{SHORT[k]} vs MATLAB: {name}", fontsize=7.5, loc="left", fontweight="bold")
            if f == "D_SUB":  # fewer ticks: 0.0025-style labels collide at this width
                ax.set_xticks([0, 1e-3, 2e-3, 3e-3])
                ax.set_xticklabels(["0", "1e-3", "2e-3", "3e-3"])
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.0e}" if y else "0"))
            if r == len(ba) - 1:
                ax.set_xlabel(f"mean {name}" + (f" ({unit})" if unit else ""), fontsize=7)
            ax.set_ylabel(f"{name} − MATLAB {name}", fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"Bland-Altman agreement with the original MATLAB fit\n({run.replace('_', ' ')}, "
                 f"{int(np.isfinite(vols['matlab']['D_SUB']).sum()):,} voxels)", x=0.01, ha="left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "bland_altman.png", dpi=180, facecolor=SURFACE)
    plt.close(fig)

    # --- Figure 3: fit time per version (log scale), from run.txt ---
    timed = [(labels[k], float(runs[k]["fit_time_s"])) for k in vols if runs[k].get("fit_time_s")]
    if timed:
        fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(timed) + 1.2), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        names = [t[0] for t in timed][::-1]
        secs = [t[1] for t in timed][::-1]
        ax.barh(names, secs, color="#2a78d6", height=0.6)
        ax.set_xscale("log")
        base = dict(timed)["MATLAB (original)"]
        for i, s in enumerate(secs):
            ratio = base / s
            rtxt = f"{ratio:,.0f}" if ratio >= 10 else f"{ratio:.2g}"
            stxt = f"{s:,.0f}" if s >= 100 else f"{s:.3g}"
            ax.text(s * 1.15, i, f"{stxt} s  ({rtxt}× vs MATLAB)" if s != base else f"{stxt} s  (baseline)",
                    va="center", fontsize=8, color=INK)
        ax.set_xlim(min(secs) / 2, max(secs) * 25)
        ax.grid(True, axis="x", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("fit time (s, log scale)")
        nvox = runs["matlab"].get("voxels", "?")
        ax.set_title(f"Fit time per version, {run.replace('_', ' ')} ({nvox} voxels)", loc="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "timing.png", dpi=180, facecolor=SURFACE)
        plt.close(fig)

    print(f"\nWrote {out}/summary.md, summary.csv, maps_slice_{slice_n}.png, bland_altman.png" +
          (", timing.png" if timed else ""))


if __name__ == "__main__":
    main()
