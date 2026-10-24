"""Strong-scaling sweep: speedup over the best sequential version against CPU threads and GPU SMs.

Runs fit_sub_best (--threads N, OpenMP) and fit_sub_cuda (--sms N) on the whole brain, takes
the median fit time of several runs per point, and writes scaling/scaling.csv and
scaling/speedup.png. Speedup is relative to fit_sub_best with 1 thread (the best sequential).

Run from the repo root, after building both projects:
    python scaling/run_scaling.py
    python scaling/run_scaling.py --repeats 3 --skip-gpu
    python scaling/run_scaling.py --plot-only          # redraw from scaling.csv
"""

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

CPU_EXE = Path("serial_cpp/build/fit_sub_best.exe")
GPU_EXE = Path("cuda/build/fit_sub_cuda.exe")
OUT_DIR = Path("scaling")
FIT_TIME = re.compile(r"Fit time: ([0-9.]+) s")


def fit_time(cmd, env):
    """Runs one fit and returns its reported fit time in seconds."""
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if out.returncode != 0:
        sys.exit(f"{' '.join(cmd)} failed:\n{out.stdout}{out.stderr}")
    m = FIT_TIME.search(out.stdout)
    if not m:
        sys.exit(f"No 'Fit time' in output of {' '.join(cmd)}:\n{out.stdout}")
    return float(m.group(1))


def sweep(device, exe, flag, counts, repeats, env):
    rows = []
    for k in counts:
        times = [fit_time([str(exe), "--all", flag, str(k)], env) for _ in range(repeats)]
        med = statistics.median(times)
        print(f"  {device} {flag} {k:>2}: median {med:.4f} s  (min {min(times):.4f}, max {max(times):.4f})")
        rows.append({"device": device, "units": k, "median_s": med, "min_s": min(times), "max_s": max(times)})
    return rows


def plot(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Reference palette (light): categorical slots 1-2, text ink, recessive grid
    CPU_C, GPU_C = "#2a78d6", "#eb6834"
    INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"

    base = next(r["median_s"] for r in rows if r["device"] == "cpu" and r["units"] == 1)
    cpu = sorted((r for r in rows if r["device"] == "cpu"), key=lambda r: r["units"])
    gpu = sorted((r for r in rows if r["device"] == "gpu"), key=lambda r: r["units"])

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
    panels = [p for p in (("cpu", cpu), ("gpu", gpu)) if p[1]]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.4), sharey=True,
                             squeeze=False, facecolor="#fcfcfb")
    ymax = max(base / r["median_s"] for r in rows) * 1.15

    for ax, (dev, data) in zip(axes[0], panels):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        x = [r["units"] for r in data]
        s = [base / r["median_s"] for r in data]
        color = CPU_C if dev == "cpu" else GPU_C
        xmax = max(x)

        if dev == "cpu":
            # Ideal: n threads give n times the single-thread speed
            ax.plot([1, xmax], [1, xmax], "--", color=MUTED, linewidth=1.2)
            ax.annotate("ideal (linear)", (xmax, min(xmax, ymax * 0.97)), color=INK2, ha="right", va="top",
                        xytext=(-4, -2), textcoords="offset points")
            ax.axvline(8, color="#c9c8c2", linewidth=1.5, zorder=0)
            ax.annotate("8 physical cores", (8, ymax * 0.03), color=INK2, ha="left",
                        xytext=(4, 0), textcoords="offset points")
            ax.set_xlabel("CPU threads (Ryzen 7 7700, OpenMP)")
            ax.set_title("CPU: speedup vs threads", loc="left", fontweight="bold")
        else:
            # Ideal: linear from the 1-SM point
            s1 = s[0]
            ax.plot([1, xmax], [s1, s1 * xmax], "--", color=MUTED, linewidth=1.2)
            # Labelled mid-line, away from the end-point label
            ax.annotate("ideal (linear from 1 SM)", (xmax * 0.55, s1 * xmax * 0.55), color=INK2, ha="right",
                        va="bottom", xytext=(-6, 6), textcoords="offset points")
            if cpu:
                best_cpu = base / min(r["median_s"] for r in cpu)
                ax.axhline(best_cpu, color=CPU_C, linewidth=1.2, linestyle=":")
                ax.annotate(f"best CPU ({best_cpu:.1f}x)", (xmax, best_cpu), color=INK2, ha="right", va="bottom",
                            xytext=(-4, 3), textcoords="offset points")
            ax.set_xlabel("GPU streaming multiprocessors (RTX 4070, 46 SMs)")
            ax.set_title("GPU: speedup vs SMs", loc="left", fontweight="bold")

        ax.plot(x, s, "-o", color=color, linewidth=2, markersize=6, markeredgecolor="#fcfcfb",
                markeredgewidth=1.5, zorder=3)
        # Direct label on the last point only
        ax.annotate(f"{s[-1]:.1f}x", (x[-1], s[-1]), color=INK, fontweight="bold", ha="right", va="bottom",
                    xytext=(-6, 6), textcoords="offset points")
        ax.set_xlim(0, xmax * 1.04)
        ax.set_ylim(0, ymax)

    axes[0][0].set_ylabel("Speedup over best sequential (1 CPU core)")
    fig.suptitle(f"Sub-diffusion fit, 180,722 voxels: best sequential = {base:.2f} s",
                 x=0.01, ha="left", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    print(f"Wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repeats", type=int, default=5, help="runs per point; the median is used (default 5)")
    p.add_argument("--threads", default="1,2,4,6,8,10,12,14,16", help="CPU thread counts")
    p.add_argument("--sms", default="1,2,4,8,12,16,23,30,38,46", help="GPU SM counts")
    p.add_argument("--skip-cpu", action="store_true")
    p.add_argument("--skip-gpu", action="store_true")
    p.add_argument("--plot-only", action="store_true", help="redraw the graph from scaling.csv")
    args = p.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    csv_path, png_path = OUT_DIR / "scaling.csv", OUT_DIR / "speedup.png"

    if args.plot_only:
        with open(csv_path, newline="") as f:
            rows = [{**r, "units": int(r["units"]), "median_s": float(r["median_s"])} for r in csv.DictReader(f)]
        plot(rows, png_path)
        return

    env = dict(os.environ)
    if os.environ.get("CUDA_PATH"):  # cudart DLL for fit_sub_cuda
        env["PATH"] = str(Path(os.environ["CUDA_PATH"]) / "bin") + os.pathsep + env["PATH"]

    rows = []
    # The CPU sweep always includes 1 thread: it is the baseline for every speedup
    threads = sorted({1, *map(int, args.threads.split(","))})
    if not args.skip_cpu:
        print(f"CPU sweep ({CPU_EXE}, {args.repeats} runs per point)")
        rows += sweep("cpu", CPU_EXE, "--threads", threads, args.repeats, env)
    else:
        print("CPU sweep skipped; measuring the 1-thread baseline only")
        rows += sweep("cpu", CPU_EXE, "--threads", [1], args.repeats, env)
    if not args.skip_gpu:
        print(f"GPU sweep ({GPU_EXE}, {args.repeats} runs per point)")
        rows += sweep("gpu", GPU_EXE, "--sms", sorted(set(map(int, args.sms.split(",")))), args.repeats, env)

    base = rows[0]["median_s"]
    for r in rows:
        r["speedup"] = base / r["median_s"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["device", "units", "median_s", "min_s", "max_s", "speedup"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")
    plot(rows, png_path)


if __name__ == "__main__":
    main()
