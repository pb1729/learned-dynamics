#!/usr/bin/env python3
"""
plot_tensorboard_scalars.py
---------------------------

Plot every scalar in one or more TensorBoard log directories.
Each run (logdir) appears as a separate trace on every plot.

Example
-------
python plot_tensorboard_scalars.py logs/exp1 logs/exp2 \
    --names "baseline" "with_augmentation" --outdir figures
"""
from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# ------------------------------------------------------------
# Scalar-reading helper  (adapted from the previous answer)
# ------------------------------------------------------------
#   Returns {tag : [(step, wall_time, value), ...]}
ScalarPoint = Tuple[int, float, float]


matplotlib.rc("font", size=16) # set big font size


def load_scalars_from_text_file(
    filepath: str | Path,
) -> Dict[str, List[ScalarPoint]]:
    """Load scalar values from a text file with format like:
    0        ℒᴰ = 0.633395
    1        ℒᴰ = 0.466986
    2        ℒᴰ = 0.401379
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Text file not found: {filepath}")

    # Regex to match lines with step and loss values
    pattern = r'^(\d+)\s+(\S+)\s*=\s*(\d+\.\d+)'

    scalars: Dict[str, List[ScalarPoint]] = {}
    with open(filepath, 'r') as f:
        for line in f:
            match = re.match(pattern, line.strip())
            if match:
                step = int(match.group(1))
                value = float(match.group(3))
                # Use step as wall_time since we don't have actual wall_time
                scalars.setdefault("loss", []).append((step, float(step), value))

    return scalars


def load_scalars(
    logdir: str | Path,
    *,
    tags: list[str] | None = None,
    size_guidance: int = 0,
) -> Dict[str, List[ScalarPoint]]:
    logdir = Path(logdir)

    # Check if it's a regular text file
    if logdir.is_file() and not str(logdir).endswith('.tfevents.'):
        return load_scalars_from_text_file(logdir)

    # Handle TensorBoard files
    event_files = (
        [logdir] if logdir.is_file() else sorted(logdir.glob("**/events.out.tfevents.*"))
    )
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {logdir}")

    scalars: Dict[str, List[ScalarPoint]] = {}
    for ev_path in event_files:
        ea = event_accumulator.EventAccumulator(
            str(ev_path), size_guidance={"scalars": size_guidance}
        )
        ea.Reload()

        for tag in ea.Tags().get("scalars", []):
            if tags is not None and tag not in tags:
                continue
            scalars.setdefault(tag, []).extend(
                (ev.step, ev.wall_time, ev.value) for ev in ea.Scalars(tag)
            )

    # Sort each tag by step so runs that resumed training are monotonic
    for tag in scalars:
        scalars[tag].sort(key=lambda p: p[0])

    return scalars


# ------------------------------------------------------------
# CLI + plotting
# ------------------------------------------------------------
def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot TensorBoard scalars from multiple runs."
    )
    p.add_argument(
        "runs",
        nargs="+",
        help="One or more log directories (each containing *.tfevents.* files).",
    )
    p.add_argument(
        "--names",
        nargs="+",
        help="Legend labels for the runs (defaults to directory names).",
    )
    p.add_argument(
        "--outdir",
        help="If given, save each figure here as PNG instead of showing interactively.",
    )
    p.add_argument(
        "--size_guidance",
        type=int,
        default=0,
        metavar="N",
        help="Pass size guidance to EventAccumulator (0 = load all points).",
    )
    p.add_argument(
      "--window",
      type=int,
      default=1,
      help="smoothing window width",
    )
    p.add_argument(
      "--logx",
      action="store_true",
      help="Set x axis to log scale",
    )
    return p.parse_args()


def main() -> None:
    args = parse_cli()

    run_paths: list[str] = args.runs
    run_labels: list[str] = (
        args.names if args.names else [Path(r).name for r in run_paths]
    )
    if len(run_labels) != len(run_paths):
        raise SystemExit("ERROR: --names must match the number of runs given.")

    # Load every run
    print("Loading scalars ...")
    run_data: list[dict[str, list[ScalarPoint]]] = [
        load_scalars(p, size_guidance=args.size_guidance) for p in run_paths
    ]

    # Union of all scalar tags across runs
    all_tags = sorted(
        set(itertools.chain.from_iterable(d.keys() for d in run_data))
    )
    if not all_tags:
        raise SystemExit("No scalar summaries found in the supplied logs.")

    # Prepare output directory if requested
    outdir = Path(args.outdir).expanduser() if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    # Plot every tag
    print(f"Creating {len(all_tags)} figure(s) ...")
    slice_start = 10 if args.logx else 0
    for tag in all_tags:
        values_mins = []
        plt.figure()
        for label, data in zip(run_labels, run_data):
            if tag not in data:
                continue
            steps = np.array([pt[0] for pt in data[tag]])
            values = np.array([pt[2] for pt in data[tag]])
            if args.window > 1:
              #values = savgol_filter(values, args.window, 0)
              values = gaussian_filter1d(values, sigma=args.window)
            values_mins.append(values[slice_start:].min())
            plt.plot(steps, values, label=label)
        plt.title(f"{'smoothed ' if args.window > 1 else ''}{tag}")
        plt.xlabel("training step")
        plt.ylabel(tag)
        if args.logx:
          plt.xscale("log")
          plt.xlim(slice_start, None)
        plt.yscale("log")
        # Get ymin as the minimum of all values across all runs for this tag
        ymin = min(values_mins)
        # Get ymax as the 99th percentile of all values across all runs for this tag
        #ymax = max([np.percentile([pt[2] for pt in data[tag][slice_start:]], 99) for data in run_data if tag in data])
        ymax = max([max([pt[2] for pt in data[tag][slice_start:]]) for data in run_data if tag in data])
        # Set y-axis limits
        plt.ylim(ymin, ymax)
        plt.legend()
        plt.tight_layout()

        # Save or display
        if outdir:
            fname = tag.replace("/", "_") + ".png"
            plt.savefig(outdir / fname, dpi=150)
            plt.close()
        else:
            plt.show()

    print("Done.")


if __name__ == "__main__":
    main()
