"""
python plot_metric_grid.py \
  --metric episode_return \
  --file "Enb 0::runs/env0.csv" \
  --file "Enb 1::runs/env1.csv" \
  --file "Enb 2::runs/env2.csv"
"""
import argparse
import math
import os

import pandas as pd
import matplotlib.pyplot as plt


def parse_file_arg(file_arg: str):
    """
    Parse 'name::path' into (name, path).
    """
    if "::" not in file_arg:
        raise ValueError(
            f'Invalid --file argument "{file_arg}". '
            'Expected format: "name::path/to/file.csv"'
        )
    name, path = file_arg.split("::", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise ValueError(f'Empty name in --file argument: "{file_arg}"')
    if not path:
        raise ValueError(f'Empty path in --file argument: "{file_arg}"')
    return name, path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot a single metric from multiple CSV files, "
            "one subplot per file (2 plots per row), with shared axis scales."
        )
    )
    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        required=True,
        help='CSV file spec in the form "name::path/to/file.csv". '
             "Can be given multiple times.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        help="Name of the metric (column) to plot from each CSV.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=14.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--height_per_row",
        type=float,
        default=3.0,
        help="Figure height per row in inches.",
    )

    args = parser.parse_args()

    metric = args.metric
    series_list = []  # list of (name, series)

    # ---------- load data ----------
    for file_arg in args.files:
        name, path = parse_file_arg(file_arg)

        if not os.path.isfile(path):
            raise FileNotFoundError(f'File for "{name}" not found: {path}')

        df = pd.read_csv(path)

        if metric not in df.columns:
            raise ValueError(
                f'Metric "{metric}" not found in file "{path}". '
                f"Available columns: {list(df.columns)}"
            )

        s = pd.to_numeric(df[metric], errors="coerce")
        if s.isna().all():
            raise ValueError(
                f'Metric "{metric}" in file "{path}" is not numeric or all NaN.'
            )

        series_list.append((name, s))

    if not series_list:
        raise ValueError("No valid files/metrics to plot.")

    # ---------- compute global scales ----------
    # x-axis: we use index (0..len-1). We want same x-range across all.
    max_len = max(len(s) for _, s in series_list)
    # global x range [0, max_len-1], then add +25% padding on the right
    if max_len > 1:
        x_max_base = max_len - 1
    else:
        x_max_base = 0
    x_range = max(x_max_base, 1)  # avoid zero range
    x_pad = 0.25 * x_range
    x_min_global = -0.5  # small left margin so first point isn't on the border
    x_max_global = x_max_base + x_pad

    # y-axis: based on metric values across all files
    y_min_global = min(s.min() for _, s in series_list)
    y_max_global = max(s.max() for _, s in series_list)
    y_range = y_max_global - y_min_global

    if y_range == 0:
        # all values same across all series; create a symmetric window around that value
        base = max(abs(y_max_global), 1.0)
        y_pad = 0.25 * base
        y_min_global = y_min_global - y_pad
        y_max_global = y_max_global + y_pad
    else:
        y_pad = 0.25 * y_range
        y_min_global = y_min_global - y_pad
        y_max_global = y_max_global + y_pad

    # ---------- prepare grid ----------
    n_plots = len(series_list)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(args.width, args.height_per_row * n_rows),
    )

    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    # ---------- plot ----------
    for i, (name, s) in enumerate(series_list):
        ax = axes[i]
        x = s.index
        ax.plot(x, s.values, marker="o")
        ax.set_title(name)
        ax.set_xlabel("Row index")
        ax.set_ylabel(metric)

        # apply global limits
        ax.set_xlim(x_min_global, x_max_global)
        ax.set_ylim(y_min_global, y_max_global)

    # hide unused axes
    for j in range(len(series_list), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
