"""Plot experiment logs from ExperimentTracker CSV output."""

import csv
import sys
from pathlib import Path
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)


def plot_log(csv_path: str, output_path: str = None):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("No data found")
        return

    keys = [k for k in rows[0].keys() if k not in ('step', 'epoch', 'timestamp')]
    steps = [int(r.get('step', i)) for i, r in enumerate(rows)]
    epochs = [int(r.get('epoch', 0)) for r in rows]

    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 3 * len(keys)), squeeze=False)
    for ax, key in zip(axes, keys):
        values = [float(r.get(key, 0)) for r in rows]
        ax.plot(steps, values, marker='.', linestyle='-', label=key)
        ax.set_xlabel('Step')
        ax.set_ylabel(key)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_path or Path(csv_path).with_suffix('.png')
    plt.savefig(out, dpi=150)
    print(f"Plot saved to {out}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_logs.py <csv_path> [output_path]")
        sys.exit(1)
    plot_log(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
