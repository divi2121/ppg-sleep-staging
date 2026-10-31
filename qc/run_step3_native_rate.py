"""Step 3: native_rate per column on every 64 Hz wristband file.

Usage: python -I run_step3_native_rate.py <data root> <table out> <figure out>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_funcs import load_wristband, native_rate

COLS = ["BVP", "ACC_X", "ACC_Y", "ACC_Z", "TEMP", "EDA", "HR", "IBI"]

root, table_out, fig_out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
files = sorted(list(root.rglob("S???_whole_df.csv")) + list(root.rglob("S006_test.csv")),
               key=lambda p: p.name)
print("files found:", len(files))

rows = []
for f in files:
    subject, df = load_wristband(f)
    row = {"subject": subject}
    for c in COLS:
        row[c] = native_rate(df[c].to_numpy())
    rows.append(row)
    print(subject, " ".join(f"{c}={row[c]:.2f}" for c in COLS), flush=True)

tab = pd.DataFrame(rows).set_index("subject").sort_index()
table_out.parent.mkdir(parents=True, exist_ok=True)
tab.to_csv(table_out)
print("\nmedian across people (changes/s):")
print(tab.median().round(3).to_string())
acc = tab[["ACC_X", "ACC_Y", "ACC_Z"]].max(axis=1)
print(f"\nACC, fastest axis per person: median {acc.median():.2f}, min {acc.min():.2f}, max {acc.max():.2f}")

plt.rcParams.update({"font.size": 8})
fig, ax = plt.subplots(figsize=(5, 3))
rng = np.random.default_rng(0)
for i, c in enumerate(COLS):
    y = tab[c].to_numpy()
    ax.scatter(i + rng.uniform(-0.15, 0.15, y.size), y, s=6, color="0.3")
for ref in (64, 32, 8, 4, 1):
    ax.axhline(ref, color="0.8", lw=0.6, zorder=0)
ax.set_xticks(range(len(COLS)))
ax.set_xticklabels(COLS)
ax.set_yscale("log")
ax.set_ylabel("value changes per second")
fig.tight_layout()
fig_out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_out, dpi=150)
print("saved", fig_out)
