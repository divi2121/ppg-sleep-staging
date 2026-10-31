"""Step 2: check_timeline on every 64 Hz wristband file, plot hours per person.

Usage: python -I run_step2_timeline_all.py <data root> <table out> <figure out>
<data root> is searched for S???_whole_df.csv and S006_test.csv.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_funcs import load_wristband, check_timeline

root, table_out, fig_out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
files = sorted(list(root.rglob("S???_whole_df.csv")) + list(root.rglob("S006_test.csv")),
               key=lambda p: p.name)
print("files found:", len(files))

rows = []
for f in files:
    subject, df = load_wristband(f)
    res = check_timeline(df["TIMESTAMP"].to_numpy())
    res["subject"] = subject
    res["ncols"] = df.shape[1]
    res["t0"] = float(df["TIMESTAMP"].iloc[0])
    rows.append(res)
    print(subject, f"{res['hours']:.2f} h", res["n_off_grid"], res["n_gaps"], flush=True)

tab = pd.DataFrame(rows).set_index("subject").sort_index()
table_out.parent.mkdir(parents=True, exist_ok=True)
tab.to_csv(table_out)
print()
print(tab.to_string())
bad = tab[(tab.n_off_grid > 0) | (tab.n_gaps > 0) | (~tab.monotonic)]
print(f"\nfiles with any off-grid step, gap or backward step: {len(bad)} of {len(tab)}")
print("duplicate subjects:", tab.index.duplicated().sum())
print(f"hours: min {tab.hours.min():.2f}, median {tab.hours.median():.2f}, max {tab.hours.max():.2f}, total {tab.hours.sum():.1f}")

plt.rcParams.update({"font.size": 8})
fig, ax = plt.subplots(figsize=(7, 2.6))
ax.bar(range(len(tab)), tab.hours, color="0.3")
ax.set_xticks(range(len(tab)))
ax.set_xticklabels(tab.index, rotation=90)
ax.set_ylabel("hours recorded")
fig.tight_layout()
fig_out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_out, dpi=150)
print("saved", fig_out)
