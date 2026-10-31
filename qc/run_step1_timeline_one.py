"""Step 1: load one wristband file, check its timeline, plot its time steps.

Usage: python -I run_step1_timeline_one.py <csv path> <figure out path>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_funcs import load_wristband, check_timeline

csv_path, fig_path = sys.argv[1], sys.argv[2]
subject, df = load_wristband(csv_path)
print(subject, df.shape)
print(df.dtypes.to_string())
res = check_timeline(df["TIMESTAMP"].to_numpy())
for k, v in res.items():
    print(f"{k}: {v}")
print("first / last timestamp:", df["TIMESTAMP"].iloc[0], df["TIMESTAMP"].iloc[-1])

steps_ms = np.diff(df["TIMESTAMP"].to_numpy()) * 1000
plt.rcParams.update({"font.size": 8})
fig, ax = plt.subplots(figsize=(4, 2.6))
ax.hist(steps_ms, bins=200, color="0.3")
ax.axvline(1000 / 64, color="tab:red", lw=0.8, ls="--")
ax.set_yscale("log")
ax.set_xlabel("time step (ms)")
ax.set_ylabel("count")
fig.tight_layout()
Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_path, dpi=150)
print("saved", fig_path)
