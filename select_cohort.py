"""Grow the DREAMT cohort from 10 to 40 people, reproducibly.

Why this exists: the cohort is a random draw, and a random draw is only
checkable if the exact rule and seed are published with the code.

Rule: 8 cells = 4 apnea bands x 2 sexes, 5 people per cell.
Bands use AHI (events per hour): normal <5, mild 5-15, moderate 15-30,
severe >=30, lower edge included. The 10 people chosen on 2026-09-21
(seed 20260921) are kept. Each cell is topped up to 5 by a random draw,
seed 20260925, from the people not yet chosen, taken in SID order.

Prints only IDs and counts. Per-person AHI stays out of the output because
the licence forbids redistributing participant data.

Usage: python select_cohort.py <path to participant_info.csv> <urls out>
"""
import sys
import numpy as np
import pandas as pd

KEPT = ["S006", "S015", "S022", "S024", "S030",
        "S035", "S042", "S064", "S093", "S103"]
PER_CELL = 5
SEED = 20260925
BANDS = ["normal", "mild", "moderate", "severe"]
URL = "https://physionet.org/files/dreamt/2.2.0/data_64Hz/{}_whole_df.csv"


def select(info):
    info = info.assign(band=pd.cut(info.AHI, [0, 5, 15, 30, np.inf],
                                   right=False, labels=BANDS))
    if info.band.isna().any():
        raise ValueError("AHI outside the band edges")
    rng = np.random.default_rng(SEED)
    new = []
    for band in BANDS:
        for sex in ["F", "M"]:
            cell = info[(info.band == band) & (info.GENDER == sex)]
            have = cell.SID.isin(KEPT).sum()
            pool = sorted(cell.SID[~cell.SID.isin(KEPT)])
            need = PER_CELL - have
            if need < 0 or need > len(pool):
                raise ValueError(f"cell {band}/{sex} cannot reach {PER_CELL}")
            new += list(rng.choice(pool, size=need, replace=False))
            print(f"{band:9s}{sex}  kept {have}  drawn {need}  pool {len(pool)}")
    return sorted(new)


if __name__ == "__main__":
    info = pd.read_csv(sys.argv[1])
    new = select(info)
    print(f"\n{len(new)} new: {' '.join(new)}")
    print(f"cohort: {len(KEPT) + len(new)}")
    with open(sys.argv[2], "w") as f:
        f.write("\n".join(URL.format(s) for s in new) + "\n")
