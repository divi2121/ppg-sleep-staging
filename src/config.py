"""All pipeline settings in one frozen object.

Logic approved in CODE_PLAN.md section 1.
"""
from dataclasses import asdict, dataclass, replace
import json


# LOGIC: one frozen dataclass holds every setting the pipeline uses. Functions take
#   their settings from it, so a run is fully described by its Config. It is saved
#   next to every result (to_json) so any number can be traced to its settings.
#   Offline and causal runs differ only by Config.causal.
# EDGE CASES: frozen=True, so a setting cannot change mid-run; with() returns a
#   changed copy instead.
# NOT HANDLED: no validation of value ranges; nonsense values fail inside the
#   function that uses them.
# CHECKED BY: tests/test_structure.py::test_config_frozen_and_roundtrip
@dataclass(frozen=True)
class Config:
    """Every setting, with the value in use on 2026-10-11."""
    # data
    fs: int = 64
    epoch_s: int = 30
    # filter
    filt_low: float = 0.5
    filt_high: float = 8.0
    filt_order: int = 2
    causal: bool = False
    # beats
    detector: str = "msptdfast"
    detect_chunk_s: int = 600
    detect_overlap_s: int = 10
    detect_pre_anchor_s: int = 300
    ibi_lo: float = 0.3
    ibi_hi: float = 2.0
    ibi_rel: float = 0.30
    ibi_win: int = 11
    match_tol: float = 0.05
    # quality gate (first guesses, to be tuned on S042 only)
    q_min_kept: float = 0.8
    q_min_template_r: float = 0.8
    q_max_motion: float = float("inf")
    # features
    freq_win_s: int = 300
    context_k: int = 4
    # label timing: "none" or "corrected" (CODE_PLAN decision 1)
    label_timing: str = "none"
    # live decoding (2026-10-24). ASSUMPTION: 3 s wait, not tuned; notebook 10
    # section 1 found 1 s already keeps 99.71% of beats (median of 5), 3 s keeps 99.80%.
    # The stage for an epoch is output live_lag_s + 1 s after the epoch ends.
    live_lag_s: float = 3.0
    live_buffer_s: int = 30
    # reproducibility
    seed: int = 20261011

    def with_(self, **changes):
        """A copy with some settings changed."""
        return replace(self, **changes)

    def to_json(self):
        """The settings as a JSON string, to save beside a result."""
        return json.dumps(asdict(self), sort_keys=True)
