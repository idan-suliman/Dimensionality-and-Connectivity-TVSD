from __future__ import annotations
import numpy as np
from core.runtime import runtime

def build_trial_matrix(*args, **kwargs):
    """
    Local wrapper for runtime.get_data_manager().build_trial_matrix.
    """
    return runtime.get_data_manager().build_trial_matrix(*args, **kwargs)

def build_groups_by_rep_or_subsets(trials: list[dict], *, k_subsets: int | None, random_state: int):
    """
    Partition trials into groups:
    - If k_subsets is None: Group by repetition index (0..29).
    - If k_subsets is Int:  Partition randomly into K subsets.
    """
    rep_arr = np.array([int(tr["allmat_row"][3]) - 1 for tr in trials], dtype=int)

    if k_subsets is None:
        rep_ids = np.unique(rep_arr)
        groups = [np.flatnonzero(rep_arr == g) for g in rep_ids]
        label_D = "Repetitions"
        id_print = (lambda g: f"rep {g:2}")
    else:
        rng = np.random.default_rng(random_state)
        idxs = np.arange(len(trials))
        rng.shuffle(idxs)
        splits = np.array_split(idxs, k_subsets)
        groups = [np.asarray(part, dtype=int) for part in splits]
        label_D = f"{k_subsets} random subsets"
        id_print = (lambda g: f"set {g:2}")

    return groups, label_D, id_print, rep_arr
