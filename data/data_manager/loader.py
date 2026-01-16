from __future__ import annotations
import pickle
import sys

def load_trials(manager):
    """
    Load trials from pickle, cached in memory.
    Expected to be called with a DataManager instance.
    """
    if manager._trials_cache is not None:
        return manager._trials_cache

    p = manager.config.get_main_data_file_path()
    if not p.exists():
        print(f"[!] Data file not found: {p}")
        print("[*] Attempting to auto-build data via DataBuilder...")
        try:
            # Import locally to avoid circular dependencies
            from ..databuilder import DataBuilder
            DataBuilder.build_if_missing()
        except Exception as e:
            print(f"[!] Auto-build failed: {e}")

    if not p.exists():
        raise FileNotFoundError(f"[!]: data file not found: {p}")
    
    with open(p, "rb") as f:
        trials = pickle.load(f)
    if trials is None:
        raise RuntimeError("Trials cache empty – run TVSDAnalysis.load_trials()")
    
    manager._trials_cache = trials
    return trials
