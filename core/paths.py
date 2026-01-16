from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Any
import numpy as np

# Avoid circular import by using TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .runtime import RUNTIME

class Paths:
    """
    Centralized path generation and management for the TVSD codebase.
    Accessed via `runtime.paths`.
    """

    def __init__(self, cfg: CONFIG, consts: CONSTANTS):
        self._cfg = cfg
        self._consts = consts
    
    @property
    def cfg(self):
        return self._cfg
    
    @property
    def consts(self):
        return self._consts

    # =========================================================================
    # Core Output Directories (Proxied from Config)
    # =========================================================================

    def get_data_path(self) -> Path:
        return self.cfg.get_data_path()

    def get_plot_dir(self, create: bool = True) -> Path:
        return self.cfg.get_plot_dir()
    
    def get_target_rrr_dir(self, create: bool = True) -> Path:
        return self.cfg.get_target_rrr_dir(create=create)

    def get_regular_rrr_dir(self, create: bool = True) -> Path:
        return self.cfg.get_regular_rrr_dir(create=create)

    def get_compare_rrr_dir(self, create: bool = True) -> Path:
        return self.cfg.get_compare_rrr_dir(create=create)

    # =========================================================================
    # Dimensionality Correlation
    # From: methods/dimensionality_correlation/utils.py
    # =========================================================================

    def get_dim_corr_path(
        self,
        monkey_name: str | None = None,
        analysis_type: str | None = None,
        group_size: int | None = None,
        region_id: int | None = None,
        src_tgt: Tuple[int, int] | None = None,
        suffix: str = "_dim_corr",
        extension: str = ".npz",
        output_dir: str | Path | None = None
    ) -> Path:
        """
        Generate consistent filename for Dimensionality Correlation.
        """
        if output_dir is None:
            output_dir = self.get_data_path()
        
        # Infer configuration if arguments are missing
        cfg = self.cfg
        if monkey_name is None: monkey_name = cfg.get_monkey_name()
        
        # Verify required arguments
        if analysis_type is None: raise ValueError("analysis_type is required")
        if group_size is None: raise ValueError("group_size is required")

        base = Path(output_dir) / "Dimensionality_vs_Correlation"
        mk = monkey_name.replace(" ", "")
        bt = f"blk{group_size}"

        # Determine directory and filename base
        if extension in {".png", ".pdf", ".svg", ".jpg"}:
            target_dir = base / "Plot"
            suff_final = suffix
        else:
            # Data paths
            if region_id is not None:
                target_dir = base / "Region"
            elif src_tgt is not None:
                target_dir = base / "Connection"
            else:
                target_dir = base # Fallback
            suff_final = suffix

        target_dir.mkdir(parents=True, exist_ok=True)

        consts = self.consts
        if region_id is not None:
            nm = consts.REGION_ID_TO_NAME.get(region_id, f"Reg{region_id}")
            fname = f"{mk}_{nm}_{analysis_type}_{bt}{suff_final}{extension}"
        elif src_tgt is not None:
            s = consts.REGION_ID_TO_NAME.get(src_tgt[0], f"Reg{src_tgt[0]}")
            t = consts.REGION_ID_TO_NAME.get(src_tgt[1], f"Reg{src_tgt[1]}")
            fname = f"{mk}_{s}_to_{t}_{analysis_type}_{bt}{suff_final}{extension}"
        else:
            # Fallback/Summary
            fname = f"{mk}_{analysis_type}_{bt}{suff_final}{extension}"

        return target_dir / fname

    # =========================================================================
    # Repetition Stability
    # From: methods/repetition_stability/utils.py
    # =========================================================================

    def get_rep_stability_path(
        self,
        monkey_name: str | None = None,
        analysis_type: str | None = None,
        group_size: int | None = None,
        region_id: int | None = None, 
        src_tgt: Tuple[int, int] | None = None, 
        suffix: str = "",
        extension: str = ".npz",
        output_dir: str | Path | None = None
    ) -> Path:
        """
        Generate consistent filename for Repetition Stability.
        """
        if output_dir is None:
            output_dir = self.get_data_path()
        if monkey_name is None: monkey_name = self.cfg.get_monkey_name()
        if analysis_type is None: raise ValueError("analysis_type is required")
        if group_size is None: raise ValueError("group_size is required")

        base = Path(output_dir) / "Repetition_Stability"
        mk = monkey_name.replace(" ", "")
        bt = f"blk{group_size}"
        
        consts = self.consts

        # Determine Subdirectory & Filename
        if extension in {".png", ".pdf", ".svg", ".jpg"}:
            target_dir = base / "Repetition_Stability_plots"
            if region_id is not None:
                 nm = consts.REGION_ID_TO_NAME.get(region_id, f"Reg{region_id}")
                 fname = f"{mk}_{nm}_{analysis_type}_{bt}{suffix}{extension}"
            elif src_tgt is not None:
                 s = consts.REGION_ID_TO_NAME.get(src_tgt[0], f"Reg{src_tgt[0]}")
                 t = consts.REGION_ID_TO_NAME.get(src_tgt[1], f"Reg{src_tgt[1]}")
                 fname = f"{mk}_{s}_to_{t}_{analysis_type}_{bt}{suffix}{extension}"
            else:
                 fname = f"{mk}_RepetitionStability_{analysis_type}_{bt}{suffix}{extension}"

        elif region_id is not None:
            target_dir = base / "Region"
            nm = consts.REGION_ID_TO_NAME.get(region_id, f"Reg{region_id}")
            fname = f"{mk}_{nm}_{analysis_type}_{bt}{suffix}{extension}"
            
        elif src_tgt is not None:
            target_dir = base / "Connection"
            s = consts.REGION_ID_TO_NAME.get(src_tgt[0], f"Reg{src_tgt[0]}")
            t = consts.REGION_ID_TO_NAME.get(src_tgt[1], f"Reg{src_tgt[1]}")
            fname = f"{mk}_{s}_to_{t}_{analysis_type}_{bt}{suffix}{extension}"
            
        else:
            raise ValueError("Must provide either region_id or src_tgt pair.")
            
        target_dir.mkdir(parents=True, exist_ok=True)
            
        return target_dir / fname

    # =========================================================================
    # Semedo Replication
    # From: methods/Semedo/utils.py
    # =========================================================================

    def get_semedo_figure_path(
        self,
        figure_key: str, 
        analysis_type: str, 
        *,
        source_region: int | None = None,
        target_region: int | None = None,
        k_subsets: int | None = None,
        n_src: int | None = None,
        n_tgt: int | None = None,
        n_runs: int | None = None,
        num_sets: int | None = None,
    ) -> Path:
        """
        Centralized path generation for Semedo figures.
        """
        cfg = self.cfg
        data_path = cfg.get_data_path()
        monkey = cfg.get_monkey_name().replace(' ', '')
        consts = self.consts
        
        if figure_key == "Figure_4":
            if source_region is None or target_region is None:
                 raise ValueError("Figure_4 requires source_region and target_region")
            tgt_name = consts.REGION_ID_TO_NAME[target_region]
            out_dir = data_path / "Semedo_plots" / "Figure_4"
            suffix = f"sub{k_subsets}" if k_subsets else "rep30"
            base_name = f"{monkey}_Figure_4_{suffix}_{analysis_type}_V1_to_{tgt_name}"
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir / f"{base_name}.npz"
            
        elif figure_key == "Figure_4_subset":
            if any(x is None for x in [source_region, target_region, n_src, n_tgt, n_runs]):
                 raise ValueError("Figure_4_subset requires region IDs and counts")
            tgt_name = consts.REGION_ID_TO_NAME[target_region]
            out_dir = data_path / "Semedo_plots" / "Figure_4_subset" / analysis_type
            base_name = (
                f"{monkey}_Figure_4_subset_"
                f"src{n_src}_tgt{n_tgt}_runs{n_runs}_"
                f"{analysis_type}_V1_to_{tgt_name}"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir / f"{base_name}.npz"
            
        elif figure_key == "Figure_5_B":
            if num_sets is None:
                 raise ValueError("Figure_5_B requires num_sets")
            out_dir = data_path / "Semedo_plots" / "figure_5_B"
            filename = f"figure_5_B_{analysis_type.upper()}_{num_sets}_SETS.png"
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir / filename

        raise ValueError(f"Unknown figure_key: {figure_key}")

    # =========================================================================
    # Matching Subset
    # From: methods/matchingSubset.py
    # =========================================================================

    def get_matching_path(
        self,
        stat_mode: str,
        source_region: str,
        target_region: str,
        suffix: str = ".npz"
    ) -> Path:
        """
        Centralized path generation for matching subset analysis.
        """
        root = self.cfg.get_data_path() / "TARGET_RRR" / stat_mode.upper()
        root.mkdir(parents=True, exist_ok=True)
        base = f"{source_region}_to_{target_region}_{stat_mode}"
        return root / f"{base}{suffix}"
