import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

@dataclass
class ReclusterConfig:
    FORCE_MIN_COVERAGE: float = 0.95
    FORCE_RT_TOL: float = 0.1
    FORCE_MASS_TOL: float = 0.0001

class Reclusterer:
    PATCH_STEM_RE = re.compile(
        r"""
        ^(?P<file_base>.+?)
        _mass(?P<mass>\d+(?:\.\d+)?)
        _Peak(?P<peak_num>\d+)
        _(?P<group>Group\d+)$
        """,
        re.VERBOSE,
    )

    def __init__(self, dirs: Dict[str, str | Path], config: Optional[ReclusterConfig] = None):
        self.dirs = {k: Path(v) for k, v in dirs.items()}
        self.cfg = config or ReclusterConfig()
        for k in ("patch", "pixel", "clustering"):
            if k not in self.dirs:
                raise KeyError(f"dirs must include '{k}'")
        self.dirs["clustering"].mkdir(parents=True, exist_ok=True)

        # NEW: cache pixel tables per file_base
        self._pixel_cache: Dict[Tuple[str, str], pd.DataFrame] = {}

    def run(self, group: str) -> pd.DataFrame:
        cluster_dir = self.dirs["clustering"]
        unclustered_csv = cluster_dir / f"unclustered_peaks_group_{group}.csv"
        summary_csv = cluster_dir / f"alignment_summary_group_{group}.csv"
        out_csv = cluster_dir / f"Feature_list_{group}.csv"

        if not summary_csv.exists():
            raise FileNotFoundError(f"Missing alignment summary: {summary_csv}")
        df_sum = pd.read_csv(summary_csv)

        if not unclustered_csv.exists():
            df_out = self._add_forced_columns(df_sum)
            df_out.to_csv(out_csv, index=False)
            return df_out

        df_un = pd.read_csv(unclustered_csv)
        if df_un.empty or "peak_id" not in df_un.columns:
            df_out = self._add_forced_columns(df_sum)
            df_out.to_csv(out_csv, index=False)
            return df_out

        # Validate expected summary columns
        required = ["Group", "m/z", "Isomer_position", "Aligned_rt_apex", "peak count"]
        missing = [c for c in required if c not in df_sum.columns]
        if missing:
            raise ValueError(
                f"alignment_summary missing required columns {missing}. Found: {list(df_sum.columns)}"
            )

        # Identify file bases from *_height columns
        file_bases = sorted({c[:-len("_height")] for c in df_sum.columns if c.endswith("_height")})
        total_files = len(file_bases)
        if total_files == 0:
            df_out = self._add_forced_columns(df_sum)
            df_out.to_csv(out_csv, index=False)
            return df_out

        # Coverage threshold
        min_required = int(np.ceil(self.cfg.FORCE_MIN_COVERAGE * total_files))

        # Normalize numeric types
        df_sum = df_sum.copy()
        df_sum["m/z"] = pd.to_numeric(df_sum["m/z"], errors="coerce")
        df_sum["Aligned_rt_apex"] = pd.to_numeric(df_sum["Aligned_rt_apex"], errors="coerce")
        df_sum["peak count"] = pd.to_numeric(df_sum["peak count"], errors="coerce").fillna(0).astype(int)
        df_sum = self._add_forced_columns(df_sum)

        # Pre-parse all unclustered peaks and build a small table with needed values
        peaks_df = self._build_unclustered_table(df_un["peak_id"].astype(str), group)
        if peaks_df.empty:
            df_sum.to_csv(out_csv, index=False)
            return df_sum

        # Only consider peaks whose file exists in summary columns
        peaks_df = peaks_df[peaks_df["file_base"].isin(file_bases)].copy()
        if peaks_df.empty:
            df_sum.to_csv(out_csv, index=False)
            return df_sum

        # For speed, create an "eligible features" view once
        eligible_features = df_sum[
            (df_sum["peak count"] >= min_required) &
            (df_sum["m/z"].notna()) &
            (df_sum["Aligned_rt_apex"].notna())
        ].copy()

        # Attach loop (fast: no file reads inside)
        for _, pk in peaks_df.iterrows():
            file_base = pk["file_base"]
            mass = float(pk["mass"])
            rt_apex = float(pk["rt_apex"])
            height = float(pk["height"])
            area = float(pk["area"])

            h_col = f"{file_base}_height"
            a_col = f"{file_base}_area"
            if h_col not in df_sum.columns or a_col not in df_sum.columns:
                continue  # can't attach into missing columns

            # Candidate features by mass + RT tolerance on eligible set
            cand = eligible_features[
                (np.abs(eligible_features["m/z"] - mass) <= self.cfg.FORCE_MASS_TOL) &
                (np.abs(eligible_features["Aligned_rt_apex"] - rt_apex) <= self.cfg.FORCE_RT_TOL)
            ]

            # Unambiguous rule
            if len(cand) != 1:
                continue

            target_idx = cand.index[0]

            # Must NOT override existing assignment
            if pd.notna(df_sum.loc[target_idx, h_col]) or pd.notna(df_sum.loc[target_idx, a_col]):
                continue

            # Attach
            df_sum.loc[target_idx, h_col] = height
            df_sum.loc[target_idx, a_col] = area

            # Mark forced flags
            df_sum.loc[target_idx, "Forced_any"] = True
            df_sum.loc[target_idx, "Forced_n"] = int(df_sum.loc[target_idx, "Forced_n"]) + 1
            prev = str(df_sum.loc[target_idx, "Forced_files"]).strip()
            df_sum.loc[target_idx, "Forced_files"] = (prev + ";" if prev else "") + file_base

        # Recompute peak count robustly from heights
        height_cols = [c for c in df_sum.columns if c.endswith("_height")]
        df_sum["peak count"] = df_sum[height_cols].notna().sum(axis=1).astype(int)

        df_sum.sort_values(["m/z", "Isomer_position"], inplace=True)
        df_sum.to_csv(out_csv, index=False)
        return df_sum

    def _build_unclustered_table(self, peak_ids: pd.Series, group: str) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for peak_id in peak_ids.tolist():
            parsed = self._parse_peak_id(peak_id)
            if parsed is None:
                continue
            file_base, mass, peak_num, grp = parsed
            if grp != group:
                continue

            # pull rt_apex/height/area from cached pixel table
            vals = self._get_peak_vals_cached(file_base, peak_num, group)
            if vals is None:
                continue
            rt_apex, height, area = vals

            rows.append({
                "peak_id": peak_id,
                "file_base": file_base,
                "mass": float(mass),
                "peak_num": int(peak_num),
                "rt_apex": float(rt_apex),
                "height": float(height),
                "area": float(area),
            })
        return pd.DataFrame(rows)

    def _get_peak_vals_cached(self, file_base: str, peak_num: int, group: str) -> Optional[Tuple[float, float, float]]:
        df = self._load_pixel_df_cached(file_base, group)
        if df is None:
            return None
        if "peak_num" not in df.columns or "RT_apex" not in df.columns or "height" not in df.columns or "area" not in df.columns:
            return None

        hits = df.loc[df["peak_num"].astype(int) == int(peak_num)]
        if hits.shape[0] != 1:
            return None
        r = hits.iloc[0]
        return float(r["RT_apex"]), float(r["height"]), float(r["area"])

    def _load_pixel_df_cached(self, file_base: str, group: str) -> Optional[pd.DataFrame]:
        key = (file_base, group)
        if key in self._pixel_cache:
            return self._pixel_cache[key]

        path = self.dirs["pixel"] / f"{file_base}_peaks_pix_{group}.csv"
        if not path.exists():
            self._pixel_cache[key] = None  # remember missing
            return None

        df = pd.read_csv(path)

        # normalize Pixel_start/Pixel_end in case other code uses it later
        if "pixel_start" not in df.columns and "Pixel_start" in df.columns:
            df = df.rename(columns={"Pixel_start": "pixel_start"})
        if "pixel_end" not in df.columns and "Pixel_end" in df.columns:
            df = df.rename(columns={"Pixel_end": "pixel_end"})

        self._pixel_cache[key] = df
        return df

    def _add_forced_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "Forced_any" not in df.columns:
            df["Forced_any"] = False
        if "Forced_n" not in df.columns:
            df["Forced_n"] = 0
        if "Forced_files" not in df.columns:
            df["Forced_files"] = ""
        return df

    def _parse_peak_id(self, peak_id: str) -> Optional[Tuple[str, float, int, str]]:
        m = self.PATCH_STEM_RE.match(str(peak_id))
        if not m:
            return None
        return (
            m.group("file_base"),
            float(m.group("mass")),
            int(m.group("peak_num")),
            m.group("group"),
        )

def process_file_recluster(dirs: Dict[str, str | Path], group_name: str, config: Optional[ReclusterConfig] = None) -> str:
    t0 = time.time()
    reclusterer = Reclusterer(dirs=dirs, config=config)
    df_out = reclusterer.run(group=group_name)
    elapsed = time.time() - t0

    forced_any = int(df_out["Forced_any"].sum()) if "Forced_any" in df_out.columns else 0
    forced_n = int(df_out["Forced_n"].sum()) if "Forced_n" in df_out.columns else 0
    return (
        f"Recluster complete for {group_name}. "
        f"Features flagged forced: {forced_any}; total forced attachments: {forced_n}. "
        f"Elapsed: {elapsed:.2f} s. Wrote Feature_list_{group_name}.csv"
    )
