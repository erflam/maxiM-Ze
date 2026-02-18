# Recluster.py
"""
Fallback reclustering for "unclustered" peaks.

Implements two fallback passes (in this order):

1) FORCE-RECLUSTER AGAINST EXISTING ALIGNMENTS
   If an unclustered peak matches (same group, same mass) AND its RT_apex from peaks_pix
   is within `rt_window_min` of an existing alignment's Aligned_rt_apex, we add it into that
   existing alignment (same isomer_position).

2) CREATE NEW ALIGNMENT ROW(S) FROM REMAINING UNCLUSTERED PEAKS
   For remaining unclustered peaks, if (same group, same mass) and their RT_apex values
   are within `rt_window_min` of each other, we create a *new* alignment group (new
   isomer_position) and add rows for those peaks.

Outputs (written into dirs['clustering']):

- peak_alignment_reclustered.csv
- alignment_summary_reclustered_group_<GroupX>.csv
- unclustered_peaks_reclustered_group_<GroupX>.csv

Notes / assumptions:
- Uses the same patch naming convention as the checkpoint code:
    <file_base>_mass<mass>_Peak<peak_num>_<GroupX>.png
  and peak_id is usually the PNG stem.
- If a peak_id includes "__MERGED", we strip that suffix when mapping back to peaks_pix.
- We pull RT_start/RT_apex/RT_end/pixel_start/pixel_end/height/area from peaks_pix CSV.
- When forcing into an existing alignment, we keep that alignment's aligned_* values as-is
  (we do NOT recompute the whole cluster mean; this is a conservative append).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PATCH_RE = re.compile(
    r"""
    ^(?P<file_base>.+?)              # everything before _mass
    _mass(?P<mass>\d+(?:\.\d+)?)     # float mass
    _Peak(?P<peak_num>\d+)           # int peak number
    _(?P<group>Group\d+)$            # GroupX
    """,
    re.VERBOSE,
)


REQUIRED_PIX_COLS = [
    "peak_num",
    "RT_start",
    "RT_apex",
    "RT_end",
    "pixel_start",
    "pixel_end",
    "height",
    "area",
]

COLUMN_ALIASES = {
    "pixel_start": ["Pixel_start", "pixelStart", "PixelStart", "pixel_start", "pixel start", "Pixel start"],
    "pixel_end": ["Pixel_end", "pixelEnd", "PixelEnd", "pixel_end", "pixel end", "Pixel end"],
    "peak_num": ["peak_num", "Peak_num", "PeakNum", "peak number", "PeakNumber"],
    "RT_start": ["RT_start", "rt_start", "Rt_start", "RT start"],
    "RT_apex": ["RT_apex", "rt_apex", "Rt_apex", "RT apex"],
    "RT_end": ["RT_end", "rt_end", "Rt_end", "RT end"],
    "height": ["height", "Height", "peak_height", "Peak_height"],
    "area": ["area", "Area", "peak_area", "Peak_area"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    rename_map: Dict[str, str] = {}

    alias_to_canon: Dict[str, str] = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            alias_to_canon[a] = canon

    existing = set(cols)
    for c in cols:
        if c in alias_to_canon:
            canon = alias_to_canon[c]
            if canon not in existing:
                rename_map[c] = canon

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def _assert_required_columns(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    df = _normalize_columns(df)
    missing = [c for c in REQUIRED_PIX_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Pixel CSV missing required columns {missing} in {path}. Found: {list(df.columns)}")
    return df


@dataclass(frozen=True)
class UnclusteredPeakKey:
    peak_id: str
    group: str
    mass: float
    file_base: str
    peak_num: int


def _parse_peak_id(peak_id: str) -> Optional[UnclusteredPeakKey]:
    """
    peak_id is expected to be the PNG stem, e.g.:
      SomeFile_mass123.45_Peak7_Group3
    It may also include "__MERGED" suffix; we strip it for parsing.
    """
    core = peak_id.split("__MERGED", 1)[0]
    m = PATCH_RE.match(core)
    if not m:
        return None
    return UnclusteredPeakKey(
        peak_id=peak_id,
        group=m.group("group"),
        mass=float(m.group("mass")),
        file_base=m.group("file_base"),
        peak_num=int(m.group("peak_num")),
    )


def _load_peak_from_pix(dirs: Dict[str, Path], key: UnclusteredPeakKey) -> Optional[Dict[str, Any]]:
    """
    Pull peak metrics from peaks_pix CSV for (file_base, group) and peak_num.
    Returns a dict with raw fields needed for an alignment row.
    """
    pixel_csv = dirs["pixel"] / f"{key.file_base}_peaks_pix_{key.group}.csv"
    if not pixel_csv.exists():
        return None

    df = pd.read_csv(pixel_csv)
    df = _assert_required_columns(df, pixel_csv)

    hits = df.loc[df["peak_num"].astype(int) == int(key.peak_num)]
    if hits.shape[0] != 1:
        return None

    r = hits.iloc[0]
    return {
        "group": key.group,
        "mass": float(key.mass),
        "file": key.file_base,
        "peak_id": key.peak_id,
        "peak_num": int(key.peak_num),
        "rt_start": float(r["RT_start"]),
        "rt_apex": float(r["RT_apex"]),
        "rt_end": float(r["RT_end"]),
        "pixel_start": int(r["pixel_start"]),
        "pixel_end": int(r["pixel_end"]),
        "height": float(r["height"]),
        "area": float(r["area"]),
    }


def _build_summary_like_checkpoint(df_align: pd.DataFrame, group: str) -> pd.DataFrame:
    """
    Lightweight summary builder matching the checkpoint’s output *shape*:
    Group, m/z, Isomer_position, Aligned_rt_apex, peak count, then <file>_height/<file>_area columns.

    (This fallback version does not expand component_peaks_json; it just uses height/area per row.)
    """
    if df_align.empty:
        return pd.DataFrame()

    df = df_align.copy()
    df = df.loc[df["group"] == group].copy()
    if df.empty:
        return pd.DataFrame()

    idx_cols = ["group", "mass", "isomer_position", "aligned_rt_apex"]

    h = df.pivot_table(index=idx_cols, columns="file", values="height", aggfunc="first")
    a = df.pivot_table(index=idx_cols, columns="file", values="area", aggfunc="first")
    h.columns = [f"{c}_height" for c in h.columns]
    a.columns = [f"{c}_area" for c in a.columns]

    df_wide = pd.concat([h, a], axis=1).reset_index()

    height_cols = [c for c in df_wide.columns if c.endswith("_height")]
    area_cols = [c for c in df_wide.columns if c.endswith("_area")]

    df_wide["peak_count"] = df_wide[height_cols].notna().sum(axis=1).astype(int)

    df_wide = df_wide.rename(
        columns={
            "group": "Group",
            "mass": "m/z",
            "isomer_position": "Isomer_position",
            "aligned_rt_apex": "Aligned_rt_apex",
            "peak_count": "peak count",
        }
    )

    meta_cols = ["Group", "m/z", "Isomer_position", "Aligned_rt_apex", "peak count"]
    height_cols2 = sorted([c for c in df_wide.columns if c.endswith("_height")])
    area_cols2 = sorted([c for c in df_wide.columns if c.endswith("_area")])
    df_wide = df_wide[meta_cols + height_cols2 + area_cols2]
    df_wide.sort_values(["m/z", "Isomer_position"], inplace=True)
    return df_wide


class Reclusterer:
    def __init__(self, dirs: Dict[str, str | Path], rt_window_min: float = 0.095):
        self.dirs = {k: Path(v) for k, v in dirs.items()}
        for k in ("pixel", "clustering"):
            if k not in self.dirs:
                raise KeyError(f"dirs must include '{k}'")
        self.rt_window_min = float(rt_window_min)

    def run(self, group: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Reads existing clustering outputs, applies fallback reclustering, and writes updated outputs.
        Returns: (df_align_reclustered, df_summary_reclustered, df_unclustered_remaining)
        """
        cluster_dir = self.dirs["clustering"]

        # Load existing alignment (if missing, treat as empty)
        align_path = cluster_dir / "peak_alignment.csv"
        if align_path.exists():
            df_align = pd.read_csv(align_path)
        else:
            df_align = pd.DataFrame()

        # Load existing unclustered list (required for this stage)
        un_path = cluster_dir / f"unclustered_peaks_group_{group}.csv"
        if not un_path.exists():
            # Nothing to do
            df_summary = _build_summary_like_checkpoint(df_align, group)
            df_un = pd.DataFrame(columns=["peak_id"])
            self._write(group, df_align, df_summary, df_un)
            return df_align, df_summary, df_un

        df_un = pd.read_csv(un_path)
        if df_un.empty or "peak_id" not in df_un.columns:
            df_summary = _build_summary_like_checkpoint(df_align, group)
            df_un2 = pd.DataFrame(columns=["peak_id"])
            self._write(group, df_align, df_summary, df_un2)
            return df_align, df_summary, df_un2

        # Parse + hydrate unclustered peaks from peaks_pix
        keys: List[UnclusteredPeakKey] = []
        for pid in df_un["peak_id"].astype(str).tolist():
            k = _parse_peak_id(pid)
            if k and k.group == group:
                keys.append(k)

        un_rows: List[Dict[str, Any]] = []
        for k in keys:
            r = _load_peak_from_pix(self.dirs, k)
            if r is not None:
                un_rows.append(r)

        if not un_rows:
            df_summary = _build_summary_like_checkpoint(df_align, group)
            df_un2 = pd.DataFrame({"peak_id": df_un["peak_id"].astype(str).tolist()})
            self._write(group, df_align, df_summary, df_un2)
            return df_align, df_summary, df_un2

        df_un_peaks = pd.DataFrame(un_rows)

        # Ensure df_align has needed columns
        # (If df_align is empty, we can only do pass #2 among unclustered peaks.)
        if df_align.empty:
            df_align = pd.DataFrame(
                columns=[
                    "group",
                    "mass",
                    "isomer_position",
                    "file",
                    "peak_id",
                    "peak_num",
                    "rt_start",
                    "rt_apex",
                    "rt_end",
                    "aligned_rt_start",
                    "aligned_rt_apex",
                    "aligned_rt_end",
                    "pixel_start",
                    "pixel_end",
                    "aligned_pixel_start",
                    "aligned_pixel_end",
                    "height",
                    "area",
                ]
            )

        # -----------------------------
        # PASS 1: force recluster to existing alignments by (mass, |rt_apex - aligned_rt_apex| <= window)
        # -----------------------------
        df_align_g = df_align.loc[df_align.get("group", "") == group].copy()

        forced_added: List[Dict[str, Any]] = []
        forced_peak_ids: set[str] = set()

        if not df_align_g.empty and "aligned_rt_apex" in df_align_g.columns:
            # Build per (mass, isomer_position) representative aligned_* values
            rep_cols = [
                "group",
                "mass",
                "isomer_position",
                "aligned_rt_start",
                "aligned_rt_apex",
                "aligned_rt_end",
                "aligned_pixel_start",
                "aligned_pixel_end",
            ]
            # Take the first row as representative per cluster
            clusters = (
                df_align_g.sort_values(["mass", "isomer_position", "aligned_rt_apex"])
                .groupby(["mass", "isomer_position"], as_index=False)[rep_cols]
                .first()
            )

            # For each unclustered peak, find best cluster within window
            for _, u in df_un_peaks.iterrows():
                mass = float(u["mass"])
                rt = float(u["rt_apex"])
                cand = clusters.loc[clusters["mass"].astype(float) == mass].copy()
                if cand.empty:
                    continue
                cand["rt_diff"] = (cand["aligned_rt_apex"].astype(float) - rt).abs()
                cand = cand.loc[cand["rt_diff"] <= self.rt_window_min].sort_values("rt_diff")
                if cand.empty:
                    continue

                best = cand.iloc[0]
                forced_peak_ids.add(str(u["peak_id"]))

                forced_added.append(
                    {
                        "group": group,
                        "mass": mass,
                        "isomer_position": int(best["isomer_position"]),
                        "file": str(u["file"]),
                        "peak_id": str(u["peak_id"]),
                        "peak_num": int(u["peak_num"]),
                        "rt_start": float(u["rt_start"]),
                        "rt_apex": rt,
                        "rt_end": float(u["rt_end"]),
                        "aligned_rt_start": float(best["aligned_rt_start"]),
                        "aligned_rt_apex": float(best["aligned_rt_apex"]),
                        "aligned_rt_end": float(best["aligned_rt_end"]),
                        "pixel_start": int(u["pixel_start"]),
                        "pixel_end": int(u["pixel_end"]),
                        "aligned_pixel_start": float(best["aligned_pixel_start"]),
                        "aligned_pixel_end": float(best["aligned_pixel_end"]),
                        "height": float(u["height"]),
                        "area": float(u["area"]),
                    }
                )

        # Append forced rows (drop duplicates if already present)
        if forced_added:
            df_forced = pd.DataFrame(forced_added)
            if not df_align.empty and "peak_id" in df_align.columns:
                df_forced = df_forced.loc[~df_forced["peak_id"].isin(df_align["peak_id"].astype(str))].copy()
            df_align = pd.concat([df_align, df_forced], ignore_index=True)

        # Remaining unclustered peaks after forced pass
        df_remaining = df_un_peaks.loc[~df_un_peaks["peak_id"].astype(str).isin(forced_peak_ids)].copy()

        # -----------------------------
        # PASS 2: create new alignments from remaining unclustered peaks
        # Rule: same mass, and RT_apex within window of each other => new isomer_position
        # -----------------------------
        created_added: List[Dict[str, Any]] = []
        created_peak_ids: set[str] = set()

        if not df_remaining.empty:
            # Determine current max isomer_position per mass (for this group)
            df_align_g2 = df_align.loc[df_align.get("group", "") == group].copy()
            if df_align_g2.empty:
                max_iso_by_mass: Dict[float, int] = {}
            else:
                tmp = df_align_g2.groupby("mass")["isomer_position"].max()
                max_iso_by_mass = {float(k): int(v) for k, v in tmp.items()}

            for mass, df_m in df_remaining.groupby("mass", sort=True):
                df_m = df_m.sort_values("rt_apex").copy()

                # cluster by rt_apex window (simple single-linkage walk)
                bucket: List[pd.Series] = []
                last_rt: Optional[float] = None

                def flush_bucket(bucket_rows: List[pd.Series]) -> None:
                    nonlocal created_added, created_peak_ids, max_iso_by_mass
                    if len(bucket_rows) < 2:
                        return  # only form a new alignment if we have at least 2 peaks

                    mass_f = float(mass)
                    new_iso = max_iso_by_mass.get(mass_f, 0) + 1
                    max_iso_by_mass[mass_f] = new_iso

                    # aligned values = mean of bucket
                    rt_start_mean = float(np.mean([float(r["rt_start"]) for r in bucket_rows]))
                    rt_apex_mean = float(np.mean([float(r["rt_apex"]) for r in bucket_rows]))
                    rt_end_mean = float(np.mean([float(r["rt_end"]) for r in bucket_rows]))
                    px_s_mean = float(np.mean([float(r["pixel_start"]) for r in bucket_rows]))
                    px_e_mean = float(np.mean([float(r["pixel_end"]) for r in bucket_rows]))

                    for r in bucket_rows:
                        created_peak_ids.add(str(r["peak_id"]))
                        created_added.append(
                            {
                                "group": group,
                                "mass": mass_f,
                                "isomer_position": int(new_iso),
                                "file": str(r["file"]),
                                "peak_id": str(r["peak_id"]),
                                "peak_num": int(r["peak_num"]),
                                "rt_start": float(r["rt_start"]),
                                "rt_apex": float(r["rt_apex"]),
                                "rt_end": float(r["rt_end"]),
                                "aligned_rt_start": rt_start_mean,
                                "aligned_rt_apex": rt_apex_mean,
                                "aligned_rt_end": rt_end_mean,
                                "pixel_start": int(r["pixel_start"]),
                                "pixel_end": int(r["pixel_end"]),
                                "aligned_pixel_start": px_s_mean,
                                "aligned_pixel_end": px_e_mean,
                                "height": float(r["height"]),
                                "area": float(r["area"]),
                            }
                        )

                for _, row in df_m.iterrows():
                    rt = float(row["rt_apex"])
                    if last_rt is None:
                        bucket = [row]
                        last_rt = rt
                        continue

                    if abs(rt - last_rt) <= self.rt_window_min:
                        bucket.append(row)
                        last_rt = rt
                    else:
                        flush_bucket(bucket)
                        bucket = [row]
                        last_rt = rt

                flush_bucket(bucket)

        if created_added:
            df_created = pd.DataFrame(created_added)
            if not df_align.empty and "peak_id" in df_align.columns:
                df_created = df_created.loc[~df_created["peak_id"].isin(df_align["peak_id"].astype(str))].copy()
            df_align = pd.concat([df_align, df_created], ignore_index=True)

        # -----------------------------
        # Final: compute remaining unclustered peak_ids
        # -----------------------------
        newly_clustered = forced_peak_ids.union(created_peak_ids)

        remaining_peak_ids = [
            pid for pid in df_un["peak_id"].astype(str).tolist()
            if pid not in newly_clustered
        ]
        df_un_out = pd.DataFrame({"peak_id": remaining_peak_ids})

        # Sort df_align
        if not df_align.empty:
            sort_cols = [c for c in ["group", "mass", "isomer_position", "file", "rt_apex"] if c in df_align.columns]
            if sort_cols:
                df_align.sort_values(sort_cols, inplace=True)

        # Build summary (group-specific)
        df_summary = _build_summary_like_checkpoint(df_align, group)

        # Write outputs
        self._write(group, df_align, df_summary, df_un_out)
        return df_align, df_summary, df_un_out

    def _write(self, group: str, df_align: pd.DataFrame, df_summary: pd.DataFrame, df_un: pd.DataFrame) -> None:
        outdir = self.dirs["clustering"]
        outdir.mkdir(parents=True, exist_ok=True)

        df_align.to_csv(outdir / "peak_alignment_reclustered.csv", index=False)
        df_summary.to_csv(outdir / f"Feature_list_{group}.csv", index=False)
        df_un.to_csv(outdir / f"unclustered_peaks_reclustered_group_{group}.csv", index=False)


def process_file_recluster_peaks(
    dirs: Dict[str, str | Path],
    group_name: str,
    rt_window_min: float = 0.095,
) -> str:
    """
    Entrypoint similar in style to your other pipeline stage functions.

    Reads:
      - <clustering>/peak_alignment.csv
      - <clustering>/unclustered_peaks_group_<GroupX>.csv

    Writes:
      - <clustering>/peak_alignment_reclustered.csv
      - <clustering>/alignment_summary_reclustered_group_<GroupX>.csv
      - <clustering>/unclustered_peaks_reclustered_group_<GroupX>.csv
    """
    recl = Reclusterer(dirs=dirs, rt_window_min=rt_window_min)
    df_align, df_summary, df_un = recl.run(group=group_name)

    return (
        f"Recluster fallback complete for {group_name}. "
        f"Aligned rows now: {0 if df_align.empty else len(df_align)}. "
        f"Summary rows: {0 if df_summary.empty else len(df_summary)}. "
        f"Remaining unclustered patches: {len(df_un)}."
    )
