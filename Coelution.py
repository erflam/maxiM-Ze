# Coelution.py
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


@dataclass(frozen=True)
class CoelutionParams:
    pixel_tolerance: int = 8
    mz_round_decimals: int = 4     # for robust m/z matching
    mz_fname_decimals: int = 4     # used in "{base}_mz{mz_str}_seg{seg_id}_{group_tag}.png"


def _to_bool(s: pd.Series) -> pd.Series:
    """Coerce TRUE/FALSE, 1/0, yes/no, etc. into boolean."""
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def mz_to_fname(mz: float, decimals: int) -> str:
    """187.0964 -> '187p0964' (filename-safe)"""
    return f"{float(mz):.{decimals}f}".replace(".", "p")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def collect_coelution_slices_for_base(
    *,
    base: str,
    tag: str,
    group_tag: str,
    pixel_csv_dir: Path,   # .../Pixel CSVs
    slice_dir: Path,       # .../Peak Slices
    out_dir: Path,         # .../Peak Coelu Slices
    params: CoelutionParams = CoelutionParams(),
    dry_run: bool = False,
) -> Tuple[int, int]:
    """
    For one base, copy qualifying slice PNGs.
    Returns (copied_count, missing_png_count)
    """
    peaks_path = pixel_csv_dir / f"{base}_peaks_pix_{tag}.csv"
    mapping_path = pixel_csv_dir / f"{base}_pixelmapping_{tag}.csv"

    if not peaks_path.exists():
        print(f"[!] Missing peaks CSV: {peaks_path}")
        return (0, 0)
    if not mapping_path.exists():
        print(f"[!] Missing pixelmapping CSV: {mapping_path}")
        return (0, 0)

    peaks = _read_csv(peaks_path)
    mapping = _read_csv(mapping_path)

    # --- Validate required columns (exact names from your examples)
    required_peaks = {"m/z", "peak_type", "is_cluster_lead", "peak_num"}
    required_map = {"m/z", "Segment_ID", "Pixel_start"}

    if not required_peaks.issubset(peaks.columns):
        missing = sorted(required_peaks - set(peaks.columns))
        print(f"[!] {peaks_path.name}: missing columns: {missing}")
        print(f"    Columns present: {list(peaks.columns)}")
        return (0, 0)

    if not required_map.issubset(mapping.columns):
        missing = sorted(required_map - set(mapping.columns))
        print(f"[!] {mapping_path.name}: missing columns: {missing}")
        print(f"    Columns present: {list(mapping.columns)}")
        return (0, 0)

    # --- Filter to coeluting cluster-lead peaks
    p = peaks.copy()
    p["_is_coelu"] = p["peak_type"].astype(str).str.strip().str.lower().eq("coeluting")
    p["_is_lead"] = _to_bool(p["is_cluster_lead"])
    p["_mz_key"] = pd.to_numeric(p["m/z"], errors="coerce").round(params.mz_round_decimals)
    p["_peak_num"] = pd.to_numeric(p["peak_num"], errors="coerce")

    p = p.loc[p["_is_coelu"] & p["_is_lead"]].dropna(subset=["_mz_key", "_peak_num"])
    if p.empty:
        return (0, 0)

    # --- Prepare mapping table
    m = mapping.copy()
    m["_mz_key"] = pd.to_numeric(m["m/z"], errors="coerce").round(params.mz_round_decimals)
    m["_seg_id"] = m["Segment_ID"]
    m["_start"] = pd.to_numeric(m["Pixel_start"], errors="coerce")
    m = m.dropna(subset=["_mz_key", "_start"])
    if m.empty:
        return (0, 0)

    # --- Join by m/z, then apply +/- pixel tolerance to segment start
    j = p.merge(m[["_mz_key", "_seg_id", "_start"]], on="_mz_key", how="inner")
    if j.empty:
        return (0, 0)

    j["_delta"] = (j["_peak_num"] - j["_start"]).abs()
    j = j.loc[j["_delta"] <= params.pixel_tolerance]
    if j.empty:
        return (0, 0)

    # Dedupe so we copy each (mz, seg) once per base
    j = j.drop_duplicates(subset=["_mz_key", "_seg_id"])

    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing_png = 0

    for _, row in j.iterrows():
        mz_val = float(row["_mz_key"])
        seg_id = row["_seg_id"]
        mz_str = mz_to_fname(mz_val, params.mz_fname_decimals)
        fname = f"{base}_mz{mz_str}_seg{seg_id}_{group_tag}.png"

        src = slice_dir / fname
        dst = out_dir / fname

        if not src.exists():
            missing_png += 1
            continue

        if dry_run:
            copied += 1
        else:
            try:
                shutil.copy2(src, dst)
                copied += 1
            except Exception as e:
                print(f"[!] Failed to copy {src.name} -> {dst.name}: {e}")

    return copied, missing_png


def run_group_coelution(
    *,
    dirs: dict,
    group_name: str,
    tag: str | None = None,
    params: CoelutionParams = CoelutionParams(),
    dry_run: bool = False,
) -> None:
    """
    Pipeline-friendly wrapper.

    dirs must include:
      - dirs['pixel'] : Pixel CSVs folder
      - dirs['slice'] : Peak Slices folder
      - dirs['coelu'] : Peak Coelu Slices folder
    """
    group_tag = str(group_name).replace(" ", "")
    tag = tag or group_tag

    pixel_csv_dir = Path(dirs["pixel"])
    slice_dir = Path(dirs["slice"])
    out_dir = Path(dirs["coelu"])

    peaks_files = sorted(pixel_csv_dir.glob(f"*_peaks_pix_{tag}.csv"))
    if not peaks_files:
        print(f"[!] No peaks CSVs found for tag={tag} in {pixel_csv_dir}")
        return

    total_copied = 0
    total_missing = 0

    for peaks_path in peaks_files:
        base = peaks_path.name[: -len(f"_peaks_pix_{tag}.csv")]
        c, m = collect_coelution_slices_for_base(
            base=base,
            tag=tag,
            group_tag=group_tag,
            pixel_csv_dir=pixel_csv_dir,
            slice_dir=slice_dir,
            out_dir=out_dir,
            params=params,
            dry_run=dry_run,
        )
        total_copied += c
        total_missing += m

    print(
        f"Coelution complete for {group_name} (tag={tag}): "
        f"copied={total_copied}, missing_png={total_missing}"
    )
