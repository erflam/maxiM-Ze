# Coelution.py
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd


@dataclass(frozen=True)
class CoelutionParams:
    pixel_tolerance: int = 8
    mz_round_decimals: int = 4      # for joining peaks <-> mapping
    mz_fname_decimals: int = 4      # for filename formatting
    verbose: bool = False


def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _mz_key(series: pd.Series, decimals: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round(decimals)


def _mz_to_fname_dot(mz: float, decimals: int) -> str:
    """Matches your slice name style: mz187.0964 (dot, fixed decimals)."""
    return f"{float(mz):.{decimals}f}"


def collect_coelution_for_base(
    *,
    base: str,
    tag: str,
    group_tag: str,
    pixel_csv_dir: Path,
    slice_dir: Path,
    out_slice_dir: Path,
    out_csv_dir: Path,
    params: CoelutionParams = CoelutionParams(),
    dry_run: bool = False,
) -> Tuple[int, int, pd.DataFrame]:
    """
    Returns (copied_png_count, missing_png_count, matches_df)
    matches_df includes ALL matches that pass tolerance; includes whether PNG exists.
    """
    peaks_path = pixel_csv_dir / f"{base}_peaks_pix_{tag}.csv"
    mapping_path = pixel_csv_dir / f"{base}_pixelmapping_{tag}.csv"

    if not peaks_path.exists() or not mapping_path.exists():
        if params.verbose:
            print(f"[v] base={base}: missing peaks or mapping CSV")
        return (0, 0, pd.DataFrame())

    peaks = pd.read_csv(peaks_path)
    mapping = pd.read_csv(mapping_path)

    # Exact columns per your examples
    required_peaks = {
        "m/z", "RT_start", "RT_apex", "RT_end", "area", "height",
        "Pixel_start", "Pixel_end", "peak_type", "cluster_id", "is_cluster_lead", "peak_num"
    }
    required_map = {"m/z", "Segment_ID", "Pixel_start", "Pixel_end"}

    if not required_peaks.issubset(peaks.columns):
        if params.verbose:
            print(f"[v] {peaks_path.name} missing: {sorted(required_peaks - set(peaks.columns))}")
        return (0, 0, pd.DataFrame())

    if not required_map.issubset(mapping.columns):
        if params.verbose:
            print(f"[v] {mapping_path.name} missing: {sorted(required_map - set(mapping.columns))}")
        return (0, 0, pd.DataFrame())

    # Filter peaks: coeluting + cluster lead
    p = peaks.copy()
    p["_is_coelu"] = p["peak_type"].astype(str).str.strip().str.lower().eq("coeluting")
    p["_is_lead"] = _to_bool(p["is_cluster_lead"])
    p["_mz_key"] = _mz_key(p["m/z"], params.mz_round_decimals)
    p["_lead_pix"] = pd.to_numeric(p["Pixel_start"], errors="coerce")

    p = p.loc[p["_is_coelu"] & p["_is_lead"]].dropna(subset=["_mz_key", "_lead_pix"])
    if p.empty:
        return (0, 0, pd.DataFrame())

    # Prep mapping
    m = mapping.copy()
    m["_mz_key"] = _mz_key(m["m/z"], params.mz_round_decimals)
    m["_seg_id"] = m["Segment_ID"]
    m["_seg_start"] = pd.to_numeric(m["Pixel_start"], errors="coerce")
    m["_seg_end"] = pd.to_numeric(m["Pixel_end"], errors="coerce")
    m = m.dropna(subset=["_mz_key", "_seg_start"])
    if m.empty:
        return (0, 0, pd.DataFrame())

    # Join and apply tolerance
    j = p.merge(
        m[["_mz_key", "_seg_id", "_seg_start", "_seg_end"]],
        on="_mz_key",
        how="inner",
    )
    if j.empty:
        return (0, 0, pd.DataFrame())

    j["_delta_pixels"] = (j["_lead_pix"] - j["_seg_start"]).abs()
    j = j.loc[j["_delta_pixels"] <= params.pixel_tolerance].copy()
    if j.empty:
        return (0, 0, pd.DataFrame())

    # Dedupe (one row per mz+seg)
    j = j.sort_values("_delta_pixels").drop_duplicates(subset=["_mz_key", "_seg_id"]).copy()

    # Build filenames and check existence
    j["base"] = base
    j["tag"] = tag
    j["group_tag"] = group_tag

    j["mz_fname"] = j["_mz_key"].astype(float).map(lambda x: _mz_to_fname_dot(x, params.mz_fname_decimals))
    j["slice_filename"] = j.apply(
        lambda r: f"{base}_mz{r['mz_fname']}_seg{r['_seg_id']}_{group_tag}.png",
        axis=1,
    )
    j["slice_path"] = j["slice_filename"].map(lambda fn: str(slice_dir / fn))
    j["slice_found"] = j["slice_filename"].map(lambda fn: (slice_dir / fn).exists())

    # Copy PNGs that exist
    out_slice_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing_png = 0

    for fn, found in zip(j["slice_filename"], j["slice_found"]):
        if not found:
            missing_png += 1
            continue
        if dry_run:
            copied += 1
            continue
        src = slice_dir / fn
        dst = out_slice_dir / fn
        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception as e:
            print(f"[!] Failed to copy {src.name} -> {dst.name}: {e}")

    # Write per-base CSV
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path = out_csv_dir / f"{base}_coelu_matches_{tag}.csv"

    # Keep a clean, explicit column order
    keep_cols = [
        "base", "tag", "group_tag",
        "m/z", "_mz_key", "mz_fname",
        "cluster_id", "peak_num",
        "RT_start", "RT_apex", "RT_end",
        "area", "height",
        "Pixel_start", "Pixel_end",
        "_seg_id", "_seg_start", "_seg_end",
        "_delta_pixels",
        "slice_filename", "slice_found",
    ]
    # Some columns (like original 'm/z') are duplicated in join; keep the peaks-side one
    # In our merge, peaks columns remain as-is, so "m/z" is peaks m/z.
    out_df = j[keep_cols].rename(
        columns={
            "_mz_key": "mz_key",
            "_seg_id": "Segment_ID",
            "_seg_start": "Pixel_start",
            "_seg_end": "Pixel_end",
            "_delta_pixels": "delta_pixels",
        }
    )

    out_df.to_csv(out_csv_path, index=False)

    return copied, missing_png, out_df


def run_group_coelution(
    *,
    dirs: dict,
    group_name: str,
    tag: str | None = None,
    params: CoelutionParams = CoelutionParams(),
    dry_run: bool = False,
) -> None:
    group_tag = str(group_name).replace(" ", "")
    tag = tag or group_tag

    pixel_csv_dir = Path(dirs["pixel"])
    slice_dir = Path(dirs["slice"])
    out_slice_dir = Path(dirs["coelu"])
    out_csv_dir = Path(dirs["coelu csv"])

    peaks_files = sorted(pixel_csv_dir.glob(f"*_peaks_pix_{tag}.csv"))
    if not peaks_files:
        print(f"[!] No peaks CSVs found for tag={tag} in {pixel_csv_dir}")
        return

    all_rows: List[pd.DataFrame] = []
    total_copied = 0
    total_missing = 0

    for peaks_path in peaks_files:
        base = peaks_path.name[: -len(f"_peaks_pix_{tag}.csv")]

        c, m, df = collect_coelution_for_base(
            base=base,
            tag=tag,
            group_tag=group_tag,
            pixel_csv_dir=pixel_csv_dir,
            slice_dir=slice_dir,
            out_slice_dir=out_slice_dir,
            out_csv_dir=out_csv_dir,
            params=params,
            dry_run=dry_run,
        )
        total_copied += c
        total_missing += m
        if not df.empty:
            all_rows.append(df)

    # Write combined group CSV
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_csv_dir / f"ALL_coelu_matches_{tag}.csv"

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(combined_path, index=False)
    else:
        # still create an empty file with headers for consistency
        pd.DataFrame().to_csv(combined_path, index=False)

    print(f"Coelution complete for {group_name} (tag={tag}): copied = {total_copied}, missing_png = {total_missing}")
    print(f"coelu_csv = {combined_path}")