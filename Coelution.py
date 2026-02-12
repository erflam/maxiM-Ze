import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import pandas as pd

@dataclass(frozen=True)
class CoelutionParams:
    pixel_tolerance: int = 8
    mz_round_decimals: int = 4      # match peaks <-> mapping robustly
    mz_fname_decimals: int = 4      # filename formatting (dot style)
    verbose: bool = False

def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})

def _mz_key(series: pd.Series, decimals: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round(decimals)

def _mz_to_fname_dot(mz: float, decimals: int) -> str:
    """Matches your slice filenames: mz187.0964 (dot, fixed decimals)."""
    return f"{float(mz):.{decimals}f}"

def copy_non_coeluting_to_patch(
    *,
    slice_dir: Path,
    coelu_dir: Path,
    patch_dir: Path,
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Copy PNGs that exist in slice_dir but NOT in coelu_dir into patch_dir.
    Returns (copied_count, skipped_count).
    """
    patch_dir.mkdir(parents=True, exist_ok=True)

    slice_pngs = list(slice_dir.glob("*.png"))
    coelu_names = {p.name for p in coelu_dir.glob("*.png")}

    copied = 0
    skipped = 0

    for p in slice_pngs:
        if p.name not in coelu_names:
            if not dry_run:
                shutil.copy2(p, patch_dir / p.name)
            copied += 1
        else:
            skipped += 1

    return copied, skipped

def collect_for_base(
    *,
    base: str,
    tag: str,
    group_tag: str,
    pixel_csv_dir: Path,
    slice_dir: Path,
    out_slice_dir: Path,
    params: CoelutionParams,
    dry_run: bool,
) -> Tuple[int, int, pd.DataFrame]:
    """
    Returns:
      copied_png_count, missing_png_count, rows_df
    rows_df contains ALL peaks belonging to cluster_id(s) that triggered a segment copy.
    """
    peaks_path = pixel_csv_dir / f"{base}_peaks_pix_{tag}.csv"
    mapping_path = pixel_csv_dir / f"{base}_pixelmapping_{tag}.csv"

    if not peaks_path.exists() or not mapping_path.exists():
        if params.verbose:
            print(f"[v] base={base}: missing peaks or mapping csv")
        return 0, 0, pd.DataFrame()

    peaks = pd.read_csv(peaks_path)
    mapping = pd.read_csv(mapping_path)

    required_peaks = {
        "m/z", "RT_apex", "Pixel_start", "Pixel_end",
        "peak_type", "cluster_id", "is_cluster_lead", "peak_num",
        # (RT_start/RT_end/area/height may exist, but we won't output them)
    }
    required_map = {"m/z", "Segment_ID", "Pixel_start", "Pixel_end"}

    if not required_peaks.issubset(peaks.columns):
        if params.verbose:
            print(f"[v] {peaks_path.name} missing: {sorted(required_peaks - set(peaks.columns))}")
        return 0, 0, pd.DataFrame()

    if not required_map.issubset(mapping.columns):
        if params.verbose:
            print(f"[v] {mapping_path.name} missing: {sorted(required_map - set(mapping.columns))}")
        return 0, 0, pd.DataFrame()

    # Normalize peaks
    p = peaks.copy()
    p["_is_coelu"] = p["peak_type"].astype(str).str.strip().str.lower().eq("coeluting")
    p["_is_lead"] = _to_bool(p["is_cluster_lead"])
    p["_mz_key"] = _mz_key(p["m/z"], params.mz_round_decimals)
    p["_pstart"] = pd.to_numeric(p["Pixel_start"], errors="coerce")

    p = p.dropna(subset=["_mz_key", "_pstart"])
    if p.empty:
        return 0, 0, pd.DataFrame()

    # Normalize mapping
    m = mapping.copy()
    m["_mz_key"] = _mz_key(m["m/z"], params.mz_round_decimals)
    m["_seg_id"] = m["Segment_ID"]
    m["_sstart"] = pd.to_numeric(m["Pixel_start"], errors="coerce")
    m["_send"] = pd.to_numeric(m["Pixel_end"], errors="coerce")
    m = m.dropna(subset=["_mz_key", "_sstart", "_send"])
    if m.empty:
        return 0, 0, pd.DataFrame()

    # Choose segments based on coeluting + lead peaks within tolerance to Pixel_start
    lead = p.loc[p["_is_coelu"] & p["_is_lead"]].copy()
    if lead.empty:
        if params.verbose:
            print(f"[v] base={base}: no coeluting cluster-lead peaks")
        return 0, 0, pd.DataFrame()

    candidates = lead.merge(m[["_mz_key", "_seg_id", "_sstart", "_send"]], on="_mz_key", how="inner")
    if candidates.empty:
        if params.verbose:
            print(f"[v] base={base}: no m/z overlap between lead peaks and mapping")
        return 0, 0, pd.DataFrame()

    candidates["_delta_to_seg_start"] = (candidates["_pstart"] - candidates["_sstart"]).abs()
    chosen = candidates.loc[candidates["_delta_to_seg_start"] <= params.pixel_tolerance].copy()
    if chosen.empty:
        if params.verbose:
            best = candidates.sort_values("_delta_to_seg_start").groupby("_mz_key").head(1)
            for _, r in best.iterrows():
                print(
                    f"[v] base={base} mz={r['_mz_key']} best seg={r['_seg_id']} "
                    f"seg_start={r['_sstart']} lead_peak_start={r['_pstart']} "
                    f"delta={r['_delta_to_seg_start']} tol={params.pixel_tolerance}"
                )
        return 0, 0, pd.DataFrame()

    # Deduplicate segments (one per mz+seg). Keep smallest delta.
    chosen = chosen.sort_values("_delta_to_seg_start").drop_duplicates(subset=["_mz_key", "_seg_id"]).copy()

    out_slice_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing_png = 0

    seg_rows: List[dict] = []
    for _, seg in chosen.iterrows():
        mz_val = float(seg["_mz_key"])
        seg_id = seg["_seg_id"]
        mz_str = _mz_to_fname_dot(mz_val, params.mz_fname_decimals)
        slice_filename = f"{base}_mz{mz_str}_seg{seg_id}_{group_tag}.png"
        slice_path = slice_dir / slice_filename
        slice_found = slice_path.exists()

        if slice_found:
            if dry_run:
                copied += 1
            else:
                try:
                    shutil.copy2(slice_path, out_slice_dir / slice_filename)
                    copied += 1
                except Exception as e:
                    print(f"[!] Failed to copy {slice_filename}: {e}")
        else:
            missing_png += 1

        seg_rows.append(
            {
                "base": base,
                "m/z": mz_val,
                "Segment_ID": seg_id,
                "Pixel_start": float(seg["_sstart"]),
                "Pixel_end": float(seg["_send"]),
                "delta_pixels": float(seg["_delta_to_seg_start"]),
                "slice_filename": slice_filename,
                "slice_found": bool(slice_found),
                "trigger_cluster_id": seg["cluster_id"],
            }
        )

    seg_df = pd.DataFrame(seg_rows)
    if seg_df.empty:
        return copied, missing_png, pd.DataFrame()
    trigger_clusters = set(seg_df["trigger_cluster_id"].astype(str).tolist())
    peaks_in_clusters = peaks.loc[peaks["cluster_id"].astype(str).isin(trigger_clusters)].copy()
    peaks_in_clusters["_cluster_sort_pixel"] = pd.to_numeric(
        peaks_in_clusters["Pixel_start"], errors="coerce"
    )
    peaks_in_clusters = peaks_in_clusters.sort_values(
        ["cluster_id", "_cluster_sort_pixel"], kind="stable"
    )
    peaks_in_clusters["Peak_num_cluster"] = (
            peaks_in_clusters.groupby("cluster_id").cumcount() + 1
    )
    if peaks_in_clusters.empty:
        return copied, missing_png, pd.DataFrame()

    # For each segment row, attach all peaks of its triggering cluster_id
    out_rows: List[dict] = []
    for _, seg in seg_df.iterrows():
        cid = str(seg["trigger_cluster_id"])
        pk = peaks_in_clusters.loc[peaks_in_clusters["cluster_id"].astype(str).eq(cid)]

        for _, r in pk.iterrows():
            out_rows.append(
                {
                    "base": base,
                    "m/z": seg["m/z"],
                    "Segment_ID": seg["Segment_ID"],
                    "Pixel_start": seg["Pixel_start"],
                    "Pixel_end": seg["Pixel_end"],
                    "delta_pixels": seg["delta_pixels"],
                    "slice_filename": seg["slice_filename"],
                    "slice_found": seg["slice_found"],
                    "Peak_num_cluster": r["Peak_num_cluster"],

                    # all peaks belonging to this cluster_id:
                    "RT_apex": r["RT_apex"],
                    "peak_pixel_start": r["Pixel_start"],
                    "peak_pixel_end": r["Pixel_end"],
                    "peak_type": r["peak_type"],
                    "cluster_id": r["cluster_id"],
                    "is_cluster_lead": r["is_cluster_lead"],
                    "peak_num": r["peak_num"],
                }
            )

    rows_df = pd.DataFrame(out_rows)
    return copied, missing_png, rows_df

def run_group_coelution(
    *,
    dirs: dict,
    group_name: str,
    tag: str | None = None,
    params: CoelutionParams = CoelutionParams(),
    dry_run: bool = False,
) -> None:
    """
    dirs must include:
      - dirs['pixel']      Pixel CSVs folder
      - dirs['slice']      Peak Slices folder
      - dirs['coelu']      Peak Coelu Slices folder
      - dirs['coelu csv']  Peak Coelu CSV folder  (NOTE SPACE)
    """
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

        c, m, df = collect_for_base(
            base=base,
            tag=tag,
            group_tag=group_tag,
            pixel_csv_dir=pixel_csv_dir,
            slice_dir=slice_dir,
            out_slice_dir=out_slice_dir,
            params=params,
            dry_run=dry_run,
        )
        total_copied += c
        total_missing += m
        if not df.empty:
            all_rows.append(df)

    out_csv_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_csv_dir / f"ALL_coelu_matches_{tag}.csv"

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)

        # Enforce removals you requested
        drop_cols = {"tag", "group_tag", "mz_key", "mz_fname", "RT_start", "RT_end", "area", "height"}
        combined = combined.drop(columns=[c for c in combined.columns if c in drop_cols], errors="ignore")

        # Optional tidy sorting
        sort_cols = [c for c in ["base", "m/z", "Segment_ID", "cluster_id", "peak_num"] if c in combined.columns]
        if sort_cols:
            combined = combined.sort_values(sort_cols, kind="stable")

        combined.to_csv(combined_path, index=False)
    else:
        # Create empty but with expected headers
        pd.DataFrame(columns=[
            "base", "m/z", "Segment_ID", "Pixel_start", "Pixel_end",
            "delta_pixels", "slice_filename", "slice_found", "Peak_num_cluster",
            "RT_apex", "peak_pixel_start", "peak_pixel_end",
            "peak_type", "cluster_id", "is_cluster_lead", "peak_num",
        ]).to_csv(combined_path, index=False)

    # Copy NON-coeluting slices to Peak Patches
    patch_dir = Path(dirs["patch"])

    c2, s2 = copy_non_coeluting_to_patch(
        slice_dir=slice_dir,
        coelu_dir=out_slice_dir,
        patch_dir=patch_dir,
        dry_run=dry_run,
    )

    if params.verbose or c2 > 0:
        print(f"Patch sync (non-coeluting): copied={c2}, skipped_coelu={s2} -> {patch_dir}")

    print(
        f"Coelution complete for {group_name} (tag={tag}): "
        f"copied={total_copied}, missing_png={total_missing} | "
        f"coelu_csv={combined_path}"
    )