import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd

@dataclass(frozen=True)
class CoelutionParams:
    pixel_tolerance: int = 50
    mz_round_decimals: int = 4      # match peaks <-> mapping robustly
    mz_fname_decimals: int = 4      # filename formatting (dot style)
    verbose: bool = False

def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})

def _mz_key(series: pd.Series, decimals: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round(decimals)

def _mz_variants_for_filename(mz: float, decimals: int) -> list[str]:
    """
    Generate safe filename variants for m/z:
    - Full precision (e.g. 250.1500)
    - Trailing-zero stripped (250.15)
    Never alters non-zero digits.
    """
    base = f"{float(mz):.{decimals}f}"

    variants = [base]

    if "." in base:
        stripped = base.rstrip("0").rstrip(".")
        if stripped != base:
            variants.append(stripped)

    return variants

def _seg_id_to_int(seg_id) -> int:
    """
    Robust Segment_ID -> int conversion for filenames.
    Handles values like 3, "3", 3.0, "3.0", pandas Int64, etc.
    """
    try:
        # pandas NA-safe
        v = pd.to_numeric(pd.Series([seg_id]), errors="coerce").iloc[0]
        if pd.isna(v):
            raise ValueError(f"Segment_ID is NaN: {seg_id!r}")
        return int(float(v))
    except Exception as e:
        raise ValueError(f"Cannot convert Segment_ID to int: {seg_id!r}") from e

def find_secondary_coelution_segments(
    *,
    peaks_norm: pd.DataFrame,
    mapping_norm: pd.DataFrame,
    rt_diff_threshold: float = 0.13,
    apex_tolerance_pixels: int = 2,
) -> pd.DataFrame:
    cols = ["_mz_key", "_seg_id", "_sstart", "_send", "_delta_to_seg_start", "cluster_id"]
    if peaks_norm.empty or mapping_norm.empty:
        return pd.DataFrame(columns=cols)

    p = peaks_norm.copy()
    m = mapping_norm.copy()

    # Ensure numeric columns
    p["_pstart"] = pd.to_numeric(p.get("Pixel_start"), errors="coerce")
    p["_pend"] = pd.to_numeric(p.get("Pixel_end"), errors="coerce")
    p["_rt"] = pd.to_numeric(p.get("RT_apex"), errors="coerce")
    p["_is_resolved"] = p["peak_type"].astype(str).str.strip().str.lower().eq("resolved")

    # define an apex pixel if not already present: midpoint of peak bounds
    p["_papex"] = (p["_pstart"] + p["_pend"]) / 2.0

    p = p.dropna(subset=["_mz_key", "_pstart", "_pend", "_papex", "_rt"])
    p = p.loc[p["_is_resolved"]].copy()
    if p.empty:
        return pd.DataFrame(columns=cols)

    # Ensure mapping numeric columns exist
    m["_sstart"] = pd.to_numeric(m.get("_sstart", m.get("Pixel_start")), errors="coerce")
    m["_send"] = pd.to_numeric(m.get("_send", m.get("Pixel_end")), errors="coerce")

    # IMPORTANT: ensure segment IDs are integer-like to avoid seg3.0 filenames
    m["_seg_id"] = pd.to_numeric(m.get("_seg_id", m.get("Segment_ID")), errors="coerce").astype("Int64")

    m = m.dropna(subset=["_mz_key", "_seg_id", "_sstart", "_send"])
    if m.empty:
        return pd.DataFrame(columns=cols)

    out_rows = []

    # Work m/z by m/z
    for mz_key, m_mz in m.groupby("_mz_key", dropna=True):
        p_mz = p.loc[p["_mz_key"].eq(mz_key)].copy()
        if len(p_mz) < 2:
            continue

        # Assign each peak to ONE segment
        assignments = []
        for _, pk in p_mz.iterrows():
            apex = float(pk["_papex"])
            tol = float(apex_tolerance_pixels)

            # candidate segments where apex is inside (with small tolerance)
            cand = m_mz.loc[
                (m_mz["_sstart"] - tol <= apex) & (apex <= m_mz["_send"] + tol)
            ].copy()

            if cand.empty:
                # fallback: choose the segment with minimal distance to segment interval
                def dist_to_seg(row):
                    s, e = float(row["_sstart"]), float(row["_send"])
                    if s <= apex <= e:
                        return 0.0
                    return min(abs(apex - s), abs(apex - e))

                cand = m_mz.copy()
                cand["_apex_dist"] = cand.apply(dist_to_seg, axis=1)
                best = cand.sort_values("_apex_dist", kind="stable").head(1)
            else:
                # choose best by distance to segment center
                cand["_center"] = (cand["_sstart"] + cand["_send"]) / 2.0
                cand["_apex_dist"] = (cand["_center"] - apex).abs()
                best = cand.sort_values("_apex_dist", kind="stable").head(1)

            seg_row = best.iloc[0]
            assignments.append(
                {
                    "cluster_id": str(pk["cluster_id"]),
                    "_rt": float(pk["_rt"]),
                    "_seg_id": int(seg_row["_seg_id"]),  # ensure int
                    "_sstart": float(seg_row["_sstart"]),
                    "_send": float(seg_row["_send"]),
                }
            )

        a = pd.DataFrame(assignments)
        if a.empty:
            continue

        # Find segments that have >=2 assigned peaks (THIS is "sliced together")
        for seg_id, grp in a.groupby("_seg_id", dropna=True):
            if len(grp) < 2:
                continue

            # Optional RT proximity gate
            if rt_diff_threshold is not None:
                rts = grp["_rt"].sort_values().to_numpy()
                close_pair = any(
                    abs(float(rts[i + 1]) - float(rts[i])) <= float(rt_diff_threshold) + 1e-12
                    for i in range(len(rts) - 1)
                )
                if not close_pair:
                    continue

            # Trigger: add one row per cluster_id in this segment
            for cid in grp["cluster_id"].unique().tolist():
                out_rows.append(
                    {
                        "_mz_key": float(mz_key),
                        "_seg_id": int(seg_id),  # ensure int
                        "_sstart": float(grp["_sstart"].iloc[0]),
                        "_send": float(grp["_send"].iloc[0]),
                        "_delta_to_seg_start": 0.0,
                        "cluster_id": cid,
                    }
                )

    if not out_rows:
        return pd.DataFrame(columns=cols)

    return pd.DataFrame(out_rows)

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
    p["_pend"] = pd.to_numeric(p["Pixel_end"], errors="coerce")

    p = p.dropna(subset=["_mz_key", "_pstart"])
    if p.empty:
        return 0, 0, pd.DataFrame()

    # Normalize mapping
    m = mapping.copy()
    m["_mz_key"] = _mz_key(m["m/z"], params.mz_round_decimals)

    # IMPORTANT: force seg_id to integer-like to match filenames (seg3 not seg3.0)
    m["_seg_id"] = pd.to_numeric(m["Segment_ID"], errors="coerce").astype("Int64")

    m["_sstart"] = pd.to_numeric(m["Pixel_start"], errors="coerce")
    m["_send"] = pd.to_numeric(m["Pixel_end"], errors="coerce")
    m = m.dropna(subset=["_mz_key", "_seg_id", "_sstart", "_send"])
    if m.empty:
        return 0, 0, pd.DataFrame()

    # Choose segments based on (A) your original coeluting-lead rule and (B) the new secondary rule
    lead = p.loc[p["_is_coelu"] & p["_is_lead"]].copy()

    secondary = find_secondary_coelution_segments(
        peaks_norm=p,
        mapping_norm=m,
        rt_diff_threshold=0.13,
        apex_tolerance_pixels=2,
    )

    if lead.empty and secondary.empty:
        if params.verbose:
            print(f"[v] base={base}: no coeluting-lead peaks AND no secondary-coelution segments")
        return 0, 0, pd.DataFrame()

    chosen_parts = []

    # Coeluting + lead peaks within tolerance to Pixel_start
    if not lead.empty:
        candidates = lead.merge(m[["_mz_key", "_seg_id", "_sstart", "_send"]], on="_mz_key", how="inner")

        if candidates.empty:
            if params.verbose:
                print(f"[v] base={base}: no m/z overlap between lead peaks and mapping (rule A skipped)")
        else:
            candidates["_delta_to_seg_start"] = (candidates["_pstart"] - candidates["_sstart"]).abs()
            chosen_a = candidates.loc[candidates["_delta_to_seg_start"] <= params.pixel_tolerance].copy()

            if not chosen_a.empty:
                # keep one per mz+seg (smallest delta)
                chosen_a = (
                    chosen_a.sort_values("_delta_to_seg_start")
                    .drop_duplicates(subset=["_mz_key", "_seg_id"])
                    .copy()
                )
                chosen_parts.append(chosen_a)

    # Secondary coelution segments
    if not secondary.empty:
        chosen_parts.append(secondary.copy())

    if not chosen_parts:
        if params.verbose:
            print(f"[v] base={base}: no segments chosen by either rule")
        return 0, 0, pd.DataFrame()

    # Combine and dedupe copies by mz+seg+cluster_id (so one segment can trigger multiple clusters)
    chosen = pd.concat(chosen_parts, ignore_index=True)

    # Normalize chosen seg_id to int for filenames
    chosen["_seg_id"] = pd.to_numeric(chosen["_seg_id"], errors="coerce")

    if "cluster_id" in chosen.columns:
        chosen = chosen.sort_values("_delta_to_seg_start", kind="stable").drop_duplicates(
            subset=["_mz_key", "_seg_id", "cluster_id"], keep="first"
        )
    else:
        chosen = chosen.sort_values("_delta_to_seg_start", kind="stable").drop_duplicates(
            subset=["_mz_key", "_seg_id"], keep="first"
        )

    out_slice_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing_png = 0

    seg_rows: List[dict] = []
    for _, seg in chosen.iterrows():
        mz_val = float(seg["_mz_key"])
        seg_id_int = _seg_id_to_int(seg["_seg_id"])

        mz_variants = _mz_variants_for_filename(
            mz_val,
            params.mz_fname_decimals
        )

        slice_found = False
        slice_path = None
        slice_filename = None

        for mz_str in mz_variants:
            candidate = f"{base}_mz{mz_str}_seg{seg_id_int}_{group_tag}.png"
            path = slice_dir / candidate
            if path.exists():
                slice_found = True
                slice_path = path
                slice_filename = candidate
                break

        # If none found, still record the expected "full precision" name
        if not slice_found:
            slice_filename = f"{base}_mz{mz_variants[0]}_seg{seg_id_int}_{group_tag}.png"

        if not slice_found and params.verbose:
            print(f"[v] missing slice png: {slice_filename}")

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
                "Segment_ID": seg_id_int,  # write int
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
    peaks_in_clusters["_cluster_sort_pixel"] = pd.to_numeric(peaks_in_clusters["Pixel_start"], errors="coerce")
    peaks_in_clusters = peaks_in_clusters.sort_values(["cluster_id", "_cluster_sort_pixel"], kind="stable")
    peaks_in_clusters["Peak_num_cluster"] = peaks_in_clusters.groupby("cluster_id").cumcount() + 1

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
      - dirs['patch']      Peak Patches folder
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
        f"copied={total_copied}, missing_png={total_missing}")

    print(f"Coelution CSV saved to coelu_csv={combined_path}")