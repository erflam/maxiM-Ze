# Clustering.py
# ------------------------------------------------------------
# Shape-similarity clustering + optional post-clustering repair
# for Peak Patches PNGs named like:
#   {base}_mz{mz}_Peak{N}_{GroupTag}.png
#
# Outputs (per group) to dirs["clustering"]:
#   - peak_alignment.csv
#   - alignment_summary_group_{group}.csv
#   - unclustered_peaks_group_{group}.csv
#   - Feature_list_{group}.csv (if recluster is enabled and can run)
#
# Also includes: export_all_group_summaries_to_excel()
# ------------------------------------------------------------

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import cv2
from scipy.stats import pearsonr

from Config import Config


# -----------------------------
# Params
# -----------------------------
@dataclass(frozen=True)
class ClusteringParams:
    similarity_threshold: float = 0.75     # Pearson corr threshold
    rt_resolution_threshold: float = 0.01  # minutes; below => treat as unresolved (optional)
    target_size: Tuple[int, int] = (50, 50)
    verbose: bool = True


@dataclass(frozen=True)
class ReclusterParams:
    min_coverage: float = 0.95  # fraction of files
    rt_tol: float = 0.10        # minutes
    mass_tol: float = 0.0001    # m/z
    verbose: bool = True


# -----------------------------
# Filename parsing
# -----------------------------
_PATCH_RE = re.compile(
    r"^(?P<base>.+?)_mz(?P<mz>\d+(?:\.\d+)?)_Peak(?P<peak>\d+)_(?P<group>[^_]+)\.png$"
)

def parse_patch_filename(stem: str) -> Optional[Tuple[str, float, int, str]]:
    """
    stem: filename without extension
    returns: (base, mz, peak_num, group_tag)
    """
    m = _PATCH_RE.match(stem + ".png")  # easiest reuse: add .png to satisfy regex
    if not m:
        return None
    base = m.group("base")
    mz = float(m.group("mz"))
    peak_num = int(m.group("peak"))
    group_tag = m.group("group")
    return base, mz, peak_num, group_tag


def _group_tag(group_name: str) -> str:
    return str(group_name).replace(" ", "")


# -----------------------------
# Robust column access for peaks_pix_{tag}.csv
# -----------------------------
def _col(df: pd.DataFrame, *names: str) -> str:
    """
    Return the first column name that exists in df among provided names.
    Raises KeyError if none exist.
    """
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"Missing columns (tried {names}); available={list(df.columns)}")


def load_peak_row(peaks_pix_csv: Path, peak_num: int) -> Optional[pd.Series]:
    if not peaks_pix_csv.exists():
        return None
    try:
        df = pd.read_csv(peaks_pix_csv)
        pn_col = _col(df, "peak_num", "Peak_num", "PeakNum")
        row = df[df[pn_col].astype(int) == int(peak_num)]
        if row.empty:
            return None
        return row.iloc[0]
    except Exception:
        return None


# -----------------------------
# Shape similarity
# -----------------------------
def _normalized_profile_from_img(gray_img: np.ndarray, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Convert image -> smoothed, normalized 1D profile (mean over rows).
    Returns None if profile is constant/invalid.
    """
    img_resized = cv2.resize(gray_img, target_size, interpolation=cv2.INTER_AREA)
    img_smooth = cv2.GaussianBlur(img_resized, (3, 3), 0)
    profile = np.mean(img_smooth.astype(np.float64), axis=0)

    if np.std(profile) < 1e-10:
        return None
    profile = (profile - profile.min()) / (profile.max() - profile.min() + 1e-10)
    if np.std(profile) < 1e-10:
        return None
    return profile


def check_peak_similarity(peaks: List[Dict[str, Any]], params: ClusteringParams) -> List[Dict[str, Any]]:
    """
    Keep peaks whose 1D profiles correlate well with a reference profile.
    Reference is the peak with highest avg correlation to others.
    """
    if not peaks:
        return []

    profiles: List[np.ndarray] = []
    valid_peaks: List[Dict[str, Any]] = []

    for peak in peaks:
        prof = _normalized_profile_from_img(peak["image"], params.target_size)
        if prof is None:
            if params.verbose:
                print(f"  - {peak['peak_id']}: skipped (constant profile)")
            continue
        profiles.append(prof)
        valid_peaks.append(peak)

    if len(valid_peaks) <= 1:
        return valid_peaks

    # correlations matrix
    n = len(valid_peaks)
    corr_mat = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                corr = float(pearsonr(profiles[i], profiles[j])[0])
                if np.isnan(corr):
                    corr = 0.0
            except Exception:
                corr = 0.0
            corr_mat[i, j] = corr

    avg_corr = np.nanmean(corr_mat, axis=1)
    ref_idx = int(np.nanargmax(avg_corr)) if np.any(~np.isnan(avg_corr)) else 0
    ref_prof = profiles[ref_idx]

    if params.verbose:
        print("\nSimilarity scores:")

    kept: List[Dict[str, Any]] = []
    for idx, peak in enumerate(valid_peaks):
        try:
            corr = float(pearsonr(ref_prof, profiles[idx])[0])
            if np.isnan(corr):
                corr = 0.0
            if params.verbose:
                print(f"  - {peak['peak_id']}: {corr:.3f}")
            if corr >= params.similarity_threshold:
                kept.append(peak)
        except Exception as e:
            if params.verbose:
                print(f"  - {peak['peak_id']}: error ({e})")
            continue

    return kept


# -----------------------------
# Resolution grouping (optional)
# -----------------------------
def check_peak_resolution(peaks_in_file: List[Dict[str, Any]], params: ClusteringParams) -> Dict[str, Any]:
    """
    Groups peaks that are too close in RT into unresolved groups.
    Returns {'resolved': bool, 'groups': List[List[int]]} where indices refer to peaks sorted by rt_apex.
    """
    if len(peaks_in_file) < 2:
        return {"resolved": True, "groups": [[0]]}

    peaks_sorted = sorted(peaks_in_file, key=lambda x: x["rt_apex"])

    current_group = [0]
    groups: List[List[int]] = []

    for i in range(len(peaks_sorted) - 1):
        rt_diff = peaks_sorted[i + 1]["rt_apex"] - peaks_sorted[i]["rt_apex"]
        if rt_diff < params.rt_resolution_threshold:
            current_group.append(i + 1)
        else:
            groups.append(current_group)
            current_group = [i + 1]

    if current_group:
        groups.append(current_group)

    return {"resolved": len(groups) == len(peaks_sorted), "groups": groups, "sorted": peaks_sorted}


# -----------------------------
# Main clustering step (pipeline)
# -----------------------------
def process_file_clustering(
    dirs: Dict[str, str],
    group_name: str,
    params: ClusteringParams = ClusteringParams(),
) -> str:
    """
    Expects dirs:
      - dirs["patch"]      : Peak Patches folder (input PNGs)
      - dirs["pixel"]      : per-file peaks_pix_{tag}.csv folder
      - dirs["clustering"] : output folder
    """
    group_tag = _group_tag(group_name)

    patch_dir = Path(dirs["patch"])
    pixel_dir = Path(dirs["pixel"])
    cluster_dir = Path(dirs["clustering"])
    cluster_dir.mkdir(parents=True, exist_ok=True)

    if not patch_dir.exists():
        return f"[!] Missing patch dir: {patch_dir}"

    # Collect peaks grouped by mass -> file_base
    mass_peaks: Dict[float, Dict[str, List[Dict[str, Any]]]] = {}

    # Parse all patch PNGs for this group
    patch_files = sorted(patch_dir.glob(f"*_{group_tag}.png"))
    if not patch_files:
        return f"[!] No patch PNGs found for {group_name} in {patch_dir}"

    for patch_file in patch_files:
        parsed = parse_patch_filename(patch_file.stem)
        if parsed is None:
            continue
        file_base, mz, peak_num, gtag = parsed
        if gtag != group_tag:
            continue

        peaks_pix_csv = pixel_dir / f"{file_base}_peaks_pix_{group_tag}.csv"
        row = load_peak_row(peaks_pix_csv, peak_num)
        if row is None:
            # If missing meta, skip (could also warn)
            continue

        try:
            rt_start_col = _col(row.to_frame().T, "RT_start", "rt_start")
            rt_apex_col  = _col(row.to_frame().T, "RT_apex", "rt_apex")
            rt_end_col   = _col(row.to_frame().T, "RT_end", "rt_end")
        except KeyError:
            # At minimum we need rt_apex for ordering; if missing, skip
            continue

        # pixel columns (support both cases)
        pstart_col = _col(row.to_frame().T, "Pixel_start", "pixel_start")
        pend_col   = _col(row.to_frame().T, "Pixel_end", "pixel_end")

        h_col = None
        a_col = None
        for cand in ("height", "Height", "peak_height"):
            if cand in row.index:
                h_col = cand
                break
        for cand in ("area", "Area", "peak_area"):
            if cand in row.index:
                a_col = cand
                break

        img = cv2.imread(str(patch_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        peak_info = {
            "peak_id": patch_file.stem,
            "file": file_base,
            "peak_num": int(peak_num),
            "mz": float(mz),
            "rt_start": float(row[rt_start_col]),
            "rt_apex": float(row[rt_apex_col]),
            "rt_end": float(row[rt_end_col]),
            "pixel_start": float(row[pstart_col]),
            "pixel_end": float(row[pend_col]),
            "peak_height": float(row[h_col]) if h_col is not None else np.nan,
            "peak_area": float(row[a_col]) if a_col is not None else np.nan,
            "image": img,
        }

        mass_peaks.setdefault(float(mz), {}).setdefault(file_base, []).append(peak_info)

    if not mass_peaks:
        return f"[!] No usable peaks found to cluster for {group_name} (check peaks_pix_{group_tag}.csv presence)"

    # Optional: handle unresolved peaks by merging close RT peaks (ported from your old code)
    for mz in list(mass_peaks.keys()):
        for file_base, peaks in list(mass_peaks[mz].items()):
            # sort by rt for resolution check
            res = check_peak_resolution(peaks, params)
            peaks_sorted = res.get("sorted", sorted(peaks, key=lambda x: x["rt_apex"]))

            if not res["resolved"]:
                if params.verbose:
                    print(f"\nHandling unresolved peaks in {file_base} for mz {mz:.4f}")
                # Merge each unresolved group with >1 peaks
                merged = list(peaks_sorted)
                for grp in res["groups"]:
                    if len(grp) <= 1:
                        continue
                    group_peaks = [peaks_sorted[i] for i in grp]
                    # weighted apex by height (fallback weight=1 if nan)
                    weights = np.array([
                        (p["peak_height"] if np.isfinite(p["peak_height"]) else 1.0) for p in group_peaks
                    ], dtype=np.float64)
                    weights = np.where(weights <= 0, 1.0, weights)
                    apex = float(np.sum([p["rt_apex"] for p in group_peaks] * weights) / np.sum(weights))

                    combined = {
                        "peak_id": f"{file_base}_mz{mz:.4f}_combined_" + "_".join(str(p["peak_num"]) for p in group_peaks),
                        "file": file_base,
                        "peak_num": int(group_peaks[0]["peak_num"]),
                        "mz": float(mz),
                        "rt_start": float(min(p["rt_start"] for p in group_peaks)),
                        "rt_apex": apex,
                        "rt_end": float(max(p["rt_end"] for p in group_peaks)),
                        "pixel_start": float(min(p["pixel_start"] for p in group_peaks)),
                        "pixel_end": float(max(p["pixel_end"] for p in group_peaks)),
                        "peak_height": float(np.nanmax([p["peak_height"] for p in group_peaks])),
                        "peak_area": float(np.nansum([p["peak_area"] for p in group_peaks])),
                        "image": group_peaks[0]["image"],
                        "component_peaks": group_peaks,
                    }

                    # mark group slots None then place combined at first index
                    for i in grp:
                        merged[i] = None
                    merged[grp[0]] = combined

                mass_peaks[mz][file_base] = [p for p in merged if p is not None]

    cluster_results: List[Dict[str, Any]] = []
    processed_peaks: set[str] = set()

    # Process each mz separately
    for mz in sorted(mass_peaks.keys()):
        if params.verbose:
            print(f"\nProcessing mz {mz:.4f}")

        # sort peaks by rt per file
        for file_base in mass_peaks[mz]:
            mass_peaks[mz][file_base].sort(key=lambda x: x["rt_apex"])

        max_isomers = max(len(v) for v in mass_peaks[mz].values()) if mass_peaks[mz] else 0
        if params.verbose:
            print(f"Maximum isomers found: {max_isomers}")

        for isomer_pos in range(max_isomers):
            isomer_peaks: List[Dict[str, Any]] = []
            for file_base in mass_peaks[mz]:
                file_peaks = mass_peaks[mz][file_base]
                if isomer_pos < len(file_peaks):
                    isomer_peaks.append(file_peaks[isomer_pos])

            if len(isomer_peaks) < 1:
                continue

            if params.verbose:
                print(f"\nValidating isomer position {isomer_pos + 1} peaks:")
                for p in isomer_peaks:
                    print(f"  - {p['peak_id']} (RT: {p['rt_apex']:.2f})")

            validated = check_peak_similarity(isomer_peaks, params)
            if len(validated) < len(isomer_peaks) and params.verbose:
                print(f"Warning: {len(isomer_peaks) - len(validated)} peaks excluded (low similarity)")

            if not validated:
                continue

            for p in validated:
                processed_peaks.add(p["peak_id"])

            # Averages from validated
            avg_rt_start = float(np.mean([p["rt_start"] for p in validated]))
            avg_rt_apex = float(np.mean([p["rt_apex"] for p in validated]))
            avg_rt_end = float(np.mean([p["rt_end"] for p in validated]))
            avg_pixel_start = float(np.mean([p["pixel_start"] for p in validated]))
            avg_pixel_end = float(np.mean([p["pixel_end"] for p in validated]))

            for p in validated:
                result = {
                    "mass": float(mz),
                    "isomer_position": int(isomer_pos + 1),
                    "file": p["file"],
                    "peak_num": int(p["peak_num"]),
                    "rt_start": float(p["rt_start"]),
                    "rt_apex": float(p["rt_apex"]),
                    "rt_end": float(p["rt_end"]),
                    "rt_start_aligned": avg_rt_start,
                    "rt_apex_aligned": avg_rt_apex,
                    "rt_end_aligned": avg_rt_end,
                    "pixel_start": float(p["pixel_start"]),
                    "pixel_end": float(p["pixel_end"]),
                    "pixel_start_aligned": avg_pixel_start,
                    "pixel_end_aligned": avg_pixel_end,
                    "peak_height": float(p["peak_height"]) if np.isfinite(p["peak_height"]) else np.nan,
                    "peak_area": float(p["peak_area"]) if np.isfinite(p["peak_area"]) else np.nan,
                }

                if "component_peaks" in p:
                    for i, comp in enumerate(p["component_peaks"]):
                        result[f"component_{i+1}_height"] = comp.get("peak_height", np.nan)
                        result[f"component_{i+1}_area"] = comp.get("peak_area", np.nan)
                        result[f"component_{i+1}_rt"] = comp.get("rt_apex", np.nan)

                cluster_results.append(result)

    # Unclustered peaks report
    all_patch_stems = {p.stem for p in patch_dir.glob(f"*_{group_tag}.png")}
    unclustered = sorted(all_patch_stems - processed_peaks)
    if unclustered:
        if params.verbose:
            print("\nWARNING: The following peaks were not clustered:")
            for pk in unclustered:
                print(f"  - {pk}")
        pd.DataFrame({"peak_id": unclustered}).to_csv(
            cluster_dir / f"unclustered_peaks_group_{group_tag}.csv", index=False
        )

    # Save raw alignment list
    df_results = pd.DataFrame(cluster_results)
    df_results.to_csv(cluster_dir / "peak_alignment.csv", index=False)

    # Build summary similar to your old pipeline (with component handling)
    expanded_results: List[Dict[str, Any]] = []
    for r in cluster_results:
        if any(k.startswith("component_") and k.endswith("_height") for k in r.keys()):
            base_r = {k: v for k, v in r.items() if not k.startswith("component_")}
            num_components = sum(1 for k in r.keys() if k.startswith("component_") and k.endswith("_height"))
            for i in range(num_components):
                comp_r = dict(base_r)
                comp_r["peak_height"] = r.get(f"component_{i+1}_height", np.nan)
                comp_r["peak_area"] = r.get(f"component_{i+1}_area", np.nan)
                comp_r["rt_apex"] = r.get(f"component_{i+1}_rt", np.nan)
                comp_r["is_component"] = True
                comp_r["component_number"] = i + 1
                expanded_results.append(comp_r)
        else:
            r2 = dict(r)
            r2["is_component"] = False
            r2["component_number"] = 0
            expanded_results.append(r2)

    df_exp = pd.DataFrame(expanded_results)

    if df_exp.empty:
        # still write an empty summary so export step won't crash
        out_summary = cluster_dir / f"alignment_summary_group_{group_tag}.csv"
        pd.DataFrame().to_csv(out_summary, index=False)
        return f"[✔] Clustering {group_tag}: no clustered peaks (empty summary written)"

    peak_heights = df_exp.pivot(
        index=["mass", "isomer_position", "component_number"],
        columns="file",
        values="peak_height",
    ).round(0)
    peak_heights.columns = [f"{c}_height" for c in peak_heights.columns]

    peak_areas = df_exp.pivot(
        index=["mass", "isomer_position", "component_number"],
        columns="file",
        values="peak_area",
    ).round(0)
    peak_areas.columns = [f"{c}_area" for c in peak_areas.columns]

    summary = df_exp.groupby(["mass", "isomer_position", "component_number"]).agg(
        file=("file", "count"),
        peak_files=("file", lambda x: ", ".join(sorted(x))),
        rt_apex_aligned=("rt_apex_aligned", "first"),
        rt_start_aligned=("rt_start_aligned", "first"),
        rt_end_aligned=("rt_end_aligned", "first"),
        is_component=("is_component", "first"),
    ).round(4)
    summary = summary.rename(columns={"file": "peak_count"})

    final_summary = pd.concat([summary, peak_heights, peak_areas], axis=1)

    # reorder columns
    cols = list(summary.columns)
    sample_cols = [c for c in final_summary.columns if c not in cols]
    height_cols = sorted([c for c in sample_cols if c.endswith("_height")])
    area_cols = sorted([c for c in sample_cols if c.endswith("_area")])
    final_summary = final_summary[cols + height_cols + area_cols]

    # reset index and force mass formatting like old patch
    final_summary = final_summary.reset_index()
    final_summary["mass"] = final_summary["mass"].apply(lambda x: f"{float(x):.4f}")
    final_summary = final_summary.set_index(["mass", "isomer_position"])

    out_summary = cluster_dir / f"alignment_summary_group_{group_tag}.csv"
    final_summary.to_csv(out_summary, float_format="%.4f")

    if params.verbose:
        print("\nAlignment Summary:")
        print(final_summary.to_string())
        print(f"\n[✔] Results saved to {out_summary}")

    return (f"[✔] Clustering {group_tag}: "
            f"patches={len(all_patch_stems)}, clustered={len(processed_peaks)}, "
            f"unclustered={len(unclustered)} | summary={out_summary.name}")


# -----------------------------
# Post-clustering repair step (forced attach)
# -----------------------------
def process_file_recluster(
    dirs: Dict[str, str],
    group_name: str,
    params: ReclusterParams = ReclusterParams(),
) -> str:
    """
    Reads:
      - alignment_summary_group_{group}.csv
      - unclustered_peaks_group_{group}.csv
    Writes:
      - Feature_list_{group}.csv
    """
    group_tag = _group_tag(group_name)
    pixel_dir = Path(dirs["pixel"])
    cluster_dir = Path(dirs["clustering"])
    cluster_dir.mkdir(parents=True, exist_ok=True)

    unclustered_csv = cluster_dir / f"unclustered_peaks_group_{group_tag}.csv"
    summary_csv = cluster_dir / f"alignment_summary_group_{group_tag}.csv"

    if not summary_csv.exists():
        return f"[!] Recluster: missing summary: {summary_csv.name}"

    df_summary = pd.read_csv(summary_csv)

    # Ensure forced columns exist
    if "forced_files" not in df_summary.columns:
        df_summary["forced_files"] = ""
    if "has_forced" not in df_summary.columns:
        df_summary["has_forced"] = False

    height_cols = [c for c in df_summary.columns if c.endswith("_height")]
    area_cols = [c for c in df_summary.columns if c.endswith("_area")]
    sample_names = sorted({c[:-7] for c in height_cols})

    total_files = len(sample_names)
    if total_files == 0 or "peak_count" not in df_summary.columns:
        out = cluster_dir / f"Feature_list_{group_tag}.csv"
        df_summary.to_csv(out, index=False)
        return f"[✔] Recluster {group_tag}: wrote Feature_list (no forcing possible) -> {out.name}"

    if "mass" not in df_summary.columns or "rt_apex_aligned" not in df_summary.columns:
        out = cluster_dir / f"Feature_list_{group_tag}.csv"
        df_summary.to_csv(out, index=False)
        return f"[✔] Recluster {group_tag}: wrote Feature_list (missing columns) -> {out.name}"

    df_summary["_coverage"] = df_summary["peak_count"].astype(float) / float(total_files)
    mass_series = pd.to_numeric(df_summary["mass"], errors="coerce")
    rt_series = pd.to_numeric(df_summary["rt_apex_aligned"], errors="coerce")

    if not unclustered_csv.exists():
        df_summary.drop(columns=["_coverage"], errors="ignore").to_csv(
            cluster_dir / f"Feature_list_{group_tag}.csv", index=False
        )
        return f"[✔] Recluster {group_tag}: no unclustered file; Feature_list written"

    df_uncl = pd.read_csv(unclustered_csv)
    if df_uncl.empty or "peak_id" not in df_uncl.columns:
        df_summary.drop(columns=["_coverage"], errors="ignore").to_csv(
            cluster_dir / f"Feature_list_{group_tag}.csv", index=False
        )
        return f"[✔] Recluster {group_tag}: no unclustered peaks; Feature_list written"

    def parse_peak_id_new(peak_id: str) -> Optional[Tuple[str, float, int]]:
        """
        Expects patch stem:
          {base}_mz{mz}_Peak{N}_{GroupTag}
        """
        m = re.match(r"(.+)_mz(\d+\.\d+)_Peak(\d+)_" + re.escape(group_tag) + r"$", peak_id)
        if not m:
            return None
        return m.group(1), float(m.group(2)), int(m.group(3))

    def load_peak_meta(file_base: str, peak_num: int) -> Optional[Dict[str, float]]:
        pix_csv = pixel_dir / f"{file_base}_peaks_pix_{group_tag}.csv"
        row = load_peak_row(pix_csv, peak_num)
        if row is None:
            return None
        try:
            rt_apex_col = _col(row.to_frame().T, "RT_apex", "rt_apex")
        except KeyError:
            return None
        h_col = "height" if "height" in row.index else ("Height" if "Height" in row.index else None)
        a_col = "area" if "area" in row.index else ("Area" if "Area" in row.index else None)
        return {
            "rt_apex": float(row[rt_apex_col]),
            "peak_height": float(row[h_col]) if h_col else np.nan,
            "peak_area": float(row[a_col]) if a_col else np.nan,
        }

    forced_events = []

    for peak_id in df_uncl["peak_id"].dropna().astype(str).tolist():
        parsed = parse_peak_id_new(peak_id)
        if parsed is None:
            continue
        file_base, peak_mass, peak_num = parsed

        # Summary columns are sample-based (file_base_height/area)
        sample_key = file_base
        hcol = f"{sample_key}_height"
        acol = f"{sample_key}_area"
        if hcol not in df_summary.columns or acol not in df_summary.columns:
            continue

        meta = load_peak_meta(file_base, peak_num)
        if meta is None:
            continue

        peak_rt = float(meta["rt_apex"])

        candidates = df_summary.index[
            (df_summary["_coverage"] >= params.min_coverage) &
            (mass_series.sub(float(peak_mass)).abs() <= params.mass_tol) &
            (rt_series.sub(float(peak_rt)).abs() <= params.rt_tol) &
            (df_summary[hcol].isna()) &
            (df_summary[acol].isna())
        ].tolist()

        if len(candidates) != 1:
            continue

        target_idx = candidates[0]
        if not (pd.isna(df_summary.loc[target_idx, hcol]) and pd.isna(df_summary.loc[target_idx, acol])):
            continue

        df_summary.loc[target_idx, hcol] = meta["peak_height"]
        df_summary.loc[target_idx, acol] = meta["peak_area"]

        if "peak_files" in df_summary.columns:
            existing_pf = df_summary.loc[target_idx, "peak_files"]
            files_list = []
            if isinstance(existing_pf, str) and existing_pf.strip():
                files_list = [x.strip() for x in existing_pf.split(",") if x.strip()]
            if file_base not in files_list:
                files_list.append(file_base)
            df_summary.loc[target_idx, "peak_files"] = ", ".join(sorted(files_list))

        try:
            df_summary.loc[target_idx, "peak_count"] = int(df_summary.loc[target_idx, "peak_count"]) + 1
        except Exception:
            pass

        existing_forced = df_summary.loc[target_idx, "forced_files"]
        forced_list = []
        if isinstance(existing_forced, str) and existing_forced.strip():
            forced_list = [x.strip() for x in existing_forced.split(",") if x.strip()]
        if peak_id not in forced_list:
            forced_list.append(peak_id)
        df_summary.loc[target_idx, "forced_files"] = ", ".join(forced_list)
        df_summary.loc[target_idx, "has_forced"] = True

        forced_events.append({
            "peak_id": peak_id,
            "file": file_base,
            "mass": float(peak_mass),
            "rt_apex": float(peak_rt),
            "target_row": int(target_idx),
        })

    df_summary.drop(columns=["_coverage"], inplace=True, errors="ignore")

    out = cluster_dir / f"Feature_list_{group_tag}.csv"
    df_summary.to_csv(out, index=False, float_format="%.4f")

    if params.verbose:
        print(f"\n[✔] Feature_list saved → {out}")
        if forced_events:
            print(f"[✔] Forced attachments: {len(forced_events)}")
        else:
            print("[ℹ] No unambiguous forced attachments were made.")

    return f"[✔] Recluster {group_tag}: Feature_list={out.name}, forced={len(forced_events)}"


# -----------------------------
# Excel export (final)
# -----------------------------
def export_all_group_summaries_to_excel():
    output_root = Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER
    excel_path = output_root / f"MassSelectionSummary_{Config.ANALYSIS_FOLDER}.xlsx"

    print(f"\n[Excel Export] Saving all group summaries to {excel_path}\n")

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        for group_name in Config.MASS_GROUPS.keys():
            cluster_dir = output_root / group_name / "Clustering"

            summary_csv = cluster_dir / f"alignment_summary_group_{group_name}.csv"
            unresolved_csv = cluster_dir / f"unresolved_peaks_group_{group_name}.csv"
            unclustered_csv = cluster_dir / f"unclustered_peaks_group_{group_name}.csv"

            if not summary_csv.exists():
                print(f"[!] Skipping {group_name}: summary file not found.")
                continue

            sheet_name = str(group_name)
            current_row = 0

            # Prefer Feature_list for export (fallback to alignment summary)
            feature_csv = cluster_dir / f"Feature_list_{group_name}.csv"
            chosen_csv = feature_csv if feature_csv.exists() else summary_csv

            df_summary = pd.read_csv(chosen_csv)
            df_summary.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
            print(f"[✔] Sheet '{sheet_name}' → Summary ({len(df_summary)} rows) [{chosen_csv.name}]")

            worksheet = writer.sheets[sheet_name]
            workbook = writer.book
            forced_fmt = workbook.add_format({"bold": True, "font_color": "red"})

            if "forced_files" in df_summary.columns and "peak_files" in df_summary.columns:
                header = list(df_summary.columns)
                col_peak_files = header.index("peak_files")

                sample_height_cols = {c[:-7]: header.index(c) for c in header if c.endswith("_height")}
                sample_area_cols = {c[:-5]: header.index(c) for c in header if c.endswith("_area")}

                for i, row in df_summary.iterrows():
                    forced_files = row.get("forced_files", "")
                    if not isinstance(forced_files, str) or not forced_files.strip():
                        continue

                    forced_samples_raw = [x.strip() for x in forced_files.split(",") if x.strip()]

                    # UPDATED normalize for both old and new naming
                    def normalize_forced_token(tok: str) -> str:
                        # old: {file_base}_mass{...}_peak{...}
                        m1 = re.match(r"(.+)_mass\d+\.\d+_peak\d+$", tok)
                        if m1:
                            return m1.group(1)
                        # new: {file_base}_mz{...}_Peak{...}_{GroupTag}
                        m2 = re.match(r"(.+)_mz\d+\.\d+_Peak\d+_.+$", tok)
                        if m2:
                            return m2.group(1)
                        return tok

                    forced_samples = [normalize_forced_token(s) for s in forced_samples_raw]
                    excel_row = current_row + 1 + i

                    worksheet.write(excel_row, col_peak_files, row.get("peak_files", ""), forced_fmt)

                    for s in forced_samples:
                        if s in sample_height_cols:
                            cidx = sample_height_cols[s]
                            worksheet.write(excel_row, cidx, row.get(header[cidx], ""), forced_fmt)
                        if s in sample_area_cols:
                            cidx = sample_area_cols[s]
                            worksheet.write(excel_row, cidx, row.get(header[cidx], ""), forced_fmt)

            current_row += len(df_summary) + 2

            # Unresolved peaks
            if unresolved_csv.exists():
                df_unresolved = pd.read_csv(unresolved_csv)
                if not df_unresolved.empty:
                    df_unresolved.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
                    print(f"    ↳ Unresolved peaks added ({len(df_unresolved)} rows)")
                    current_row += len(df_unresolved) + 2
                else:
                    print("    ↳ Skipped unresolved peaks: file is empty")
            else:
                print("    ↳ No unresolved peaks file found")

            # Unclustered peaks
            if unclustered_csv.exists():
                df_unclustered = pd.read_csv(unclustered_csv)
                if not df_unclustered.empty:
                    df_unclustered.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
                    print(f"    ↳ Unclustered peaks added ({len(df_unclustered)} rows)")
                    current_row += len(df_unclustered) + 2
                else:
                    print("    ↳ Skipped unclustered peaks: file is empty")
            else:
                print("    ↳ No unclustered peaks file found")

    print(f"\n[✔] Excel export complete → {excel_path}")
