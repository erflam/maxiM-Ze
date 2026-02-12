# Clustering.py
# Align / cluster Peak patches by SHAPE similarity + build summary tables + optional RT/mass recluster + final Excel export
#
# Expected patch filenames (stem, no .png):
#   {file_base}_mz{mz}_Peak{peaknum}_{GroupTag}
# Example:
#   OE_EF_IsmailBaseline_POS_C007_0002_mz187.0964_Peak2_Group1.png
#
# Expected pixel CSV per file (in dirs["pixel"]):
#   {file_base}_peaks_pix.csv
# Must contain columns: peak_num, RT_start, RT_apex, RT_end, pixel_start, pixel_end, height, area

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from Config import Config


# ----------------------------
# Filename parsing
# ----------------------------

_PATCH_RE = re.compile(r"(.+)_mz(\d+\.\d+)_Peak(\d+)_(Group\d+)$")


def parse_patch_stem(stem: str) -> Optional[Tuple[str, float, int, str]]:
    """
    Returns: (file_base, mz, peak_num, group_tag)
    """
    m = _PATCH_RE.match(stem)
    if not m:
        return None
    file_base, mz_s, peak_s, group_tag = m.groups()
    return file_base, float(mz_s), int(peak_s), group_tag


# ----------------------------
# Core peak objects
# ----------------------------

@dataclass
class PeakInfo:
    peak_id: str          # patch stem
    file_base: str
    mz: float
    peak_num: int
    group_tag: str

    rt_start: float
    rt_apex: float
    rt_end: float
    pixel_start: float
    pixel_end: float
    peak_height: float
    peak_area: float

    image: np.ndarray     # grayscale


# ----------------------------
# Similarity helpers
# ----------------------------

def _profile_from_image(img_gray: np.ndarray, target_size=(50, 50)) -> Optional[np.ndarray]:
    """
    Produces a normalized 1D intensity profile for correlation.
    Returns None if profile is constant/invalid.
    """
    try:
        img_resized = cv2.resize(img_gray, target_size, interpolation=cv2.INTER_AREA)
        img_smooth = cv2.GaussianBlur(img_resized, (3, 3), 0)
        profile = np.mean(img_smooth, axis=0).astype(np.float64)

        if np.std(profile) < 1e-10:
            return None

        # Min-max normalize
        mn, mx = float(profile.min()), float(profile.max())
        profile = (profile - mn) / (mx - mn + 1e-10)

        if np.std(profile) < 1e-10:
            return None

        return profile
    except Exception:
        return None


def filter_by_shape_similarity(
    peaks: List[PeakInfo],
    similarity_threshold: float = 0.75,
    target_size=(50, 50),
    debug_print: bool = True,
) -> List[PeakInfo]:
    """
    Keeps peaks whose profile correlates with the most representative peak (highest avg corr).
    """
    if not peaks:
        return []

    profiles: List[np.ndarray] = []
    valid_peaks: List[PeakInfo] = []

    for p in peaks:
        prof = _profile_from_image(p.image, target_size=target_size)
        if prof is None:
            if debug_print:
                print(f"  - {p.peak_id}: skipped (constant/invalid profile)")
            continue
        profiles.append(prof)
        valid_peaks.append(p)

    if len(valid_peaks) <= 1:
        return valid_peaks

    n = len(valid_peaks)
    corr_mat = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                r = pearsonr(profiles[i], profiles[j])[0]
                if r is None or np.isnan(r):
                    r = 0.0
            except Exception:
                r = 0.0
            corr_mat[i, j] = r

    avg_corr = np.nanmean(corr_mat, axis=1)
    ref_idx = int(np.nanargmax(avg_corr)) if len(avg_corr) else 0
    ref_profile = profiles[ref_idx]

    if debug_print:
        print("\nSimilarity scores:")

    kept: List[PeakInfo] = []
    for i, p in enumerate(valid_peaks):
        try:
            r = pearsonr(ref_profile, profiles[i])[0]
            if r is None or np.isnan(r):
                r = 0.0
                if debug_print:
                    print(f"  - {p.peak_id}: nan")
            else:
                if debug_print:
                    print(f"  - {p.peak_id}: {r:.3f}")

            if float(r) >= float(similarity_threshold):
                kept.append(p)
        except Exception as e:
            if debug_print:
                print(f"  - {p.peak_id}: error ({e})")

    return kept


# ----------------------------
# Loading patches + pixel metadata
# ----------------------------

def load_peaks_from_patch_dir(
    patch_dir: Path,
    pixel_dir: Path,
    expected_group_tag: str,
) -> Dict[float, Dict[str, List[PeakInfo]]]:
    """
    Returns structure:
      mass_peaks[mz][file_base] -> list of PeakInfo
    Only loads patches that match expected_group_tag.
    """
    mass_peaks: Dict[float, Dict[str, List[PeakInfo]]] = {}

    for patch_file in patch_dir.glob("*.png"):
        stem = patch_file.stem
        parsed = parse_patch_stem(stem)
        if parsed is None:
            continue

        file_base, mz, peak_num, group_tag = parsed
        if group_tag != expected_group_tag:
            continue

        pix_csv = pixel_dir / f"{file_base}_peaks_pix.csv"
        if not pix_csv.exists():
            continue

        try:
            df = pd.read_csv(pix_csv)
            row = df[df["peak_num"] == peak_num]
            if row.empty:
                continue
            row = row.iloc[0]
        except Exception:
            continue

        img = cv2.imread(str(patch_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        p = PeakInfo(
            peak_id=stem,
            file_base=file_base,
            mz=float(mz),
            peak_num=int(peak_num),
            group_tag=group_tag,

            rt_start=float(row["RT_start"]),
            rt_apex=float(row["RT_apex"]),
            rt_end=float(row["RT_end"]),
            pixel_start=float(row["pixel_start"]),
            pixel_end=float(row["pixel_end"]),
            peak_height=float(row["height"]) if "height" in df.columns else float("nan"),
            peak_area=float(row["area"]) if "area" in df.columns else float("nan"),

            image=img,
        )

        mass_peaks.setdefault(float(mz), {}).setdefault(file_base, []).append(p)

    # Sort within each file by RT apex
    for mz in mass_peaks:
        for fb in mass_peaks[mz]:
            mass_peaks[mz][fb].sort(key=lambda x: x.rt_apex)

    return mass_peaks


# ----------------------------
# Summary building
# ----------------------------

def build_alignment_summary(
    cluster_results: List[dict],
    group_tag: str,
) -> pd.DataFrame:
    """
    Produces a wide table:
      mass, isomer_position, component_number, peak_count, peak_files,
      rt_apex_aligned, rt_start_aligned, rt_end_aligned, is_component,
      <sample>_height, <sample>_area...
    """
    if not cluster_results:
        # Still return a valid empty df
        return pd.DataFrame(columns=[
            "mass", "isomer_position", "component_number",
            "peak_count", "peak_files",
            "rt_apex_aligned", "rt_start_aligned", "rt_end_aligned",
            "is_component"
        ])

    df = pd.DataFrame(cluster_results)

    # component_number/is_component always exist for consistency
    if "component_number" not in df.columns:
        df["component_number"] = 0
    if "is_component" not in df.columns:
        df["is_component"] = False

    # Wide height/area pivots
    heights = df.pivot_table(
        index=["mass", "isomer_position", "component_number"],
        columns="file",
        values="peak_height",
        aggfunc="first"
    )
    heights.columns = [f"{c}_height" for c in heights.columns]

    areas = df.pivot_table(
        index=["mass", "isomer_position", "component_number"],
        columns="file",
        values="peak_area",
        aggfunc="first"
    )
    areas.columns = [f"{c}_area" for c in areas.columns]

    # Summary rows
    summary = df.groupby(["mass", "isomer_position", "component_number"]).agg(
        peak_count=("file", "count"),
        peak_files=("file", lambda x: ", ".join(sorted(set(map(str, x))))),
        rt_apex_aligned=("rt_apex_aligned", "first"),
        rt_start_aligned=("rt_start_aligned", "first"),
        rt_end_aligned=("rt_end_aligned", "first"),
        is_component=("is_component", "first"),
    )

    final = pd.concat([summary, heights, areas], axis=1).reset_index()

    # Force 4-decimal string mass for stable output
    final["mass"] = final["mass"].apply(lambda x: f"{float(x):.4f}")

    # Column ordering: summary first, then heights, then areas
    base_cols = [
        "mass", "isomer_position", "component_number",
        "peak_count", "peak_files",
        "rt_apex_aligned", "rt_start_aligned", "rt_end_aligned",
        "is_component"
    ]
    height_cols = sorted([c for c in final.columns if c.endswith("_height")])
    area_cols = sorted([c for c in final.columns if c.endswith("_area")])

    final = final[base_cols + height_cols + area_cols]
    return final


# ----------------------------
# Public API: clustering per group
# ----------------------------

def cluster_group(
    dirs: dict,
    group_name: str,
    similarity_threshold: float = 0.75,
    target_size=(50, 50),
    debug_print: bool = True,
) -> str:
    """
    Runs shape-based clustering for ONE group.
    Writes:
      - peak_alignment.csv
      - unclustered_peaks_group_{group_tag}.csv
      - alignment_summary_group_{group_tag}.csv
    Returns status string.
    """
    group_tag = str(group_name).replace(" ", "")  # ALWAYS use Group1 format in filenames

    patch_dir = Path(dirs["patch"])
    pixel_dir = Path(dirs["pixel"])
    cluster_dir = Path(dirs["clustering"])
    cluster_dir.mkdir(exist_ok=True, parents=True)

    mass_peaks = load_peaks_from_patch_dir(
        patch_dir=patch_dir,
        pixel_dir=pixel_dir,
        expected_group_tag=group_tag,
    )

    all_patch_stems = {p.stem for p in patch_dir.glob(f"*_{group_tag}.png")}

    processed_peaks: set[str] = set()
    cluster_results: List[dict] = []

    for mz in sorted(mass_peaks.keys()):
        if debug_print:
            print(f"\nProcessing mz {mz:.4f}")

        # Determine max isomers across files for this mz
        max_isomers = 0
        for fb, peaks in mass_peaks[mz].items():
            max_isomers = max(max_isomers, len(peaks))

        if debug_print:
            print(f"Maximum isomers found: {max_isomers}")

        for isomer_pos in range(max_isomers):
            isomer_peaks: List[PeakInfo] = []
            for fb, peaks in mass_peaks[mz].items():
                if isomer_pos < len(peaks):
                    isomer_peaks.append(peaks[isomer_pos])

            if not isomer_peaks:
                continue

            if debug_print:
                print(f"\nValidating isomer position {isomer_pos + 1} peaks:")
                for p in isomer_peaks:
                    print(f"  - {p.peak_id} (RT: {p.rt_apex:.2f})")

            validated = filter_by_shape_similarity(
                isomer_peaks,
                similarity_threshold=similarity_threshold,
                target_size=target_size,
                debug_print=debug_print,
            )

            if len(validated) < len(isomer_peaks) and debug_print:
                print(f"Warning: {len(isomer_peaks) - len(validated)} peaks excluded (low similarity)")

            if not validated:
                continue

            for p in validated:
                processed_peaks.add(p.peak_id)

            # Compute aligned averages using validated peaks only
            avg_rt_start = float(np.mean([p.rt_start for p in validated]))
            avg_rt_apex = float(np.mean([p.rt_apex for p in validated]))
            avg_rt_end = float(np.mean([p.rt_end for p in validated]))
            avg_px_start = float(np.mean([p.pixel_start for p in validated]))
            avg_px_end = float(np.mean([p.pixel_end for p in validated]))

            for p in validated:
                cluster_results.append({
                    "mass": float(mz),
                    "isomer_position": int(isomer_pos + 1),
                    "component_number": 0,
                    "is_component": False,

                    "file": p.file_base,
                    "peak_id": p.peak_id,
                    "peak_num": int(p.peak_num),

                    "rt_start": p.rt_start,
                    "rt_apex": p.rt_apex,
                    "rt_end": p.rt_end,

                    "rt_start_aligned": avg_rt_start,
                    "rt_apex_aligned": avg_rt_apex,
                    "rt_end_aligned": avg_rt_end,

                    "pixel_start": p.pixel_start,
                    "pixel_end": p.pixel_end,
                    "pixel_start_aligned": avg_px_start,
                    "pixel_end_aligned": avg_px_end,

                    "peak_height": p.peak_height,
                    "peak_area": p.peak_area,
                })

    # Unclustered peaks report
    unclustered = sorted(all_patch_stems - processed_peaks)
    if unclustered and debug_print:
        print("\nWARNING: The following peaks were not clustered:")
        for u in unclustered:
            print(f"  - {u}")

    unclustered_csv = cluster_dir / f"unclustered_peaks_group_{group_tag}.csv"
    pd.DataFrame({"peak_id": unclustered}).to_csv(unclustered_csv, index=False)

    # Save raw per-peak alignment list
    df_results = pd.DataFrame(cluster_results)
    df_results.to_csv(cluster_dir / "peak_alignment.csv", index=False)

    # Build + save alignment summary
    summary_df = build_alignment_summary(cluster_results, group_tag=group_tag)
    summary_csv = cluster_dir / f"alignment_summary_group_{group_tag}.csv"
    summary_df.to_csv(summary_csv, index=False, float_format="%.4f")

    if debug_print:
        print("\nAlignment Summary:")
        # don't explode the console too much
        print(summary_df.head(20).to_string(index=False))
        print(f"\n[✔] Results saved to {summary_csv}")

    return (f"[✔] Clustering {group_tag}: patches={len(all_patch_stems)}, "
            f"clustered={len(processed_peaks)}, unclustered={len(unclustered)} | "
            f"summary={summary_csv.name}")


# ----------------------------
# Recluster / force-attach (RT+mass only) — FIXED for new peak_id format
# ----------------------------

def recluster_group(
    dirs: dict,
    group_name: str,
) -> str:
    """
    Post-clustering repair step (RT+mass only; NO shape check).
    Reads:
      - alignment_summary_group_{group_tag}.csv
      - unclustered_peaks_group_{group_tag}.csv
    Writes:
      - Feature_list_{group_tag}.csv
    """
    group_tag = str(group_name).replace(" ", "")

    pixel_dir = Path(dirs["pixel"])
    cluster_dir = Path(dirs["clustering"])
    cluster_dir.mkdir(exist_ok=True, parents=True)

    FORCE_MIN_COVERAGE = float(getattr(Config, "FORCE_MIN_COVERAGE", 0.95))  # fraction (0-1)
    FORCE_RT_TOL = float(getattr(Config, "FORCE_RT_TOL", 0.1))              # minutes
    FORCE_MASS_TOL = float(getattr(Config, "FORCE_MASS_TOL", 0.0001))       # m/z

    summary_csv = cluster_dir / f"alignment_summary_group_{group_tag}.csv"
    unclustered_csv = cluster_dir / f"unclustered_peaks_group_{group_tag}.csv"

    if not summary_csv.exists():
        return f"[!] Recluster {group_tag}: missing summary {summary_csv.name}"

    df_summary = pd.read_csv(summary_csv)

    # Ensure forced columns exist
    if "forced_files" not in df_summary.columns:
        df_summary["forced_files"] = ""
    if "has_forced" not in df_summary.columns:
        df_summary["has_forced"] = False

    height_cols = [c for c in df_summary.columns if c.endswith("_height")]
    area_cols = [c for c in df_summary.columns if c.endswith("_area")]
    sample_names = sorted({c[:-7] for c in height_cols})  # strip _height
    total_files = len(sample_names)

    if total_files == 0 or "peak_count" not in df_summary.columns:
        feature_list_path = cluster_dir / f"Feature_list_{group_tag}.csv"
        df_summary.to_csv(feature_list_path, index=False)
        return f"[✔] Recluster {group_tag}: wrote Feature_list (no sample columns)"

    if "mass" not in df_summary.columns or "rt_apex_aligned" not in df_summary.columns:
        feature_list_path = cluster_dir / f"Feature_list_{group_tag}.csv"
        df_summary.to_csv(feature_list_path, index=False)
        return f"[✔] Recluster {group_tag}: wrote Feature_list (missing mass/rt columns)"

    df_summary["_coverage"] = df_summary["peak_count"].astype(float) / float(total_files)
    mass_series = df_summary["mass"].astype(float)
    rt_series = df_summary["rt_apex_aligned"].astype(float)

    if not unclustered_csv.exists():
        feature_list_path = cluster_dir / f"Feature_list_{group_tag}.csv"
        df_summary.drop(columns=["_coverage"], errors="ignore").to_csv(feature_list_path, index=False)
        return f"[✔] Recluster {group_tag}: wrote Feature_list (no unclustered file)"

    df_uncl = pd.read_csv(unclustered_csv)
    if df_uncl.empty or "peak_id" not in df_uncl.columns:
        feature_list_path = cluster_dir / f"Feature_list_{group_tag}.csv"
        df_summary.drop(columns=["_coverage"], errors="ignore").to_csv(feature_list_path, index=False)
        return f"[✔] Recluster {group_tag}: wrote Feature_list (no unclustered peaks)"

    def parse_new_peak_id(peak_id: str) -> Optional[Tuple[str, float, int, str]]:
        parsed = parse_patch_stem(peak_id)
        return parsed

    def load_peak_meta(file_base: str, peak_num: int) -> Optional[dict]:
        pix_csv = pixel_dir / f"{file_base}_peaks_pix.csv"
        if not pix_csv.exists():
            return None
        try:
            dfp = pd.read_csv(pix_csv)
            row = dfp[dfp["peak_num"] == peak_num]
            if row.empty:
                return None
            row = row.iloc[0]
            return {
                "rt_apex": float(row["RT_apex"]),
                "peak_height": float(row["height"]) if "height" in dfp.columns else np.nan,
                "peak_area": float(row["area"]) if "area" in dfp.columns else np.nan,
            }
        except Exception:
            return None

    forced_events = []

    for peak_id in df_uncl["peak_id"].dropna().astype(str).tolist():
        parsed = parse_new_peak_id(peak_id)
        if parsed is None:
            continue

        file_base, peak_mass, peak_num, peak_group_tag = parsed
        if peak_group_tag != group_tag:
            continue

        # Summary uses sample columns by file_base (not full peak_id)
        if f"{file_base}_height" not in df_summary.columns or f"{file_base}_area" not in df_summary.columns:
            continue

        hcol = f"{file_base}_height"
        acol = f"{file_base}_area"

        meta = load_peak_meta(file_base, peak_num)
        if meta is None:
            continue

        peak_rt = float(meta["rt_apex"])

        candidates = df_summary.index[
            (df_summary["_coverage"] >= FORCE_MIN_COVERAGE) &
            (mass_series.sub(float(peak_mass)).abs() <= FORCE_MASS_TOL) &
            (rt_series.sub(float(peak_rt)).abs() <= FORCE_RT_TOL) &
            (df_summary[hcol].isna()) &
            (df_summary[acol].isna())
        ].tolist()

        if len(candidates) != 1:
            continue

        idx = candidates[0]

        # Apply forced attachment
        df_summary.loc[idx, hcol] = meta["peak_height"]
        df_summary.loc[idx, acol] = meta["peak_area"]

        # Update peak_files / peak_count if present
        if "peak_files" in df_summary.columns:
            existing = df_summary.loc[idx, "peak_files"]
            files = [x.strip() for x in str(existing).split(",") if x.strip()] if isinstance(existing, str) else []
            if file_base not in files:
                files.append(file_base)
            df_summary.loc[idx, "peak_files"] = ", ".join(sorted(set(files)))

        try:
            df_summary.loc[idx, "peak_count"] = int(df_summary.loc[idx, "peak_count"]) + 1
        except Exception:
            pass

        # forced flags
        forced_list = []
        existing_forced = df_summary.loc[idx, "forced_files"]
        if isinstance(existing_forced, str) and existing_forced.strip():
            forced_list = [x.strip() for x in existing_forced.split(",") if x.strip()]
        if peak_id not in forced_list:
            forced_list.append(peak_id)

        df_summary.loc[idx, "forced_files"] = ", ".join(forced_list)
        df_summary.loc[idx, "has_forced"] = True

        forced_events.append({
            "peak_id": peak_id,
            "file_base": file_base,
            "mass": float(peak_mass),
            "rt_apex": float(peak_rt),
            "target_row": int(idx),
        })

    df_summary.drop(columns=["_coverage"], errors="ignore", inplace=True)

    feature_list_path = cluster_dir / f"Feature_list_{group_tag}.csv"
    df_summary.to_csv(feature_list_path, index=False, float_format="%.4f")

    return f"[✔] Recluster {group_tag}: Feature_list={feature_list_path.name}, forced={len(forced_events)}"


# ----------------------------
# FINAL Excel export (FIXED to handle Group1 vs Group 1 folder/name mismatches)
# ----------------------------

def export_all_group_summaries_to_excel():
    output_root = Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER
    excel_path = output_root / f"MassSelectionSummary_{Config.ANALYSIS_FOLDER}.xlsx"

    print(f"\n[Excel Export] Saving all group summaries to {excel_path}\n")

    def group_variants(g: str) -> List[str]:
        g = str(g)
        # Always include both the original key and no-space version
        no_space = g.replace(" ", "")
        out = []
        if g not in out:
            out.append(g)
        if no_space not in out:
            out.append(no_space)
        return out

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        for group_name in Config.MASS_GROUPS.keys():
            variants = group_variants(group_name)

            # Find the first variant that has a summary or feature list
            found = None  # (variant, cluster_dir)
            for gv in variants:
                cluster_dir = output_root / gv / "Clustering"
                if (cluster_dir / f"Feature_list_{gv}.csv").exists() or (cluster_dir / f"alignment_summary_group_{gv}.csv").exists():
                    found = (gv, cluster_dir)
                    break

            if found is None:
                print(f"[!] Skipping {group_name}: no clustering outputs found for variants {variants}")
                continue

            gv, cluster_dir = found

            summary_csv = cluster_dir / f"alignment_summary_group_{gv}.csv"
            unresolved_csv = cluster_dir / f"unresolved_peaks_group_{gv}.csv"
            unclustered_csv = cluster_dir / f"unclustered_peaks_group_{gv}.csv"

            if not summary_csv.exists():
                print(f"[!] Skipping {group_name}: summary file not found in {cluster_dir}")
                continue

            sheet_name = str(group_name)
            current_row = 0

            feature_csv = cluster_dir / f"Feature_list_{gv}.csv"
            chosen_csv = feature_csv if feature_csv.exists() else summary_csv

            df_summary = pd.read_csv(chosen_csv)
            df_summary.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
            print(f"[✔] Sheet '{sheet_name}' → Summary ({len(df_summary)} rows) [{chosen_csv.name}]")

            # Forced formatting (bold + red)
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

                    forced_peak_ids = [x.strip() for x in forced_files.split(",") if x.strip()]

                    # Convert forced peak ids -> file_base (your summary columns are file_base_height/area)
                    def forced_to_file_base(tok: str) -> str:
                        parsed = parse_patch_stem(tok)
                        return parsed[0] if parsed else tok

                    forced_samples = [forced_to_file_base(x) for x in forced_peak_ids]
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
                    print(f"    ↳ Skipped unresolved peaks: file is empty")
            else:
                print(f"    ↳ No unresolved peaks file found")

            # Unclustered peaks
            if unclustered_csv.exists():
                df_unclustered = pd.read_csv(unclustered_csv)
                if not df_unclustered.empty:
                    df_unclustered.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
                    print(f"    ↳ Unclustered peaks added ({len(df_unclustered)} rows)")
                    current_row += len(df_unclustered) + 2
                else:
                    print(f"    ↳ Skipped unclustered peaks: file is empty")
            else:
                print(f"    ↳ No unclustered peaks file found")

    print(f"\n[✔] Excel export complete → {excel_path}")
