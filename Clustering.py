import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import pandas as pd
import cv2
from scipy.stats import pearsonr


# -----------------------------
# Parsing / data structures
# -----------------------------

PATCH_RE = re.compile(
    r"""
    ^
    (?P<base>.+?)                       # base
    _mass(?P<mz>[-+]?\d*\.?\d+)         # mz float-ish
    _Peak(?P<peak>\d+)                  # Peak index
    _(?P<group_tag>.+?)                 # group tag
    $
    """,
    re.VERBOSE,
)

@dataclass
class PeakInfo:
    peak_id: str
    file_base: str
    mass: float
    peak_num: int
    rt_start: float
    rt_apex: float
    rt_end: float
    pixel_start: float
    pixel_end: float
    peak_height: float
    peak_area: float
    image: np.ndarray
    # Optional: for combined peaks
    component_peaks: Optional[List["PeakInfo"]] = None


# -----------------------------
# Core logic (ported from your snippet)
# -----------------------------

def check_peak_resolution(
    peaks_in_file: List[PeakInfo],
    rt_resolution_threshold: float,
) -> Dict[str, Any]:
    """Check if peaks are well resolved and group unresolved peaks by RT apex proximity."""
    if len(peaks_in_file) < 2:
        return {"resolved": True, "groups": [[0]]}

    peaks_sorted = sorted(peaks_in_file, key=lambda x: x.rt_apex)

    current_group = [0]
    groups: List[List[int]] = []

    for i in range(len(peaks_sorted) - 1):
        rt_diff = peaks_sorted[i + 1].rt_apex - peaks_sorted[i].rt_apex
        if rt_diff < rt_resolution_threshold:
            current_group.append(i + 1)
        else:
            groups.append(current_group)
            current_group = [i + 1]

    if current_group:
        groups.append(current_group)

    return {"resolved": len(groups) == len(peaks_sorted), "groups": groups, "peaks_sorted": peaks_sorted}


def check_peak_similarity(
    peaks: List[PeakInfo],
    similarity_threshold: float,
    target_size: Tuple[int, int],
) -> List[PeakInfo]:
    """Validate peaks have similar shapes via Pearson correlation of 1D profile."""
    if not peaks:
        return []

    profiles: List[np.ndarray] = []
    valid_peaks: List[PeakInfo] = []

    for peak in peaks:
        img_resized = cv2.resize(peak.image, target_size, interpolation=cv2.INTER_AREA)
        img_smooth = cv2.GaussianBlur(img_resized, (3, 3), 0)

        # NOTE: your comment said "vertical profile" but code uses axis=0 mean -> profile along x
        profile = np.mean(img_smooth, axis=0)

        if np.std(profile) < 1e-10:
            print(f"  - {peak.peak_id}: skipped (constant profile)")
            continue

        profile = (profile - profile.min()) / (profile.max() - profile.min() + 1e-10)

        if np.std(profile) < 1e-10:
            print(f"  - {peak.peak_id}: skipped (constant after normalization)")
            continue

        profiles.append(profile.astype(np.float64))
        valid_peaks.append(peak)

    if len(profiles) < 1:
        return []
    if len(profiles) == 1:
        return valid_peaks

    correlations = np.zeros((len(valid_peaks), len(valid_peaks)), dtype=np.float64)
    for i in range(len(valid_peaks)):
        for j in range(len(valid_peaks)):
            if i == j:
                continue
            try:
                corr_val = float(pearsonr(profiles[i], profiles[j])[0])
                if np.isnan(corr_val):
                    corr_val = 0.0
                correlations[i, j] = corr_val
            except Exception:
                correlations[i, j] = 0.0

    avg_correlations = np.nanmean(correlations, axis=1)
    reference_idx = int(np.nanargmax(avg_correlations)) if len(avg_correlations) else 0
    reference_profile = profiles[reference_idx]

    similar_peaks: List[PeakInfo] = []
    print("\nSimilarity scores:")

    for idx, peak in enumerate(valid_peaks):
        try:
            corr_val = float(pearsonr(reference_profile, profiles[idx])[0])
            if np.isnan(corr_val):
                corr_val = 0.0
                print(f"  - {peak.peak_id}: nan")
            else:
                print(f"  - {peak.peak_id}: {corr_val:.3f}")

            if corr_val >= similarity_threshold:
                similar_peaks.append(peak)
            else:
                # Secondary width-ratio check (as in your code)
                ref_peak = valid_peaks[reference_idx]
                ref_width = ref_peak.rt_end - ref_peak.rt_start
                peak_width = peak.rt_end - peak.rt_start
                width_ratio = min(ref_width, peak_width) / max(ref_width, peak_width, 1e-12)
                if width_ratio > 0.75:
                    similar_peaks.append(peak)

        except Exception as e:
            print(f"  - {peak.peak_id}: error ({str(e)})")
            continue

    return similar_peaks


def _read_peak_row(df_pixels: pd.DataFrame, peak_num: int) -> Optional[pd.Series]:
    rows = df_pixels[df_pixels["peak_num"] == peak_num]
    if rows.empty:
        return None
    return rows.iloc[0]


def _load_pixels_csv(pixel_dir: Path, file_base: str) -> Optional[pd.DataFrame]:
    pixel_csv = pixel_dir / f"{file_base}_peaks_pix.csv"
    if not pixel_csv.exists():
        return None
    try:
        return pd.read_csv(pixel_csv)
    except Exception:
        return None


def _parse_patch_name(stem: str) -> Optional[Tuple[str, float, int, str]]:
    """
    Parse stem like:
      {base}_mass{mz}_Peak{i}_{group_tag}
    """
    m = PATCH_RE.match(stem)
    if not m:
        return None
    base = m.group("base")
    mz = float(m.group("mz"))
    peak = int(m.group("peak"))
    group_tag = m.group("group_tag")
    return base, mz, peak, group_tag


# -----------------------------
# Public API
# -----------------------------

def cluster_peaks(
    patch_dir: Path | str,
    pixel_dir: Path | str,
    out_dir: Path | str,
    *,
    group_tag: Optional[str] = None,
    similarity_threshold: float = 0.75,
    rt_resolution_threshold: float = 0.01,
    target_size: Tuple[int, int] = (50, 50),
) -> Dict[str, Path]:
    """
    Standalone clustering/alignment based on your checkpoint5 code.
    Returns paths of output files.
    """
    patch_dir = Path(patch_dir)
    pixel_dir = Path(pixel_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect peaks per mass per file_base
    mass_peaks: Dict[float, Dict[str, List[PeakInfo]]] = {}
    all_patch_stems: Set[str] = set()
    used_group_tags: Set[str] = set()

    for patch_file in patch_dir.glob("*.png"):
        stem = patch_file.stem
        all_patch_stems.add(stem)

        parsed = _parse_patch_name(stem)
        if not parsed:
            continue

        file_base, mass, peak_num, this_group_tag = parsed
        used_group_tags.add(this_group_tag)

        if group_tag is not None and this_group_tag != group_tag:
            continue

        df_pixels = _load_pixels_csv(pixel_dir, file_base)
        if df_pixels is None:
            continue

        row = _read_peak_row(df_pixels, peak_num)
        if row is None:
            continue

        img = cv2.imread(str(patch_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Expect these columns (as per your snippet)
        required_cols = ["RT_start", "RT_apex", "RT_end", "pixel_start", "pixel_end", "height", "area", "peak_num"]
        for c in required_cols:
            if c not in df_pixels.columns:
                raise ValueError(
                    f"Missing column '{c}' in {pixel_dir / (file_base + '_peaks_pix.csv')}. "
                    f"Found columns: {list(df_pixels.columns)}"
                )

        peak_info = PeakInfo(
            peak_id=stem,
            file_base=file_base,
            mass=float(mass),
            peak_num=int(peak_num),
            rt_start=float(row["RT_start"]),
            rt_apex=float(row["RT_apex"]),
            rt_end=float(row["RT_end"]),
            pixel_start=float(row["pixel_start"]),
            pixel_end=float(row["pixel_end"]),
            peak_height=float(row["height"]),
            peak_area=float(row["area"]),
            image=img,
        )

        mass_peaks.setdefault(peak_info.mass, {}).setdefault(file_base, []).append(peak_info)

    # If group_tag not given, but multiple found, we still run across all.
    if group_tag is None and len(used_group_tags) > 1:
        print(f"[i] Detected multiple group tags in patch dir: {sorted(used_group_tags)}")
        print("[i] Running clustering across ALL tags (group_tag=None). "
              "If you want per-group outputs, call cluster_peaks(..., group_tag='Group1')")

    # Handle unresolved peaks by combining
    for mass in list(mass_peaks.keys()):
        for file_base, peaks in list(mass_peaks[mass].items()):
            # Use the same logic but we need sorted peaks returned
            res = check_peak_resolution(peaks, rt_resolution_threshold)
            if res["resolved"]:
                continue

            peaks_sorted: List[PeakInfo] = res["peaks_sorted"]
            print(f"\nHandling unresolved peaks in {file_base} for mass {mass:.4f}")

            # We'll build a new list, combining groups
            new_peaks: List[PeakInfo] = []
            for grp in res["groups"]:
                if len(grp) == 1:
                    new_peaks.append(peaks_sorted[grp[0]])
                    continue

                group_peaks = [peaks_sorted[i] for i in grp]
                # intensity-weighted RT apex
                denom = sum(p.peak_height for p in group_peaks) or 1e-12
                rt_apex_w = sum(p.rt_apex * p.peak_height for p in group_peaks) / denom

                combined = PeakInfo(
                    peak_id=f"{file_base}_mass{mass:.4f}_combined" + "_".join(str(p.peak_num) for p in group_peaks),
                    file_base=file_base,
                    mass=mass,
                    peak_num=group_peaks[0].peak_num,
                    rt_start=min(p.rt_start for p in group_peaks),
                    rt_apex=float(rt_apex_w),
                    rt_end=max(p.rt_end for p in group_peaks),
                    pixel_start=min(p.pixel_start for p in group_peaks),
                    pixel_end=max(p.pixel_end for p in group_peaks),
                    peak_height=max(p.peak_height for p in group_peaks),
                    peak_area=sum(p.peak_area for p in group_peaks),
                    image=group_peaks[0].image,
                    component_peaks=group_peaks,
                )
                new_peaks.append(combined)

            mass_peaks[mass][file_base] = new_peaks

    cluster_results: List[Dict[str, Any]] = []
    processed_peaks: Set[str] = set()

    # Process each mass separately
    for mass in sorted(mass_peaks.keys()):
        print(f"\nProcessing mass {mass:.4f}")

        # Sort peaks by RT within each file
        for file_base in mass_peaks[mass]:
            mass_peaks[mass][file_base].sort(key=lambda x: x.rt_apex)

        max_isomers = max((len(peaks) for peaks in mass_peaks[mass].values()), default=0)
        print(f"Maximum isomers found: {max_isomers}")

        for isomer_pos in range(max_isomers):
            isomer_peaks: List[PeakInfo] = []
            for file_base in mass_peaks[mass]:
                file_peaks = mass_peaks[mass][file_base]
                if isomer_pos < len(file_peaks):
                    isomer_peaks.append(file_peaks[isomer_pos])

            if len(isomer_peaks) < 1:
                continue

            print(f"\nValidating isomer position {isomer_pos + 1} peaks:")
            for p in isomer_peaks:
                print(f"  - {p.peak_id} (RT: {p.rt_apex:.2f})")

            validated_peaks = check_peak_similarity(
                isomer_peaks,
                similarity_threshold=similarity_threshold,
                target_size=target_size,
            )

            if len(validated_peaks) < len(isomer_peaks):
                print(f"Warning: {len(isomer_peaks) - len(validated_peaks)} peaks excluded due to low similarity")

            if not validated_peaks:
                continue

            for peak in validated_peaks:
                processed_peaks.add(peak.peak_id)

            avg_rt_start = float(np.mean([p.rt_start for p in validated_peaks]))
            avg_rt_apex = float(np.mean([p.rt_apex for p in validated_peaks]))
            avg_rt_end = float(np.mean([p.rt_end for p in validated_peaks]))
            avg_pixel_start = float(np.mean([p.pixel_start for p in validated_peaks]))
            avg_pixel_end = float(np.mean([p.pixel_end for p in validated_peaks]))

            for peak in validated_peaks:
                result: Dict[str, Any] = {
                    "mass": float(mass),
                    "isomer_position": int(isomer_pos + 1),
                    "file": peak.file_base,
                    "peak_num": int(peak.peak_num),
                    "rt_start": float(peak.rt_start),
                    "rt_apex": float(peak.rt_apex),
                    "rt_end": float(peak.rt_end),
                    "rt_start_aligned": avg_rt_start,
                    "rt_apex_aligned": avg_rt_apex,
                    "rt_end_aligned": avg_rt_end,
                    "pixel_start": float(peak.pixel_start),
                    "pixel_end": float(peak.pixel_end),
                    "pixel_start_aligned": avg_pixel_start,
                    "pixel_end_aligned": avg_pixel_end,
                    "peak_height": float(peak.peak_height),
                    "peak_area": float(peak.peak_area),
                }

                if peak.component_peaks:
                    for i, comp_peak in enumerate(peak.component_peaks):
                        result[f"component_{i+1}_height"] = float(comp_peak.peak_height)
                        result[f"component_{i+1}_area"] = float(comp_peak.peak_area)
                        result[f"component_{i+1}_rt"] = float(comp_peak.rt_apex)

                cluster_results.append(result)

    # Save unclustered
    unclustered = {stem for stem in all_patch_stems} - {pid for pid in processed_peaks}
    unclustered_csv = out_dir / f"unclustered_peaks_group_{group_tag or 'ALL'}.csv"
    if unclustered:
        print("\nWARNING: The following peaks were not clustered:")
        for peak in sorted(unclustered):
            print(f"  - {peak}")
        pd.DataFrame({"peak_id": sorted(unclustered)}).to_csv(unclustered_csv, index=False)
    else:
        # still write an empty file? usually not necessary; skip
        unclustered_csv = None  # type: ignore

    # Save peak_alignment.csv
    peak_alignment_csv = out_dir / "peak_alignment.csv"
    df_results = pd.DataFrame(cluster_results)
    df_results.to_csv(peak_alignment_csv, index=False)

    # Build expanded summary (your exact approach)
    expanded_results: List[Dict[str, Any]] = []
    for result in cluster_results:
        if any(k.startswith("component_") and k.endswith("_height") for k in result.keys()):
            base_result = {k: v for k, v in result.items() if not k.startswith("component_")}
            num_components = sum(1 for k in result.keys() if k.startswith("component_") and k.endswith("_height"))
            for i in range(num_components):
                component_result = dict(base_result)
                component_result["peak_height"] = result.get(f"component_{i+1}_height")
                component_result["peak_area"] = result.get(f"component_{i+1}_area")
                component_result["rt_apex"] = result.get(f"component_{i+1}_rt")
                component_result["is_component"] = True
                component_result["component_number"] = i + 1
                expanded_results.append(component_result)
        else:
            r2 = dict(result)
            r2["is_component"] = False
            r2["component_number"] = 0
            expanded_results.append(r2)

    df_expanded = pd.DataFrame(expanded_results)

    summary_csv = out_dir / f"alignment_summary_group_{group_tag or 'ALL'}.csv"
    if not df_expanded.empty:
        peak_heights = df_expanded.pivot(
            index=["mass", "isomer_position", "component_number"],
            columns="file",
            values="peak_height",
        ).round(0)
        peak_heights.columns = [f"{col}_height" for col in peak_heights.columns]

        peak_areas = df_expanded.pivot(
            index=["mass", "isomer_position", "component_number"],
            columns="file",
            values="peak_area",
        ).round(0)
        peak_areas.columns = [f"{col}_area" for col in peak_areas.columns]

        summary = df_expanded.groupby(["mass", "isomer_position", "component_number"]).agg(
            file=("file", "count"),
            peak_files=("file", lambda x: ", ".join(sorted(x))),
            rt_apex_aligned=("rt_apex_aligned", "first"),
            rt_start_aligned=("rt_start_aligned", "first"),
            rt_end_aligned=("rt_end_aligned", "first"),
            is_component=("is_component", "first"),
        ).round(4)

        summary.rename(columns={"file": "peak_count"}, inplace=True)

        final_summary = pd.concat([summary, peak_heights, peak_areas], axis=1)

        cols = list(summary.columns)
        sample_cols = [c for c in final_summary.columns if c not in cols]
        height_cols = sorted([c for c in sample_cols if c.endswith("_height")])
        area_cols = sorted([c for c in sample_cols if c.endswith("_area")])
        final_summary = final_summary[cols + height_cols + area_cols]

        final_summary = final_summary.reset_index()
        final_summary["component_label"] = final_summary.apply(
            lambda x: f" (Component {int(x['component_number'])})" if bool(x["is_component"]) else "",
            axis=1,
        )

        # Force 4-decimal mass string (your patch)
        final_summary["mass"] = final_summary["mass"].apply(lambda x: f"{float(x):.4f}")
        final_summary = final_summary.set_index(["mass", "isomer_position"])

        final_summary.to_csv(summary_csv, float_format="%.4f")
        print("\nAlignment Summary:")
        print(final_summary.to_string())
        print(f"\n[✔] Results saved to {summary_csv}")
    else:
        # still write an empty summary for pipeline stability
        pd.DataFrame().to_csv(summary_csv, index=False)
        print(f"[i] No results; wrote empty summary to {summary_csv}")

    outputs = {
        "peak_alignment_csv": peak_alignment_csv,
        "summary_csv": summary_csv,
    }
    if unclustered_csv:
        outputs["unclustered_csv"] = unclustered_csv
    return outputs

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Peak clustering/alignment by shape similarity.")
    ap.add_argument("--patch", required=True, help="Path to patch directory containing PNG slices")
    ap.add_argument("--pixel", required=True, help="Path to pixel directory containing *_peaks_pix.csv")
    ap.add_argument("--out", required=True, help="Output directory (e.g., clustering)")
    ap.add_argument("--group-tag", default=None, help="Only process this group tag (e.g., Group1). Default: all.")
    ap.add_argument("--sim", type=float, default=0.75, help="Similarity threshold (Pearson)")
    ap.add_argument("--rt-res", type=float, default=0.01, help="RT resolution threshold (minutes)")
    ap.add_argument("--size", default="50,50", help="Resize target size like '50,50'")

    args = ap.parse_args()
    w, h = (int(x.strip()) for x in args.size.split(","))

    cluster_peaks(
        patch_dir=args.patch,
        pixel_dir=args.pixel,
        out_dir=args.out,
        group_tag=args.group_tag,
        similarity_threshold=args.sim,
        rt_resolution_threshold=args.rt_res,
        target_size=(w, h),
    )
