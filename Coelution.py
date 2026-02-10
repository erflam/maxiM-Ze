"""
Coelution.py
Resolves coeluting isomers using watershed segmentation and valley detection.

Reads Peak Info CSVs and EIC PNGs from EICBuilder output.
Identifies which peaks are truly resolved vs. coeluting based on:
- RT separation
- Valley depth between apices
- Pixel separation

Outputs resolved isomer CSVs with proper peak boundaries and cluster assignments.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
import cv2
from PIL import Image
from multiprocessing import Pool, cpu_count

from Config import Config


# ========================================
# Coelution Resolution Parameters
# ========================================
MIN_RT_DIFF = 0.095  # Minimum RT separation (minutes) to consider peaks resolved
VALLEY_DROP_MIN = 0.40  # Minimum valley drop (40%) between apices to call peaks resolved
MIN_PIXEL_SEP = 6  # Minimum pixel separation between apices
MARGIN_FRAC = 0.08  # Fraction of image to exclude as margin when estimating plot bounds

# Morphological kernel for edge dilation
KERNEL_3x3 = np.ones((3, 3), np.uint8)


def estimate_plot_bounds(h: int, w: int) -> Tuple[int, int, int, int]:
    """
    Estimate the actual plot area within the image (excluding margins).
    
    Returns: (top, bottom, left, right) indices
    """
    t = int(MARGIN_FRAC * h)
    l = int(MARGIN_FRAC * w)
    return max(0, t), h - t, max(0, l), w - l


def build_elevation_from_alpha(alpha: np.ndarray) -> np.ndarray:
    """
    Build an elevation map from alpha channel for watershed segmentation.
    
    High values = peaks/ridges (signal)
    Low values = valleys (background/between peaks)
    
    Args:
        alpha: Alpha channel (2D array)
    
    Returns:
        Elevation map (uint8)
    """
    # Detect edges
    edges = cv2.Canny(alpha, 0, 0)
    edges = cv2.dilate(edges, KERNEL_3x3, iterations=1)
    
    # Distance transform (distance from edges)
    dist = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 3)
    
    # Normalize to 0-255
    normalized = (dist / max(1.0, dist.max()) * 255).astype(np.uint8)
    
    # Invert (so peaks are high) and smooth
    return 255 - cv2.medianBlur(normalized, 5)


def watershed_labels(alpha_roi: np.ndarray, seed_cols_rel: np.ndarray) -> np.ndarray:
    """
    Perform watershed segmentation on the alpha channel ROI.
    
    Args:
        alpha_roi: Alpha channel region of interest
        seed_cols_rel: Column indices for watershed seeds (apex positions)
    
    Returns:
        Label array (same shape as alpha_roi)
    """
    h, w = alpha_roi.shape
    if w < 2 or h < 2 or not np.any(seed_cols_rel):
        return np.zeros((h, w), dtype=np.int32)
    
    # Build elevation map
    elev = build_elevation_from_alpha(alpha_roi)
    
    # Create markers (seed points)
    markers = np.zeros((h, w), np.int32)
    y_range = slice(h // 4, 3 * h // 4)  # Place seeds in middle vertical region
    
    for i, xc in enumerate(seed_cols_rel, 1):
        xc = int(np.clip(xc, 0, w - 1))
        markers[y_range, xc] = i
    
    # Run watershed
    color = cv2.cvtColor(elev, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)
    
    # Clean up watershed boundaries (marked as -1)
    markers[markers < 0] = 0
    
    return markers


def column_labels(labels: np.ndarray) -> np.ndarray:
    """
    Convert 2D watershed labels to 1D column-wise labels.
    Each column gets the most common label in that column.
    
    Args:
        labels: 2D label array from watershed
    
    Returns:
        1D array of labels per column
    """
    h, w = labels.shape
    out = np.zeros(w, dtype=np.int32)
    
    for x in range(w):
        col = labels[:, x]
        col = col[col > 0]  # Ignore background
        if col.size:
            vals, counts = np.unique(col, return_counts=True)
            out[x] = vals[np.argmax(counts)]
    
    # Fill gaps (forward/backward fill)
    nz = np.nonzero(out)[0]
    if nz.size:
        out[:nz[0]] = out[nz[0]]
        out[nz[-1] + 1:] = out[nz[-1]]
    
    return out


def smooth_profile(profile: np.ndarray) -> np.ndarray:
    """
    Smooth 1D intensity profile using Gaussian blur.
    
    Args:
        profile: 1D intensity profile
    
    Returns:
        Smoothed profile
    """
    if profile.ndim == 1:
        p2 = cv2.GaussianBlur(
            profile.reshape(1, -1).astype(np.float32),
            (1, 21),
            0
        ).ravel()
        return p2
    return profile


def valley_metrics(
    profile_smooth: np.ndarray,
    xL: int,
    xR: int
) -> Tuple[int, float, float, float]:
    """
    Calculate valley metrics between two apex positions.
    
    Args:
        profile_smooth: Smoothed intensity profile
        xL: Left apex position (pixel)
        xR: Right apex position (pixel)
    
    Returns:
        (valley_x, valley_y, left_apex_y, right_apex_y)
    """
    if xL > xR:
        xL, xR = xR, xL
    
    xL = max(0, xL)
    xR = min(len(profile_smooth) - 1, xR)
    
    seg = profile_smooth[xL:xR + 1]
    if seg.size == 0:
        return (xL + xR) // 2, 0.0, 0.0, 0.0
    
    # Find valley (minimum) in segment
    v_local_idx = int(np.argmin(seg))
    valley_x = xL + v_local_idx
    valley_y = profile_smooth[valley_x]
    
    # Apex intensities
    left_apex_y = profile_smooth[xL]
    right_apex_y = profile_smooth[xR]
    
    return valley_x, float(valley_y), float(left_apex_y), float(right_apex_y)


def find_boundary_or_dip(
    lbl1d: np.ndarray,
    profile: np.ndarray,
    x0: int,
    x1: int,
    id1: int,
    id2: int
) -> int:
    """
    Find the boundary between two peaks using watershed labels or intensity dip.
    
    Args:
        lbl1d: 1D watershed label array
        profile: Intensity profile
        x0: Start position
        x1: End position
        id1: Label of first peak
        id2: Label of second peak
    
    Returns:
        Boundary position (pixel)
    """
    if x0 > x1:
        x0, x1 = x1, x0
    
    # Try to find watershed boundary first
    if np.any(lbl1d):
        seg = lbl1d[x0:x1 + 1]
        diff_idx = np.where(np.diff(seg) != 0)[0]
        
        for k in diff_idx:
            l = seg[k]
            r = seg[k + 1] if k + 1 < seg.size else 0
            if (l == id1 and r == id2) or (l == id2 and r == id1):
                return x0 + k + 1
    
    # Fallback: find intensity dip
    s = profile[max(0, x0):min(len(profile), x1 + 1)]
    if len(s) < 3:
        return (x0 + x1) // 2
    
    # Find local minimum (valley)
    d = np.diff(s)
    neg_pos = (d[:-1] < 0) & (d[1:] > 0)  # Negative slope followed by positive
    idxs = np.where(neg_pos)[0]
    
    if idxs.size:
        return x0 + int(idxs[0] + 1)
    else:
        return x0 + int(np.argmin(s))


def resolve_coeluting_peaks_for_file(
    peakinfo_csv: Path,
    png_path: Path,
    output_csv: Path,
    group_name: str
) -> Tuple[str, int, int]:
    """
    Resolve coeluting peaks for a single file.
    
    Args:
        peakinfo_csv: Path to peakinfo CSV from EICBuilder
        png_path: Path to EIC PNG from EICBuilder
        output_csv: Path to output resolved isomers CSV
        group_name: Mass group name
    
    Returns:
        (base_name, num_peaks, num_resolved)
    """
    base = peakinfo_csv.stem.replace(f"_peakinfo_{group_name}", "")
    
    # Skip if already processed
    if output_csv.exists():
        return (base, 0, 0)
    
    # Check required files exist
    if not peakinfo_csv.exists() or not png_path.exists():
        return (base, 0, 0)
    
    # Load peak info
    df = pd.read_csv(peakinfo_csv)
    
    # Ensure required columns
    required = ['m/z', 'RT_start', 'RT_apex', 'RT_end', 'Peak Area', 'Height']
    if not all(col in df.columns for col in required):
        return (base, 0, 0)
    
    # Rename for consistency
    df = df.rename(columns={'Peak Area': 'area', 'Height': 'height'})
    
    # Filter by duration (0.05 to 0.75 minutes)
    df = df[(df['RT_end'] - df['RT_start']).between(0.05, 0.75)]
    
    if df.empty:
        # Create empty output
        pd.DataFrame(columns=[
            'File Name', 'm/z', 'RT_apex', 'pixel_start', 'pixel_end',
            'peak_type', 'cluster_id', 'is_cluster_lead'
        ]).to_csv(output_csv, index=False)
        return (base, 0, 0)
    
    # Load PNG image
    im = Image.open(png_path).convert('RGBA')
    W, H = im.size
    A = np.array(im)[..., 3].astype(np.uint8)  # Alpha channel
    
    # Calculate RT-pixel conversion
    rt_min = df['RT_start'].min() - 0.1
    rt_max = df['RT_end'].max() + 0.1
    rt_range = rt_max - rt_min
    width_factor = (W - 1) / rt_range
    
    def rt_to_px(rt_vals):
        return np.clip(
            ((np.asarray(rt_vals) - rt_min) * width_factor).astype(int),
            0,
            W - 1
        )
    
    def px_to_rt(px_vals):
        return rt_min + (np.asarray(px_vals) / (W - 1)) * rt_range
    
    # Get plot bounds and intensity profile
    top, bot, left, right = estimate_plot_bounds(H, W)
    alpha_roi = A[top:bot, left:right]
    profile_raw = A[top:bot, left:right].sum(axis=0).astype(np.float32)
    profile = smooth_profile(profile_raw)
    
    # Group by m/z and sort by RT_apex
    df['_mz_key'] = df['m/z'].astype(str)
    df = df.sort_values(['_mz_key', 'RT_apex']).reset_index(drop=True)
    
    rows_out = []
    
    for mz_key, grp in df.groupby('_mz_key', sort=False):
        grp = grp.sort_values('RT_apex').reset_index(drop=True)
        
        if len(grp) == 1:
            # Single peak - no coelution
            row = grp.iloc[0].to_dict()
            ps, pe = rt_to_px([row['RT_start'], row['RT_end']])
            row.update({
                'File Name': base,
                'pixel_start': int(ps),
                'pixel_end': int(pe),
                'peak_type': 'resolved',
                'cluster_id': f'{mz_key}_0',
                'is_cluster_lead': True
            })
            rows_out.append(row)
            continue
        
        # Multiple peaks for this m/z - check for coelution
        apex_px = rt_to_px(grp['RT_apex'].values)
        l_px = rt_to_px(grp['RT_start'].values)
        r_px = rt_to_px(grp['RT_end'].values)
        seeds_rel = np.clip(apex_px - left, 0, right - left - 1)
        
        # Watershed segmentation
        labels = watershed_labels(alpha_roi, seeds_rel)
        lbl1d = column_labels(labels)
        
        # Determine which adjacent pairs are resolved
        resolved_pair = []
        for i in range(len(grp) - 1):
            rt_diff = float(grp['RT_apex'].iloc[i + 1] - grp['RT_apex'].iloc[i])
            px_diff = int(abs(apex_px[i + 1] - apex_px[i]))
            
            # Calculate valley metrics
            r1 = int(np.clip(apex_px[i] - left, 0, right - left - 1))
            r2 = int(np.clip(apex_px[i + 1] - left, 0, right - left - 1))
            rL, rR = (r1, r2) if r1 <= r2 else (r2, r1)
            vx, vy, yL, yR = valley_metrics(profile, rL, rR)
            
            min_apex_y = max(1.0, min(yL, yR))
            valley_drop = 1.0 - float(vy) / float(min_apex_y)
            
            # Check resolution criteria
            is_resolved = (
                (rt_diff >= MIN_RT_DIFF) and
                (px_diff >= MIN_PIXEL_SEP) and
                (valley_drop >= VALLEY_DROP_MIN)
            )
            
            cut = None
            if is_resolved:
                cut = left + find_boundary_or_dip(lbl1d, profile, rL, rR, i + 1, i + 2)
                cut = int(np.clip(cut, 0, W - 1))
            
            resolved_pair.append((is_resolved, cut))
        
        # Build clusters of coeluting peaks
        clusters = []
        current = [0]
        for i, (ok, _) in enumerate(resolved_pair):
            if ok:
                clusters.append(current)
                current = [i + 1]
            else:
                current.append(i + 1)
        clusters.append(current)
        
        # Process each cluster
        for cidx, members in enumerate(clusters):
            member_idxs = np.array(members, dtype=int)
            ps = int(np.min(l_px[member_idxs]))
            pe = int(np.max(r_px[member_idxs]))
            
            # Constrain by watershed label extents
            w_lefts, w_rights = [], []
            for j in member_idxs:
                label_id = j + 1
                mask = (lbl1d == label_id)
                if np.any(mask):
                    idxs = np.where(mask)[0]
                    w_lefts.append(int(left + idxs[0]))
                    w_rights.append(int(left + idxs[-1]))
            
            if w_lefts and w_rights:
                ps = max(ps, min(w_lefts))
                pe = min(pe, max(w_rights))
            
            ps = int(np.clip(ps, 0, W - 2))
            pe = int(np.clip(pe, ps + 1, W))
            
            if len(member_idxs) == 1:
                # Single peak in cluster (fully resolved)
                i = member_idxs[0]
                ps_single = max(ps, int(l_px[i]))
                pe_single = min(pe, int(r_px[i]))
                row = grp.iloc[i].to_dict()
                row.update({
                    'File Name': base,
                    'RT_start': float(px_to_rt(ps_single)),
                    'RT_end': float(px_to_rt(pe_single)),
                    'pixel_start': int(ps_single),
                    'pixel_end': int(pe_single),
                    'peak_type': 'resolved',
                    'cluster_id': f'{mz_key}_{cidx}',
                    'is_cluster_lead': True
                })
                rows_out.append(row)
            else:
                # Multiple peaks in cluster (coeluting)
                for k, i in enumerate(member_idxs):
                    row = grp.iloc[i].to_dict()
                    row.update({
                        'File Name': base,
                        'RT_start': float(px_to_rt(ps)),
                        'RT_end': float(px_to_rt(pe)),
                        'pixel_start': int(ps),
                        'pixel_end': int(pe),
                        'peak_type': 'coeluting',
                        'cluster_id': f'{mz_key}_{cidx}',
                        'is_cluster_lead': bool(k == 0)
                    })
                    rows_out.append(row)
    
    # Create output dataframe
    df_out = pd.DataFrame(rows_out)
    
    # Ensure required columns
    required_cols = [
        'File Name', 'm/z', 'RT_start', 'RT_apex', 'RT_end',
        'area', 'height', 'pixel_start', 'pixel_end',
        'peak_type', 'cluster_id', 'is_cluster_lead'
    ]
    for col in required_cols:
        if col not in df_out.columns:
            if col == 'is_cluster_lead':
                df_out[col] = False
            else:
                df_out[col] = np.nan
    
    # Sort by m/z then RT_apex
    df_out = df_out.sort_values(['m/z', 'RT_apex'])
    
    # Save output
    df_out.to_csv(output_csv, index=False)
    
    num_resolved = (df_out['peak_type'] == 'resolved').sum()
    num_coeluting = (df_out['peak_type'] == 'coeluting').sum()
    
    return (base, len(df_out), num_resolved)


def process_single_file_wrapper(args: Tuple[Path, Path, Path, str]) -> str:
    """
    Wrapper for parallel processing.
    
    Args:
        args: (peakinfo_csv, png_path, output_csv, group_name)
    
    Returns:
        Status message
    """
    peakinfo_csv, png_path, output_csv, group_name = args
    
    try:
        base, total, resolved = resolve_coeluting_peaks_for_file(
            peakinfo_csv,
            png_path,
            output_csv,
            group_name
        )
        
        if total == 0:
            if output_csv.exists():
                return f"[↷] {base} (cached)"
            else:
                return f"[!] {base} (missing files or no peaks)"
        
        coeluting = total - resolved
        return f"[✔] {base} ({total} peaks: {resolved} resolved, {coeluting} coeluting)"
        
    except Exception as e:
        return f"[!] {Path(peakinfo_csv).stem}: {str(e)[:80]}"


def run_for_group(group_name: str, n_processes: int = None) -> List[str]:
    """
    Resolve coeluting peaks for all files in a mass group.
    
    Args:
        group_name: Mass group name
        n_processes: Number of parallel processes
    
    Returns:
        List of status messages
    """
    Config.set_mass_group(group_name)
    
    group_dir = Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER / str(Config.CURRENT_GROUP)
    peakinfo_dir = group_dir / "EIC CSVs" / "Peak Info CSVs"
    png_dir = group_dir / "EIC PNGs"
    output_dir = group_dir / "Resolved Isomers"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all peakinfo CSVs
    peakinfo_files = list(peakinfo_dir.glob(f"*_peakinfo_{group_name}.csv"))
    
    if not peakinfo_files:
        print(f"[!] No peakinfo CSVs found for {group_name} in {peakinfo_dir}")
        return []
    
    # Build argument list
    args_list = []
    for peakinfo_csv in peakinfo_files:
        base = peakinfo_csv.stem.replace(f"_peakinfo_{group_name}", "")
        png_path = png_dir / f"EIC_{base}_plotly.png"
        output_csv = output_dir / f"{base}_resolved_{group_name}.csv"
        args_list.append((peakinfo_csv, png_path, output_csv, group_name))
    
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)
    
    print(f"\n{'=' * 70}")
    print(f"Resolving Coeluting Isomers for {group_name}")
    print(f"{'=' * 70}")
    print(f"Processing {len(peakinfo_files)} files using {n_processes} processes")
    print(f"Output directory: {output_dir}\n")
    
    import time
    start_time = time.time()
    
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_single_file_wrapper, args_list)
    
    elapsed_time = time.time() - start_time
    
    print("\nCoelution Resolution Results:")
    for result in results:
        print(result)
    
    successful = sum(1 for r in results if r.startswith("[✔]"))
    cached = sum(1 for r in results if r.startswith("[↷]"))
    failed = sum(1 for r in results if r.startswith("[!]"))
    
    print(f"\nSummary: {successful} processed, {cached} cached, {failed} failed")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    if len(peakinfo_files) > 0:
        print(f"Average time per file: {elapsed_time / len(peakinfo_files):.2f} seconds")
    
    return results


def run_for_all_groups(n_processes: int = None) -> Dict[str, List[str]]:
    """
    Resolve coeluting peaks for all mass groups.
    
    Args:
        n_processes: Number of parallel processes
    
    Returns:
        Dictionary of results per group
    """
    import time
    
    total_start_time = time.time()
    all_results = {}
    
    print("\n" + "=" * 70)
    print("COELUTION RESOLUTION - ALL GROUPS")
    print("=" * 70)
    
    for group_name in Config.MASS_GROUPS.keys():
        results = run_for_group(group_name, n_processes)
        all_results[group_name] = results
    
    total_elapsed = time.time() - total_start_time
    
    print("\n" + "=" * 70)
    print("COELUTION RESOLUTION COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_elapsed:.2f} seconds")
    
    return all_results


if __name__ == "__main__":
    run_for_all_groups()
