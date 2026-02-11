"""
EICBuilder.py
Builds PNG images of Extracted Ion Chromatograms (EICs) for specific masses
with peak detection.
Reads from pre-filtered CSV files for efficiency.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from scipy.integrate import trapezoid
from scipy.signal import find_peaks, peak_prominences, peak_widths, savgol_filter, argrelextrema
from scipy.ndimage import gaussian_filter1d

import plotly.io as pio
import plotly.graph_objects as go
from PIL import Image

from multiprocessing import Pool, cpu_count
import colorsys

from Config import Config
from FileUtils import FileUtils  # (unused but leaving as-is)

# ----------------------------
# Peak detection parameters (MATCH ORIGINAL)
# ----------------------------
REL_HEIGHT_BY_MASS = {104.1069: 0.985, 187.0964: 0.985, 119.0896: 0.98}
DEFAULT_REL_HEIGHT = 0.99
MAX_PEAK_DURATION = 1.5

# Shoulder detection parameters (kept from your newer pipeline)
SHOULDER_MIN_HEIGHT_FRAC = 0.02
SHOULDER_MIN_NOISE_MULT = 3.0
SHOULDER_MIN_SEP_MIN = 0.06
SHOULDER_VALLEY_DROP_FRAC = 0.85
SHOULDER_LOCALMAX_ORDER = 2


def init_worker():
    """Initialize worker process for parallel execution"""
    import matplotlib
    matplotlib.use('Agg')
    os.environ['KALEIDO_DISABLE'] = '1'
    os.environ['PLOTLY_RENDERER'] = 'json'


# ----------------------------
# Your improved peak cutting (unchanged)
# ----------------------------
def improved_peak_cutting(intensity_vals_smooth, rt_vals, peaks, width_results, specific_mass):
    """
    Improved peak cutting algorithm to split overlapping peaks
    """
    MIN_APEX_SEP_MIN = 0.015
    MAX_SEG_WIDTH_MIN = 0.8
    MIN_SEG_WIDTH_MIN = 0.015

    all_minima = set()
    smoothed = gaussian_filter1d(intensity_vals_smooth, sigma=1.0)
    minima = argrelextrema(smoothed, np.less, order=2)[0]
    all_minima.update(minima)

    all_minima = np.array(sorted(all_minima))
    if len(all_minima) == 0:
        return peaks, width_results

    rt_delta = float(np.mean(np.diff(rt_vals))) if len(rt_vals) > 1 else 0.0

    def seg_time_ok(l_i, r_i):
        if rt_delta <= 0:
            return True
        wmin = (r_i - l_i) * rt_delta
        return MIN_SEG_WIDTH_MIN <= wmin <= MAX_SEG_WIDTH_MIN

    def calculate_peak_quality(apex_idx, left_idx, right_idx):
        if right_idx <= left_idx or apex_idx < left_idx or apex_idx > right_idx:
            return 0.0
        segment = intensity_vals_smooth[left_idx:right_idx + 1]
        apex_val = intensity_vals_smooth[apex_idx]
        left_min = np.min(segment[:apex_idx - left_idx + 1]) if apex_idx > left_idx else apex_val
        right_min = np.min(segment[apex_idx - left_idx:]) if apex_idx < right_idx else apex_val
        prominence = apex_val - max(left_min, right_min)
        return prominence

    try:
        width_results_split = peak_widths(intensity_vals_smooth, peaks, rel_height=0.85)
        valid_split = width_results_split[0] > 0
    except:
        valid_split = np.zeros(len(peaks), dtype=bool)
        width_results_split = (
            np.zeros(len(peaks)), np.zeros(len(peaks)),
            np.zeros(len(peaks)), np.zeros(len(peaks))
        )

    new_peaks = []

    for i_pk, pk in enumerate(peaks):
        if valid_split[i_pk]:
            left_ip, right_ip = width_results_split[2][i_pk], width_results_split[3][i_pk]
        else:
            left_ip, right_ip = width_results[2][i_pk], width_results[3][i_pk]

        left_idx = max(0, int(np.floor(left_ip)))
        right_idx = min(len(intensity_vals_smooth) - 1, int(np.ceil(right_ip)))

        internal_valleys = [v for v in all_minima if left_idx < v < right_idx]
        if not internal_valleys:
            new_peaks.append(pk)
            continue

        significant_valleys = []
        peak_intensity = intensity_vals_smooth[pk]

        for valley in internal_valleys:
            valley_intensity = intensity_vals_smooth[valley]
            left_peak_h = np.max(intensity_vals_smooth[left_idx:valley + 1])
            right_peak_h = np.max(intensity_vals_smooth[valley:right_idx + 1])

            if valley_intensity < 0.7 * min(left_peak_h, right_peak_h):
                valley_rt = rt_vals[valley]
                peak_rt = rt_vals[pk]
                if abs(valley_rt - peak_rt) >= MIN_APEX_SEP_MIN / 2:
                    significant_valleys.append(valley)

        if not significant_valleys:
            new_peaks.append(pk)
            continue

        seg_starts = [left_idx] + significant_valleys
        seg_ends = significant_valleys + [right_idx]

        candidate_peaks = []
        for seg_l, seg_r in zip(seg_starts, seg_ends):
            if seg_r - seg_l < 3 or not seg_time_ok(seg_l, seg_r):
                continue
            local_segment = intensity_vals_smooth[seg_l:seg_r + 1]
            local_apex_offset = np.argmax(local_segment)
            apex_idx = seg_l + local_apex_offset
            quality = calculate_peak_quality(apex_idx, seg_l, seg_r)
            if quality > peak_intensity * 0.1:
                candidate_peaks.append((apex_idx, quality))

        if len(candidate_peaks) >= 2:
            candidate_peaks.sort(key=lambda x: x[1], reverse=True)
            selected_peaks = [candidate_peaks[0]]
            for candidate in candidate_peaks[1:]:
                apex_idx, quality = candidate
                min_sep = min(abs(rt_vals[apex_idx] - rt_vals[selected[0]]) for selected in selected_peaks)
                if min_sep >= MIN_APEX_SEP_MIN:
                    selected_peaks.append(candidate)
            for apex_idx, _ in selected_peaks:
                new_peaks.append(apex_idx)
        elif len(candidate_peaks) == 1:
            new_peaks.append(candidate_peaks[0][0])
        else:
            new_peaks.append(pk)

    if new_peaks:
        new_peaks = np.unique(np.array(new_peaks, dtype=int))
        try:
            new_width_results = peak_widths(intensity_vals_smooth, new_peaks, rel_height=0.99)
            valid_mask = new_width_results[0] > 0
            new_peaks = new_peaks[valid_mask]
            new_width_results = tuple(arr[valid_mask] for arr in new_width_results)
            return new_peaks, new_width_results
        except:
            return peaks, width_results

    return peaks, width_results


# ----------------------------
# FULL RT AXIS LOADER (key to matching original)
# ----------------------------
def load_rt_axis_for_file(peaks_csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load full scan/rt axis for this file.

    Expects a file named: {base}_rtaxis.csv
    in the same folder as the peaks CSV.

    Columns required: scan, rt
    """
    stem = peaks_csv_path.stem
    if "_peaks_" in stem:
        base = stem.split("_peaks_")[0]
    else:
        base = stem

    rtaxis_path = peaks_csv_path.parent / f"{base}_rtaxis.csv"
    if not rtaxis_path.exists():
        raise FileNotFoundError(f"Missing RT axis file: {rtaxis_path}")

    axis_df = pd.read_csv(rtaxis_path)
    if not {"scan", "rt"}.issubset(axis_df.columns):
        raise ValueError(f"{rtaxis_path.name} must have columns: scan, rt")

    axis_df = axis_df.sort_values("scan")
    scans = axis_df["scan"].to_numpy(dtype=int)
    rts = axis_df["rt"].to_numpy(dtype=float)
    return scans, rts


def reconstruct_intensity_profile_from_csv(
        csv_path: str,
        mass_list: List[float]
) -> Tuple[np.ndarray, Dict[float, np.ndarray]]:
    """
    Reconstruct intensity profiles for each mass from the filtered CSV.

    IMPORTANT:
    - If {base}_rtaxis.csv exists (scan, rt), we use the FULL scan axis (closest to original mzXML behavior).
    - If the peaks CSV also has a 'scan' column, mapping is exact.
    - If not, we fall back to RT-nearest mapping.
    - If rtaxis is missing, we fall back to old unique-rt behavior (not identical to original).
    """
    peaks_csv_path = Path(csv_path)
    df = pd.read_csv(peaks_csv_path)

    if len(df) == 0:
        return np.array([]), {}

    # Try full-axis reconstruction
    try:
        scans_axis, rt_axis = load_rt_axis_for_file(peaks_csv_path)
        scan_to_idx = {int(s): i for i, s in enumerate(scans_axis)}
        intensity_by_mass = {mass: np.zeros(len(rt_axis), dtype=float) for mass in mass_list}

        has_scan = "scan" in df.columns
        if has_scan:
            for _, row in df.iterrows():
                scan = int(row["scan"])
                mass = float(row["mass"])
                intensity = float(row["intensity"])
                i = scan_to_idx.get(scan, None)
                if i is None:
                    continue
                for target_mass in mass_list:
                    if abs(mass - target_mass) <= Config.MASS_TOLERANCE:
                        intensity_by_mass[target_mass][i] += intensity
                        break
        else:
            # RT-nearest fallback on full axis (less exact than scan mapping)
            rt_axis_arr = np.asarray(rt_axis, dtype=float)
            for _, row in df.iterrows():
                rt = float(row["rt"])
                mass = float(row["mass"])
                intensity = float(row["intensity"])
                i = int(np.argmin(np.abs(rt_axis_arr - rt)))
                for target_mass in mass_list:
                    if abs(mass - target_mass) <= Config.MASS_TOLERANCE:
                        intensity_by_mass[target_mass][i] += intensity
                        break

        return rt_axis, intensity_by_mass

    except Exception:
        # Fall back to old behavior (not identical to original)
        unique_rts = np.sort(df['rt'].unique())
        intensity_by_mass = {mass: np.zeros(len(unique_rts), dtype=float) for mass in mass_list}
        rt_to_idx = {rt: idx for idx, rt in enumerate(unique_rts)}

        for _, row in df.iterrows():
            rt = row['rt']
            mass = row['mass']
            intensity = row['intensity']

            for target_mass in mass_list:
                if abs(mass - target_mass) <= Config.MASS_TOLERANCE:
                    rt_idx = rt_to_idx[rt]
                    intensity_by_mass[target_mass][rt_idx] += float(intensity)
                    break

        return unique_rts, intensity_by_mass


# ----------------------------
# Shoulder splitting (kept from your pipeline)
# ----------------------------
def split_shoulders_in_window(
        rt_vals: np.ndarray,
        intensity_raw: np.ndarray,
        intensity_smooth: np.ndarray,
        left_idx: int,
        right_idx: int,
        noise_level: float,
        mass_str: str,
        base_name: str,
        scan_start: int,
        scan_end: int,
) -> List[Dict]:
    """
    Detect and split shoulder peaks within a peak window.
    """
    x_win = rt_vals[left_idx:right_idx + 1]
    y_raw_win = intensity_raw[left_idx:right_idx + 1]
    y_smooth_win = intensity_smooth[left_idx:right_idx + 1]

    def _record(apex_idx: int, area: float) -> Dict:
        return {
            'File Name': base_name,
            'm/z': mass_str,
            'RT_start': round(float(x_win[0]), 4),
            'RT_apex': round(float(rt_vals[apex_idx]), 4),
            'RT_end': round(float(x_win[-1]), 4),
            'Scan_start': int(scan_start),
            'Scan_end': int(scan_end),
            'Peak Area': round(float(area), 2),
            'Height': round(float(intensity_raw[apex_idx]), 2),
        }

    if len(y_smooth_win) < 7:
        apex_local = int(np.argmax(y_raw_win))
        apex_idx = left_idx + apex_local
        area = trapezoid(y_raw_win, x_win)
        return [_record(apex_idx, area)]

    main_local = int(np.argmax(y_smooth_win))
    main_idx = left_idx + main_local
    main_height = float(intensity_raw[main_idx])

    local_maxima = argrelextrema(y_smooth_win, np.greater, order=SHOULDER_LOCALMAX_ORDER)[0]

    candidates = []
    for lm in local_maxima:
        abs_idx = left_idx + int(lm)
        if abs_idx == main_idx:
            continue
        if abs(float(rt_vals[abs_idx]) - float(rt_vals[main_idx])) < SHOULDER_MIN_SEP_MIN:
            continue
        cand_height = float(intensity_raw[abs_idx])
        if cand_height < (SHOULDER_MIN_HEIGHT_FRAC * main_height):
            continue
        if cand_height < (SHOULDER_MIN_NOISE_MULT * float(noise_level)):
            continue
        candidates.append(abs_idx)

    if not candidates:
        area = trapezoid(y_raw_win, x_win)
        return [_record(main_idx, area)]

    best = None
    best_height = -1.0
    best_valley_idx = None

    for cand_idx in candidates:
        lo = min(cand_idx, main_idx)
        hi = max(cand_idx, main_idx)
        if hi - lo < 3:
            continue

        seg = intensity_smooth[lo:hi + 1]
        valley_ofs = int(np.argmin(seg))
        valley_idx = lo + valley_ofs

        cand_h = float(intensity_raw[cand_idx])
        main_h = float(intensity_raw[main_idx])
        valley_h = float(intensity_raw[valley_idx])

        smaller = min(cand_h, main_h)
        if smaller <= 0:
            continue
        if valley_h > (SHOULDER_VALLEY_DROP_FRAC * smaller):
            continue

        if cand_h > best_height:
            best = cand_idx
            best_height = cand_h
            best_valley_idx = valley_idx

    if best is None or best_valley_idx is None:
        area = trapezoid(y_raw_win, x_win)
        return [_record(main_idx, area)]

    valley_idx = best_valley_idx
    left_l, left_r = left_idx, valley_idx
    right_l, right_r = valley_idx, right_idx

    left_s = intensity_smooth[left_l:left_r + 1]
    right_s = intensity_smooth[right_l:right_r + 1]

    left_apex = left_l + int(np.argmax(left_s))
    right_apex = right_l + int(np.argmax(right_s))

    x_left = rt_vals[left_l:left_r + 1]
    y_left = intensity_raw[left_l:left_r + 1]
    area_left = trapezoid(y_left, x_left)

    x_right = rt_vals[right_l:right_r + 1]
    y_right = intensity_raw[right_l:right_r + 1]
    area_right = trapezoid(y_right, x_right)

    if float(intensity_raw[left_apex]) >= float(intensity_raw[right_apex]):
        main_apex_idx = left_apex
        shoulder_apex_idx = right_apex
        main_area = area_left
        shoulder_area = area_right
    else:
        main_apex_idx = right_apex
        shoulder_apex_idx = left_apex
        main_area = area_right
        shoulder_area = area_left

    return [
        _record(shoulder_apex_idx, shoulder_area),
        _record(main_apex_idx, main_area),
    ]


# ----------------------------
# Pixel conversion helper (unchanged)
# ----------------------------
def rt_to_pixel_x(rt: float, x_min: float, x_max: float, img_width: int) -> int:
    if img_width <= 1 or x_max <= x_min:
        return 0
    frac = (rt - x_min) / (x_max - x_min)
    px = int(round(frac * (img_width - 1)))
    return max(0, min(img_width - 1, px))


def write_peakinfo_csv(peaks_out: List[Dict], out_csv_path: Path) -> None:
    if not peaks_out:
        return

    df = pd.DataFrame(peaks_out)

    cols = [
        "File Name", "m/z",
        "RT_start", "RT_apex", "RT_end",
        "Scan_start", "Scan_end",
        "Peak Area", "Height",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[cols]
    df["RT_apex"] = pd.to_numeric(df["RT_apex"], errors="coerce")
    df["Peak Area"] = pd.to_numeric(df["Peak Area"], errors="coerce")
    df = df.sort_values(["m/z", "RT_apex", "Peak Area"], ascending=[True, True, False])

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)


def generate_mass_colors(mass_list: List[float]) -> Dict[str, str]:
    """
    Generate distinct DARK colors per m/z (repeatable).
    Uses HSV with low value to keep traces dark.
    MUST MATCH PixelMapping.py if you use image-only mapping.
    """
    import colorsys
    mass_strs = [f"{m:.4f}" for m in mass_list]
    n = max(1, len(mass_strs))
    colors = {}
    for i, mz_str in enumerate(mass_strs):
        hue = i / n
        saturation = 0.75
        value = 0.40
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors[mz_str] = '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
    return colors


# ----------------------------
# MAIN: analyze + render (MATCH ORIGINAL PEAKING + ORIGINAL TAIL CROP)
# ----------------------------
def analyze_csv_and_generate_eic(
        csv_path: str,
        output_image_path: str,
        file_colors: Dict[str, str],
        group_name: str
) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns:
        (peaks_out, pixel_rows)
    """
    Config.set_mass_group(group_name)

    peaks_out: List[Dict] = []
    pixel_rows: List[Dict] = []

    csv_path = Path(csv_path) if isinstance(csv_path, str) else csv_path
    output_image_path = Path(output_image_path) if isinstance(output_image_path, str) else output_image_path

    base = csv_path.stem.replace(f"_peaks_{group_name}", "")

    if output_image_path.exists():
        return peaks_out, pixel_rows

    mass_list = list(Config.MASS_LIST)
    mass_list_str = [f"{m:.4f}" for m in mass_list]

    rt_vals, intensity_by_mass = reconstruct_intensity_profile_from_csv(str(csv_path), mass_list)
    if len(rt_vals) == 0:
        return peaks_out, pixel_rows

    fig = go.Figure()
    mass_colors = generate_mass_colors(mass_list)
    all_peak_rts: List[float] = []

    for mass_idx, specific_mass in enumerate(mass_list):
        intensity_vals = np.asarray(intensity_by_mass[specific_mass], dtype=float)
        if len(intensity_vals) < 3 or float(np.max(intensity_vals)) == 0.0:
            continue

        # Smooth (same structure as original)
        window_length = min(7, len(intensity_vals))
        if window_length % 2 == 0:
            window_length -= 1

        if window_length >= 3:
            try:
                intensity_vals_smooth = savgol_filter(intensity_vals, window_length=window_length, polyorder=2)
            except:
                intensity_vals_smooth = intensity_vals
        else:
            intensity_vals_smooth = intensity_vals

        noise_level = float(np.std(intensity_vals_smooth[:min(20, len(intensity_vals_smooth))]))
        max_intensity = float(np.max(intensity_vals_smooth))

        # MATCH ORIGINAL thresholds
        min_prom = max(max_intensity * 0.01, noise_level * 2.0)
        min_height = max(max_intensity * 0.04, noise_level * 3.0)

        try:
            # MATCH ORIGINAL find_peaks call
            peaks, _ = find_peaks(
                intensity_vals_smooth,
                prominence=min_prom,
                height=min_height,
                width=(None, None),
                threshold=noise_level * 1.0,
                distance=1
            )
            if len(peaks) == 0:
                continue

            # MATCH ORIGINAL prominence filtering
            prom_results = peak_prominences(intensity_vals_smooth, peaks)
            peak_heights = intensity_vals_smooth[peaks]
            valid_mask = (prom_results[0] >= min_prom) & (peak_heights >= min_height)
            peaks = peaks[valid_mask]
            if len(peaks) == 0:
                continue

            # MATCH ORIGINAL rel_height per mass
            rel_height = REL_HEIGHT_BY_MASS.get(round(float(specific_mass), 4), DEFAULT_REL_HEIGHT)
            width_results = peak_widths(intensity_vals_smooth, peaks, rel_height=rel_height)

            valid_width = width_results[0] > 0
            peaks = peaks[valid_width]
            width_results = (
                width_results[0][valid_width],
                width_results[1][valid_width],
                width_results[2][valid_width],
                width_results[3][valid_width]
            )
            if len(peaks) == 0:
                continue

            # Keep your overlap splitting
            try:
                peaks, width_results = improved_peak_cutting(
                    intensity_vals_smooth, rt_vals, peaks, width_results, specific_mass
                )
            except:
                pass

        except:
            continue

        # Render each peak window
        for i, idx in enumerate(peaks):
            try:
                left_ip, right_ip = width_results[2][i], width_results[3][i]
                left_idx = max(0, int(np.floor(left_ip)))
                right_idx = min(len(rt_vals) - 1, int(np.ceil(right_ip)))

                rt_start = float(rt_vals[left_idx])
                rt_end = float(rt_vals[right_idx])
                duration = rt_end - rt_start

                # MATCH ORIGINAL truncate if too long
                if duration > Config.MAX_PEAK_DURATION:
                    max_rt_end = rt_start + Config.MAX_PEAK_DURATION
                    valid_idxs = np.where(rt_vals <= max_rt_end)[0]
                    if valid_idxs.size:
                        trunc_idx = int(valid_idxs[valid_idxs >= left_idx].max())
                        right_idx = trunc_idx
                    else:
                        continue
                    rt_end = float(rt_vals[right_idx])
                    duration = rt_end - rt_start

                # MATCH ORIGINAL duration filter
                if not (0.05 <= duration <= 0.75):
                    continue

                x_peak = rt_vals[left_idx:right_idx + 1]
                y_peak = intensity_vals[left_idx:right_idx + 1]

                # ============================
                # EXACT ORIGINAL tail crop block
                # ============================
                if len(y_peak) >= 12:
                    apex_idx = int(np.argmax(y_peak))
                    peak_height = float(y_peak[apex_idx])
                    post_y = y_peak[apex_idx + 1:]
                    slope = np.abs(np.diff(post_y))

                    slope_thresh = 0.01 * peak_height  # flat-ish
                    height_thresh = 0.015 * peak_height  # must be low enough
                    stable_len = 5

                    for j in range(len(slope) - stable_len):
                        window = slope[j:j + stable_len]
                        if np.all(window < slope_thresh):
                            crop_candidate_idx = apex_idx + 1 + j

                            if float(y_peak[crop_candidate_idx]) < height_thresh:
                                buffer = int(0.01 * len(y_peak))  # 1% of points
                                safe_idx = max(apex_idx + 1, crop_candidate_idx - buffer)

                                x_peak = x_peak[:safe_idx]
                                y_peak = y_peak[:safe_idx]

                                # Keep indices consistent for CSVs
                                right_idx = left_idx + safe_idx - 1
                                rt_end = float(rt_vals[right_idx])
                                break

                # Your pipeline wants peak records + shoulder splitting
                new_records = split_shoulders_in_window(
                    rt_vals=rt_vals,
                    intensity_raw=intensity_vals,
                    intensity_smooth=intensity_vals_smooth,
                    left_idx=left_idx,
                    right_idx=right_idx,
                    noise_level=noise_level,
                    mass_str=mass_list_str[mass_idx],
                    base_name=base,
                    scan_start=left_idx,
                    scan_end=right_idx,
                )
                peaks_out.extend(new_records)

                mz_str = mass_list_str[mass_idx]
                mz_color = mass_colors.get(mz_str, "#ffffff")

                fig.add_trace(go.Scatter(
                    x=x_peak, y=y_peak,
                    mode='lines',
                    line=dict(color=mz_color, width=3),
                    showlegend=False
                ))

                all_peak_rts.extend(x_peak.tolist())

            except:
                continue

    if not fig.data or len(all_peak_rts) == 0:
        return peaks_out, pixel_rows

    x_min = min(all_peak_rts) - 0.1
    x_max = max(all_peak_rts) + 0.1

    fig.update_xaxes(range=[x_min, x_max], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_layout(
        width=1600, height=900,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    wrote_png = False
    try:
        pio.write_image(fig, str(output_image_path), format='png', engine='kaleido', scale=4)
        wrote_png = True
    except:
        try:
            raw_png = str(output_image_path).replace('.png', '_raw.png')
            pio.write_image(fig, raw_png, format='png', engine='kaleido', scale=4)
            with Image.open(raw_png) as img:
                img.save(str(output_image_path), optimize=True, compress_level=9)
            os.remove(raw_png)
            wrote_png = True
        except:
            wrote_png = False

    del fig

    # De-dup (your existing logic)
    if peaks_out:
        dfp = pd.DataFrame(peaks_out)
        if all(col in dfp.columns for col in ['m/z', 'RT_apex', 'Peak Area']):
            dfp['RT_apex'] = pd.to_numeric(dfp['RT_apex'], errors='coerce')
            dfp['Peak Area'] = pd.to_numeric(dfp['Peak Area'], errors='coerce')
            dfp = dfp.sort_values(['m/z', 'RT_apex', 'Peak Area'], ascending=[True, True, False])
            dfp = dfp.drop_duplicates(subset=['m/z', 'RT_apex'], keep='first')
            peaks_out = dfp.to_dict('records')

    # Build pixelRT mapping only if PNG exists
    if wrote_png and output_image_path.exists():
        try:
            with Image.open(output_image_path) as im:
                img_width, _ = im.size

            rows = []
            for rec in peaks_out:
                try:
                    rt_s = float(rec["RT_start"])
                    rt_e = float(rec["RT_end"])
                    px_s = rt_to_pixel_x(rt_s, x_min, x_max, img_width)
                    px_e = rt_to_pixel_x(rt_e, x_min, x_max, img_width)
                    if px_e < px_s:
                        px_s, px_e = px_e, px_s

                    rows.append({
                        "File Name": rec["File Name"],
                        "m/z": rec["m/z"],
                        "RT_start": rt_s,
                        "RT_end": rt_e,
                        "Pixel_start": int(px_s),
                        "Pixel_end": int(px_e),
                    })
                except:
                    continue
            pixel_rows = rows
        except:
            pixel_rows = []

    return peaks_out, pixel_rows


# ----------------------------
# Worker wrapper
# ----------------------------
def process_csv_for_eic(args: Tuple[str, str, Dict[str, str], str]) -> str:
    csv_path, output_image_path, file_colors, group_name = args

    try:
        csv_path = Path(csv_path) if isinstance(csv_path, str) else csv_path
        output_image_path = Path(output_image_path) if isinstance(output_image_path, str) else output_image_path

        base = csv_path.stem.replace(f"_peaks_{group_name}", "")

        peakinfo_dir = csv_path.parent / "Peak Info CSVs"
        peakinfo_dir.mkdir(parents=True, exist_ok=True)
        peakinfo_csv_path = peakinfo_dir / f"{base}_peakinfo_{group_name}.csv"

        pixel_dir = Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER / str(Config.CURRENT_GROUP) / "Pixel CSVs"
        pixel_dir.mkdir(parents=True, exist_ok=True)
        pixel_csv_path = pixel_dir / f"{base}_pixelRT_{group_name}.csv"

        if output_image_path.exists() and peakinfo_csv_path.exists() and pixel_csv_path.exists():
            return f"[↷] {base} (cached)"

        peaks, pixel_rows = analyze_csv_and_generate_eic(
            str(csv_path),
            str(output_image_path),
            file_colors,
            group_name
        )

        if pixel_rows:
            pd.DataFrame(pixel_rows).to_csv(pixel_csv_path, index=False)
        else:
            pd.DataFrame(columns=["File Name", "m/z", "RT_start", "RT_end", "Pixel_start", "Pixel_end"]).to_csv(
                pixel_csv_path, index=False
            )

        write_peakinfo_csv(peaks, peakinfo_csv_path)

        return f"[✔] {base} ({len(peaks)} peaks)"

    except Exception as e:
        return f"[!] {Path(csv_path).name}: {str(e)[:80]}"


def generate_file_colors(csv_files: List[Path]) -> Dict[str, str]:
    """Generate DARK colors for each file (not currently used for traces)"""
    file_colors = {}
    n_files = len(csv_files)
    group_name = Config.CURRENT_GROUP

    for i, csv_path in enumerate(csv_files):
        csv_path = Path(csv_path) if isinstance(csv_path, str) else csv_path
        base = csv_path.stem.replace(f"_peaks_{group_name}", "")

        hue = i / max(n_files, 1)
        saturation = 0.7
        value = 0.4
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        file_colors[base] = hex_color

    return file_colors


def build_eics_for_group(group_name: str, n_processes: int = None) -> List[str]:
    Config.set_mass_group(group_name)

    csv_dir = Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER / str(Config.CURRENT_GROUP) / "EIC CSVs"
    csv_files = list(csv_dir.glob(f"*_peaks_{group_name}.csv"))

    if not csv_files:
        print(f"[!] No CSV files found for {group_name} in {csv_dir}")
        return []

    file_colors = generate_file_colors(csv_files)

    output_dir = Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER / str(Config.CURRENT_GROUP) / "EIC PNGs"
    output_dir.mkdir(parents=True, exist_ok=True)

    args_list = []
    for csv_path in csv_files:
        base_name = csv_path.stem.replace(f"_peaks_{group_name}", "")
        output_image = output_dir / f"EIC_{base_name}_plotly.png"
        args_list.append((str(csv_path), str(output_image), file_colors, group_name))

    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)

    print(f"\n{'=' * 70}")
    print(f"Building EIC Images for {group_name}")
    print(f"{'=' * 70}")
    print(f"Processing {len(csv_files)} files using {n_processes} processes")
    print(f"Output directory: {output_dir}\n")

    import time
    start_time = time.time()

    with Pool(processes=n_processes, initializer=init_worker) as pool:
        results = pool.map(process_csv_for_eic, args_list)

    elapsed_time = time.time() - start_time

    print("\nEIC Generation Results:")
    for result in results:
        print(result)

    successful = sum(1 for r in results if r.startswith("[✔]"))
    cached = sum(1 for r in results if r.startswith("[↷]"))
    failed = sum(1 for r in results if r.startswith("[!]"))

    print(f"\nSummary: {successful} processed, {cached} cached, {failed} failed")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    if len(csv_files) > 0:
        print(f"Average time per file: {elapsed_time / len(csv_files):.2f} seconds")

    return results


def build_eics_for_all_groups(n_processes: int = None) -> Dict[str, List[str]]:
    import time

    total_start_time = time.time()
    all_results = {}

    print("\n" + "=" * 70)
    print("EIC IMAGE GENERATION - ALL GROUPS")
    print("=" * 70)

    for group_name in Config.MASS_GROUPS.keys():
        results = build_eics_for_group(group_name, n_processes)
        all_results[group_name] = results

    total_elapsed = time.time() - total_start_time

    print("\n" + "=" * 70)
    print("EIC GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_elapsed:.2f} seconds")

    return all_results


if __name__ == "__main__":
    build_eics_for_all_groups()
