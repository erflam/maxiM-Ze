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
from typing import List, Dict, Tuple
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, peak_widths, savgol_filter, argrelextrema
from scipy.ndimage import gaussian_filter1d
import plotly.io as pio
import plotly.graph_objects as go
from PIL import Image
from multiprocessing import Pool, cpu_count
import colorsys

from Config import Config
from FileUtils import FileUtils

# Peak detection parameters
DEFAULT_REL_HEIGHT = 0.98
MAX_PEAK_DURATION = 1.5

# Shoulder detection parameters
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


def improved_peak_cutting(intensity_vals_smooth, rt_vals, peaks, width_results, specific_mass):
    """
    Improved peak cutting algorithm to split overlapping peaks

    Args:
        intensity_vals_smooth: Smoothed intensity values
        rt_vals: Retention time values
        peaks: Initial peak indices
        width_results: Peak width results from scipy
        specific_mass: Mass being analyzed

    Returns:
        tuple: (new_peaks, new_width_results)
    """

    MIN_APEX_SEP_MIN = 0.015
    MAX_SEG_WIDTH_MIN = 0.8
    MIN_SEG_WIDTH_MIN = 0.015

    all_minima = set()
    for sigma in [0.5, 1.0, 2.0]:
        smoothed = gaussian_filter1d(intensity_vals_smooth, sigma=sigma)
        for order in [2, 3]:
            minima = argrelextrema(smoothed, np.less, order=order)[0]
            all_minima.update(minima)

    second_deriv = np.gradient(np.gradient(intensity_vals_smooth))
    inflection_valleys = np.where((second_deriv[:-2] < 0) & (second_deriv[2:] > 0))[0] + 1
    all_minima.update(inflection_valleys)

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
        """Calculate peak quality metrics for better selection"""
        if right_idx <= left_idx or apex_idx < left_idx or apex_idx > right_idx:
            return 0.0

        segment = intensity_vals_smooth[left_idx:right_idx + 1]
        apex_val = intensity_vals_smooth[apex_idx]

        left_min = np.min(segment[:apex_idx - left_idx + 1]) if apex_idx > left_idx else apex_val
        right_min = np.min(segment[apex_idx - left_idx:]) if apex_idx < right_idx else apex_val
        prominence = apex_val - max(left_min, right_min)

        left_wing = segment[:apex_idx - left_idx]
        right_wing = segment[apex_idx - left_idx + 1:]
        min_len = min(len(left_wing), len(right_wing))
        if min_len > 0:
            symmetry = np.corrcoef(left_wing[-min_len:], right_wing[:min_len][::-1])[0, 1]
            symmetry = max(0, symmetry)
        else:
            symmetry = 0

        return prominence * (1 + 0.3 * symmetry)

    rel_height_split = 0.85
    try:
        width_results_split = peak_widths(intensity_vals_smooth, peaks, rel_height=rel_height_split)
        valid_split = width_results_split[0] > 0
    except:
        valid_split = np.zeros(len(peaks), dtype=bool)
        width_results_split = (np.zeros(len(peaks)), np.zeros(len(peaks)),
                               np.zeros(len(peaks)), np.zeros(len(peaks)))

    new_peaks = []

    for i_pk, pk in enumerate(peaks):
        # Determine search window
        if valid_split[i_pk]:
            left_ip, right_ip = width_results_split[2][i_pk], width_results_split[3][i_pk]
        else:
            left_ip, right_ip = width_results[2][i_pk], width_results[3][i_pk]

        left_idx = max(0, int(np.floor(left_ip)))
        right_idx = min(len(intensity_vals_smooth) - 1, int(np.ceil(right_ip)))

        # Find valleys within the search window
        internal_valleys = [v for v in all_minima if left_idx < v < right_idx]

        if not internal_valleys:
            new_peaks.append(pk)
            continue

        significant_valleys = []
        peak_intensity = intensity_vals_smooth[pk]

        for valley in internal_valleys:
            valley_intensity = intensity_vals_smooth[valley]

            local_window = slice(max(0, valley - 3), min(len(intensity_vals_smooth), valley + 4))
            local_max = np.max(intensity_vals_smooth[local_window])

            left_peak_h = np.max(intensity_vals_smooth[left_idx:valley + 1])
            right_peak_h = np.max(intensity_vals_smooth[valley:right_idx + 1])
            ratio = min(left_peak_h, right_peak_h) / max(left_peak_h, right_peak_h)
            adaptive_threshold = 0.6 + (0.3 * ratio)

            if valley_intensity <= adaptive_threshold * local_max:
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
                candidate_peaks.append((apex_idx, quality, seg_l, seg_r))

        if len(candidate_peaks) >= 2:
            candidate_peaks.sort(key=lambda x: x[1], reverse=True)

            selected_peaks = [candidate_peaks[0]]
            for candidate in candidate_peaks[1:]:
                apex_idx, quality, seg_l, seg_r = candidate

                min_sep = min(abs(rt_vals[apex_idx] - rt_vals[selected[0]])
                              for selected in selected_peaks)

                if min_sep >= MIN_APEX_SEP_MIN:
                    selected_peaks.append(candidate)

            for apex_idx, _, _, _ in selected_peaks:
                new_peaks.append(apex_idx)

        elif len(candidate_peaks) == 1:
            new_peaks.append(candidate_peaks[0][0])
        else:
            deepest_valley = significant_valleys[np.argmin(intensity_vals_smooth[significant_valleys])]

            if deepest_valley - left_idx >= 3:
                left_segment = intensity_vals_smooth[left_idx:deepest_valley + 1]
                left_apex = left_idx + np.argmax(left_segment)
                if calculate_peak_quality(left_apex, left_idx, deepest_valley) > peak_intensity * 0.05:
                    new_peaks.append(left_apex)

            if right_idx - deepest_valley >= 3:
                right_segment = intensity_vals_smooth[deepest_valley:right_idx + 1]
                right_apex = deepest_valley + np.argmax(right_segment)
                if calculate_peak_quality(right_apex, deepest_valley, right_idx) > peak_intensity * 0.05:
                    new_peaks.append(right_apex)

            if not any(abs(rt_vals[p] - rt_vals[pk]) < MIN_APEX_SEP_MIN * 2 for p in new_peaks[-2:]):
                new_peaks.append(pk)

    if new_peaks:
        new_peaks = np.unique(np.array(new_peaks, dtype=int))

        try:
            new_width_results = peak_widths(intensity_vals_smooth, new_peaks, rel_height=0.99)

            valid_mask = new_width_results[0] > 0
            new_peaks = new_peaks[valid_mask]
            new_width_results = tuple(arr[valid_mask] for arr in new_width_results)

            return new_peaks, new_width_results

        except Exception as e:
            print(f"Error recalculating widths: {e}")
            return peaks, width_results

    return peaks, width_results


def reconstruct_intensity_profile_from_csv(csv_path: str, mass_list: List[float]) -> Tuple[
    np.ndarray, Dict[float, np.ndarray]]:
    """
    Reconstruct intensity profiles for each mass from the filtered CSV

    Args:
        csv_path: Path to the filtered peaks CSV file
        mass_list: List of target masses

    Returns:
        Tuple of (rt_values array, dict mapping mass to intensity array)
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        return np.array([]), {}

    # Get unique retention times and sort them
    unique_rts = np.sort(df['rt'].unique())

    # Initialize intensity arrays for each mass
    intensity_by_mass = {mass: np.zeros(len(unique_rts)) for mass in mass_list}

    # Create RT index mapping
    rt_to_idx = {rt: idx for idx, rt in enumerate(unique_rts)}

    # Fill in intensities
    for _, row in df.iterrows():
        rt = row['rt']
        mass = row['mass']
        intensity = row['intensity']

        # Find which target mass this belongs to
        for target_mass in mass_list:
            if abs(mass - target_mass) <= Config.MASS_TOLERANCE:
                rt_idx = rt_to_idx[rt]
                intensity_by_mass[target_mass][rt_idx] += intensity
                break

    return unique_rts, intensity_by_mass


def split_shoulders_in_window(
        rt_vals: np.ndarray,
        intensity_raw: np.ndarray,
        intensity_smooth: np.ndarray,
        left_idx: int,
        right_idx: int,
        noise_level: float,
        mass_str: str,
        base_name: str
) -> List[Dict]:
    """
    Detect and split shoulder peaks within a peak window

    Args:
        rt_vals: Retention time array
        intensity_raw: Raw intensity values
        intensity_smooth: Smoothed intensity values
        left_idx: Left boundary index
        right_idx: Right boundary index
        noise_level: Noise threshold
        mass_str: Mass as string
        base_name: File base name

    Returns:
        List of peak dictionaries
    """
    x_win = rt_vals[left_idx:right_idx + 1]
    y_raw_win = intensity_raw[left_idx:right_idx + 1]
    y_smooth_win = intensity_smooth[left_idx:right_idx + 1]

    # Handle small windows
    if len(y_smooth_win) < 7:
        apex_local = int(np.argmax(y_raw_win))
        apex_idx = left_idx + apex_local
        area = trapezoid(y_raw_win, x_win)
        return [{
            'File': base_name,
            'm/z': mass_str,
            'RT_start': round(float(x_win[0]), 4),
            'RT_apex': round(float(rt_vals[apex_idx]), 4),
            'RT_end': round(float(x_win[-1]), 4),
            'Peak Area': round(float(area), 2),
            'height': round(float(intensity_raw[apex_idx]), 2)
        }]

    # Find main peak
    main_local = int(np.argmax(y_smooth_win))
    main_idx = left_idx + main_local
    main_height = float(intensity_raw[main_idx])

    # Find local maxima (potential shoulders)
    local_maxima = argrelextrema(y_smooth_win, np.greater, order=SHOULDER_LOCALMAX_ORDER)[0]

    # Filter candidates
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

    # No valid shoulders found
    if not candidates:
        area = trapezoid(y_raw_win, x_win)
        return [{
            'File': base_name,
            'm/z': mass_str,
            'RT_start': round(float(x_win[0]), 4),
            'RT_apex': round(float(rt_vals[main_idx]), 4),
            'RT_end': round(float(x_win[-1]), 4),
            'Peak Area': round(float(area), 2),
            'height': round(float(intensity_raw[main_idx]), 2)
        }]

    # Find best shoulder candidate
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

    # No valid shoulder found after filtering
    if best is None or best_valley_idx is None:
        area = trapezoid(y_raw_win, x_win)
        return [{
            'File': base_name,
            'm/z': mass_str,
            'RT_start': round(float(x_win[0]), 4),
            'RT_apex': round(float(rt_vals[main_idx]), 4),
            'RT_end': round(float(x_win[-1]), 4),
            'Peak Area': round(float(area), 2),
            'height': round(float(intensity_raw[main_idx]), 2)
        }]

    # Split at valley
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

    # Determine main vs shoulder based on height
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
        {
            'File': base_name,
            'm/z': mass_str,
            'RT_start': round(float(rt_vals[left_idx]), 4),
            'RT_apex': round(float(rt_vals[shoulder_apex_idx]), 4),
            'RT_end': round(float(rt_vals[right_idx]), 4),
            'Peak Area': round(float(shoulder_area), 2),
            'height': round(float(intensity_raw[shoulder_apex_idx]), 2)
        },
        {
            'File': base_name,
            'm/z': mass_str,
            'RT_start': round(float(rt_vals[left_idx]), 4),
            'RT_apex': round(float(rt_vals[main_apex_idx]), 4),
            'RT_end': round(float(rt_vals[right_idx]), 4),
            'Peak Area': round(float(main_area), 2),
            'height': round(float(intensity_raw[main_apex_idx]), 2)
        }
    ]


def analyze_csv_and_generate_eic(
        csv_path: str,
        output_image_path: str,
        file_colors: Dict[str, str],
        group_name: str
) -> List[Dict]:
    """
    Analyze filtered CSV and generate EIC visualization with peak detection

    Args:
        csv_path: Path to filtered peaks CSV
        output_image_path: Path to save PNG
        file_colors: Dictionary mapping file names to colors
        group_name: Mass group name

    Returns:
        List of detected peak dictionaries
    """
    # Set current group
    Config.set_mass_group(group_name)

    # Convert to Path for easier manipulation
    csv_path = Path(csv_path) if isinstance(csv_path, str) else csv_path
    output_image_path = Path(output_image_path) if isinstance(output_image_path, str) else output_image_path

    # Get base name
    base = csv_path.stem.replace(f"_peaks_{group_name}", "")

    # Skip if output already exists
    if output_image_path.exists():
        return []

    mass_list = list(Config.MASS_LIST)
    mass_list_str = [f"{m:.4f}" for m in mass_list]

    # Reconstruct intensity profiles from CSV (pass as string)
    rt_values, intensity_by_mass = reconstruct_intensity_profile_from_csv(str(csv_path), mass_list)

    if len(rt_values) == 0:
        print(f"[!] No data in {base}")
        return []

    # Create plotly figure
    fig = go.Figure()
    color = file_colors.get(base, '#1f77b4')
    peaks_out = []
    all_peak_rts = []

    # Process each mass
    for mass_idx, specific_mass in enumerate(mass_list):
        intensity_vals = intensity_by_mass[specific_mass]

        if len(intensity_vals) < 3 or np.max(intensity_vals) == 0:
            continue

        rt_vals = rt_values

        # Smooth the data
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

        # Calculate noise level and thresholds
        noise_level = np.std(intensity_vals_smooth[:min(20, len(intensity_vals_smooth))])
        max_intensity = np.max(intensity_vals_smooth)
        min_height = max(max_intensity * 0.04, noise_level * 3)

        # Find peaks
        try:
            peaks, properties = find_peaks(
                intensity_vals_smooth,
                height=min_height,
                distance=1
            )

            if len(peaks) == 0:
                continue

            # Calculate peak widths using DEFAULT_REL_HEIGHT
            width_results = peak_widths(intensity_vals_smooth, peaks, rel_height=DEFAULT_REL_HEIGHT)

            # Filter valid widths
            valid_width_mask = width_results[0] > 0
            peaks = peaks[valid_width_mask]
            width_results = tuple(wr[valid_width_mask] for wr in width_results)

            if len(peaks) == 0:
                continue

            # Apply improved peak cutting
            try:
                peaks, width_results = improved_peak_cutting(
                    intensity_vals_smooth, rt_vals, peaks, width_results, specific_mass
                )
            except Exception as e:
                print(f"Peak cutting failed for mass {mass_list_str[mass_idx]}: {e}")

        except Exception as e:
            print(f"Error finding peaks for mass {mass_list_str[mass_idx]}: {str(e)}")
            continue

        # Process and output peaks
        for i, idx in enumerate(peaks):
            try:
                left_ip, right_ip = width_results[2][i], width_results[3][i]
                left_idx = max(0, int(np.floor(left_ip)))
                right_idx = min(len(rt_vals) - 1, int(np.ceil(right_ip)))

                rt_start = rt_vals[left_idx]
                rt_end = rt_vals[right_idx]
                duration = rt_end - rt_start

                # Check duration limits
                if duration > Config.MAX_PEAK_DURATION:
                    max_rt_end = rt_start + Config.MAX_PEAK_DURATION
                    valid_idxs = np.where(rt_vals <= max_rt_end)[0]
                    if valid_idxs.size:
                        trunc_idx = valid_idxs[valid_idxs >= left_idx].max()
                        right_idx = int(trunc_idx)
                    else:
                        continue
                    rt_end = rt_vals[right_idx]
                    duration = rt_end - rt_start

                if not (0.03 <= duration <= 0.75):
                    continue

                x_peak = rt_vals[left_idx:right_idx + 1]
                y_peak = intensity_vals[left_idx:right_idx + 1]

                # Crop trailing flat regions for long peaks
                if len(y_peak) >= 12:
                    apex_idx_local = np.argmax(y_peak)
                    peak_height = y_peak[apex_idx_local]
                    post_y = y_peak[apex_idx_local + 1:]
                    slope = np.abs(np.diff(post_y))

                    slope_thresh = 0.01 * peak_height
                    height_thresh = 0.015 * peak_height
                    stable_len = 5

                    for j in range(len(slope) - stable_len):
                        window = slope[j:j + stable_len]
                        if np.all(window < slope_thresh):
                            crop_candidate_idx = apex_idx_local + 1 + j
                            if y_peak[crop_candidate_idx] < height_thresh:
                                buffer = int(0.01 * len(y_peak))
                                safe_idx = max(apex_idx_local + 1, crop_candidate_idx - buffer)
                                x_peak = x_peak[:safe_idx]
                                y_peak = y_peak[:safe_idx]
                                right_idx = left_idx + safe_idx - 1
                                break

                # Split shoulders if present
                new_records = split_shoulders_in_window(
                    rt_vals=rt_vals,
                    intensity_raw=intensity_vals,
                    intensity_smooth=intensity_vals_smooth,
                    left_idx=left_idx,
                    right_idx=right_idx,
                    noise_level=noise_level,
                    mass_str=mass_list_str[mass_idx],
                    base_name=base
                )
                peaks_out.extend(new_records)

                # Add to plot
                fig.add_trace(go.Scatter(
                    x=x_peak, y=y_peak,
                    mode='lines',
                    line=dict(color=color, width=3),
                    showlegend=False
                ))
                all_peak_rts.extend(x_peak.tolist())

            except Exception as e:
                print(f"Error processing peak {i} for mass {mass_list_str[mass_idx]}: {str(e)}")
                continue

    # Check if we have any peaks to plot
    if not fig.data:
        print(f"[!] No peaks in {base}; skipping.")
        return []

    # Update figure layout
    fig.update_xaxes(
        range=[min(all_peak_rts) - 0.1, max(all_peak_rts) + 0.1],
        showgrid=False, zeroline=False, showticklabels=False
    )
    fig.update_yaxes(
        showgrid=False, zeroline=False, showticklabels=False
    )
    fig.update_layout(
        width=1600, height=900,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # Save figure
    try:
        pio.write_image(fig, str(output_image_path), format='png', engine='kaleido', scale=4)
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        try:
            raw_png = str(output_image_path).replace('.png', '_raw.png')
            pio.write_image(fig, raw_png, format='png', engine='kaleido', scale=4)
            with Image.open(raw_png) as img:
                img.save(str(output_image_path), optimize=True, compress_level=9)
            os.remove(raw_png)
        except Exception as e2:
            print(f"Fallback image saving also failed: {str(e2)}")

    del fig

    # Clean up duplicate peaks
    if peaks_out:
        df = pd.DataFrame(peaks_out)
        if all(col in df.columns for col in ['m/z', 'RT_apex', 'Peak Area']):
            df['RT_apex'] = pd.to_numeric(df['RT_apex'], errors='coerce')
            df['Peak Area'] = pd.to_numeric(df['Peak Area'], errors='coerce')
            df = df.sort_values(['m/z', 'RT_apex', 'Peak Area'], ascending=[True, True, False])
            df = df.drop_duplicates(subset=['m/z', 'RT_apex'], keep='first')
            peaks_out = df.to_dict('records')

    return peaks_out


def process_csv_for_eic(args: Tuple[str, str, Dict[str, str], str]) -> str:
    """
    Process a single CSV file to generate EIC (for parallel execution)

    Args:
        args: Tuple of (csv_path, output_image_path, file_colors, group_name)

    Returns:
        Status message
    """
    csv_path, output_image_path, file_colors, group_name = args

    try:
        # Convert to Path objects if they're strings
        csv_path = Path(csv_path) if isinstance(csv_path, str) else csv_path
        output_image_path = Path(output_image_path) if isinstance(output_image_path, str) else output_image_path

        base = csv_path.stem.replace(f"_peaks_{group_name}", "")

        # Check if already exists
        if output_image_path.exists():
            return f"[↷] {base} (cached)"

        # Generate EIC and detect peaks (pass as string)
        peaks = analyze_csv_and_generate_eic(str(csv_path), str(output_image_path), file_colors, group_name)

        return f"[✔] {base} ({len(peaks)} peaks)"

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"\nFull error for {csv_path}:\n{error_details}")
        return f"[!] {Path(csv_path).name}: {str(e)[:50]}"


def generate_file_colors(csv_files: List[Path]) -> Dict[str, str]:
    """
    Generate unique colors for each file

    Args:
        csv_files: List of CSV file paths

    Returns:
        Dictionary mapping base names to hex colors
    """
    file_colors = {}
    n_files = len(csv_files)

    # Get current group name to remove from filenames
    group_name = Config.CURRENT_GROUP

    for i, csv_path in enumerate(csv_files):
        # Ensure it's a Path object
        csv_path = Path(csv_path) if isinstance(csv_path, str) else csv_path
        base = csv_path.stem.replace(f"_peaks_{group_name}", "")
        hue = i / max(n_files, 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        file_colors[base] = hex_color

    return file_colors


def build_eics_for_group(group_name: str, n_processes: int = None) -> List[str]:
    """
    Build EIC PNG images for all files in a specific mass group

    Args:
        group_name: Mass group name (e.g., 'Group 1')
        n_processes: Number of parallel processes (default: CPU count - 1)

    Returns:
        List of status messages
    """
    # Set current group
    Config.set_mass_group(group_name)

    # Get CSV directory
    csv_dir = Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER / str(Config.CURRENT_GROUP) / "EIC CSVs"

    # Find all filtered CSV files for this group
    csv_files = list(csv_dir.glob(f"*_peaks_{group_name}.csv"))

    if not csv_files:
        print(f"[!] No CSV files found for {group_name} in {csv_dir}")
        return []

    # Generate colors
    file_colors = generate_file_colors(csv_files)

    # Create output directory
    output_dir = Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER / str(Config.CURRENT_GROUP) / "EIC PNGs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments
    args_list = []
    for csv_path in csv_files:
        base_name = csv_path.stem.replace(f"_peaks_{group_name}", "")
        output_image = output_dir / f"EIC_{base_name}_plotly.png"
        args_list.append((str(csv_path), str(output_image), file_colors, group_name))

    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)

    print(f"\n{'=' * 70}")
    print(f"Building EIC Images for {group_name}")
    print(f"{'=' * 70}")
    print(f"Processing {len(csv_files)} files using {n_processes} processes")
    print(f"CSV directory: {csv_dir}")
    print(f"Output directory: {output_dir}\n")

    import time
    start_time = time.time()

    # Process in parallel
    with Pool(processes=n_processes, initializer=init_worker) as pool:
        results = pool.map(process_csv_for_eic, args_list)

    elapsed_time = time.time() - start_time

    # Print results
    print("\nEIC Generation Results:")
    for result in results:
        print(result)

    # Summary
    successful = sum(1 for r in results if r.startswith("[✔]"))
    cached = sum(1 for r in results if r.startswith("[↷]"))
    failed = sum(1 for r in results if r.startswith("[!]"))

    print(f"\nSummary: {successful} processed, {cached} cached, {failed} failed")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    if len(csv_files) > 0:
        print(f"Average time per file: {elapsed_time / len(csv_files):.2f} seconds")

    return results


def build_eics_for_all_groups(n_processes: int = None) -> Dict[str, List[str]]:
    """
    Build EIC PNG images for all mass groups

    Args:
        n_processes: Number of parallel processes (default: CPU count - 1)

    Returns:
        Dictionary mapping group names to status results
    """
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
    # Build EICs for all groups
    build_eics_for_all_groups()