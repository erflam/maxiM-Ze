from FileReader import *
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, peak_widths, savgol_filter, argrelextrema
from scipy.ndimage import gaussian_filter1d
import plotly.io as pio
import plotly.graph_objects as go
from PIL import Image
import os
from Config import Config
import colorsys

REL_HEIGHT_BY_MASS = {104.1069: 0.985, 187.0964: 0.985, 119.0896: 0.98}
DEFAULT_REL_HEIGHT = 0.99
MIN_WINDOW_MINUTES = 6.0          # always show at least 6 minutes on x-axis
X_PADDING_MINUTES = 0.10          # little padding on each side

SHOULDER_MIN_HEIGHT_FRAC = 0.02
SHOULDER_MIN_NOISE_MULT = 3.0
SHOULDER_MIN_SEP_MIN = 0.06
SHOULDER_VALLEY_DROP_FRAC = 0.85
SHOULDER_LOCALMAX_ORDER = 2

DEBUG_PEAK_FILTER = False
EXPORT_DEBUG_CSV = False
BYPASS_CACHE_WHEN_DEBUG = False


def _dark_hex_palette(n: int):
    """Generate n distinct dark-ish hex colors."""
    if n <= 0:
        return []
    colors = []
    for i in range(n):
        h = (i / n) % 1.0
        s = 0.85
        v = 0.25  # darker
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


def improved_peak_cutting(intensity_vals_smooth, rt_vals, peaks, width_results, specific_mass):
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

        if min_len >= 2:
            a = left_wing[-min_len:]
            b = right_wing[:min_len][::-1]
            if np.std(a) == 0 or np.std(b) == 0:
                symmetry = 0
            else:
                symmetry = np.corrcoef(a, b)[0, 1]
                symmetry = max(0, symmetry)
        else:
            symmetry = 0

        return prominence * (1 + 0.3 * symmetry)

    rel_height_split = 0.85
    try:
        width_results_split = peak_widths(intensity_vals_smooth, peaks, rel_height=rel_height_split)
        valid_split = width_results_split[0] > 0
    except Exception:
        valid_split = np.zeros(len(peaks), dtype=bool)
        width_results_split = (
            np.zeros(len(peaks)),
            np.zeros(len(peaks)),
            np.zeros(len(peaks)),
            np.zeros(len(peaks)),
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
            apex_idx = seg_l + int(np.argmax(local_segment))
            quality = calculate_peak_quality(apex_idx, seg_l, seg_r)

            if quality > peak_intensity * 0.1:
                candidate_peaks.append((apex_idx, quality))

        if len(candidate_peaks) >= 2:
            candidate_peaks.sort(key=lambda x: x[1], reverse=True)
            selected = [candidate_peaks[0][0]]
            for apex_idx, q in candidate_peaks[1:]:
                if min(abs(rt_vals[apex_idx] - rt_vals[s]) for s in selected) >= MIN_APEX_SEP_MIN:
                    selected.append(apex_idx)
            for s in selected:
                new_peaks.append(s)
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
        except Exception as e:
            print(f"Error recalculating widths: {e}")
            return peaks, width_results

    return peaks, width_results


def looks_like_real_peak(y_raw_window: np.ndarray):
    """
    Returns:
      (is_peak, frac_up, frac_down, apex, left_end, right_end)

    This ALWAYS returns 6 values (so your CSV logging never breaks).
    """
    nan = float("nan")

    if y_raw_window is None:
        return True, nan, nan, nan, nan, nan

    if len(y_raw_window) < 7:
        y = np.asarray(y_raw_window, dtype=float)
        apex = float(np.max(y)) if len(y) else 0.0
        left_end = float(y[0]) if len(y) else 0.0
        right_end = float(y[-1]) if len(y) else 0.0
        return True, nan, nan, apex, left_end, right_end

    y = np.asarray(y_raw_window, dtype=float)
    apex_i = int(np.argmax(y))
    apex = float(y[apex_i])

    left_end = float(y[0])
    right_end = float(y[-1])

    if apex <= 0:
        return False, nan, nan, apex, left_end, right_end

    left = y[:apex_i + 1]
    right = y[apex_i:]

    if len(left) < 3 or len(right) < 3:
        return True, nan, nan, apex, left_end, right_end

    MIN_RISE_FRAC = 0.25
    MIN_FALL_FRAC = 0.25

    if (apex - left_end) / apex < MIN_RISE_FRAC:
        return False, nan, nan, apex, left_end, right_end
    if (apex - right_end) / apex < MIN_FALL_FRAC:
        return False, nan, nan, apex, left_end, right_end

    left_diff = np.diff(left)
    frac_up = float(np.mean(left_diff > 0)) if left_diff.size else 0.0

    right_diff = np.diff(right)
    frac_down = float(np.mean(right_diff < 0)) if right_diff.size else 0.0

    if DEBUG_PEAK_FILTER:
        print(
            f"[PeakCheck] apex={apex:.2f} "
            f"start={left_end:.2f} end={right_end:.2f} "
            f"frac_up={frac_up:.2f} frac_down={frac_down:.2f}"
        )

    MIN_FRAC_UP = 0.7
    MIN_FRAC_DOWN = 0.7

    is_peak = True
    if frac_up < MIN_FRAC_UP:
        is_peak = False
    if frac_down < MIN_FRAC_DOWN:
        is_peak = False

    return is_peak, frac_up, frac_down, apex, left_end, right_end


def analyze_ms_file_plotly(file_path, output_image_path, file_colors, axis_meta_csv=None):
    """Analyze MS file and generate plotly visualization."""

    group_tag = Config.CURRENT_GROUP.replace(" ", "")

    # --- CACHING ---
    use_cache = True
    if BYPASS_CACHE_WHEN_DEBUG and (DEBUG_PEAK_FILTER or EXPORT_DEBUG_CSV):
        use_cache = False

    if use_cache and os.path.exists(output_image_path):
        base = os.path.splitext(os.path.basename(file_path))[0]
        peaks_csv = os.path.join(
            os.path.dirname(os.path.dirname(output_image_path)),
            'csv',
            f"{base}_peaks_{group_tag}.csv"
        )
        if os.path.exists(peaks_csv):
            try:
                return pd.read_csv(peaks_csv).to_dict('records')
            except Exception:
                pass

    analyzer = MSFileAnalyzer(file_path)
    base = analyzer.base_name

    mass_list = list(Config.MASS_LIST)
    mass_list_str = [f"{m:.4f}" for m in mass_list]

    rt_values = []
    scan_numbers = []
    intensity_by_mass = {mass: [] for mass in mass_list}

    with analyzer.get_reader() as reader:
        for scan in reader:
            try:
                rt = analyzer.get_retention_time(scan)
                scan_num = scan.get('num', None)

                mzs = np.asarray(scan['m/z array'], dtype=np.float32)
                ints = np.asarray(scan['intensity array'], dtype=np.float32)

                for mass in mass_list:
                    mask = np.abs(np.round(mzs, 4) - mass) <= Config.MASS_TOLERANCE
                    intensity = np.sum(ints[mask]) if np.any(mask) else 0.0
                    intensity_by_mass[mass].append(intensity)

                rt_values.append(rt)
                scan_numbers.append(scan_num)

            except KeyError:
                continue
            except Exception as e:
                print(f"Error processing scan: {str(e)}")
                continue

    fig = go.Figure()

    # Keep different color for each mass
    mass_colors = _dark_hex_palette(len(mass_list))
    mass_color_map = {mass_list[i]: mass_colors[i] for i in range(len(mass_list))}

    peaks_out = []          # peaks that pass the noise filter — written to peaks_{group}.csv
    peaks_prefilter = []    # ALL detected peaks before noise filter — written to peaks_prefilter_{group}.csv
    debug_rows = []
    all_peak_rts = []

    rt_vals = np.array(rt_values)
    scan_numbers_arr = np.array(scan_numbers, dtype=object)

    # Noise threshold from GUI
    gui_noise_level = Config.GROUP_NOISE_LEVEL

    def split_shoulders_in_window(
        rt_vals, scan_numbers_arr, intensity_raw, intensity_smooth,
        left_idx, right_idx, noise_level, mass_str, base_name
    ):
        x_win = rt_vals[left_idx:right_idx + 1]
        y_raw_win = intensity_raw[left_idx:right_idx + 1]
        y_smooth_win = intensity_smooth[left_idx:right_idx + 1]

        scan_start = scan_numbers_arr[left_idx]
        scan_end = scan_numbers_arr[right_idx]

        if len(y_smooth_win) < 7:
            apex_local = int(np.argmax(y_raw_win))
            apex_idx = left_idx + apex_local
            area = trapezoid(y_raw_win, x_win)
            return [{
                'File': base_name, 'm/z': mass_str,
                'RT_start': round(float(x_win[0]), 4),
                'RT_apex': round(float(rt_vals[apex_idx]), 4),
                'RT_end': round(float(x_win[-1]), 4),
                'scan_start': scan_start,
                'scan_end': scan_end,
                'Peak Area': round(float(area), 2),
                'height': round(float(intensity_raw[apex_idx]), 2)
            }]

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
            return [{
                'File': base_name, 'm/z': mass_str,
                'RT_start': round(float(x_win[0]), 4),
                'RT_apex': round(float(rt_vals[main_idx]), 4),
                'RT_end': round(float(x_win[-1]), 4),
                'scan_start': scan_start,
                'scan_end': scan_end,
                'Peak Area': round(float(area), 2),
                'height': round(float(intensity_raw[main_idx]), 2)
            }]

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
            return [{
                'File': base_name, 'm/z': mass_str,
                'RT_start': round(float(x_win[0]), 4),
                'RT_apex': round(float(rt_vals[main_idx]), 4),
                'RT_end': round(float(x_win[-1]), 4),
                'scan_start': scan_start,
                'scan_end': scan_end,
                'Peak Area': round(float(area), 2),
                'height': round(float(intensity_raw[main_idx]), 2)
            }]

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
            {
                'File': base_name, 'm/z': mass_str,
                'RT_start': round(float(rt_vals[left_idx]), 4),
                'RT_apex': round(float(rt_vals[shoulder_apex_idx]), 4),
                'RT_end': round(float(rt_vals[right_idx]), 4),
                'scan_start': scan_start,
                'scan_end': scan_end,
                'Peak Area': round(float(shoulder_area), 2),
                'height': round(float(intensity_raw[shoulder_apex_idx]), 2)
            },
            {
                'File': base_name, 'm/z': mass_str,
                'RT_start': round(float(rt_vals[left_idx]), 4),
                'RT_apex': round(float(rt_vals[main_apex_idx]), 4),
                'RT_end': round(float(rt_vals[right_idx]), 4),
                'scan_start': scan_start,
                'scan_end': scan_end,
                'Peak Area': round(float(main_area), 2),
                'height': round(float(intensity_raw[main_apex_idx]), 2)
            }
        ]

    for mass_idx, specific_mass in enumerate(mass_list):
        intensity_vals = np.array(intensity_by_mass[specific_mass])
        if len(intensity_vals) < 3 or np.max(intensity_vals) == 0:
            continue

        window_length = min(7, len(intensity_vals))
        if window_length % 2 == 0:
            window_length -= 1
        if window_length >= 3:
            try:
                intensity_vals_smooth = savgol_filter(intensity_vals, window_length=window_length, polyorder=2)
            except Exception:
                intensity_vals_smooth = intensity_vals
        else:
            intensity_vals_smooth = intensity_vals

        noise_level = np.std(intensity_vals_smooth[:min(20, len(intensity_vals_smooth))])
        max_intensity = np.max(intensity_vals_smooth)
        min_height = max(max_intensity * 0.04, noise_level * 3)

        try:
            peaks, properties = find_peaks(intensity_vals_smooth, height=min_height, distance=1)
            if len(peaks) == 0:
                continue

            rel_height = REL_HEIGHT_BY_MASS.get(round(float(specific_mass), 4), DEFAULT_REL_HEIGHT)
            width_results = peak_widths(intensity_vals_smooth, peaks, rel_height=rel_height)

            valid_width_mask = width_results[0] > 0
            peaks = peaks[valid_width_mask]
            width_results = (
                width_results[0][valid_width_mask],
                width_results[1][valid_width_mask],
                width_results[2][valid_width_mask],
                width_results[3][valid_width_mask],
            )

            if len(peaks) == 0:
                continue

            try:
                peaks, width_results = improved_peak_cutting(
                    intensity_vals_smooth, rt_vals, peaks, width_results, specific_mass
                )
            except Exception as e:
                print(f"Peak cutting failed for mass {mass_list_str[mass_idx]}: {e}")

        except Exception as e:
            print(f"Error finding peaks for mass {mass_list_str[mass_idx]}: {str(e)}")
            continue

        for i, idx in enumerate(peaks):
            try:
                left_ip, right_ip = width_results[2][i], width_results[3][i]
                left_idx = max(0, int(np.floor(left_ip)))
                right_idx = min(len(rt_vals) - 1, int(np.ceil(right_ip)))

                rt_start = rt_vals[left_idx]
                rt_end = rt_vals[right_idx]
                duration = rt_end - rt_start

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

                is_peak, frac_up, frac_down, apex, left_end, right_end = looks_like_real_peak(y_peak)

                debug_rows.append({
                    "File": base,
                    "m/z": float(specific_mass),
                    "scan_start": scan_numbers_arr[left_idx],
                    "scan_end": scan_numbers_arr[right_idx],
                    "RT_start": float(rt_vals[left_idx]),
                    "RT_end": float(rt_vals[right_idx]),
                    "apex_intensity": float(apex) if np.isfinite(apex) else np.nan,
                    "start_intensity": float(left_end) if np.isfinite(left_end) else np.nan,
                    "end_intensity": float(right_end) if np.isfinite(right_end) else np.nan,
                    "frac_up": float(frac_up) if np.isfinite(frac_up) else np.nan,
                    "frac_down": float(frac_down) if np.isfinite(frac_down) else np.nan,
                    "passed_filter": bool(is_peak),
                })

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

                new_records = split_shoulders_in_window(
                    rt_vals=rt_vals,
                    scan_numbers_arr=scan_numbers_arr,
                    intensity_raw=intensity_vals,
                    intensity_smooth=intensity_vals_smooth,
                    left_idx=left_idx,
                    right_idx=right_idx,
                    noise_level=noise_level,
                    mass_str=mass_list_str[mass_idx],
                    base_name=base
                )

                # ── Always add to prefilter list ──────────────────────
                peaks_prefilter.extend(new_records)

                # ── Only plot and keep peaks above the GUI noise level ─
                passing_records = [r for r in new_records if r.get('height', 0) >= gui_noise_level]
                if passing_records:
                    peaks_out.extend(passing_records)
                    fig.add_trace(go.Scatter(
                        x=x_peak, y=y_peak,
                        mode='lines',
                        line=dict(color=mass_color_map[specific_mass], width=3),
                        showlegend=False
                    ))
                    all_peak_rts.extend(x_peak.tolist())

            except Exception as e:
                print(f"Error processing peak {i} for mass {mass_list_str[mass_idx]}: {str(e)}")
                continue

    if not fig.data:
        print(f"[!] No peaks above noise ({gui_noise_level:.0f}) in {base}; skipping.")
        if EXPORT_DEBUG_CSV:
            debug_csv = os.path.join(
                os.path.dirname(os.path.dirname(output_image_path)),
                'csv',
                f"{base}_peak_debug_{group_tag}.csv"
            )
            os.makedirs(os.path.dirname(debug_csv), exist_ok=True)
            pd.DataFrame(debug_rows).to_csv(debug_csv, index=False)
            print(f"[DEBUG] wrote {len(debug_rows)} rows to {debug_csv}")
        return peaks_out, peaks_prefilter

    # --- FORCE SAME X-AXIS SIZE (>= 6 minutes) FOR EVERY IMAGE ---
    min_rt = float(np.min(rt_vals))
    max_rt = float(np.max(rt_vals))

    if all_peak_rts:
        peaks_min = float(np.min(all_peak_rts))
        peaks_max = float(np.max(all_peak_rts))
    else:
        peaks_min = min_rt
        peaks_max = max_rt

    center = 0.5 * (peaks_min + peaks_max)
    half_width = 0.5 * max(MIN_WINDOW_MINUTES, (peaks_max - peaks_min))

    x0 = center - half_width - X_PADDING_MINUTES
    x1 = center + half_width + X_PADDING_MINUTES

    x0 = max(x0, min_rt)
    x1 = min(x1, max_rt)

    if (x1 - x0) < MIN_WINDOW_MINUTES:
        needed = MIN_WINDOW_MINUTES - (x1 - x0)
        add_right = min(needed, max_rt - x1)
        x1 += add_right
        needed -= add_right
        add_left = min(needed, x0 - min_rt)
        x0 -= add_left

    if axis_meta_csv is not None:
        os.makedirs(os.path.dirname(axis_meta_csv), exist_ok=True)
        pd.DataFrame([{
            "x0": float(x0),
            "x1": float(x1),
            "png_width": 1600,
            "png_height": 900,
        }]).to_csv(axis_meta_csv, index=False)

    fig.update_xaxes(range=[x0, x1], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_layout(
        width=1600, height=900,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    try:
        pio.write_image(fig, output_image_path, format='png', engine='kaleido', scale=4)
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        try:
            raw_png = output_image_path.replace('.png', '_raw.png')
            pio.write_image(fig, raw_png, format='png', engine='kaleido', scale=4)
            with Image.open(raw_png) as img:
                img.save(output_image_path, optimize=True, compress_level=9)
            os.remove(raw_png)
        except Exception as e2:
            print(f"Fallback image saving also failed: {str(e2)}")

    del fig

    if peaks_out:
        df = pd.DataFrame(peaks_out)
        if ('m/z' in df.columns) and ('RT_apex' in df.columns) and ('Peak Area' in df.columns):
            df['RT_apex'] = pd.to_numeric(df['RT_apex'], errors='coerce')
            df['Peak Area'] = pd.to_numeric(df['Peak Area'], errors='coerce')
            df = df.sort_values(['m/z', 'RT_apex', 'Peak Area'], ascending=[True, True, False])
            df = df.drop_duplicates(subset=['m/z', 'RT_apex'], keep='first')
            peaks_out = df.to_dict('records')

    if peaks_prefilter:
        df_pre = pd.DataFrame(peaks_prefilter)
        if ('m/z' in df_pre.columns) and ('RT_apex' in df_pre.columns) and ('Peak Area' in df_pre.columns):
            df_pre['RT_apex'] = pd.to_numeric(df_pre['RT_apex'], errors='coerce')
            df_pre['Peak Area'] = pd.to_numeric(df_pre['Peak Area'], errors='coerce')
            df_pre = df_pre.sort_values(['m/z', 'RT_apex', 'Peak Area'], ascending=[True, True, False])
            df_pre = df_pre.drop_duplicates(subset=['m/z', 'RT_apex'], keep='first')
            peaks_prefilter = df_pre.to_dict('records')

    if EXPORT_DEBUG_CSV:
        debug_csv = os.path.join(
            os.path.dirname(os.path.dirname(output_image_path)),
            'csv',
            f"{base}_peak_debug_{group_tag}.csv"
        )
        os.makedirs(os.path.dirname(debug_csv), exist_ok=True)
        pd.DataFrame(debug_rows).to_csv(debug_csv, index=False)
        print(f"[DEBUG] wrote {len(debug_rows)} rows to {debug_csv}")

    return peaks_out, peaks_prefilter


def process_file_checkpoint2(fp, dirs, file_colors, group_name):
    """Checkpoint 2: Build EIC PNG + peaks CSV (group-specific filename)."""
    Config.set_mass_group(group_name)
    try:
        base = os.path.splitext(os.path.basename(fp))[0]
        group_tag = Config.CURRENT_GROUP.replace(" ", "")

        png_path = os.path.join(dirs['png'], f"EIC_{base}_{group_tag}.png")
        peaks_csv = os.path.join(dirs['csv'], f"{base}_peaks_{group_tag}.csv")
        peaks_prefilter_csv = os.path.join(dirs['csv'], f"{base}_peaks_prefilter_{group_tag}.csv")

        if os.path.exists(png_path) and os.path.exists(peaks_csv):
            return f"[↷] {base} (png+peaks cached)"

        axis_meta_csv = os.path.join(dirs['csv'], f"{base}_axis_{group_tag}.csv")
        peaks, peaks_prefilter = analyze_ms_file_plotly(fp, png_path, file_colors, axis_meta_csv=axis_meta_csv)

        # Save prefilter CSV (all detected peaks, no noise filter applied)
        if peaks_prefilter:
            pd.DataFrame(peaks_prefilter).to_csv(peaks_prefilter_csv, index=False, float_format='%.3f')

        # Save filtered CSV (only peaks with height >= GUI noise level)
        if peaks:
            pd.DataFrame(peaks).to_csv(peaks_csv, index=False, float_format='%.3f')

        n_total = len(peaks_prefilter)
        n_kept = len(peaks)
        n_dropped = n_total - n_kept
        return f"[✔] {base} (png+peaks) — {n_kept} peaks kept, {n_dropped} below noise ({Config.GROUP_NOISE_LEVEL:.0f})"

    except Exception as e:
        return f"[!] Error: {os.path.basename(fp)}: {str(e)[:50]}"