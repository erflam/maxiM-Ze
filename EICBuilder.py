"""
EICBuilder.py
Builds PNG images of Extracted Ion Chromatograms (EICs) for specific masses
with peak detection and debug visualizations.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, peak_widths, savgol_filter, argrelextrema
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing import Pool, cpu_count

from Config import Config
from FileUtils import FileUtils
from MSFileAnalyzer import MSFileAnalyzer
from improved_peak_cutting import improved_peak_cutting


# Peak detection parameters
REL_HEIGHT_BY_MASS = {104.1069: 0.985, 187.0964: 0.985, 119.0896: 0.98}
DEFAULT_REL_HEIGHT = 0.99

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


def generate_debug_plot(
        base_name: str,
        all_mass_debug_data: List[Dict],
        output_image_path: str,
        debug_dir: Path
) -> None:
    """
    Generate debug plot showing peak detection for all masses combined
    
    Args:
        base_name: File base name
        all_mass_debug_data: Debug data for all masses
        output_image_path: Path to main EIC image
        debug_dir: Directory for debug plots
    """
    if not all_mass_debug_data or not any(
            len(d['detected_peaks']) > 0 or len(d['rejected_peaks']) > 0 
            for d in all_mass_debug_data):
        return

    fig_debug, axes = plt.subplots(2, 1, figsize=(20, 12), sharex=False)
    fig_debug.suptitle(f'{base_name} - Peak Detection Debug', fontsize=16, fontweight='bold', y=0.995)

    # Panel 0: EIC PNG Image at the top
    axes[0].axis('off')
    if os.path.exists(output_image_path):
        try:
            img = plt.imread(output_image_path)
            axes[0].imshow(img, aspect='auto')
        except Exception as e:
            axes[0].text(0.5, 0.5, f'Could not load EIC image',
                         ha='center', va='center', transform=axes[0].transAxes)
    else:
        axes[0].text(0.5, 0.5, 'EIC image not yet generated',
                     ha='center', va='center', transform=axes[0].transAxes)

    # Panel 1: Combined trace with all masses
    ax = axes[1]

    # Combine all masses into single trace
    all_rt = all_mass_debug_data[0]['rt']
    combined_raw = np.zeros_like(all_rt, dtype=float)
    combined_smooth = np.zeros_like(all_rt, dtype=float)

    for debug_data in all_mass_debug_data:
        combined_raw += debug_data['raw']
        combined_smooth += debug_data['smooth']

    # Normalize to 0-1
    max_val = max(combined_raw.max(), combined_smooth.max())
    if max_val > 0:
        raw_norm = combined_raw / max_val
        smooth_norm = combined_smooth / max_val
    else:
        raw_norm = combined_raw
        smooth_norm = combined_smooth

    # Plot combined profiles
    ax.plot(all_rt, raw_norm, color='lightblue', linewidth=1.5, label='Raw profile', alpha=0.7)
    ax.plot(all_rt, smooth_norm, color='blue', linewidth=2, label='Smoothed profile')

    # Collect all peaks from all masses
    all_detected = []
    all_rejected = []
    all_final = []

    for debug_data in all_mass_debug_data:
        all_detected.extend(debug_data['detected_peaks'])
        all_rejected.extend(debug_data['rejected_peaks'])
        all_final.extend(debug_data['final_peaks'])

    # Green shaded regions for detected peaks
    for peak in all_detected:
        ax.axvspan(all_rt[peak['left']], all_rt[peak['right']], alpha=0.15, color='green')

    # Red dots at detected peak apexes
    for i, peak in enumerate(all_detected):
        peak_height_norm = smooth_norm[peak['idx']]
        ax.plot(all_rt[peak['idx']], peak_height_norm, 'ro', markersize=10,
                markeredgecolor='darkred', markeredgewidth=1.5,
                label='Detected peaks' if i == 0 else '')

    # Red shaded regions for rejected peaks with reasons
    for peak in all_rejected:
        ax.axvspan(all_rt[peak['left']], all_rt[peak['right']], alpha=0.15, color='red')
        mid_rt = (all_rt[peak['left']] + all_rt[peak['right']]) / 2
        peak_height_norm = smooth_norm[peak['idx']]
        ax.text(mid_rt, peak_height_norm * 1.15, peak['reason'],
                fontsize=7, ha='center', color='red', rotation=0,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))

    # Dark green fill for final kept peaks
    for peak in all_final:
        x_region = all_rt[peak['left']:peak['right'] + 1]
        y_region = raw_norm[peak['left']:peak['right'] + 1]
        ax.fill_between(x_region, 0, y_region, alpha=0.3, color='darkgreen')

    ax.set_ylabel('Intensity (normalized)', fontsize=11)
    ax.set_xlabel('Retention Time (min)', fontsize=11)
    ax.set_title(f'Intensity Profile Analysis - All Masses Combined | '
                 f'Detected: {len(all_detected)}, '
                 f'Rejected: {len(all_rejected)}, '
                 f'Kept: {len(all_final)}',
                 fontsize=12, pad=10)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Remove duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    debug_path = debug_dir / f'{base_name}_debug.png'
    plt.savefig(debug_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[DEBUG] Saved: {debug_path}")


def analyze_ms_file_and_generate_eic(
        file_path: str,
        output_image_path: str,
        file_colors: Dict[str, str],
        group_name: str
) -> List[Dict]:
    """
    Analyze MS file and generate EIC visualization with peak detection
    
    Args:
        file_path: Path to mzXML file
        output_image_path: Path to save PNG
        file_colors: Dictionary mapping file names to colors
        group_name: Mass group name
        
    Returns:
        List of detected peak dictionaries
    """
    # Set current group
    Config.set_mass_group(group_name)
    
    # Skip if output already exists
    base = os.path.splitext(os.path.basename(file_path))[0]
    if os.path.exists(output_image_path):
        # Try to load existing peaks
        peaks_csv = Path(output_image_path).parent.parent / 'EIC CSVs' / f"{base}_peaks_{group_name}.csv"
        if peaks_csv.exists():
            try:
                return pd.read_csv(peaks_csv).to_dict('records')
            except:
                pass

    analyzer = MSFileAnalyzer(file_path)
    mass_list = list(Config.MASS_LIST)
    mass_list_str = [f"{m:.4f}" for m in mass_list]

    # Setup debug directory
    dirs = Config.setup_directories()
    debug_dir = dirs['debugpng']

    # Extract intensity data for each mass
    rt_values = []
    intensity_by_mass = {mass: [] for mass in mass_list}

    with analyzer.get_reader() as reader:
        for scan in reader:
            try:
                rt = analyzer.get_retention_time(scan)
                mzs = np.asarray(scan['m/z array'], dtype=np.float32)
                ints = np.asarray(scan['intensity array'], dtype=np.float32)
                
                for mass in mass_list:
                    mask = np.abs(np.round(mzs, 4) - mass) <= Config.MASS_TOLERANCE
                    intensity = np.sum(ints[mask]) if np.any(mask) else 0.0
                    intensity_by_mass[mass].append(intensity)
                    
                rt_values.append(rt)
            except (KeyError, Exception) as e:
                continue

    # Create plotly figure
    fig = go.Figure()
    color = file_colors.get(base, '#1f77b4')
    peaks_out = []
    all_peak_rts = []
    all_mass_debug_data = []

    # Process each mass
    for mass_idx, specific_mass in enumerate(mass_list):
        intensity_vals = np.array(intensity_by_mass[specific_mass])
        if len(intensity_vals) < 3 or np.max(intensity_vals) == 0:
            continue

        rt_vals = np.array(rt_values)
        
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

        # Create debug data structure
        debug_data = {
            'mass': specific_mass,
            'mass_str': mass_list_str[mass_idx],
            'rt': rt_vals,
            'raw': intensity_vals,
            'smooth': intensity_vals_smooth,
            'noise': noise_level,
            'min_height': min_height,
            'detected_peaks': [],
            'rejected_peaks': [],
            'final_peaks': []
        }
        all_mass_debug_data.append(debug_data)

        # Find peaks
        try:
            peaks, properties = find_peaks(
                intensity_vals_smooth,
                height=min_height,
                distance=1
            )
            
            if len(peaks) == 0:
                continue

            # Calculate peak widths
            rel_height = REL_HEIGHT_BY_MASS.get(round(float(specific_mass), 4), DEFAULT_REL_HEIGHT)
            width_results = peak_widths(intensity_vals_smooth, peaks, rel_height=rel_height)

            # Filter valid widths
            valid_width_mask = width_results[0] > 0
            peaks = peaks[valid_width_mask]
            width_results = tuple(wr[valid_width_mask] for wr in width_results)

            if len(peaks) == 0:
                continue

            # Record detected peaks
            for i, idx in enumerate(peaks):
                left_idx = max(0, int(np.floor(width_results[2][i])))
                right_idx = min(len(rt_vals) - 1, int(np.ceil(width_results[3][i])))
                debug_data['detected_peaks'].append({
                    'idx': idx,
                    'left': left_idx,
                    'right': right_idx,
                    'height': intensity_vals_smooth[idx]
                })

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
                        debug_data['rejected_peaks'].append({
                            'idx': idx, 'left': left_idx, 'right': right_idx,
                            'reason': f'Duration {duration:.3f} > {Config.MAX_PEAK_DURATION}'
                        })
                        continue
                    rt_end = rt_vals[right_idx]
                    duration = rt_end - rt_start

                if not (0.03 <= duration <= 0.75):
                    debug_data['rejected_peaks'].append({
                        'idx': idx, 'left': left_idx, 'right': right_idx,
                        'reason': f'Duration {duration:.3f} outside 0.03-0.75'
                    })
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

                # Record final peak
                debug_data['final_peaks'].append({
                    'left': left_idx,
                    'right': right_idx,
                    'apex': left_idx + np.argmax(intensity_vals[left_idx:right_idx + 1])
                })

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

    # Generate debug plot
    generate_debug_plot(base, all_mass_debug_data, output_image_path, debug_dir)

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


def process_file_for_eic(args: Tuple[str, str, Dict[str, str], str]) -> str:
    """
    Process a single file to generate EIC (for parallel execution)
    
    Args:
        args: Tuple of (file_path, output_image_path, file_colors, group_name)
        
    Returns:
        Status message
    """
    file_path, output_image_path, file_colors, group_name = args
    
    try:
        base = os.path.splitext(os.path.basename(file_path))[0]
        
        # Check if already exists
        if os.path.exists(output_image_path):
            peaks_csv = Path(output_image_path).parent.parent / 'EIC CSVs' / f"{base}_peaks_{group_name}.csv"
            if peaks_csv.exists():
                return f"[↷] {base} (cached)"
        
        # Generate EIC and detect peaks
        peaks = analyze_ms_file_and_generate_eic(file_path, output_image_path, file_colors, group_name)
        
        # Save peaks to CSV
        if peaks:
            peaks_csv = Path(output_image_path).parent.parent / 'EIC CSVs' / f"{base}_peaks_{group_name}.csv"
            peaks_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(peaks).to_csv(peaks_csv, index=False)
        
        return f"[✔] {base} ({len(peaks)} peaks)"
        
    except Exception as e:
        return f"[!] {os.path.basename(file_path)}: {str(e)[:50]}"


def generate_file_colors(file_paths: List[str]) -> Dict[str, str]:
    """
    Generate unique colors for each file
    
    Args:
        file_paths: List of file paths
        
    Returns:
        Dictionary mapping base names to hex colors
    """
    import colorsys
    
    file_colors = {}
    n_files = len(file_paths)
    
    for i, fp in enumerate(file_paths):
        base = os.path.splitext(os.path.basename(fp))[0]
        hue = i / max(n_files, 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
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
    
    # Get file paths
    input_files = FileUtils.get_file_paths()
    
    # Generate colors
    file_colors = generate_file_colors(input_files)
    
    # Create output directory
    output_dir = Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER / str(Config.CURRENT_GROUP) / "EIC PNGs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare arguments
    args_list = []
    for file_path in input_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_image = output_dir / f"EIC_{base_name}_plotly.png"
        args_list.append((file_path, str(output_image), file_colors, group_name))
    
    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)
    
    print(f"\n{'='*70}")
    print(f"Building EIC Images for {group_name}")
    print(f"{'='*70}")
    print(f"Processing {len(input_files)} files using {n_processes} processes")
    print(f"Output directory: {output_dir}\n")
    
    import time
    start_time = time.time()
    
    # Process in parallel
    with Pool(processes=n_processes, initializer=init_worker) as pool:
        results = pool.map(process_file_for_eic, args_list)
    
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
    if len(input_files) > 0:
        print(f"Average time per file: {elapsed_time / len(input_files):.2f} seconds")
    
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
    
    print("\n" + "="*70)
    print("EIC IMAGE GENERATION - ALL GROUPS")
    print("="*70)
    
    for group_name in Config.MASS_GROUPS.keys():
        results = build_eics_for_group(group_name, n_processes)
        all_results[group_name] = results
    
    total_elapsed = time.time() - total_start_time
    
    print("\n" + "="*70)
    print("EIC GENERATION COMPLETE")
    print("="*70)
    print(f"Total time: {total_elapsed:.2f} seconds")
    
    return all_results


if __name__ == "__main__":
    # Build EICs for all groups
    build_eics_for_all_groups()
