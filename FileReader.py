"""
FileReader.py
Efficiently reads mzXML files and creates filtered CSVs for mass groups.
Optimized with Numba for high-performance peak detection.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from pyteomics import mzxml
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
from numba import njit
import os
import time

from Config import Config
from FileUtils import FileUtils

# Configuration
NOISE_LEVEL = 5000.0


@njit
def fast_centroid_scan(mzs: np.ndarray, intensities: np.ndarray, noise_level: float):
    """
    Numba-accelerated peak detection in a single scan

    Returns: Arrays of (mz, intensity) for detected peaks
    """
    n = len(intensities)
    if n < 3:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    # Pre-filter by noise level
    mask = intensities > noise_level
    if not np.any(mask):
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    mzs_filtered = mzs[mask]
    ints_filtered = intensities[mask]
    n_filtered = len(ints_filtered)

    if n_filtered < 3:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    # Peak detection
    peak_mask = np.zeros(n_filtered, dtype=np.bool_)

    # Interior points - check if greater than neighbors
    for i in range(1, n_filtered - 1):
        if ints_filtered[i] > ints_filtered[i - 1] and ints_filtered[i] > ints_filtered[i + 1]:
            peak_mask[i] = True

    # Check endpoints
    if ints_filtered[0] > ints_filtered[1]:
        peak_mask[0] = True
    if ints_filtered[-1] > ints_filtered[-2]:
        peak_mask[-1] = True

    # Return peak m/z and intensity values
    return mzs_filtered[peak_mask], ints_filtered[peak_mask]


@njit
def fast_filter_by_masses(mzs: np.ndarray, intensities: np.ndarray, mass_list: np.ndarray, tolerance: float):
    """
    Numba-accelerated filtering of peaks by target masses

    Returns: Boolean mask indicating which peaks match target masses
    """
    n_peaks = len(mzs)
    n_masses = len(mass_list)
    mask = np.zeros(n_peaks, dtype=np.bool_)

    for i in range(n_peaks):
        for j in range(n_masses):
            if abs(mzs[i] - mass_list[j]) <= tolerance:
                mask[i] = True
                break

    return mask


def process_single_mzxml(args: Tuple[str, str, float]) -> str:
    """
    Process a single mzXML file with Numba acceleration

    Args:
        args: Tuple of (input_file_path, output_csv_path, noise_level)

    Returns:
        Status message
    """
    file_path, output_csv, noise_level = args

    try:
        if os.path.exists(output_csv):
            return f"[↷] {os.path.basename(file_path)} (already exists)"

        # Pre-allocate lists
        rt_list = []
        mass_list = []
        intensity_list = []
        scan_list = []

        with mzxml.read(file_path) as reader:
            for scan in reader:
                if scan.get('msLevel', 0) == 1:
                    scan_number = int(scan['num'])
                    rt = float(scan['retentionTime'])

                    # Use float32 for efficiency
                    mzs = np.array(scan['m/z array'], dtype=np.float32)
                    intensities = np.array(scan['intensity array'], dtype=np.float32)

                    if len(mzs) > 0:
                        # Use Numba-accelerated peak detection
                        peak_mzs, peak_ints = fast_centroid_scan(mzs, intensities, noise_level)

                        # Batch append
                        n_peaks = len(peak_mzs)
                        rt_list.extend([rt] * n_peaks)
                        mass_list.extend(peak_mzs)
                        intensity_list.extend(peak_ints)
                        scan_list.extend([scan_number] * n_peaks)

        # Create DataFrame efficiently
        df = pd.DataFrame({
            'rt': rt_list,
            'mass': mass_list,
            'intensity': intensity_list,
            'scan': scan_list
        })

        # Fast CSV write
        df.to_csv(output_csv, index=False, float_format='%.4f')

        return f"[✔] {os.path.basename(file_path)} ({len(df)} peaks)"

    except Exception as e:
        return f"[!] {os.path.basename(file_path)}: {str(e)[:50]}"


def filter_peaks_by_mass_group_parallel(args: Tuple[str, str, str, List[float], float]) -> Tuple[str, int]:
    """
    Filter peaks from a raw CSV for a specific group with Numba acceleration

    Args:
        args: Tuple of (raw_csv_path, output_csv_path, group_name, mass_list, mass_tolerance)

    Returns:
        Tuple of (group_name, number of peaks)
    """
    raw_csv_path, output_csv_path, group_name, mass_list, mass_tolerance = args

    try:
        if os.path.exists(output_csv_path):
            # Quick count without full read
            with open(output_csv_path, 'r') as f:
                count = sum(1 for _ in f) - 1
            return (group_name, count)

        # Read CSV efficiently with specific dtypes
        df = pd.read_csv(
            raw_csv_path,
            dtype={'rt': np.float32, 'mass': np.float32, 'intensity': np.float32, 'scan': np.int32}
        )

        if len(df) == 0:
            with open(output_csv_path, 'w') as f:
                f.write('rt,mass,intensity,scan\n')
            return (group_name, 0)

        # Use Numba for fast filtering
        mass_array = df['mass'].values.astype(np.float32)
        target_masses = np.array(mass_list, dtype=np.float32)

        mask = fast_filter_by_masses(mass_array, mass_array, target_masses, mass_tolerance)

        filtered_df = df[mask]

        if len(filtered_df) > 0:
            # Sort and save efficiently
            filtered_df = filtered_df.sort_values(['rt', 'mass'])
            filtered_df.to_csv(output_csv_path, index=False, float_format='%.4f')
        else:
            with open(output_csv_path, 'w') as f:
                f.write('rt,mass,intensity,scan\n')

        return (group_name, len(filtered_df))

    except Exception as e:
        return (group_name, 0)


def process_all_raw_files(noise_level: float = NOISE_LEVEL, n_processes: int = None):
    """
    Process all mzXML files once and save raw peak data to Mass Detection folder

    Args:
        noise_level: Minimum intensity threshold for peak detection
        n_processes: Number of parallel processes (default: CPU count - 2 for cooler operation)
    """
    input_files = FileUtils.get_file_paths()

    output_dir = Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER / "Mass Detection"
    output_dir.mkdir(parents=True, exist_ok=True)

    args_list = []
    for file_path in input_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_csv = output_dir / f"{base_name}_EIC_raw.csv"
        args_list.append((file_path, str(output_csv), noise_level))

    # Use fewer processes to reduce heat
    if n_processes is None:
        n_processes = max(1, cpu_count() - 2)

    print("=" * 70)
    print("mzXML to CSV Converter - Raw Peak Detection (Numba-Accelerated)")
    print("=" * 70)
    print(f"Processing {len(input_files)} files using {n_processes} processes...")
    print(f"Input directory: {Config.BASE_DIR / Config.INPUT_SUBDIR}")
    print(f"Output directory: {output_dir}\n")

    start_time = time.time()

    with Pool(processes=n_processes) as pool:
        results = pool.map(process_single_mzxml, args_list)

    elapsed_time = time.time() - start_time

    print("\nRaw Processing Results:")
    for result in results:
        print(result)

    successful = sum(1 for r in results if r.startswith("[✔]"))
    cached = sum(1 for r in results if r.startswith("[↷]"))
    failed = sum(1 for r in results if r.startswith("[!]"))

    print(f"\nSummary: {successful} processed, {cached} cached, {failed} failed")
    print(f"Processing time: {elapsed_time:.2f} seconds")

    if len(input_files) > 0:
        print(f"Average time per file: {elapsed_time / len(input_files):.2f} seconds")

    return results


def create_all_group_filtered_csvs(n_processes: int = None):
    """
    Create filtered CSV files for all mass groups from raw data with Numba acceleration

    Args:
        n_processes: Number of parallel processes (default: CPU count - 2)
    """
    print("\n" + "=" * 70)
    print("Creating Filtered CSVs for All Mass Groups (Numba-Accelerated)")
    print("=" * 70)

    raw_dir = Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER / "Mass Detection"
    raw_csv_files = list(raw_dir.glob("*_EIC_raw.csv"))

    if not raw_csv_files:
        print(f"No raw CSV files found in {raw_dir}")
        return

    mass_tolerance = Config.MASS_TOLERANCE

    # Use fewer processes to reduce heat
    if n_processes is None:
        n_processes = max(1, cpu_count() - 2)

    filtering_args = []

    for group_name in Config.MASS_GROUPS.keys():
        Config.set_mass_group(group_name)
        mass_list = Config.MASS_GROUPS[group_name]

        group_output_dir = Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER / str(
            Config.CURRENT_GROUP) / "EIC CSVs"
        group_output_dir.mkdir(parents=True, exist_ok=True)

        for raw_csv in raw_csv_files:
            base_name = raw_csv.stem.replace("_EIC_raw", "")
            filtered_csv = group_output_dir / f"{base_name}_peaks_{group_name}.csv"

            filtering_args.append((
                str(raw_csv),
                str(filtered_csv),
                group_name,
                mass_list,
                mass_tolerance
            ))

    print(f"Filtering {len(raw_csv_files)} files across {len(Config.MASS_GROUPS)} groups...")
    print(f"Total tasks: {len(filtering_args)}")
    print(f"Using {n_processes} processes\n")

    start_time = time.time()

    with Pool(processes=n_processes) as pool:
        results = pool.map(filter_peaks_by_mass_group_parallel, filtering_args)

    elapsed_time = time.time() - start_time

    group_stats = {}
    for group_name, num_peaks in results:
        if group_name not in group_stats:
            group_stats[group_name] = {'files': 0, 'total_peaks': 0}
        group_stats[group_name]['files'] += 1
        group_stats[group_name]['total_peaks'] += num_peaks

    print("\nFiltering Results:")
    for group_name in Config.MASS_GROUPS.keys():
        if group_name in group_stats:
            stats = group_stats[group_name]
            avg_peaks = stats['total_peaks'] / stats['files'] if stats['files'] > 0 else 0
            print(
                f"[✔] {group_name}: {stats['files']} files, {stats['total_peaks']:,} total peaks, {avg_peaks:,.0f} avg/file")

    print(f"\nFiltering time: {elapsed_time:.2f} seconds")


def process_all_groups(noise_level: float = NOISE_LEVEL, n_processes: int = None):
    """
    Complete workflow: Process raw files once, then create filtered CSVs for all groups

    Args:
        noise_level: Minimum intensity threshold for peak detection
        n_processes: Number of parallel processes (default: CPU count - 2)
    """
    total_start_time = time.time()

    process_all_raw_files(noise_level=noise_level, n_processes=n_processes)
    create_all_group_filtered_csvs(n_processes=n_processes)

    total_elapsed = time.time() - total_start_time

    print("\n" + "=" * 70)
    print("COMPLETE PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total processing time: {total_elapsed:.2f} seconds")
    print(f"Raw data location: {Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER / 'Mass Detection'}")
    print(
        f"Filtered data location: {Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER / '[Group Name]' / 'EIC CSVs'}")


def analyze_results_for_group(group_name: str):
    """Analyze the generated CSV files for a specific group and print statistics"""
    Config.set_mass_group(group_name)
    output_dir = Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER / str(Config.CURRENT_GROUP) / "EIC CSVs"

    csv_files = list(output_dir.glob(f"*_peaks_{group_name}.csv"))

    if not csv_files:
        print(f"\nNo CSV files found for {group_name}!")
        return

    print(f"\n{'=' * 70}")
    print(f"ANALYSIS: {group_name}")
    print(f"{'=' * 70}")

    total_peaks = 0
    rt_min, rt_max = float('inf'), 0
    mass_min, mass_max = float('inf'), 0
    int_min, int_max = float('inf'), 0

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        total_peaks += len(df)

        if len(df) > 0:
            rt_min = min(rt_min, df['rt'].min())
            rt_max = max(rt_max, df['rt'].max())
            mass_min = min(mass_min, df['mass'].min())
            mass_max = max(mass_max, df['mass'].max())
            int_min = min(int_min, df['intensity'].min())
            int_max = max(int_max, df['intensity'].max())

    print(f"Total files processed: {len(csv_files)}")
    print(f"Total peaks detected: {total_peaks:,}")
    print(f"Average peaks per file: {total_peaks / len(csv_files):,.0f}")

    if total_peaks > 0:
        print(f"\nRetention time range: {rt_min:.3f} - {rt_max:.3f} minutes")
        print(f"Mass range: {mass_min:.4f} - {mass_max:.4f} m/z")
        print(f"Intensity range: {int_min:.0f} - {int_max:.0f}")


if __name__ == "__main__":
    process_all_groups(
        noise_level=NOISE_LEVEL,
        n_processes=None
    )

    print("\n" + "=" * 70)
    print("GROUP STATISTICS")
    print("=" * 70)
    for group_name in Config.MASS_GROUPS.keys():
        analyze_results_for_group(group_name)