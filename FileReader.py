import numpy as np
import pandas as pd
from pathlib import Path
from pyteomics import mzxml
from typing import List, Tuple, Dict
from multiprocessing import Pool, cpu_count
import os
import time

# Import your modules
from Config import Config
from FileUtils import FileUtils

# Configuration
NOISE_LEVEL = 5000.0


def centroid_scan(scan_idx: int, mzs: np.ndarray, intensities: np.ndarray, noise_level: float) -> List[
    Tuple[int, float, float]]:
    """Detect peaks in a single scan"""
    if len(intensities) < 3:
        return []

    # Pre-filter by noise level
    mask = intensities > noise_level
    if not np.any(mask):
        return []

    mzs = mzs[mask]
    intensities = intensities[mask]

    # Fast local maxima detection
    peak_mask = np.zeros(len(intensities), dtype=bool)

    # Interior points - check if greater than neighbors
    peak_mask[1:-1] = (intensities[1:-1] > intensities[:-2]) & (intensities[1:-1] > intensities[2:])

    # Add very intense peaks (>10x noise)
    intense_mask = intensities > (noise_level * 10)
    peak_mask = peak_mask | intense_mask

    # Check endpoints
    if len(intensities) > 1:
        if intensities[0] > intensities[1]:
            peak_mask[0] = True
        if intensities[-1] > intensities[-2]:
            peak_mask[-1] = True

    # Return detected peaks
    peak_indices = np.where(peak_mask)[0]
    return [(scan_idx, float(mzs[i]), float(intensities[i])) for i in peak_indices]


def process_single_mzxml(args: Tuple[str, str, float]) -> str:
    """
    Process a single mzXML file (designed for parallel execution)

    Args:
        args: Tuple of (input_file_path, output_csv_path, noise_level)

    Returns:
        Status message
    """
    file_path, output_csv, noise_level = args

    try:
        # Check if output already exists
        if os.path.exists(output_csv):
            return f"[↷] {os.path.basename(file_path)} (already exists)"

        data_rows = []

        with mzxml.read(file_path) as reader:
            for scan in reader:
                # Only process MS1 scans
                if scan.get('msLevel', 0) == 1:
                    scan_number = int(scan['num'])
                    rt = float(scan['retentionTime'])  # Retention time in minutes

                    # Convert to numpy arrays
                    mzs = np.array(scan['m/z array'], dtype=np.float64)  # Higher precision for mass
                    intensities = np.array(scan['intensity array'], dtype=np.float32)

                    if len(mzs) > 0:
                        # Detect peaks in this scan
                        peaks = centroid_scan(scan_number, mzs, intensities, noise_level)

                        # Add each peak to output data
                        for _, mz, intensity in peaks:
                            data_rows.append({
                                'rt': rt,
                                'mass': mz,
                                'intensity': intensity,
                                'scan': scan_number
                            })

        # Create DataFrame and save to CSV with specific formatting
        df = pd.DataFrame(data_rows)

        # Write with custom float formatting
        with open(output_csv, 'w') as f:
            f.write('rt,mass,intensity,scan\n')
            for _, row in df.iterrows():
                f.write(f"{row['rt']:.3f},{row['mass']:.4f},{row['intensity']:.0f},{row['scan']}\n")

        return f"[✔] {os.path.basename(file_path)} ({len(df)} peaks)"

    except Exception as e:
        return f"[!] {os.path.basename(file_path)}: {str(e)[:50]}"


def filter_peaks_by_mass_group_parallel(args: Tuple[str, str, str, List[float], float]) -> Tuple[str, int]:
    """
    Filter peaks from a raw CSV for a specific group (parallelizable)

    Args:
        args: Tuple of (raw_csv_path, output_csv_path, group_name, mass_list, mass_tolerance)

    Returns:
        Tuple of (group_name, number of peaks)
    """
    raw_csv_path, output_csv_path, group_name, mass_list, mass_tolerance = args

    try:
        # Check if already exists
        if os.path.exists(output_csv_path):
            df = pd.read_csv(output_csv_path)
            return (group_name, len(df))

        # Read the raw CSV
        df = pd.read_csv(raw_csv_path)

        if len(df) == 0:
            # Create empty file with correct columns
            with open(output_csv_path, 'w') as f:
                f.write('rt,mass,intensity,scan\n')
            return (group_name, 0)

        # Filter peaks that match any of the target masses - vectorized approach
        mass_array = df['mass'].values
        mask = np.zeros(len(mass_array), dtype=bool)

        for target_mass in mass_list:
            mask |= np.abs(mass_array - target_mass) <= mass_tolerance

        filtered_df = df[mask].copy()

        if len(filtered_df) > 0:
            # Sort by retention time, then by mass
            filtered_df = filtered_df.sort_values(['rt', 'mass']).reset_index(drop=True)

            # Write with custom float formatting
            with open(output_csv_path, 'w') as f:
                f.write('rt,mass,intensity,scan\n')
                for _, row in filtered_df.iterrows():
                    f.write(f"{row['rt']:.3f},{row['mass']:.4f},{row['intensity']:.0f},{row['scan']}\n")
        else:
            # Create empty file
            with open(output_csv_path, 'w') as f:
                f.write('rt,mass,intensity,scan\n')

        return (group_name, len(filtered_df))

    except Exception as e:
        print(f"Error filtering {os.path.basename(raw_csv_path)} for {group_name}: {e}")
        return (group_name, 0)


def process_all_raw_files(noise_level: float = NOISE_LEVEL, n_processes: int = None):
    """
    Process all mzXML files once and save raw peak data to Mass Detection folder

    Args:
        noise_level: Minimum intensity threshold for peak detection
        n_processes: Number of parallel processes (default: CPU count - 1)
    """
    # Get input file paths from FileUtils
    input_files = FileUtils.get_file_paths()

    # Create output directory in Mass Detection folder
    output_dir = Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER / "Mass Detection"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments for each file
    args_list = []
    for file_path in input_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_csv = output_dir / f"{base_name}_EIC_raw.csv"
        args_list.append((file_path, str(output_csv), noise_level))

    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)

    print("=" * 70)
    print("mzXML to CSV Converter - Raw Peak Detection")
    print("=" * 70)
    print(f"Processing {len(input_files)} files using {n_processes} processes...")
    print(f"Input directory: {Config.BASE_DIR / Config.INPUT_SUBDIR}")
    print(f"Output directory: {output_dir}\n")

    start_time = time.time()

    # Process files in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_single_mzxml, args_list)

    elapsed_time = time.time() - start_time

    # Print results
    print("\nRaw Processing Results:")
    for result in results:
        print(result)

    # Summary statistics
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
    Create filtered CSV files for all mass groups from raw data (PARALLELIZED)

    Args:
        n_processes: Number of parallel processes (default: CPU count - 1)
    """
    print("\n" + "=" * 70)
    print("Creating Filtered CSVs for All Mass Groups (Parallel)")
    print("=" * 70)

    # Get the raw data directory (Mass Detection)
    raw_dir = Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER / "Mass Detection"

    # Find all raw CSV files
    raw_csv_files = list(raw_dir.glob("*_EIC_raw.csv"))

    if not raw_csv_files:
        print(f"No raw CSV files found in {raw_dir}")
        return

    # Get mass tolerance
    mass_tolerance = Config.MASS_TOLERANCE

    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)

    # Prepare all filtering tasks
    filtering_args = []

    for group_name in Config.MASS_GROUPS.keys():
        # Set the current group in Config
        Config.set_mass_group(group_name)

        # Get the mass list for this group
        mass_list = Config.MASS_GROUPS[group_name]

        # Get the output directory for this group
        group_output_dir = Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER / str(
            Config.CURRENT_GROUP) / "EIC CSVs"
        group_output_dir.mkdir(parents=True, exist_ok=True)

        # Add tasks for each file in this group
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

    # Process all filtering tasks in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.map(filter_peaks_by_mass_group_parallel, filtering_args)

    elapsed_time = time.time() - start_time

    # Organize results by group
    group_stats = {}
    for group_name, num_peaks in results:
        if group_name not in group_stats:
            group_stats[group_name] = {'files': 0, 'total_peaks': 0}
        group_stats[group_name]['files'] += 1
        group_stats[group_name]['total_peaks'] += num_peaks

    # Print summary for each group
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
        n_processes: Number of parallel processes (default: CPU count - 1)
    """
    total_start_time = time.time()

    # Step 1: Process all raw files (only once)
    process_all_raw_files(noise_level=noise_level, n_processes=n_processes)

    # Step 2: Create filtered CSVs for all groups (PARALLELIZED)
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
    # Process all files once and create filtered CSVs for all groups
    process_all_groups(
        noise_level=NOISE_LEVEL,
        n_processes=None  # Auto-detect CPU cores
    )

    # Analyze results for each group
    print("\n" + "=" * 70)
    print("GROUP STATISTICS")
    print("=" * 70)
    for group_name in Config.MASS_GROUPS.keys():
        analyze_results_for_group(group_name)