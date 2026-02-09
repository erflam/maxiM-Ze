import numpy as np
import pandas as pd
from pathlib import Path
from pyteomics import mzxml
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
import os
import time
import shutil

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
                    mzs = np.array(scan['m/z array'], dtype=np.float32)
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

        # Create DataFrame and save to CSV
        df = pd.DataFrame(data_rows)
        df.to_csv(output_csv, index=False, float_format='%.3f')

        return f"[✔] {os.path.basename(file_path)} ({len(df)} peaks)"

    except Exception as e:
        return f"[!] {os.path.basename(file_path)}: {str(e)[:50]}"


def process_all_files_for_group(group_name, noise_level: float = NOISE_LEVEL, n_processes: int = None):
    """
    Process all mzXML files for a specific mass group

    Args:
        group_name: The mass group to process (e.g., 1, 2, 3, etc.)
        noise_level: Minimum intensity threshold for peak detection
        n_processes: Number of parallel processes (default: CPU count - 1)
    """
    # Set the current group in Config
    Config.set_mass_group(group_name)

    # Get input file paths from FileUtils
    input_files = FileUtils.get_file_paths()

    # Create output directory using Config's directory structure
    output_dir = Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER / str(Config.CURRENT_GROUP) / "EIC CSVs"
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

    print(f"Processing Group: {group_name}")
    print(f"Processing {len(input_files)} files using {n_processes} processes...")
    print(f"Input directory: {Config.BASE_DIR / Config.INPUT_SUBDIR}")
    print(f"Output directory: {output_dir}\n")

    # Process files in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_single_mzxml, args_list)

    # Print results
    print("Processing complete:")
    for result in results:
        print(result)

    # Summary statistics
    successful = sum(1 for r in results if r.startswith("[✔]"))
    cached = sum(1 for r in results if r.startswith("[↷]"))
    failed = sum(1 for r in results if r.startswith("[!]"))

    print(f"Summary: {successful} processed, {cached} cached, {failed} failed")

    return results


def process_all_groups(noise_level: float = NOISE_LEVEL, n_processes: int = None):
    """
    Process all mass groups defined in Config.MASS_GROUPS

    Args:
        noise_level: Minimum intensity threshold for peak detection
        n_processes: Number of parallel processes (default: CPU count - 1)
    """
    total_start_time = time.time()

    print("mzXML to CSV Converter - Parallel Processing")
    print("Processing All Mass Groups")

    all_results = {}

    for group_name in Config.MASS_GROUPS.keys():
        group_start_time = time.time()

        results = process_all_files_for_group(
            group_name=group_name,
            noise_level=noise_level,
            n_processes=n_processes
        )

        group_elapsed = time.time() - group_start_time
        all_results[group_name] = {
            'results': results,
            'time': group_elapsed
        }

        print(f"\nGroup {group_name} completed in {group_elapsed:.2f} seconds")
        print(f"Average time per file: {group_elapsed / len(FileUtils.get_file_paths()):.2f} seconds")

    total_elapsed = time.time() - total_start_time

    print("ALL GROUPS PROCESSING SUMMARY")
    for group_name, data in all_results.items():
        print(f"Group {group_name}: {data['time']:.2f} seconds")
    print(f"\nTotal processing time: {total_elapsed:.2f} seconds")
    return all_results


def analyze_results_for_group(group_name):
    """Analyze the generated CSV files for a specific group and print statistics"""
    Config.set_mass_group(group_name)
    output_dir = Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER / str(Config.CURRENT_GROUP) / "EIC CSVs"

    csv_files = list(output_dir.glob("*_EIC_raw.csv"))

    if not csv_files:
        print(f"No CSV files found for group {group_name}!")
        return

    print(f"\nAnalyzing {len(csv_files)} CSV files for group {group_name}...\n")

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

    print(f"GROUP {group_name} STATISTICS")
    print(f"Total files processed: {len(csv_files)}")
    print(f"Total peaks detected: {total_peaks:,}")
    print(f"Average peaks per file: {total_peaks / len(csv_files):,.0f}")
    print(f"\nRetention time range: {rt_min:.2f} - {rt_max:.2f} minutes")
    print(f"Mass range: {mass_min:.2f} - {mass_max:.2f} m/z")
    print(f"Intensity range: {int_min:.0f} - {int_max:.0f}")

if __name__ == "__main__":
    # Process all groups
    process_all_groups(
        noise_level=NOISE_LEVEL,
        n_processes=None  # Auto-detect CPU cores
    )

    # Analyze results for each group
    for group_name in Config.MASS_GROUPS.keys():
        analyze_results_for_group(group_name)