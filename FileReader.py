import numpy as np
import pandas as pd
from pathlib import Path
from pyteomics import mzxml
from typing import List, Tuple, Dict
from multiprocessing import Pool, cpu_count
import os

# Configuration
NOISE_LEVEL = 5000.0
MZ_TOLERANCE = 0.0005
MIN_CONSEC_SCANS = 7


def centroid_scan(scan_idx: int, mzs: np.ndarray, intensities: np.ndarray, noise_level: float) -> List[Tuple[int, float, float]]:
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


def process_mzxml_files_parallel(input_files: List[str], output_dir: str, 
                                 noise_level: float = NOISE_LEVEL, 
                                 n_processes: int = None):
    """
    Process multiple mzXML files in parallel
    
    Args:
        input_files: List of paths to input mzXML files
        output_dir: Directory to save output CSV files
        noise_level: Minimum intensity threshold for peak detection
        n_processes: Number of parallel processes (default: CPU count - 1)
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare arguments for each file
    args_list = []
    for file_path in input_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_csv = os.path.join(output_dir, f"{base_name}_raw.csv")
        args_list.append((file_path, output_csv, noise_level))
    
    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)
    
    print(f"Processing {len(input_files)} files using {n_processes} processes...")
    
    # Process files in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_single_mzxml, args_list)
    
    # Print results
    print("\nProcessing complete:")
    for result in results:
        print(result)
    
    return results


def process_single_mzxml_file(input_file: str, output_csv: str, noise_level: float = NOISE_LEVEL):
    """
    Process a single mzXML file (non-parallel version for single file use)
    
    Args:
        input_file: Path to input mzXML file
        output_csv: Path to output CSV file
        noise_level: Minimum intensity threshold for peak detection
    """
    result = process_single_mzxml((input_file, output_csv, noise_level))
    print(result)
    
    # Load and return the DataFrame
    if os.path.exists(output_csv):
        return pd.read_csv(output_csv)
    return None


# Example usage
if __name__ == "__main__":
    # Example 1: Process multiple files in parallel
    input_files = [
        "path/to/file1.mzXML",
        "path/to/file2.mzXML",
        "path/to/file3.mzXML",
        # Add more files...
    ]
    
    output_directory = "output_csvs"
    
    # Process all files in parallel
    process_mzxml_files_parallel(
        input_files=input_files,
        output_dir=output_directory,
        noise_level=NOISE_LEVEL,
        n_processes=None  # Auto-detect (CPU count - 1)
    )
    
    # Example 2: Process a single file
    # df = process_single_mzxml_file(
    #     input_file="path/to/single_file.mzXML",
    #     output_csv="output.csv",
    #     noise_level=NOISE_LEVEL
    # )
    # print(df.head())
    
    # Example 3: Read from a directory
    # from glob import glob
    # input_files = glob("path/to/mzxml_files/*.mzXML")
    # process_mzxml_files_parallel(input_files, "output_csvs")
