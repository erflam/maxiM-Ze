import random
import numpy as np
import pandas as pd
from pathlib import Path
from pyteomics import mzxml
from typing import List, Tuple, Set, Dict
from collections import defaultdict
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

SEED = 12345
RNG = random.Random(SEED)
np.random.seed(SEED)

BASE_DIR = Path.home() / "Desktop/maxiMiZe Tests"
INPUT_SUBDIR = Path("maxiMiZe Files")
OUTPUT_ROOT = Path("maxiMiZe Checkpoints")

# Analysis Parameters
NOISE_LEVEL = 5000.0
MZ_TOLERANCE = 0.0005
MIN_CONSEC_SCANS = 7
MIN_SAMPLE_PRESENCE = 1

USE_STUDY_DESIGN = False
N_FILES_TO_PROCESS = 6
TARGET_GROUP = "target"   # set to None if no target group

SAMPLE_GROUPS: Dict[str, List[str]] = {
    "Group 1": [
        "OE_EF_IsmailBaseline_POS_C072_0005.mzXML",
        "OE_EF_IsmailBaseline_POS_C072_0004.mzXML",
        "OE_EF_IsmailBaseline_POS_C072_0003.mzXML",
        "OE_EF_IsmailBaseline_POS_C068_0006.mzXML",
        "OE_EF_IsmailBaseline_POS_C065_0005.mzXML",
        "OE_EF_IsmailBaseline_POS_C065_0004.mzXML",
        "OE_EF_IsmailBaseline_POS_C065_0002.mzXML",
        "OE_EF_IsmailBaseline_POS_C063_0003.mzXML",
        "OE_EF_IsmailBaseline_POS_C062_0003.mzXML",
        "OE_EF_IsmailBaseline_POS_C062_0002.mzXML",
        "OE_EF_IsmailBaseline_POS_C057_0001.mzXML",
    ],
    "Group 2": [
        "OE_EF_IsmailBaseline_POS_C048_0008.mzXML",
        "OE_EF_IsmailBaseline_POS_C048_0004.mzXML",
        "OE_EF_IsmailBaseline_POS_C047_0001.mzXML",
        "OE_EF_IsmailBaseline_POS_C046_0002.mzXML",
        "OE_EF_IsmailBaseline_POS_C039_0006.mzXML",
        "OE_EF_IsmailBaseline_POS_C039_0002.mzXML",
        "OE_EF_IsmailBaseline_POS_C039_0001.mzXML",
        "OE_EF_IsmailBaseline_POS_C037_0006.mzXML",
        "OE_EF_IsmailBaseline_POS_C034_0005.mzXML",
        "OE_EF_IsmailBaseline_POS_C034_0003.mzXML",
        "OE_EF_IsmailBaseline_POS_C034_0001.mzXML",
    ],
    "target": [
        "OE_EF_IsmailBaseline_POS_C032_0003.mzXML",
        "OE_EF_IsmailBaseline_POS_C032_0002.mzXML",
        "OE_EF_IsmailBaseline_POS_C031_0006.mzXML",
        "OE_EF_IsmailBaseline_POS_C031_0005.mzXML",
        "OE_EF_IsmailBaseline_POS_C031_0003.mzXML",
    ],
}

REUSE_EXISTING_SELECTION = True  # if manifest exists, re-use it exactly
SELECTION_MANIFEST_NAME = f"selected_files_seed_{SEED}.txt"

def _manifest_path() -> Path:
    out_dir = BASE_DIR / OUTPUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / SELECTION_MANIFEST_NAME


def load_selection_manifest() -> List[str] | None:
    p = _manifest_path()
    if REUSE_EXISTING_SELECTION and p.exists():
        lines = [line.strip() for line in p.read_text().splitlines() if line.strip()]
        return lines if lines else None
    return None

def save_selection_manifest(selected: List[str]) -> None:
    p = _manifest_path()
    p.write_text("\n".join(selected) + "\n")


def select_files_with_study_design(n_files: int = N_FILES_TO_PROCESS) -> List[str]:
    selected: List[str] = []
    remaining_slots = n_files

    # Step 1: pick 1 target file
    if TARGET_GROUP is not None:
        target_files = SAMPLE_GROUPS.get(TARGET_GROUP)
        if not target_files:
            raise ValueError(f"TARGET_GROUP '{TARGET_GROUP}' is empty or not found in SAMPLE_GROUPS.")
        target_file = RNG.choice(target_files)  # keep your list order
        selected.append(str(BASE_DIR / INPUT_SUBDIR / target_file))
        remaining_slots -= 1
        print(f"  [target]         {target_file}")

    # Step 2: sample equally from non-target groups
    non_target_groups = {k: v for k, v in SAMPLE_GROUPS.items() if k != TARGET_GROUP}
    if not non_target_groups:
        raise ValueError("No non-target groups defined in SAMPLE_GROUPS.")

    # Stable group ordering (does not change pool order inside each group)
    group_names = sorted(non_target_groups.keys())

    base_per_group = remaining_slots // len(group_names)
    extras = remaining_slots % len(group_names)

    for i, group_name in enumerate(group_names):
        n_from_group = base_per_group + (1 if i < extras else 0)
        pool = non_target_groups[group_name]  # keep your list order

        if len(pool) < n_from_group:
            raise ValueError(
                f"Group '{group_name}' has only {len(pool)} files but {n_from_group} are needed."
            )

        chosen = RNG.sample(pool, n_from_group)
        for f in chosen:
            selected.append(str(BASE_DIR / INPUT_SUBDIR / f))
            padding = " " * max(1, 15 - len(group_name))
            print(f"  [{group_name}]{padding}{f}")

    return selected

def select_random_files(n_files: int = N_FILES_TO_PROCESS) -> List[str]:
    """Deterministic random selection from directory, given SEED."""
    all_files = sorted((BASE_DIR / INPUT_SUBDIR).glob("*.mzXML"))  # stable filesystem order
    if len(all_files) < n_files:
        raise ValueError(f"Not enough mzXML files. Found {len(all_files)}, need {n_files}")
    return [str(f) for f in RNG.sample(all_files, n_files)]

def select_files() -> List[str]:
    """Top-level selector with manifest reuse."""
    existing = load_selection_manifest()
    if existing is not None:
        print(f"Seed: {SEED} (reusing manifest)")
        print("Selected files (manifest):")
        for f in existing:
            print(f"  {Path(f).name}")
        return existing

    print(f"Seed: {SEED}")
    print("Selected files:")
    if USE_STUDY_DESIGN:
        selected = select_files_with_study_design(N_FILES_TO_PROCESS)
    else:
        selected = select_random_files(N_FILES_TO_PROCESS)
        for f in selected:
            print(f"  {Path(f).name}")

    save_selection_manifest(selected)
    return selected

def centroid_scan(scan_idx: int, mzs: np.ndarray, intensities: np.ndarray, noise_level: float) -> List[Tuple[int, float, float]]:
    if len(intensities) < 3:
        return []

    mask = intensities > noise_level
    if not np.any(mask):
        return []

    mzs = mzs[mask]
    intensities = intensities[mask]

    peak_mask = np.zeros(len(intensities), dtype=bool)
    peak_mask[1:-1] = (intensities[1:-1] > intensities[:-2]) & (intensities[1:-1] > intensities[2:])

    intense_mask = intensities > (noise_level * 10)
    peak_mask = peak_mask | intense_mask

    if len(intensities) > 1:
        if intensities[0] > intensities[1]:
            peak_mask[0] = True
        if intensities[-1] > intensities[-2]:
            peak_mask[-1] = True

    peak_indices = np.where(peak_mask)[0]
    return [(scan_idx, float(mzs[i]), float(intensities[i])) for i in peak_indices]

def longest_consecutive_run(scan_indices: Set[int]) -> int:
    if not scan_indices:
        return 0
    scans = np.array(sorted(scan_indices), dtype=int)
    gaps = np.diff(scans)
    run_starts = np.where(gaps != 1)[0] + 1
    splits = np.concatenate(([0], run_starts, [len(scans)]))
    run_lengths = np.diff(splits)
    return int(run_lengths.max())

class MassCluster:
    def __init__(self, scan_idx: int, mz: float, intensity: float):
        self.scan_indices = {scan_idx}
        self.mz_values = [mz]
        self.intensities: Dict[int, float] = {scan_idx: intensity}
        self.max_intensity = intensity
        self.retention_times: Dict[int, float] = {}
        self.apex_scan_idx = scan_idx

    @property
    def mean_mz(self) -> float:
        return float(np.mean(self.mz_values))

    @property
    def consecutive_scans(self) -> int:
        return longest_consecutive_run(self.scan_indices)

    def add_point(self, scan_idx: int, mz: float, intensity: float):
        self.scan_indices.add(scan_idx)
        self.mz_values.append(mz)
        self.intensities[scan_idx] = intensity
        if intensity > self.max_intensity:
            self.max_intensity = intensity
            self.apex_scan_idx = scan_idx

def find_mass_traces(
    centroids: List[Tuple[int, float, float]],
    mz_tol: float,
    min_consec_scans: int,
    retention_times: Dict[int, float],
) -> List[Dict]:
    clusters: List[MassCluster] = []
    centroids.sort(key=lambda x: x[1])

    for scan_idx, mz, intensity in centroids:
        left, right = 0, len(clusters)
        placed = False

        while left < right:
            mid = (left + right) // 2
            cluster = clusters[mid]
            diff = cluster.mean_mz - mz

            if abs(diff) <= mz_tol:
                cluster.add_point(scan_idx, mz, intensity)
                cluster.retention_times[scan_idx] = retention_times[scan_idx]
                placed = True
                break
            elif diff < 0:
                left = mid + 1
            else:
                right = mid

        if not placed:
            new_cluster = MassCluster(scan_idx, mz, intensity)
            new_cluster.retention_times[scan_idx] = retention_times[scan_idx]
            clusters.insert(left, new_cluster)

    valid_clusters: List[Dict] = []
    for cluster in clusters:
        if cluster.consecutive_scans >= min_consec_scans:
            apex_rt = cluster.retention_times[cluster.apex_scan_idx]
            rt_values = list(cluster.retention_times.values())
            valid_clusters.append({
                'mz': cluster.mean_mz,
                'rt': apex_rt,
                'rt_start': min(rt_values),
                'rt_end': max(rt_values),
                'consecutive_scans': cluster.consecutive_scans,
                'intensity': cluster.max_intensity
            })

    return valid_clusters

def process_file_centroids(file_path: str, noise_level: float) -> Tuple[List[Tuple[int, float, float]], Dict[int, float]]:
    centroids: List[Tuple[int, float, float]] = []
    retention_times: Dict[int, float] = {}

    with mzxml.read(file_path) as reader:
        for scan in reader:
            if scan.get('msLevel', 0) == 1:
                scan_number = int(scan['num'])
                mzs = np.array(scan['m/z array'], dtype=np.float32)
                intensities = np.array(scan['intensity array'], dtype=np.float32)
                rt = float(scan['retentionTime'])

                if len(mzs) > 0:
                    scan_centroids = centroid_scan(scan_number, mzs, intensities, noise_level)
                    centroids.extend(scan_centroids)
                    retention_times[scan_number] = rt

    return centroids, retention_times

def process_single_file(
    file_path: str,
    noise_level: float = NOISE_LEVEL,
    mz_tolerance: float = MZ_TOLERANCE,
    min_consec_scans: int = MIN_CONSEC_SCANS,
    verbose: bool = False,   # 👈 ADD THIS LINE
) -> Tuple[str, Set[float], List[Dict]]:

    if verbose:  # 👈 ADD THIS
        print(f"Processing {Path(file_path).name}...")
    centroids, retention_times = process_file_centroids(file_path, noise_level)
    features = find_mass_traces(centroids, mz_tolerance, min_consec_scans, retention_times)
    unique_masses = {feature['mz'] for feature in features}
    return file_path, unique_masses, features

def process_files() -> List[Dict]:
    selected_files = select_files()

    masses_by_file: Dict[str, Set[float]] = {}
    all_features: List[Dict] = []

    with ProcessPoolExecutor(max_workers=min(N_FILES_TO_PROCESS, multiprocessing.cpu_count())) as executor:
        futures = [executor.submit(process_single_file, f) for f in selected_files]
        for fut in futures:
            file_path, masses, features = fut.result()
            masses_by_file[file_path] = masses
            all_features.extend(features)

    mass_counts = defaultdict(int)
    for masses in masses_by_file.values():
        unique_masses = np.array(list(masses), dtype=float)
        rounded_masses = np.round(unique_masses, decimals=6)
        for mz in rounded_masses:
            mass_counts[float(mz)] += 1

    valid_masses = {mz for mz, count in mass_counts.items() if count >= MIN_SAMPLE_PRESENCE}
    print(f"\nUnique masses across selected files (pre MIN_SAMPLE_PRESENCE): {len(mass_counts)}")
    print(f"Masses kept with MIN_SAMPLE_PRESENCE={MIN_SAMPLE_PRESENCE}: {len(valid_masses)}")

    valid_features = []
    for feature in all_features:
        mz_round = round(feature['mz'], 6)
        if mz_round in valid_masses:
            valid_features.append(feature)

    unique_features: Dict[float, Dict] = {}
    for feature in sorted(valid_features, key=lambda x: x['intensity'], reverse=True):
        mz_round = round(feature['mz'], 6)
        if mz_round not in unique_features:
            unique_features[mz_round] = feature

    return sorted(unique_features.values(), key=lambda x: x['mz'])

def is_same_mass(mz1: float, mz2: float, tolerance: float = 0.0005) -> bool:
    return abs(mz1 - mz2) <= tolerance

def check_mass_compatibility(
    group: List[Dict],
    new_mass: Dict,
    max_rt_window: float = 15.0,
    min_intensity_ratio: float = 0.95,
    min_ppm_diff: float = 5.0
) -> bool:
    new_rt_start = new_mass['rt_start']
    new_rt_end = new_mass['rt_end']
    new_mz = new_mass['mz']
    new_intensity = new_mass['intensity']

    for mass in group:
        ppm_diff = abs(new_mz - mass['mz']) / mass['mz'] * 1e6
        if ppm_diff < min_ppm_diff:
            return False

        if new_rt_start <= mass['rt_end'] and new_rt_end >= mass['rt_start']:
            return False

        if abs(new_mass['rt'] - mass['rt']) > max_rt_window:
            return False

        intensity_ratio = min(new_intensity, mass['intensity']) / max(new_intensity, mass['intensity'])
        if intensity_ratio < min_intensity_ratio:
            return False

    return True

def find_groups(masses_df: pd.DataFrame, min_group_size: int = 3, max_group_size: int = 5) -> List[List[Dict]]:
    sorted_masses = masses_df.sort_values('intensity', ascending=False)

    available_masses: List[Dict] = []
    seen_mzs: List[float] = []
    for row in sorted_masses.itertuples():
        if any(is_same_mass(row.mz, seen) for seen in seen_mzs):
            continue
        available_masses.append({
            'mz': row.mz,
            'rt': row.rt,
            'rt_start': row.rt_start,
            'rt_end': row.rt_end,
            'intensity': row.intensity
        })
        seen_mzs.append(row.mz)

    groups: List[List[Dict]] = []
    unplaced = list(available_masses)

    while unplaced:
        seed = unplaced.pop(0)
        current_group = [seed]
        remaining = []

        for candidate in unplaced:
            if len(current_group) >= max_group_size:
                remaining.append(candidate)
                continue
            if check_mass_compatibility(current_group, candidate):
                current_group.append(candidate)
            else:
                remaining.append(candidate)

        groups.append(current_group)
        unplaced = remaining

    return groups

def build_mass_groups_from_files(
    file_paths: List[str],
    *,
    noise_level: float = NOISE_LEVEL,
    mz_tolerance: float = MZ_TOLERANCE,
    min_consec_scans: int = MIN_CONSEC_SCANS,
    min_sample_presence: int = MIN_SAMPLE_PRESENCE,
    min_group_size: int = 3,
    max_group_size: int = 5,
    verbose: bool = False,
) -> Dict[str, List[float]]:
    masses_by_file: Dict[str, Set[float]] = {}
    all_features: List[Dict] = []

    with ProcessPoolExecutor(max_workers=min(len(file_paths), multiprocessing.cpu_count())) as executor:
        futures = [
            executor.submit(process_single_file, f, noise_level, mz_tolerance, min_consec_scans, verbose)
            for f in file_paths
        ]
        for fut in futures:
            file_path, masses, features = fut.result()
            masses_by_file[file_path] = masses
            all_features.extend(features)

    mass_counts = defaultdict(int)
    for masses in masses_by_file.values():
        unique_masses = np.array(list(masses), dtype=float)
        rounded_masses = np.round(unique_masses, decimals=6)
        for mz in rounded_masses:
            mass_counts[float(mz)] += 1

    valid_masses = {mz for mz, count in mass_counts.items() if count >= min_sample_presence}

    valid_features = []
    for feature in all_features:
        mz_round = round(feature['mz'], 6)
        if mz_round in valid_masses:
            valid_features.append(feature)

    unique_features: Dict[float, Dict] = {}
    for feature in sorted(valid_features, key=lambda x: x['intensity'], reverse=True):
        mz_round = round(feature['mz'], 6)
        if mz_round not in unique_features:
            unique_features[mz_round] = feature

    features_sorted = sorted(unique_features.values(), key=lambda x: x['mz'])
    df = pd.DataFrame(features_sorted)

    mass_groups = find_groups(df, min_group_size=min_group_size, max_group_size=max_group_size)

    out: Dict[str, List[float]] = {}
    for i, group in enumerate(mass_groups, 1):
        out[f"Group{i}"] = [float(m["mz"]) for m in group]

    return out

def main():
    features = process_files()

    output_dir = BASE_DIR / OUTPUT_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(features)
    top_50 = df.nlargest(50, 'intensity')

    output_file = output_dir / 'mass_features_4.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All Features', index=False)
        top_50.to_excel(writer, sheet_name='Top 50 Intense Peaks', index=False)

    print(f"\nFound {len(features)} features")
    print(f"Results saved to {output_file}")

    if len(features) > 0:
        print("\nData ranges:")
        print(f"m/z:       {df['mz'].min():.4f} - {df['mz'].max():.4f}")
        print(f"RT:        {df['rt'].min():.2f} - {df['rt'].max():.2f} min")
        print(f"RT start:  {df['rt_start'].min():.2f} - {df['rt_start'].max():.2f} min")
        print(f"RT end:    {df['rt_end'].min():.2f} - {df['rt_end'].max():.2f} min")
        print(f"Intensity: {df['intensity'].min():.0f} - {df['intensity'].max():.0f}")

    min_group_size = 3
    max_group_size = 5

    mass_groups = find_groups(df, min_group_size=min_group_size, max_group_size=max_group_size)

    group_data = []
    total_grouped_masses = 0
    all_used_masses: List[float] = []

    for i, group in enumerate(mass_groups, 1):
        for mass in group:
            if any(is_same_mass(mass['mz'], used) for used in all_used_masses):
                print(f"Warning: Mass {mass['mz']:.6f} appears multiple times in groups!")
                continue
            group_data.append({
                'Group': f'Group {i}',
                'Mass (m/z)': mass['mz'],
                'RT (min)': mass['rt'],
                'RT Start (min)': mass['rt_start'],
                'RT End (min)': mass['rt_end'],
                'Intensity': mass['intensity']
            })
            all_used_masses.append(mass['mz'])
            total_grouped_masses += 1

    groups_df = pd.DataFrame(group_data)
    groups_df = groups_df.sort_values(
        by='Group',
        key=lambda col: col.str.extract(r'(\d+)').astype(int)[0]
    )

    formatted_groups = []
    for i, group in enumerate(mass_groups, 1):
        group_masses = [f"{mass['mz']:.4f}" for mass in group]
        formatted_groups.append({
            'Group': f"'Group {i}': ",
            'Size': len(group),
            'Masses': f"[{', '.join(group_masses)}]"
        })

    formatted_df = pd.DataFrame(formatted_groups)

    sizes = [len(g) for g in mass_groups]
    size_counts = pd.Series(sizes).value_counts().sort_index()
    n_in_target = sum(1 for s in sizes if min_group_size <= s <= max_group_size)
    pct_in_target = 100 * n_in_target / len(sizes) if sizes else 0

    print(f"\nFound {len(mass_groups)} mass groups")
    print(f"Total masses in groups: {total_grouped_masses}")
    print(f"Average group size: {total_grouped_masses / len(mass_groups):.1f}")
    print(f"Groups with 3-5 members: {n_in_target}/{len(mass_groups)} ({pct_in_target:.1f}%)")
    print("\nGroup size distribution:")
    for size, count in size_counts.items():
        print(f"  Size {size}: {count} group(s)")

    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        groups_df.to_excel(writer, sheet_name='Mass Groups', index=False)
        formatted_df.to_excel(writer, sheet_name='Formatted Groups', index=False)

    print(f"\nGrouping results appended to {output_file}")
    print(f"Selection manifest saved at: {_manifest_path()}")

if __name__ == "__main__":
    main()
