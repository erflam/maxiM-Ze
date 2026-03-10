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

REUSE_EXISTING_SELECTION = True
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

    if TARGET_GROUP is not None:
        target_files = SAMPLE_GROUPS.get(TARGET_GROUP)
        if not target_files:
            raise ValueError(f"TARGET_GROUP '{TARGET_GROUP}' is empty or not found in SAMPLE_GROUPS.")
        target_file = RNG.choice(target_files)
        selected.append(str(BASE_DIR / INPUT_SUBDIR / target_file))
        remaining_slots -= 1
        print(f"  [target]         {target_file}")

    non_target_groups = {k: v for k, v in SAMPLE_GROUPS.items() if k != TARGET_GROUP}
    if not non_target_groups:
        raise ValueError("No non-target groups defined in SAMPLE_GROUPS.")

    group_names = sorted(non_target_groups.keys())
    base_per_group = remaining_slots // len(group_names)
    extras = remaining_slots % len(group_names)

    for i, group_name in enumerate(group_names):
        n_from_group = base_per_group + (1 if i < extras else 0)
        pool = non_target_groups[group_name]

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


def select_random_files(
    n_files: int = N_FILES_TO_PROCESS,
    all_input_files: List[str] | None = None,
    target_files: List[str] | None = None,
) -> List[str]:
    """
    Select n_files for mass grouping.

    If target_files are provided, exactly one is guaranteed to be included.
    The remaining slots are filled randomly from the non-target files.

    Parameters
    ----------
    all_input_files
        Full list of input file paths supplied by the GUI (Option B).
        Falls back to scanning BASE_DIR/INPUT_SUBDIR when None.
    target_files
        Paths of pooled-control / target files.  One will always be picked
        when this list is non-empty.
    """
    target_files = target_files or []

    if all_input_files is not None:
        # GUI-supplied file list
        target_set = set(target_files)
        non_target = [f for f in all_input_files if f not in target_set]
    else:
        # Fallback: scan the input directory
        found = sorted((BASE_DIR / INPUT_SUBDIR).glob("*.mzXML"))
        target_set = set(target_files)
        non_target = [str(f) for f in found if str(f) not in target_set]

    selected: List[str] = []

    # Guarantee one target file is included
    if target_files:
        chosen_target = RNG.choice(target_files)
        selected.append(chosen_target)
        print(f"  [target]  {Path(chosen_target).name}")
        n_files -= 1

    if len(non_target) < n_files:
        raise ValueError(
            f"Not enough non-target files for mass grouping. "
            f"Found {len(non_target)}, need {n_files} "
            f"(1 slot already filled by target)."
        )

    for f in RNG.sample(non_target, n_files):
        selected.append(f)
        print(f"  {Path(f).name}")

    return selected


def select_files(
    all_input_files: List[str] | None = None,
    target_files: List[str] | None = None,
) -> List[str]:
    """
    Entry point for file selection before mass grouping.

    Parameters
    ----------
    all_input_files
        Full GUI file list (Option B).  Ignored when USE_STUDY_DESIGN is True.
    target_files
        Pooled-control file paths.  When provided (and USE_STUDY_DESIGN is
        False) one is always included in the grouping selection.
    """
    # Resolve target_files from Config if not passed explicitly
    if target_files is None:
        try:
            from Config import Config
            target_files = list(Config.TARGET_FILES) if Config.TARGET_FILES else []
        except Exception:
            target_files = []

    # Only reuse the manifest if the current file list and target files are
    # consistent with what was selected before.  When the GUI supplies a fresh
    # file list we always detect fresh — the manifest is for the standalone
    # script workflow only.
    if all_input_files is None and not target_files:
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
        selected = select_random_files(
            N_FILES_TO_PROCESS,
            all_input_files=all_input_files,
            target_files=target_files,
        )

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


class _Cluster:
    """Lightweight cluster used only inside find_mass_traces."""
    __slots__ = ("scan_indices", "mz_sum", "mz_count", "rep_mz",
                 "intensities", "max_intensity", "apex_scan",
                 "rt_values", "rt_min", "rt_max")

    def __init__(self, scan_idx: int, mz: float, intensity: float, rt: float):
        self.scan_indices: List[int] = [scan_idx]
        self.mz_sum = mz
        self.mz_count = 1
        self.rep_mz = mz
        self.intensities = {scan_idx: intensity}
        self.max_intensity = intensity
        self.apex_scan = scan_idx
        self.rt_values = {scan_idx: rt}
        self.rt_min = rt
        self.rt_max = rt

    def add(self, scan_idx: int, mz: float, intensity: float, rt: float):
        self.scan_indices.append(scan_idx)
        self.mz_sum += mz
        self.mz_count += 1
        self.rep_mz = self.mz_sum / self.mz_count
        self.intensities[scan_idx] = intensity
        if intensity > self.max_intensity:
            self.max_intensity = intensity
            self.apex_scan = scan_idx
        self.rt_values[scan_idx] = rt
        if rt < self.rt_min:
            self.rt_min = rt
        if rt > self.rt_max:
            self.rt_max = rt


def find_mass_traces(
    centroids: List[Tuple[int, float, float]],
    mz_tol: float,
    min_consec_scans: int,
    retention_times: Dict[int, float],
) -> List[Dict]:
    if not centroids:
        return []

    centroids.sort(key=lambda x: x[1])

    clusters: List[_Cluster] = []
    rep_mzs = np.empty(0, dtype=np.float64)

    for scan_idx, mz, intensity in centroids:
        rt = retention_times[scan_idx]

        pos = np.searchsorted(rep_mzs, mz)

        placed = False
        for idx in (pos - 1, pos):
            if 0 <= idx < len(clusters):
                if abs(clusters[idx].rep_mz - mz) <= mz_tol:
                    old_rep = clusters[idx].rep_mz
                    clusters[idx].add(scan_idx, mz, intensity, rt)
                    new_rep = clusters[idx].rep_mz
                    rep_mzs[idx] = new_rep
                    placed = True
                    break

        if not placed:
            new_cluster = _Cluster(scan_idx, mz, intensity, rt)
            clusters.insert(pos, new_cluster)
            rep_mzs = np.insert(rep_mzs, pos, mz)

    valid_clusters: List[Dict] = []
    for cluster in clusters:
        scans = sorted(cluster.scan_indices)
        if len(scans) < min_consec_scans:
            continue
        arr = np.array(scans, dtype=np.int32)
        gaps = np.diff(arr)
        run_starts = np.where(gaps != 1)[0] + 1
        splits = np.concatenate(([0], run_starts, [len(arr)]))
        max_run = int(np.diff(splits).max())
        if max_run >= min_consec_scans:
            apex_rt = cluster.rt_values[cluster.apex_scan]
            valid_clusters.append({
                'mz': cluster.rep_mz,
                'rt': apex_rt,
                'rt_start': cluster.rt_min,
                'rt_end': cluster.rt_max,
                'consecutive_scans': max_run,
                'intensity': cluster.max_intensity,
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
    noise_level: float = None,
    mz_tolerance: float = None,
    min_consec_scans: int = None,
    verbose: bool = False,
) -> Tuple[str, Set[float], List[Dict]]:
    # Always resolve from Config so the GUI-set values are used in workers
    try:
        from Config import Config
        if noise_level is None:
            noise_level = Config.GROUP_NOISE_LEVEL
        if mz_tolerance is None:
            mz_tolerance = Config.GROUP_MZ_TOLERANCE
        if min_consec_scans is None:
            min_consec_scans = Config.GROUP_MIN_CONSEC_SCANS
    except Exception:
        if noise_level is None:
            noise_level = NOISE_LEVEL
        if mz_tolerance is None:
            mz_tolerance = MZ_TOLERANCE
        if min_consec_scans is None:
            min_consec_scans = MIN_CONSEC_SCANS

    if verbose:
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


def check_mass_compatibility_vec(
    group_mzs: np.ndarray,
    group_rt_starts: np.ndarray,
    group_rt_ends: np.ndarray,
    group_rts: np.ndarray,
    group_intensities: np.ndarray,
    new_mass: Dict,
    max_rt_window: float = 15.0,
    min_intensity_ratio: float = 0.90,
    min_ppm_diff: float = 5.0,
) -> bool:
    if len(group_mzs) == 0:
        return True

    new_mz = new_mass['mz']
    new_rt = new_mass['rt']
    new_rt_start = new_mass['rt_start']
    new_rt_end = new_mass['rt_end']
    new_intensity = new_mass['intensity']

    ppm_diffs = np.abs(new_mz - group_mzs) / group_mzs * 1e6
    if np.any(ppm_diffs < min_ppm_diff):
        return False

    overlaps = (new_rt_start <= group_rt_ends) & (new_rt_end >= group_rt_starts)
    if np.any(overlaps):
        return False

    if np.any(np.abs(new_rt - group_rts) < 1.0):
        return False

    lo = np.minimum(new_intensity, group_intensities)
    hi = np.maximum(new_intensity, group_intensities)
    ratios = lo / hi
    if np.any(ratios < min_intensity_ratio):
        return False

    return True


def find_groups(masses_df: pd.DataFrame, min_group_size: int = 3, max_group_size: int = 5) -> List[List[Dict]]:
    sorted_df = masses_df.sort_values('intensity', ascending=False)

    seen_mzs: List[float] = []
    available_masses: List[Dict] = []
    for row in sorted_df.itertuples():
        if any(is_same_mass(row.mz, s) for s in seen_mzs):
            continue
        available_masses.append({
            'mz': row.mz,
            'rt': row.rt,
            'rt_start': row.rt_start,
            'rt_end': row.rt_end,
            'intensity': row.intensity,
        })
        seen_mzs.append(row.mz)

    n = len(available_masses)
    if n == 0:
        return []

    mzs       = np.array([m['mz']        for m in available_masses], dtype=np.float64)
    rts       = np.array([m['rt']        for m in available_masses], dtype=np.float64)
    rt_starts = np.array([m['rt_start']  for m in available_masses], dtype=np.float64)
    rt_ends   = np.array([m['rt_end']    for m in available_masses], dtype=np.float64)
    intensities = np.array([m['intensity'] for m in available_masses], dtype=np.float64)

    available = np.ones(n, dtype=bool)

    groups: List[List[Dict]] = []

    def _try_fill_group(group_indices, g_mzs, g_rt_starts, g_rt_ends, g_rts, g_intensities,
                        min_intensity_ratio, min_ppm_diff):
        for cand_idx in range(seed_idx + 1, n):
            if not available[cand_idx] or cand_idx in group_indices:
                continue
            if len(group_indices) >= max_group_size:
                break
            cand = available_masses[cand_idx]
            if check_mass_compatibility_vec(
                g_mzs, g_rt_starts, g_rt_ends, g_rts, g_intensities, cand,
                min_intensity_ratio=min_intensity_ratio,
                min_ppm_diff=min_ppm_diff,
            ):
                group_indices.append(cand_idx)
                g_mzs         = np.append(g_mzs,         mzs[cand_idx])
                g_rt_starts   = np.append(g_rt_starts,   rt_starts[cand_idx])
                g_rt_ends     = np.append(g_rt_ends,     rt_ends[cand_idx])
                g_rts         = np.append(g_rts,         rts[cand_idx])
                g_intensities = np.append(g_intensities, intensities[cand_idx])
        return group_indices, g_mzs, g_rt_starts, g_rt_ends, g_rts, g_intensities

    for seed_idx in range(n):
        if not available[seed_idx]:
            continue

        group_indices = [seed_idx]
        g_mzs         = mzs[seed_idx:seed_idx+1].copy()
        g_rt_starts   = rt_starts[seed_idx:seed_idx+1].copy()
        g_rt_ends     = rt_ends[seed_idx:seed_idx+1].copy()
        g_rts         = rts[seed_idx:seed_idx+1].copy()
        g_intensities = intensities[seed_idx:seed_idx+1].copy()

        group_indices, g_mzs, g_rt_starts, g_rt_ends, g_rts, g_intensities = _try_fill_group(
            group_indices, g_mzs, g_rt_starts, g_rt_ends, g_rts, g_intensities,
            min_intensity_ratio=0.90, min_ppm_diff=5.0,
        )

        if len(group_indices) < min_group_size:
            group_indices, g_mzs, g_rt_starts, g_rt_ends, g_rts, g_intensities = _try_fill_group(
                group_indices, g_mzs, g_rt_starts, g_rt_ends, g_rts, g_intensities,
                min_intensity_ratio=0.65, min_ppm_diff=2.0,
            )

        for idx in group_indices:
            available[idx] = False

        groups.append([available_masses[i] for i in group_indices])

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

    # Hard noise floor: drop any feature whose max intensity is below the
    # noise level before grouping.  This catches clusters that scraped past
    # the per-scan filter but whose overall intensity is still sub-noise.
    above_noise = {
        mz: feat for mz, feat in unique_features.items()
        if feat['intensity'] >= noise_level
    }
    n_dropped = len(unique_features) - len(above_noise)
    if n_dropped:
        print(f"[MassGrouping] Dropped {n_dropped} feature(s) below noise floor "
              f"({noise_level:.0f}) before grouping.")
    unique_features = above_noise

    features_sorted = sorted(unique_features.values(), key=lambda x: x['mz'])
    df = pd.DataFrame(features_sorted)

    mass_groups = find_groups(df, min_group_size=min_group_size, max_group_size=max_group_size)

    out: Dict[str, List[float]] = {}
    out_intensities: Dict[str, List[float]] = {}
    for i, group in enumerate(mass_groups, 1):
        out[f"Group{i}"] = [float(m["mz"]) for m in group]
        out_intensities[f"Group{i}"] = [float(m["intensity"]) for m in group]

    return out, out_intensities

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