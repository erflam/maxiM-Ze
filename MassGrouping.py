import bisect
import random
import numpy as np
import pandas as pd
from pathlib import Path
from pyteomics import mzxml
from typing import List, Tuple, Set, Dict
from collections import defaultdict
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time

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
MIN_SAMPLE_PRESENCE = 2

# Mass grouping parameters
# IMPORTANT:
# RT grouping rule below means APEX RT must be AT LEAST 1.0 minute apart.
MIN_GROUP_SIZE = 3
MAX_GROUP_SIZE = 5
RELAXED_MIN_GROUP_SIZE = 2
RELAXED_MAX_GROUP_SIZE = 5

GROUP_RT_APEX_SEPARATION = 1.0       # minutes; apex RT difference MUST be >= 1.0
GROUP_MIN_INTENSITY_RATIO = 0.55     # keep at 0.55
GROUP_REQUIRE_NO_RT_OVERLAP = False  # False = only enforce apex RT >= 1.0 min.
                                     # True can block many otherwise valid RT-separated groups.

USE_STUDY_DESIGN = False
N_FILES_TO_PROCESS = 6
TARGET_GROUP = "target"  # set to None if no target group

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
    target_files = target_files or []

    if all_input_files is not None:
        target_set = set(target_files)
        non_target = [f for f in all_input_files if f not in target_set]
    else:
        found = sorted((BASE_DIR / INPUT_SUBDIR).glob("*.mzXML"))
        target_set = set(target_files)
        non_target = [str(f) for f in found if str(f) not in target_set]

    selected: List[str] = []

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
    if target_files is None:
        try:
            from Config import Config
            target_files = list(Config.TARGET_FILES) if Config.TARGET_FILES else []
        except Exception:
            target_files = []

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


def centroid_scan(scan_idx: int, mzs: np.ndarray, intensities: np.ndarray, noise_level: float) -> List[
    Tuple[int, float, float]]:
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
                    clusters[idx].add(scan_idx, mz, intensity, rt)
                    rep_mzs[idx] = clusters[idx].rep_mz
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


def process_file_centroids(file_path: str, noise_level: float) -> Tuple[
    List[Tuple[int, float, float]], Dict[int, float]]:
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
        future_to_file = {executor.submit(process_single_file, f): f for f in selected_files}
        raw_results: Dict[str, tuple] = {}
        for fut, f in future_to_file.items():
            raw_results[f] = fut.result()

    for f in selected_files:
        file_path, masses, features = raw_results[f]
        masses_by_file[file_path] = masses
        all_features.extend(features)

    mass_counts = defaultdict(int)
    for f in selected_files:
        for mz in sorted(masses_by_file[f]):
            mass_counts[round(mz, 6)] += 1

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


def build_compatibility_matrix(
        mass_data: np.ndarray,
        rt_apex_separation: float,
        min_intensity_ratio: float,
        require_no_rt_overlap: bool,
) -> np.ndarray:
    """
    Precompute compatibility once.

    mass_data columns:
        [rt, intensity, mz, rt_start, rt_end]

    Required rule:
        abs(RT_i - RT_j) >= 1.0 minute

    Optional:
        If require_no_rt_overlap=True, RT windows must also not overlap.
        Default is False because the user's stated rule is apex RT >= 1 minute.
    """
    rts = mass_data[:, 0]
    intensities = mass_data[:, 1]
    rt_starts = mass_data[:, 3]
    rt_ends = mass_data[:, 4]

    rt_ok = np.abs(rts[:, None] - rts[None, :]) >= rt_apex_separation

    intensity_ratio = (
        np.minimum(intensities[:, None], intensities[None, :]) /
        np.maximum(intensities[:, None], intensities[None, :])
    )
    intensity_ok = intensity_ratio >= min_intensity_ratio

    if require_no_rt_overlap:
        overlap = (
            (rt_starts[:, None] <= rt_ends[None, :]) &
            (rt_starts[None, :] <= rt_ends[:, None])
        )
        rt_window_ok = ~overlap
    else:
        rt_window_ok = np.ones_like(rt_ok, dtype=bool)

    compat = rt_ok & intensity_ok & rt_window_ok
    np.fill_diagonal(compat, False)
    return compat


def group_from_available_seed_repack(
        available: np.ndarray,
        available_masses: List[Dict],
        mass_data: np.ndarray,
        compat_matrix: np.ndarray,
        min_group_size: int,
        max_group_size: int,
) -> List[List[Dict]]:
    """
    Fast seed-repack grouping.

    This does exactly the requested logic:
    1. Start from highest-intensity remaining mass.
    2. Move down the intensity list.
    3. Add masses that fit tolerances with the seed.
    4. Skip masses that do not fit; they stay available.
    5. Once a group is made, remove those masses.
    6. Restart from the top.
    7. Stop when no more groups can be made.
    """
    groups: List[List[Dict]] = []
    n = len(available)

    while True:
        made_group = False

        for seed_idx in range(n):
            if not available[seed_idx]:
                continue

            compatible = np.where(compat_matrix[seed_idx] & available)[0]

            if len(compatible) + 1 < min_group_size:
                continue

            candidate_indices = np.concatenate(([seed_idx], compatible))

            # Keep highest intensity candidates.
            # Because available_masses/mass_data are already intensity sorted,
            # this preserves the user's "go down the list" behavior.
            final_indices = sorted(
                candidate_indices,
                key=lambda i: mass_data[i][1],
                reverse=True
            )[:max_group_size]

            for idx in final_indices:
                available[idx] = False

            groups.append([available_masses[i] for i in final_indices])
            made_group = True

            # Restart from top after successful grouping.
            break

        if not made_group:
            break

    return groups


def find_groups_two_tier(
        masses_df: pd.DataFrame,
        min_group_size: int = MIN_GROUP_SIZE,
        max_group_size: int = MAX_GROUP_SIZE,
        rt_apex_separation: float = GROUP_RT_APEX_SEPARATION,
        min_intensity_ratio: float = GROUP_MIN_INTENSITY_RATIO,
        require_no_rt_overlap: bool = GROUP_REQUIRE_NO_RT_OVERLAP,
) -> Tuple[List[List[Dict]], List[List[Dict]]]:
    """
    FAST grouping strategy.

    Phase 1:
        Build 3-5 groups.

    Phase 2:
        Repack leftovers into 2-5 groups using the same tolerances.

    Phase 3:
        Remaining masses become true singleton groups.
    """
    start_time = time.time()

    if masses_df.empty:
        return [], []

    sorted_df = masses_df.sort_values('intensity', ascending=False)

    seen_mzs_sorted: List[float] = []
    available_masses: List[Dict] = []
    _tol = MZ_TOLERANCE

    for row in sorted_df.itertuples():
        mz = float(row.mz)
        pos = bisect.bisect_left(seen_mzs_sorted, mz - _tol)

        duplicate = False
        for i in range(pos, len(seen_mzs_sorted)):
            if seen_mzs_sorted[i] > mz + _tol:
                break
            if abs(seen_mzs_sorted[i] - mz) <= _tol:
                duplicate = True
                break

        if duplicate:
            continue

        bisect.insort(seen_mzs_sorted, mz)
        available_masses.append({
            'mz': mz,
            'rt': float(row.rt),
            'rt_start': float(row.rt_start),
            'rt_end': float(row.rt_end),
            'intensity': float(row.intensity),
        })

    n = len(available_masses)
    if n == 0:
        return [], []

    mass_data = np.array([
        [m['rt'], m['intensity'], m['mz'], m['rt_start'], m['rt_end']]
        for m in available_masses
    ], dtype=np.float64)

    print("  Building compatibility matrix...")
    print(f"    Required apex RT separation: >= {rt_apex_separation:.4f} min")
    print(f"    Required intensity ratio:    >= {min_intensity_ratio:.4f}")
    print(f"    Require no RT window overlap: {require_no_rt_overlap}")

    compat_matrix = build_compatibility_matrix(
        mass_data=mass_data,
        rt_apex_separation=rt_apex_separation,
        min_intensity_ratio=min_intensity_ratio,
        require_no_rt_overlap=require_no_rt_overlap,
    )

    available = np.ones(n, dtype=bool)

    print(f"  Phase 1: Building {MIN_GROUP_SIZE}-{MAX_GROUP_SIZE} mass groups...")

    strict_groups = group_from_available_seed_repack(
        available=available,
        available_masses=available_masses,
        mass_data=mass_data,
        compat_matrix=compat_matrix,
        min_group_size=MIN_GROUP_SIZE,
        max_group_size=MAX_GROUP_SIZE,
    )

    print(f"  Phase 2: Repacking leftovers into {RELAXED_MIN_GROUP_SIZE}-{RELAXED_MAX_GROUP_SIZE} mass groups...")

    relaxed_multi_groups = group_from_available_seed_repack(
        available=available,
        available_masses=available_masses,
        mass_data=mass_data,
        compat_matrix=compat_matrix,
        min_group_size=RELAXED_MIN_GROUP_SIZE,
        max_group_size=RELAXED_MAX_GROUP_SIZE,
    )

    print("  Phase 3: Remaining true singletons...")

    singleton_groups = [
        [available_masses[i]]
        for i in np.where(available)[0]
    ]

    relaxed_groups = relaxed_multi_groups + singleton_groups

    elapsed = time.time() - start_time

    print(f"  Grouping completed in {elapsed:.2f} seconds")
    print(f"    Strict 3-5 groups:     {len(strict_groups)}")
    print(f"    Relaxed 2-5 groups:    {len(relaxed_multi_groups)}")
    print(f"    Singleton groups:      {len(singleton_groups)}")

    return strict_groups, relaxed_groups


def build_mass_groups_from_files(
        file_paths: List[str],
        noise_level: float,
        mz_tolerance: float,
        min_consec_scans: int,
        min_sample_presence: int,
        min_group_size: int,
        max_group_size: int,
        verbose: bool = True,
) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """Build mass groups and return masses, intensities, RTs, RT starts, and RT ends."""
    masses_by_file: Dict[str, Set[float]] = {}
    all_features: List[Dict] = []

    print("Processing files in parallel...")
    with ProcessPoolExecutor(max_workers=min(len(file_paths), multiprocessing.cpu_count())) as executor:
        future_to_file = {
            executor.submit(process_single_file, f, noise_level, mz_tolerance, min_consec_scans, verbose): f
            for f in file_paths
        }
        raw_results: Dict[str, tuple] = {}
        for fut, f in future_to_file.items():
            raw_results[f] = fut.result()

    for f in file_paths:
        file_path, masses, features = raw_results[f]
        masses_by_file[file_path] = masses
        all_features.extend(features)

    mass_counts = defaultdict(int)
    for f in file_paths:
        for mz in sorted(masses_by_file[f]):
            mass_counts[round(mz, 6)] += 1

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

    above_noise = {
        mz: feat for mz, feat in unique_features.items()
        if feat['intensity'] >= noise_level
    }
    n_dropped = len(unique_features) - len(above_noise)
    if n_dropped:
        print(f"[MassGrouping] Dropped {n_dropped} feature(s) below noise floor ({noise_level:.0f})")
    unique_features = above_noise

    features_sorted = sorted(unique_features.values(), key=lambda x: x['mz'])
    df = pd.DataFrame(features_sorted)

    print(f"\nGrouping {len(df)} eligible masses...")
    strict_groups, relaxed_groups = find_groups_two_tier(
        df,
        min_group_size=min_group_size,
        max_group_size=max_group_size,
        rt_apex_separation=GROUP_RT_APEX_SEPARATION,
        min_intensity_ratio=GROUP_MIN_INTENSITY_RATIO,
        require_no_rt_overlap=GROUP_REQUIRE_NO_RT_OVERLAP,
    )

    all_groups = strict_groups + relaxed_groups

    print("  Sorting groups by total intensity...")
    all_groups_sorted = sorted(
        all_groups,
        key=lambda g: sum(m['intensity'] for m in g),
        reverse=True
    )

    out: Dict[str, List[float]] = {}
    out_intensities: Dict[str, List[float]] = {}
    out_rts: Dict[str, List[float]] = {}
    out_rt_starts: Dict[str, List[float]] = {}
    out_rt_ends: Dict[str, List[float]] = {}

    for i, group in enumerate(all_groups_sorted, 1):
        group_name = f"Group{i}"
        out[group_name] = [float(m["mz"]) for m in group]
        out_intensities[group_name] = [float(m["intensity"]) for m in group]
        out_rts[group_name] = [float(m["rt"]) for m in group]
        out_rt_starts[group_name] = [float(m["rt_start"]) for m in group]
        out_rt_ends[group_name] = [float(m["rt_end"]) for m in group]

    return out, out_intensities, out_rts, out_rt_starts, out_rt_ends


def build_formatted_groups_dataframe(all_groups_sorted: List[List[Dict]], strict_groups: List[List[Dict]]) -> pd.DataFrame:
    """Create a wide group table with Mass1, RT1, Intensity1, Mass2, RT2, etc."""
    strict_group_ids = {id(g) for g in strict_groups}
    max_group_size = max((len(g) for g in all_groups_sorted), default=0)

    rows = []
    for i, group in enumerate(all_groups_sorted, 1):
        group_type = 'Strict' if id(group) in strict_group_ids else 'Relaxed/Singleton'
        row = {
            'Group': f"'Group {i}':",
            'Type': group_type,
            'Size': len(group),
            'Masses': '[' + ', '.join(f"{float(m['mz']):.4f}" for m in group) + ']',
            'Intensities': '[' + ', '.join(f"{float(m['intensity']):.0f}" for m in group) + ']',
            'Total Intensity': sum(float(m['intensity']) for m in group),
        }

        for j in range(max_group_size):
            k = j + 1
            if j < len(group):
                mass = group[j]
                row[f'Mass{k}'] = float(mass['mz'])
                row[f'RT{k}'] = float(mass['rt'])
                row[f'RT_Start{k}'] = float(mass['rt_start'])
                row[f'RT_End{k}'] = float(mass['rt_end'])
                row[f'Intensity{k}'] = float(mass['intensity'])
            else:
                row[f'Mass{k}'] = ''
                row[f'RT{k}'] = ''
                row[f'RT_Start{k}'] = ''
                row[f'RT_End{k}'] = ''
                row[f'Intensity{k}'] = ''

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    overall_start = time.time()

    features = process_files()

    output_dir = BASE_DIR / OUTPUT_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(features)
    top_50 = df.nlargest(50, 'intensity')

    output_file = output_dir / 'mass_features_FINAL.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All Features', index=False)
        top_50.to_excel(writer, sheet_name='Top 50 Intense Peaks', index=False)

    print(f"\nFound {len(features)} features")
    print(f"Results saved to {output_file}")

    if len(features) > 0:
        print("\nData ranges:")
        print(f"m/z:       {df['mz'].min():.4f} - {df['mz'].max():.4f}")
        print(f"RT:        {df['rt'].min():.2f} - {df['rt'].max():.2f} min")
        print(f"Intensity: {df['intensity'].min():.0f} - {df['intensity'].max():.0f}")

    strict_groups, relaxed_groups = find_groups_two_tier(
        df,
        min_group_size=MIN_GROUP_SIZE,
        max_group_size=MAX_GROUP_SIZE,
        rt_apex_separation=GROUP_RT_APEX_SEPARATION,
        min_intensity_ratio=GROUP_MIN_INTENSITY_RATIO,
        require_no_rt_overlap=GROUP_REQUIRE_NO_RT_OVERLAP,
    )

    all_groups = strict_groups + relaxed_groups
    all_groups_sorted = sorted(
        all_groups,
        key=lambda g: sum(m['intensity'] for m in g),
        reverse=True
    )

    group_data = []
    total_grouped_masses = 0
    all_used_masses: List[float] = []

    strict_group_set = {id(g) for g in strict_groups}

    for i, group in enumerate(all_groups_sorted, 1):
        group_type = 'Strict' if id(group) in strict_group_set else 'Relaxed/Singleton'

        for mass in group:
            group_data.append({
                'Group': f'Group {i}',
                'Type': group_type,
                'Size': len(group),
                'Mass (m/z)': mass['mz'],
                'RT (min)': mass['rt'],
                'RT Start (min)': mass['rt_start'],
                'RT End (min)': mass['rt_end'],
                'Intensity': mass['intensity']
            })
            all_used_masses.append(mass['mz'])
            total_grouped_masses += 1

    groups_df = pd.DataFrame(group_data)
    formatted_groups_df = build_formatted_groups_dataframe(all_groups_sorted, strict_groups)

    formatted_csv = output_dir / 'MassGroups_Formatted.csv'
    formatted_groups_df.to_csv(formatted_csv, index=False)

    strict_sizes = [len(g) for g in strict_groups]
    relaxed_sizes = [len(g) for g in relaxed_groups]

    print(f"\n=== GROUPING SUMMARY ===")
    print(f"Total eligible masses: {len(df)}")
    print(f"Total masses grouped: {total_grouped_masses}")
    print(f"Coverage: {100 * total_grouped_masses / len(df):.1f}%")
    print(f"\nGroups sorted by total intensity (highest first)")
    print(f"Group 1 = highest total intensity")

    print(f"\nStrict groups ({MIN_GROUP_SIZE}-{MAX_GROUP_SIZE} masses): {len(strict_groups)}")
    if strict_sizes:
        print(f"  Total masses in strict groups: {sum(strict_sizes)}")
        print(f"  Average size: {sum(strict_sizes) / len(strict_sizes):.1f}")
        size_counts = pd.Series(strict_sizes).value_counts().sort_index()
        for size, count in size_counts.items():
            print(f"    Size {size}: {count} group(s)")

    print(f"\nRelaxed/Singleton groups: {len(relaxed_groups)}")
    if relaxed_sizes:
        print(f"  Total masses in relaxed/singleton groups: {sum(relaxed_sizes)}")
        size_counts = pd.Series(relaxed_sizes).value_counts().sort_index()
        for size, count in size_counts.items():
            print(f"    Size {size}: {count} group(s)")

    ungrouped = len(df) - total_grouped_masses
    if ungrouped > 0:
        print(f"\n⚠ WARNING: {ungrouped} masses were not grouped!")
    else:
        print(f"\n✓ All {len(df)} eligible masses were grouped!")

    print(f"\n=== TOP 10 GROUPS BY TOTAL INTENSITY ===")
    group_summary = groups_df.groupby('Group').agg({
        'Intensity': 'sum',
        'Size': 'first',
        'Type': 'first'
    }).reset_index()
    group_summary = group_summary.sort_values('Intensity', ascending=False).head(10)
    for _, row in group_summary.iterrows():
        print(f"{row['Group']}: {row['Intensity']:.0f} (Size={row['Size']}, Type={row['Type']})")

    print(f"\n=== VALIDATING GROUPS ===")
    print(
        f"Criteria: apex RT separation >= {GROUP_RT_APEX_SEPARATION} min, "
        f"Intensity ratio >= {GROUP_MIN_INTENSITY_RATIO:.0%}, "
        f"No RT window overlap={GROUP_REQUIRE_NO_RT_OVERLAP}, "
        f"Strict size {MIN_GROUP_SIZE}-{MAX_GROUP_SIZE}, "
        f"Relaxed size {RELAXED_MIN_GROUP_SIZE}-{RELAXED_MAX_GROUP_SIZE}"
    )

    overlap_count = 0
    intensity_violations = 0
    rt_sep_violations = 0

    for group_name in groups_df['Group'].unique():
        group_masses = groups_df[groups_df['Group'] == group_name]

        if len(group_masses) == 1:
            continue

        for i, row1 in group_masses.iterrows():
            for j, row2 in group_masses.iterrows():
                if i >= j:
                    continue

                rt_diff = abs(row1['RT (min)'] - row2['RT (min)'])
                if rt_diff < GROUP_RT_APEX_SEPARATION:
                    print(f"  ⚠ {group_name}: RT apex separation {rt_diff:.4f} < {GROUP_RT_APEX_SEPARATION}")
                    rt_sep_violations += 1

                if GROUP_REQUIRE_NO_RT_OVERLAP and (row1['RT Start (min)'] <= row2['RT End (min)']) and (
                        row2['RT Start (min)'] <= row1['RT End (min)']):
                    print(f"  ⚠ {group_name}: RT overlap!")
                    overlap_count += 1

                ratio = min(row1['Intensity'], row2['Intensity']) / max(row1['Intensity'], row2['Intensity'])
                if ratio < GROUP_MIN_INTENSITY_RATIO:
                    print(f"  ⚠ {group_name}: Intensity ratio {ratio:.2%} < {GROUP_MIN_INTENSITY_RATIO:.0%}")
                    intensity_violations += 1

    if overlap_count == 0 and intensity_violations == 0 and rt_sep_violations == 0:
        print("  ✓ All groups pass validation!")

    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        groups_df.to_excel(writer, sheet_name='Mass Groups', index=False)
        formatted_groups_df.to_excel(writer, sheet_name='Mass Groups Formatted', index=False)

    overall_elapsed = time.time() - overall_start
    print(f"\n=== TOTAL RUNTIME: {overall_elapsed:.2f} seconds ===")
    print(f"Results saved to {output_file}")
    print(f"Formatted CSV saved to {formatted_csv}")
    print(f"Selection manifest: {_manifest_path()}")


if __name__ == "__main__":
    main()