from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil
import numpy as np
import pandas as pd
import cv2
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import re

SAVGOL_WINDOW_FRAC = 0.025
SAVGOL_POLY = 3
PROM_FRAC = 0.05
MIN_SEP_FRAC = 0.10
NOISE_DIST_PIXELS = 15
NOISE_INT_DIFF_RATIO = 0.10
DEBUG_PRINT = True

def _smooth_1d(y: np.ndarray) -> np.ndarray:
    n = len(y)
    if n < 5:
        return y.astype(np.float64, copy=False)

    win = max(5, int(round(SAVGOL_WINDOW_FRAC * n)))
    # force odd
    win = win | 1
    if win >= n:
        win = (n - 1) | 1
    if win < 5:
        return y.astype(np.float64, copy=False)

    poly = min(SAVGOL_POLY, win - 1)
    return savgol_filter(y.astype(np.float64, copy=False), window_length=win, polyorder=poly)

def detect_peaks(profile_smooth: np.ndarray,
                 prom_frac: float = PROM_FRAC,
                 min_height_frac: float = 0.05,
                 min_sep_frac: float = MIN_SEP_FRAC) -> np.ndarray:
    if profile_smooth.size == 0:
        return np.array([], dtype=int)

    prom = max(prom_frac * (profile_smooth.max() - profile_smooth.min()), 1.0)
    dist = max(1, int(min_sep_frac * len(profile_smooth)))
    min_height = min_height_frac * float(profile_smooth.max())

    peaks, props = find_peaks(profile_smooth, prominence=prom, distance=dist, height=min_height)
    return np.array(peaks, dtype=int)

def filter_peaks(peaks: np.ndarray, profile_smooth: np.ndarray) -> np.ndarray:
    if peaks is None or len(peaks) <= 1:
        return np.array(peaks if peaks is not None else [], dtype=int)

    peaks = np.array(peaks, dtype=int)
    final_peaks: List[int] = []
    cluster: List[int] = [int(peaks[0])]

    for i in range(1, len(peaks)):
        prev_peak = cluster[-1]
        curr_peak = int(peaks[i])

        if (curr_peak - prev_peak <= NOISE_DIST_PIXELS and
            abs(profile_smooth[curr_peak] - profile_smooth[prev_peak]) <=
            NOISE_INT_DIFF_RATIO * float(profile_smooth.max())):
            cluster.append(curr_peak)
        else:
            rep = max(cluster, key=lambda p: profile_smooth[p])
            final_peaks.append(int(rep))
            cluster = [curr_peak]

    if cluster:
        rep = max(cluster, key=lambda p: profile_smooth[p])
        final_peaks.append(int(rep))

    return np.array(final_peaks, dtype=int)

def _interpolate_nans_1d(y: np.ndarray) -> np.ndarray:
    x = np.arange(len(y))
    mask = ~np.isnan(y)
    if mask.sum() == 0:
        return np.zeros_like(y, dtype=np.float64)
    y2 = y.astype(np.float64, copy=True)
    y2[~mask] = np.interp(x[~mask], x[mask], y2[mask])
    return y2

def extract_height_profile_from_alpha(img_rgba: np.ndarray) -> np.ndarray:
    if img_rgba.ndim != 3 or img_rgba.shape[2] < 4:
        raise ValueError("RGBA image required for alpha-based profiling.")

    alpha = img_rgba[:, :, 3]
    a = (alpha > 0).astype(np.uint8) * 255

    # light cleanup
    a = cv2.medianBlur(a, 3)
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    H, W = a.shape
    y_med = np.full(W, np.nan, dtype=np.float64)

    for x in range(W):
        ys = np.where(a[:, x] > 0)[0]
        if ys.size:
            y_med[x] = float(np.median(ys))

    y_med = _interpolate_nans_1d(y_med)
    height_from_bottom = (H - 1) - y_med
    return height_from_bottom.astype(np.float64)

def _profile_and_peaks(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if img.ndim == 3 and img.shape[2] == 4:
        profile_raw = extract_height_profile_from_alpha(img)
    else:
        # fallback: grayscale sum profile
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        profile_raw = np.sum(gray.astype(np.float64), axis=0)

    profile_smooth = _smooth_1d(profile_raw)

    peaks = detect_peaks(profile_smooth, prom_frac=PROM_FRAC, min_height_frac=0.05, min_sep_frac=MIN_SEP_FRAC)
    peaks = filter_peaks(peaks, profile_smooth)

    # fallback sensitivity if we didn't get enough peaks
    if peaks is None or len(peaks) < 2:
        peaks2 = detect_peaks(profile_smooth, prom_frac=0.01, min_height_frac=0.01, min_sep_frac=0.05)
        peaks2 = filter_peaks(peaks2, profile_smooth)
        if peaks2 is not None and len(peaks2) >= 2:
            peaks = peaks2

    return profile_smooth, (np.array(peaks, dtype=int) if peaks is not None else np.array([], dtype=int))

def find_split_index_from_profile(profile_smooth: np.ndarray, peaks: np.ndarray) -> int:
    """
    Given >=2 peaks, find the lowest valley between adjacent peaks.
    """
    W = len(profile_smooth)
    if W < 3:
        return max(1, W // 2)

    peaks = np.sort(np.array(peaks, dtype=int))
    valleys: List[int] = []

    for i in range(len(peaks) - 1):
        left, right = int(peaks[i]), int(peaks[i + 1])
        if right <= left + 1:
            continue
        seg = profile_smooth[left:right]
        v = left + int(np.argmin(seg))
        valleys.append(int(v))

    if valleys:
        # choose the *lowest* valley overall
        return int(min(valleys, key=lambda idx: profile_smooth[idx]))

    # last fallback: lowest point in central region
    lo = int(0.05 * W)
    hi = int(0.95 * W)
    if hi > lo + 2:
        return int(lo + int(np.argmin(profile_smooth[lo:hi])))

    return int(W // 2)

def _split_once_by_valley(img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
    W = img.shape[1]
    if W < 3:
        return img, None, False

    profile_smooth, peaks = _profile_and_peaks(img)
    if peaks is None or len(peaks) < 2:
        return img, None, False

    valley_idx = find_split_index_from_profile(profile_smooth, peaks)
    valley_idx = max(1, min(int(valley_idx), W - 1))

    left = img[:, :valley_idx + 1]
    right = img[:, valley_idx:]
    return left, right, True

def _split_into_n(img: np.ndarray, n: int, max_iters: int = 50) -> List[np.ndarray]:
    if n <= 1:
        return [img]

    pieces: List[np.ndarray] = [img]

    iters = 0
    while len(pieces) < n and iters < max_iters:
        iters += 1

        best_i = None
        best_score = None  # (has_two_peaks, width)

        for i, p in enumerate(pieces):
            W = p.shape[1]
            if W < 3:
                continue

            _, peaks = _profile_and_peaks(p)
            has_two = peaks is not None and len(peaks) >= 2
            score = (1 if has_two else 0, W)

            if best_score is None or score > best_score:
                best_score = score
                best_i = i

        if best_i is None or best_score is None or best_score[0] == 0:
            break

        left, right, ok = _split_once_by_valley(pieces[best_i])
        if not ok or right is None:
            break

        pieces = pieces[:best_i] + [left, right] + pieces[best_i + 1 :]

    return pieces

def _group_tag(group_name: str) -> str:
    return str(group_name).replace(" ", "")

def copy_coelu_sliced_to_patch(
    *,
    coelu_sliced_dir: Path,
    patch_dir: Path,
    overwrite: bool = False,
) -> Tuple[int, int]:
    patch_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for p in coelu_sliced_dir.glob("*.png"):
        dst = patch_dir / p.name
        if dst.exists() and not overwrite:
            skipped += 1
            continue
        shutil.copy2(p, dst)
        copied += 1

    return copied, skipped

def rename_patch_to_peak_numbers(
    *,
    dirs: Dict[str, str],
    group_name: str,
    overwrite: bool = False,
) -> Tuple[int, int]:
    patch_dir = Path(dirs["patch"])
    group_tag = str(group_name).replace(" ", "")

    pat = re.compile(
        r"^(?P<base>.+?)_mz(?P<mz>\d+(?:\.\d+)?)_seg(?P<seg>\d+)_(?P<group>[^_]+)(?:_p(?P<p>\d+))?\.png$"
    )

    files = []
    for p in patch_dir.glob("*.png"):
        m = pat.match(p.name)
        if not m:
            continue
        if m.group("group") != group_tag:
            continue
        seg = int(m.group("seg"))
        pnum = int(m.group("p")) if m.group("p") else 1
        files.append((m.group("base"), seg, pnum, m.group("mz"), p))

    if not files:
        return 0, 0

    # Group by base sample (so Peak1 restarts for each base)
    by_base: Dict[str, list] = {}
    for base, seg, pnum, mz, path in files:
        by_base.setdefault(base, []).append((seg, pnum, mz, path))

    renamed = 0
    skipped = 0

    for base, items in by_base.items():
        items.sort(key=lambda t: (t[0], t[1]))  # (seg, p)

        # Two-phase rename to avoid collisions
        temp_moves = []
        for i, (seg, pnum, mz, old_path) in enumerate(items, start=1):
            new_name = f"{base}_mass{mz}_Peak{i}_{group_tag}.png"
            new_path = patch_dir / new_name

            if new_path.exists() and not overwrite:
                skipped += 1
                continue

            tmp_path = patch_dir / f"{old_path.stem}__TMP__.png"
            old_path.rename(tmp_path)
            temp_moves.append((tmp_path, new_path))

        for tmp_path, new_path in temp_moves:
            if new_path.exists() and overwrite:
                new_path.unlink()
            tmp_path.rename(new_path)
            renamed += 1

    return renamed, skipped

def process_file_coelution_sliced(dirs: Dict[str, str], group_name: str) -> str:
    tag = _group_tag(group_name)

    coelu_slices_dir = Path(dirs["coelu"])
    coelu_csv_dir = Path(dirs["coelu csv"])
    out_dir = Path(dirs["coelu sliced"])
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = coelu_csv_dir / f"ALL_coelu_matches_{tag}.csv"
    if not csv_path.exists():
        return f"[!] Missing CSV: {csv_path.name} in {coelu_csv_dir}"

    df = pd.read_csv(csv_path)

    if "slice_filename" not in df.columns or "Peak_num_cluster" not in df.columns:
        return "[!] CSV missing required columns: slice_filename, Peak_num_cluster"

    # optional filter
    if "slice_found" in df.columns:
        sf = df["slice_found"].astype(str).str.upper()
        df = df[sf.isin(["TRUE", "1", "YES"])]

    groups = df.groupby("slice_filename", sort=False)

    total_in = 0
    total_out = 0
    missing = 0
    warnings = 0

    for slice_filename, g in groups:
        total_in += 1
        slice_path = coelu_slices_dir / str(slice_filename)

        if not slice_path.exists():
            missing += 1
            if DEBUG_PRINT:
                print(f"[!] Missing slice image: {slice_path}")
            continue

        # Coeluting case: Peak_num_cluster works.
        # Resolved-sliced-together case: Peak_num_cluster often collapses to [1] because each peak is in a different cluster.
        peak_types = g.get("peak_type", pd.Series([], dtype=str)).astype(str).str.strip().str.lower()
        has_resolved = peak_types.eq("resolved").any()

        if has_resolved:
            # Build unique peaks within this slice using available per-peak columns
            # Prefer pixel ordering if present; fallback to RT_apex.
            cols_needed = [c for c in ["cluster_id", "peak_num", "peak_pixel_start", "peak_pixel_end", "RT_apex"] if
                           c in g.columns]
            peaks_u = g[cols_needed].drop_duplicates().copy()

            if "peak_pixel_start" in peaks_u.columns:
                peaks_u["_sort"] = pd.to_numeric(peaks_u["peak_pixel_start"], errors="coerce")
            elif "RT_apex" in peaks_u.columns:
                peaks_u["_sort"] = pd.to_numeric(peaks_u["RT_apex"], errors="coerce")
            else:
                # last resort: keep original row order
                peaks_u["_sort"] = np.arange(len(peaks_u), dtype=float)

            peaks_u = peaks_u.sort_values("_sort", kind="stable")
            n_peaks = int(len(peaks_u))

            # labels 1..n left->right (these become _p1, _p2, ...)
            labels = list(range(1, n_peaks + 1))

        else:
            peak_nums = sorted(pd.unique(g["Peak_num_cluster"]))
            n_peaks = len(peak_nums)
            labels = [int(x) for x in peak_nums]

        img = cv2.imread(str(slice_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            warnings += 1
            if DEBUG_PRINT:
                print(f"[!] Could not read: {slice_path.name}")
            continue

        pieces = _split_into_n(img, n=n_peaks)

        expected = n_peaks
        produced = len(pieces)

        if produced != expected:
            warnings += 1

            if produced < expected:
                print(
                    f"[WARN] {slice_path.name}: expected {expected} slices "
                    f"(peaks={peak_nums}), but only produced {produced}. "
                    f"Saved what was produced."
                )
            else:
                print(
                    f"[WARN] {slice_path.name}: produced MORE slices than expected "
                    f"({produced} > {expected})."
                )

        # Save left->right pieces using Peak_num_cluster for names
        for i, piece in enumerate(pieces):
            label = labels[i] if i < len(labels) else (i + 1)
            out_name = f"{slice_path.stem}_p{label}{slice_path.suffix}"
            cv2.imwrite(str(out_dir / out_name), piece)
            total_out += 1

    patch_dir = Path(dirs["patch"])
    c2, s2 = copy_coelu_sliced_to_patch(
        coelu_sliced_dir=out_dir,
        patch_dir=patch_dir,
        overwrite=False,
    )
    if DEBUG_PRINT or c2 > 0:
        print(f"[Patch sync] coelu sliced -> patch: copied={c2}, skipped={s2} -> {patch_dir}")

    r, s = rename_patch_to_peak_numbers(dirs=dirs, group_name=group_name, overwrite=False)
    if DEBUG_PRINT:
        print(f"[Patch rename] renamed={r}, skipped={s}")

    return (f"[✔] CoelutionSliced {tag}: "
            f"{total_in} files processed, {total_out} slices written, "
            f"{missing} missing images, {warnings} warnings")