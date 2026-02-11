# Slicing.py
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from Config import Config


# ----------------------------
# Color utilities (must match)
# ----------------------------
def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    s = hex_color.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Bad hex color: {hex_color}")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def generate_mass_colors(mass_list: List[float]) -> Dict[str, str]:
    """
    MUST MATCH EICBuilder.py logic EXACTLY.
    EICBuilder uses _dark_hex_palette(h=i/n, s=0.85, v=0.25).
    """
    import colorsys

    mass_strs = [f"{m:.4f}" for m in mass_list]
    n = max(1, len(mass_strs))

    colors: Dict[str, str] = {}
    for i, mz_str in enumerate(mass_strs):
        h = (i / n) % 1.0
        s = 0.85
        v = 0.25
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors[mz_str] = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
    return colors


# ----------------------------
# Mask building (transparent bg)
# ----------------------------
def build_color_mask_rgba(
    rgba: np.ndarray,
    target_rgb: Tuple[int, int, int],
    rgb_tolerance: int = 60,
    alpha_threshold: int = 1,
) -> np.ndarray:
    """
    Returns boolean mask of pixels matching the mass color line.
    Uses alpha to ignore fully transparent background.
    """
    rgb = rgba[:, :, :3].astype(np.int16)
    a = rgba[:, :, 3].astype(np.int16)
    tr, tg, tb = target_rgb

    drawn = a >= alpha_threshold
    dist = np.abs(rgb[:, :, 0] - tr) + np.abs(rgb[:, :, 1] - tg) + np.abs(rgb[:, :, 2] - tb)
    return drawn & (dist <= rgb_tolerance)


def crop_transparent(rgba: np.ndarray, pad: int = 2) -> np.ndarray:
    a = rgba[:, :, 3]
    ys, xs = np.where(a > 0)
    if xs.size == 0 or ys.size == 0:
        return rgba
    y0 = max(0, ys.min() - pad)
    y1 = min(rgba.shape[0], ys.max() + 1 + pad)
    x0 = max(0, xs.min() - pad)
    x1 = min(rgba.shape[1], xs.max() + 1 + pad)
    return rgba[y0:y1, x0:x1]


# ----------------------------
# Coelution splitting inside a span
# ----------------------------
def connected_components_x_spans(mask: np.ndarray, x0: int, x1: int, min_area: int = 30) -> List[Tuple[int, int]]:
    """
    Finds connected components inside the [x0, x1] window and returns their x-spans.
    """
    h, w = mask.shape
    x0 = int(max(0, min(x0, w - 1)))
    x1 = int(max(0, min(x1, w - 1)))
    if x1 < x0:
        x0, x1 = x1, x0

    sub = mask[:, x0:x1 + 1].astype(np.uint8)
    if sub.sum() == 0:
        return []

    num, labels, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    spans: List[Tuple[int, int]] = []

    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        left = int(stats[lab, cv2.CC_STAT_LEFT]) + x0
        width = int(stats[lab, cv2.CC_STAT_WIDTH])
        right = left + width - 1
        spans.append((left, right))

    spans.sort(key=lambda t: (t[0], t[1]))
    return spans


def height_profile_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    For each x, estimate a 'height' from bottom based on the median y of True pixels.
    Works well for line traces on transparent background.
    """
    h, w = mask.shape
    prof = np.zeros(w, dtype=np.float64)

    ys_by_x = [None] * w
    for x in range(w):
        ys = np.where(mask[:, x])[0]
        ys_by_x[x] = ys

    for x in range(w):
        ys = ys_by_x[x]
        if ys is None or ys.size == 0:
            prof[x] = np.nan
        else:
            y_med = float(np.median(ys))
            prof[x] = (h - 1) - y_med  # higher value == higher on plot

    # interpolate NaNs
    x = np.arange(w)
    good = ~np.isnan(prof)
    if good.sum() == 0:
        return np.zeros(w, dtype=np.float64)
    if good.sum() == 1:
        prof[~good] = prof[good][0]
    else:
        prof[~good] = np.interp(x[~good], x[good], prof[good])
    return prof


def smooth_1d(y: np.ndarray, ksize: int = 21) -> np.ndarray:
    if y.size < 5:
        return y.copy()
    k = int(ksize)
    if k % 2 == 0:
        k += 1
    k = max(5, min(k, y.size if y.size % 2 == 1 else y.size - 1))
    return cv2.GaussianBlur(y.reshape(1, -1).astype(np.float32), (1, k), 0).ravel().astype(np.float64)


def valley_cut_index(profile: np.ndarray) -> int:
    """
    Split at the deepest valley between the two strongest peaks in the profile.
    """
    w = profile.size
    if w < 5:
        return max(1, w // 2)

    # find 2 strongest maxima with separation
    idx_sorted = np.argsort(profile)[::-1]
    peaks = []
    min_sep = max(10, int(0.15 * w))
    for idx in idx_sorted:
        if all(abs(int(idx) - int(p)) >= min_sep for p in peaks):
            peaks.append(int(idx))
        if len(peaks) == 2:
            break

    if len(peaks) < 2:
        return max(1, w // 2)

    a, b = sorted(peaks)
    if b <= a + 1:
        return max(1, w // 2)

    seg = profile[a:b + 1]
    v = a + int(np.argmin(seg))
    v = int(np.clip(v, 1, w - 2))
    return v


def split_span_if_needed(mask: np.ndarray, x0: int, x1: int) -> List[Tuple[int, int]]:
    """
    Returns 1+ x-spans. Tries connected components first.
    If only 1 span, attempts a valley split.
    """
    spans = connected_components_x_spans(mask, x0, x1, min_area=30)

    # if CC found 2+ components => resolved
    if len(spans) >= 2:
        return spans

    # fallback: valley split inside the window
    h, w = mask.shape
    x0 = int(max(0, min(x0, w - 2)))
    x1 = int(max(1, min(x1, w - 1)))
    if x1 <= x0:
        return [(x0, x1)]

    sub = mask[:, x0:x1 + 1]
    if sub.sum() == 0:
        return [(x0, x1)]

    prof = height_profile_from_mask(sub)
    prof_s = smooth_1d(prof, ksize=21)

    cut_rel = valley_cut_index(prof_s)
    cut = x0 + int(cut_rel)

    # enforce non-empty
    cut = int(np.clip(cut, x0 + 1, x1 - 1))
    return [(x0, cut), (cut, x1)]


# ----------------------------
# Core slicing per file
# ----------------------------
def slice_one_file(
    png_path: Path,
    pixelmap_csv: Path,
    out_dir: Path,
    mass_colors: Dict[str, str],
    rgb_tolerance: int = 60,
) -> int:
    if not png_path.exists():
        return 0
    if not pixelmap_csv.exists():
        return 0

    rgba = np.array(Image.open(png_path).convert("RGBA"))
    h, w, _ = rgba.shape

    df = pd.read_csv(pixelmap_csv)
    if df.empty:
        return 0

    saved = 0
    base = png_path.stem
    group_tag = Config.CURRENT_GROUP.replace(" ", "")

    # Expect columns: File Name, m/z, Segment_ID, Pixel_start, Pixel_end
    for mz_str, grp in df.groupby("m/z", sort=False):
        mz_str = str(mz_str)
        if mz_str not in mass_colors:
            continue

        rgb = hex_to_rgb(mass_colors[mz_str])
        mask = build_color_mask_rgba(rgba, rgb, rgb_tolerance=rgb_tolerance, alpha_threshold=1)

        # Each mapping segment may still contain coeluting peaks -> split inside it
        for _, row in grp.iterrows():
            try:
                x0 = int(row["Pixel_start"])
                x1 = int(row["Pixel_end"])
            except Exception:
                continue

            x0 = max(0, min(x0, w - 1))
            x1 = max(0, min(x1, w - 1))
            if x1 < x0:
                x0, x1 = x1, x0

            subspans = split_span_if_needed(mask, x0, x1)

            for sub_id, (sx0, sx1) in enumerate(subspans, start=1):
                sx0 = int(np.clip(sx0, 0, w - 2))
                sx1 = int(np.clip(sx1, sx0 + 1, w - 1))

                # Apply x-window to mask, zero everything else
                out = rgba.copy()
                keep = np.zeros((h, w), dtype=bool)
                keep[:, sx0:sx1 + 1] = mask[:, sx0:sx1 + 1]
                out[..., 3] = np.where(keep, out[..., 3], 0).astype(np.uint8)

                if np.count_nonzero(out[..., 3]) == 0:
                    continue

                out2 = crop_transparent(out, pad=2)

                # Name: sample_massXXXX_segY_GroupTag.png
                # (You can change this naming later to match your final scheme)
                fname = f"{base}_mass{mz_str}_seg{sub_id}_{group_tag}.png"
                Image.fromarray(out2).save(out_dir / fname)
                saved += 1

    return saved


# ----------------------------
# Pipeline entrypoint (checkpoint)
# ----------------------------
def process_file_checkpoint5(fp: str, dirs: Dict[str, str], group_name: str) -> str:
    """
    Checkpoint 4: Slice EIC PNGs into per-mass (and coelution-resolved) slices using PixelMapping.
    """
    Config.set_mass_group(group_name)
    base = os.path.splitext(os.path.basename(fp))[0]
    group_tag = Config.CURRENT_GROUP.replace(" ", "")

    png_path = Path(dirs["png"]) / f"EIC_{base}_{group_tag}.png"

    # PixelMapping output (you currently write these into Pixel CSVs folder)
    pixelmap_csv = Path(dirs["pixel"]) / f"{base}_pixelmapping_{group_tag}.csv"

    out_dir = Path(dirs["slice"])
    out_dir.mkdir(parents=True, exist_ok=True)

    mass_colors = generate_mass_colors(list(Config.MASS_LIST))

    if not png_path.exists():
        return f"[!] Missing PNG: {png_path.name}"
    if not pixelmap_csv.exists():
        return f"[!] Missing PixelMapping CSV: {pixelmap_csv.name}"

    n = slice_one_file(
        png_path=png_path,
        pixelmap_csv=pixelmap_csv,
        out_dir=out_dir,
        mass_colors=mass_colors,
        rgb_tolerance=60,
    )

    return f"[✔] {base}: saved {n} slices"


if __name__ == "__main__":
    # Manual run for current group (single-process)
    group = Config.CURRENT_GROUP
    dirs = Config.setup_directories()
    # This assumes your mzXML names in FileUtils match fp base.
    # If you just want to slice existing outputs, iterate over pixelmap CSVs instead.
    print("Slicing checkpoint is meant to be called from Pipeline with process_file_checkpoint4().")
