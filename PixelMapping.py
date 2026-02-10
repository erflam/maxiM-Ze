from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

from Config import Config


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Bad hex color: {hex_color}")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return r, g, b


def generate_mass_colors(mass_list: List[float]) -> Dict[str, str]:
    """
    MUST MATCH EICBuilder.py EXACTLY.
    Generates the same DARK colors per m/z.
    """
    import colorsys

    mass_strs = [f"{m:.4f}" for m in mass_list]
    n = max(1, len(mass_strs))

    colors = {}
    for i, mz_str in enumerate(mass_strs):
        hue = i / n
        saturation = 0.75
        value = 0.40
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors[mz_str] = '#{:02x}{:02x}{:02x}'.format(
            int(r * 255),
            int(g * 255),
            int(b * 255)
        )

    return colors


def boolean_runs_to_segments(col_has: np.ndarray) -> List[Tuple[int, int]]:
    """
    Convert a boolean array into contiguous (start, end) segments where True.
    """
    segments: List[Tuple[int, int]] = []
    w = col_has.shape[0]
    in_run = False
    start = 0

    for x in range(w):
        if col_has[x] and not in_run:
            in_run = True
            start = x
        elif (not col_has[x]) and in_run:
            in_run = False
            end = x - 1
            segments.append((start, end))

    if in_run:
        segments.append((start, w - 1))

    return segments


def merge_close_segments(segments: List[Tuple[int, int]], max_gap_px: int = 2) -> List[Tuple[int, int]]:
    """
    Merge segments that are separated by small gaps (anti-aliasing / tiny breaks).
    Example: (100,120) and (123,140) with max_gap_px=2 => merge into (100,140) because gap=2.
    """
    if not segments:
        return []

    segments = sorted(segments, key=lambda t: t[0])
    merged = [segments[0]]

    for s, e in segments[1:]:
        ps, pe = merged[-1]
        gap = s - pe - 1
        if gap <= max_gap_px:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    return merged


def filter_short_segments(segments: List[Tuple[int, int]], min_width_px: int = 3) -> List[Tuple[int, int]]:
    """
    Drop tiny segments (noise).
    width = end-start+1 must be >= min_width_px
    """
    out = []
    for s, e in segments:
        if (e - s + 1) >= min_width_px:
            out.append((s, e))
    return out


def find_segments_for_color(
    rgba: np.ndarray,
    target_rgb: Tuple[int, int, int],
    alpha_threshold: int = 1,
    rgb_tolerance: int = 40,
    max_gap_px: int = 2,
    min_width_px: int = 3,
) -> List[Tuple[int, int]]:
    """
    Return a list of (Pixel_start, Pixel_end) segments for pixels matching target_rgb.
    Uses alpha to ignore transparent background and tolerance to survive anti-aliasing.
    Produces multiple segments (for multiple peaks/isomers).
    """
    rgb = rgba[:, :, :3].astype(np.int16)
    alpha = rgba[:, :, 3].astype(np.int16)

    tr, tg, tb = target_rgb

    drawn = alpha >= alpha_threshold
    dist = np.abs(rgb[:, :, 0] - tr) + np.abs(rgb[:, :, 1] - tg) + np.abs(rgb[:, :, 2] - tb)
    mask = drawn & (dist <= rgb_tolerance)

    col_has = mask.any(axis=0)
    if not col_has.any():
        return []

    segments = boolean_runs_to_segments(col_has)
    segments = merge_close_segments(segments, max_gap_px=max_gap_px)
    segments = filter_short_segments(segments, min_width_px=min_width_px)
    return segments


def run_for_group(group_name: str) -> None:
    Config.set_mass_group(group_name)

    group_dir = Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER / str(Config.CURRENT_GROUP)
    png_dir = group_dir / "EIC PNGs"
    out_dir = group_dir / "Pixel CSVs"
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(png_dir.glob("EIC_*_plotly.png"))
    if not pngs:
        print(f"[!] No PNGs found in: {png_dir}")
        return

    mass_list = list(Config.MASS_LIST)
    mass_strs = [f"{m:.4f}" for m in mass_list]
    mass_colors = generate_mass_colors(mass_list)

    # You can tweak these if needed:
    RGB_TOL = 50       # 40-70 if needed
    MAX_GAP = 3        # merge tiny breaks
    MIN_WIDTH = 4      # drop tiny noise segments

    for png_path in pngs:
        stem = png_path.stem  # EIC_xxx_plotly
        if not (stem.startswith("EIC_") and stem.endswith("_plotly")):
            continue
        base = stem[len("EIC_"):-len("_plotly")]

        with Image.open(png_path) as im:
            im = im.convert("RGBA")
            rgba = np.array(im)

        rows = []
        for mz_str in mass_strs:
            color_hex = mass_colors[mz_str]
            rgb = hex_to_rgb(color_hex)

            segments = find_segments_for_color(
                rgba,
                target_rgb=rgb,
                alpha_threshold=1,
                rgb_tolerance=RGB_TOL,
                max_gap_px=MAX_GAP,
                min_width_px=MIN_WIDTH,
            )

            # Write one row per segment (for isomers / multiple peaks)
            for seg_id, (px_start, px_end) in enumerate(segments, start=1):
                rows.append({
                    "File Name": base,
                    "m/z": mz_str,
                    "Segment_ID": int(seg_id),
                    "Pixel_start": int(px_start),
                    "Pixel_end": int(px_end),
                })

        out_path = out_dir / f"{base}_pixelmapping_{group_name}.csv"
        pd.DataFrame(
            rows,
            columns=["File Name", "m/z", "Segment_ID", "Pixel_start", "Pixel_end"]
        ).to_csv(out_path, index=False)

        print(f"[✔] {out_path.name} ({len(rows)} segments)")


if __name__ == "__main__":
    run_for_group(Config.CURRENT_GROUP)
