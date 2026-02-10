"""
PixelMapping.py

PURE IMAGE-ONLY pixel bounds per m/z.

This works because EICBuilder.py now draws each m/z trace in a unique color.
We detect each trace by its RGB color (with tolerance) + alpha > 0.

Outputs:
{File}_pixelmapping_{Group}.csv

Columns:
File Name, m/z, Pixel_start, Pixel_end
"""

from __future__ import annotations

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


def find_bounds_for_color(
    rgba: np.ndarray,
    target_rgb: Tuple[int, int, int],
    alpha_threshold: int = 1,
    rgb_tolerance: int = 40,
) -> Optional[Tuple[int, int]]:
    """
    Returns (pixel_start, pixel_end) for pixels matching target_rgb.
    Uses alpha to ignore transparent background and tolerance to survive anti-aliasing.
    """
    # rgba shape: (H, W, 4)
    rgb = rgba[:, :, :3].astype(np.int16)  # avoid overflow
    alpha = rgba[:, :, 3].astype(np.int16)

    tr, tg, tb = target_rgb

    # only consider drawn pixels (alpha > threshold)
    drawn = alpha >= alpha_threshold

    # color distance (L1) to tolerate anti-aliasing
    dist = np.abs(rgb[:, :, 0] - tr) + np.abs(rgb[:, :, 1] - tg) + np.abs(rgb[:, :, 2] - tb)

    # match mask
    mask = drawn & (dist <= rgb_tolerance)

    # per-column presence
    col_has = mask.any(axis=0)
    if not col_has.any():
        return None

    xs = np.where(col_has)[0]
    return int(xs.min()), int(xs.max())


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

    for png_path in pngs:
        # Extract base from "EIC_{base}_plotly.png"
        stem = png_path.stem  # EIC_xxx_plotly
        if not (stem.startswith("EIC_") and stem.endswith("_plotly")):
            continue
        base = stem[len("EIC_"):-len("_plotly")]

        # Read image
        with Image.open(png_path) as im:
            im = im.convert("RGBA")
            rgba = np.array(im)

        rows = []
        for mz_str in mass_strs:
            color_hex = mass_colors[mz_str]
            rgb = hex_to_rgb(color_hex)

            bounds = find_bounds_for_color(
                rgba,
                target_rgb=rgb,
                alpha_threshold=1,
                rgb_tolerance=40,   # you can tweak this if needed (30-60 range)
            )

            if bounds is None:
                # No pixels found for this mass in this file
                continue

            px_start, px_end = bounds
            rows.append({
                "File Name": base,
                "m/z": mz_str,
                "Pixel_start": int(px_start),
                "Pixel_end": int(px_end),
            })

        out_path = out_dir / f"{base}_pixelmapping_{group_name}.csv"
        pd.DataFrame(rows, columns=["File Name", "m/z", "Pixel_start", "Pixel_end"]).to_csv(out_path, index=False)
        print(f"[✔] {out_path.name} ({len(rows)} masses found)")


if __name__ == "__main__":
    run_for_group(Config.CURRENT_GROUP)
