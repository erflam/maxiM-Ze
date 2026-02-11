from pathlib import Path
import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import colorsys

from Config import Config

def _group_tag(group_name: str) -> str:
    """Convert 'Group 1' -> 'Group1' (must match your EICBuilder naming)."""
    return str(group_name).replace(" ", "")

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Bad hex color: {hex_color}")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return r, g, b

def _dark_hex_palette(n: int) -> List[str]:
    if n <= 0:
        return []
    colors: List[str] = []
    for i in range(n):
        h = (i / n) % 1.0
        s = 0.85
        v = 0.25
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors

def generate_mass_colors(mass_list: List[float]) -> Dict[str, str]:
    mass_strs = [f"{m:.4f}" for m in mass_list]
    palette = _dark_hex_palette(len(mass_strs))
    return {mass_strs[i]: palette[i] for i in range(len(mass_strs))}

def boolean_runs_to_segments(col_has: np.ndarray) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    w = int(col_has.shape[0])
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
    out: List[Tuple[int, int]] = []
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


def _normalize_base_for_png_lookup(base: str, group_tag: str) -> str:
    if base.startswith("EIC_"):
        base = base[len("EIC_") :]
    suffix = f"_{group_tag}"
    if base.endswith(suffix):
        base = base[: -len(suffix)]

    return base

def process_file_checkpoint4(fp: str, dirs: Dict[str, str], group_name: str) -> str:
    Config.set_mass_group(group_name)
    tag = _group_tag(group_name)
    base = os.path.splitext(os.path.basename(fp))[0]
    base = _normalize_base_for_png_lookup(base, tag)
    png_path = os.path.join(dirs["png"], f"EIC_{base}_{tag}.png")
    out_path = os.path.join(dirs["pixel"], f"{base}_pixelmapping_{tag}.csv")
    if not os.path.exists(png_path):
        return f"[!] Missing PNG: {os.path.basename(png_path)}"

    mass_list = list(Config.MASS_LIST)
    mass_strs = [f"{m:.4f}" for m in mass_list]
    mass_colors = generate_mass_colors(mass_list)

    # Knobs
    RGB_TOL = 60        # tolerance for anti-aliasing
    MAX_GAP = 3         # merge tiny breaks
    MIN_WIDTH = 4       # drop tiny noise segments
    ALPHA_TH = 1        # background is transparent

    with Image.open(png_path) as im:
        rgba = np.array(im.convert("RGBA"))

    rows: List[Dict[str, object]] = []
    for mz_str in mass_strs:
        color_hex = mass_colors[mz_str]
        rgb = hex_to_rgb(color_hex)

        segments = find_segments_for_color(
            rgba,
            target_rgb=rgb,
            alpha_threshold=ALPHA_TH,
            rgb_tolerance=RGB_TOL,
            max_gap_px=MAX_GAP,
            min_width_px=MIN_WIDTH,
        )

        for seg_id, (px_start, px_end) in enumerate(segments, start=1):
            rows.append(
                {
                    "File Name": base,
                    "m/z": mz_str,
                    "Segment_ID": int(seg_id),
                    "Pixel_start": int(px_start),
                    "Pixel_end": int(px_end),
                }
            )

    pd.DataFrame(
        rows,
        columns=["File Name", "m/z", "Segment_ID", "Pixel_start", "Pixel_end"],
    ).to_csv(out_path, index=False)

    return f"[✔] {os.path.basename(out_path)} ({len(rows)} segments)"

def run_for_group(group_name: str) -> None:
    Config.set_mass_group(group_name)
    dirs = Config.setup_directories()
    png_dir = dirs["png"]
    pngs = sorted(Path(png_dir).glob(f"EIC_*_{_group_tag(group_name)}.png"))
    if not pngs:
        print(f"[!] No PNGs found for {group_name} in {png_dir}")
        return

    for p in pngs:
        msg = process_file_checkpoint4(str(p), dirs, group_name)
        print(msg)

if __name__ == "__main__":
    run_for_group(Config.CURRENT_GROUP)