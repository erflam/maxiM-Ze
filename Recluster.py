import time
import pandas as pd
import numpy as np
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter


# ── Path configuration ────────────────────────────────────────────────────────
# Mirror the class-level path constants used in the rest of the project.
# Adjust BASE_DIR / OUTPUT_ROOT / ANALYSIS_FOLDER / CURRENT_GROUP as needed.

class Config:
    BASE_DIR        = Path(".")          # change to match your project root
    OUTPUT_ROOT     = "Output"
    ANALYSIS_FOLDER = "Analysis"
    CURRENT_GROUP   = "Group1"           # or pass in dynamically

    @classmethod
    def clustering_dir(cls) -> Path:
        return cls.BASE_DIR / cls.OUTPUT_ROOT / cls.ANALYSIS_FOLDER / cls.CURRENT_GROUP / "Clustering"

    @classmethod
    def pixel_dir(cls) -> Path:
        return cls.BASE_DIR / cls.OUTPUT_ROOT / cls.ANALYSIS_FOLDER / cls.CURRENT_GROUP / "Pixel CSVs"


RT_TOLERANCE = 0.08   # seconds / minutes – same unit as RT_apex column


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample_name_from_col(col: str) -> str:
    """Strip trailing _height or _area suffix to get the bare sample name."""
    for suffix in ("_height", "_area"):
        if col.endswith(suffix):
            return col[: -len(suffix)]
    return col


def _sample_name_from_peak_id(peak_id: str) -> str:
    """
    peak_id looks like:  OE_EF_..._C012_0001_mass297.1672_Peak6_Group1
    We want:             OE_EF_..._C012_0001
    Strategy: drop everything from '_mass' onwards.
    """
    idx = peak_id.find("_mass")
    if idx != -1:
        return peak_id[:idx]
    # fallback – return as-is
    return peak_id


def _height_col(sample: str) -> str:
    return f"{sample}_height"


def _area_col(sample: str) -> str:
    return f"{sample}_area"


# ── Core reclustering logic ───────────────────────────────────────────────────

def recluster_group(group: str, config: type = Config) -> None:
    clustering_dir = config.clustering_dir()
    pixel_dir      = config.pixel_dir()

    alignment_path    = clustering_dir / f"alignment_summary_group_{group}.csv"
    unclustered_path  = clustering_dir / f"unclustered_peaks_group_{group}.csv"
    output_path       = clustering_dir / f"Group_Summary_{group}.xlsx"

    # ── Load input files ──────────────────────────────────────────────────────
    alignment  = pd.read_csv(alignment_path)
    unclustered = pd.read_csv(unclustered_path)

    # Identify all sample columns present in the alignment summary
    height_cols = [c for c in alignment.columns if c.endswith("_height")]
    area_cols   = [c for c in alignment.columns if c.endswith("_area")]
    all_samples = [_sample_name_from_col(c) for c in height_cols]

    # Add Recluster column (default FALSE)
    alignment["Recluster"] = False

    # Track which (row_idx, col_name) cells were newly filled so we can bold them
    newly_filled: set[tuple[int, str]] = set()

    # Track which unclustered peak_ids were successfully matched
    matched_peak_ids: set = set()

    # ── Iterate over every blank height cell in the alignment summary ─────────
    for row_idx, row in alignment.iterrows():
        aligned_rt  = row["Aligned_rt_apex"]
        mz          = row["m/z"]

        for sample in all_samples:
            h_col = _height_col(sample)
            a_col = _area_col(sample)

            # Only attempt to fill if the height cell is blank / NaN / 0
            height_val = row.get(h_col, np.nan)
            if pd.notna(height_val) and height_val != 0:
                continue

            # Find candidate unclustered peaks for this sample + m/z
            candidates = unclustered[
                (unclustered["m/z"] == mz) &
                (unclustered["peak_id"].apply(_sample_name_from_peak_id) == sample) &
                (abs(unclustered["RT_apex"] - aligned_rt) <= RT_TOLERANCE)
            ].copy()

            if candidates.empty:
                continue

            # Pick the candidate with the closest RT_apex
            candidates["_rt_diff"] = abs(candidates["RT_apex"] - aligned_rt)
            best = candidates.loc[candidates["_rt_diff"].idxmin()]

            # Fill height
            alignment.at[row_idx, h_col] = best["height"]
            newly_filled.add((row_idx, h_col))

            # Fill area if column exists
            if a_col in alignment.columns:
                alignment.at[row_idx, a_col] = best["area"]
                newly_filled.add((row_idx, a_col))

            alignment.at[row_idx, "Recluster"] = True
            matched_peak_ids.add(best["peak_id"])

    # ── Recalculate peak count and Aligned_rt_apex ────────────────────────────
    for row_idx, row in alignment.iterrows():
        if not row["Recluster"]:
            continue

        # Count non-null, non-zero height values
        heights = [row.get(_height_col(s), np.nan) for s in all_samples]
        new_count = sum(1 for h in heights if pd.notna(h) and h != 0)
        alignment.at[row_idx, "peak count"] = new_count

        # Recalculate Aligned_rt_apex from the pixel CSVs for this group
        # We collect RT_apex values for all filled samples by reading pixel files.
        rt_values = []
        mz_val = row["m/z"]
        for sample in all_samples:
            h_col = _height_col(sample)
            if pd.notna(row.get(h_col, np.nan)) and row.get(h_col, 0) != 0:
                pixel_file = pixel_dir / f"{sample}_peaks_pix_{group}.csv"
                if pixel_file.exists():
                    try:
                        pix = pd.read_csv(pixel_file)
                        match = pix[
                            (pix["m/z"] == mz_val) &
                            (abs(pix["RT_apex"] - row["Aligned_rt_apex"]) <= RT_TOLERANCE + 0.05)
                        ]
                        if not match.empty:
                            # pick the row whose RT_apex is closest to current aligned RT
                            best_rt = match.loc[
                                (match["RT_apex"] - row["Aligned_rt_apex"]).abs().idxmin(), "RT_apex"
                            ]
                            rt_values.append(best_rt)
                    except Exception:
                        pass

        if rt_values:
            alignment.at[row_idx, "Aligned_rt_apex"] = float(np.mean(rt_values))

    # ── Still-unclustered peaks ───────────────────────────────────────────────
    still_unclustered = unclustered[~unclustered["peak_id"].isin(matched_peak_ids)].copy()

    # ── Build the Excel workbook ──────────────────────────────────────────────
    _write_excel(
        output_path   = output_path,
        alignment     = alignment,
        unclustered   = still_unclustered,
        newly_filled  = newly_filled,
        height_cols   = height_cols,
        area_cols     = area_cols,
    )

    print(f"Done. Output written to: {output_path}")
    print(f"  Rows reclustered : {alignment['Recluster'].sum()}")
    print(f"  Peaks matched    : {len(matched_peak_ids)}")
    print(f"  Still unclustered: {len(still_unclustered)}")


# ── Excel writer ──────────────────────────────────────────────────────────────

def _write_excel(
    output_path:  Path,
    alignment:    pd.DataFrame,
    unclustered:  pd.DataFrame,
    newly_filled: set[tuple[int, str]],
    height_cols:  list[str],
    area_cols:    list[str],
) -> None:

    wb = Workbook()

    # ── Sheet 1: reclustered summary ──────────────────────────────────────────
    ws1 = wb.active
    ws1.title = "Summary"

    bold_font   = Font(name="Arial", bold=True)
    normal_font = Font(name="Arial", bold=False)
    header_font = Font(name="Arial", bold=True)

    # Reorder columns so Recluster appears right after peak count
    cols = list(alignment.columns)
    if "Recluster" in cols:
        cols.remove("Recluster")
        pc_idx = cols.index("peak count") if "peak count" in cols else len(cols) - 1
        cols.insert(pc_idx + 1, "Recluster")
    alignment = alignment[cols]

    # Write header row
    for col_idx, col_name in enumerate(alignment.columns, start=1):
        cell = ws1.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", wrap_text=True)

    # Map DataFrame column name → Excel column index (1-based)
    col_name_to_excel_col = {name: i + 1 for i, name in enumerate(alignment.columns)}

    # Write data rows
    for df_row_idx, (_, row) in enumerate(alignment.iterrows()):
        excel_row = df_row_idx + 2          # +1 for header, +1 for 1-based
        orig_df_idx = alignment.index[df_row_idx]   # original DataFrame index used in newly_filled

        for col_name in alignment.columns:
            value = row[col_name]
            # Convert numpy booleans / scalars
            if isinstance(value, (np.bool_,)):
                value = bool(value)
            elif isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = float(value) if not np.isnan(value) else None

            excel_col = col_name_to_excel_col[col_name]
            cell = ws1.cell(row=excel_row, column=excel_col, value=value)
            cell.font = normal_font

            # Bold newly filled cells
            if (orig_df_idx, col_name) in newly_filled:
                cell.font = bold_font

    # Auto-size columns (capped at 40)
    for col_cells in ws1.columns:
        max_len = max(
            (len(str(c.value)) if c.value is not None else 0) for c in col_cells
        )
        ws1.column_dimensions[get_column_letter(col_cells[0].column)].width = min(max_len + 2, 40)

    # ── Sheet 2: still-unclustered peaks ─────────────────────────────────────
    ws2 = wb.create_sheet(title="unclustered")

    if not unclustered.empty:
        for col_idx, col_name in enumerate(unclustered.columns, start=1):
            cell = ws2.cell(row=1, column=col_idx, value=col_name)
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", wrap_text=True)

        for row_idx, (_, row) in enumerate(unclustered.iterrows(), start=2):
            for col_idx, col_name in enumerate(unclustered.columns, start=1):
                value = row[col_name]
                if isinstance(value, (np.bool_,)):
                    value = bool(value)
                elif isinstance(value, (np.integer,)):
                    value = int(value)
                elif isinstance(value, (np.floating,)):
                    value = float(value) if not np.isnan(value) else None
                cell = ws2.cell(row=row_idx, column=col_idx, value=value)
                cell.font = normal_font

        for col_cells in ws2.columns:
            max_len = max(
                (len(str(c.value)) if c.value is not None else 0) for c in col_cells
            )
            ws2.column_dimensions[get_column_letter(col_cells[0].column)].width = min(max_len + 2, 40)
    else:
        ws2.cell(row=1, column=1, value="No unclustered peaks remaining.")

    wb.save(output_path)


# ── Pipeline interface ────────────────────────────────────────────────────────

def process_file_recluster(dirs: dict, group_name: str) -> str:
    """
    Thin wrapper around recluster_group() that accepts the standard pipeline
    `dirs` dict (as returned by Config.setup_directories()) and returns a
    status message string.

    Reads 'clustering' and 'pixel' paths directly from `dirs` so the correct
    absolute paths from Config.py are always used.
    """

    class _RuntimeConfig(Config):
        @classmethod
        def clustering_dir(cls) -> Path:
            return Path(dirs["clustering"])

        @classmethod
        def pixel_dir(cls) -> Path:
            return Path(dirs["pixel"])

    recluster_group(group=group_name, config=_RuntimeConfig)
    output_path = Path(dirs["clustering"]) / f"Group_Summary_{group_name}.xlsx"
    return f"Reclustering complete for {group_name}. Output: {output_path}"

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recluster unclustered peaks into alignment summary.")
    parser.add_argument("group", help="Group identifier, e.g. Group1")
    parser.add_argument("--base-dir",        default=".",        help="Project base directory")
    parser.add_argument("--output-root",     default="Output",   help="Output root folder name")
    parser.add_argument("--analysis-folder", default="Analysis", help="Analysis folder name")
    args = parser.parse_args()

    # Patch Config with CLI arguments
    Config.BASE_DIR        = Path(args.base_dir)
    Config.OUTPUT_ROOT     = args.output_root
    Config.ANALYSIS_FOLDER = args.analysis_folder
    Config.CURRENT_GROUP   = args.group

    recluster_group(group=args.group, config=Config)