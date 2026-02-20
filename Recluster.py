import time
import pandas as pd
import numpy as np
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter


# ── Path configuration ────────────────────────────────────────────────────────

class Config:
    BASE_DIR        = Path(".")
    OUTPUT_ROOT     = "Output"
    ANALYSIS_FOLDER = "Analysis"
    CURRENT_GROUP   = "Group1"

    @classmethod
    def clustering_dir(cls) -> Path:
        return cls.BASE_DIR / cls.OUTPUT_ROOT / cls.ANALYSIS_FOLDER / cls.CURRENT_GROUP / "Clustering"

    @classmethod
    def pixel_dir(cls) -> Path:
        return cls.BASE_DIR / cls.OUTPUT_ROOT / cls.ANALYSIS_FOLDER / cls.CURRENT_GROUP / "Pixel CSVs"


RT_TOLERANCE = 0.08


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample_name_from_col(col: str) -> str:
    for suffix in ("_height", "_area"):
        if col.endswith(suffix):
            return col[: -len(suffix)]
    return col


def _sample_name_from_peak_id(peak_id: str) -> str:
    idx = peak_id.find("_mass")
    if idx != -1:
        return peak_id[:idx]
    return peak_id


def _height_col(sample: str) -> str:
    return f"{sample}_height"


def _area_col(sample: str) -> str:
    return f"{sample}_area"


# ── cluster_patch helpers ─────────────────────────────────────────────────────

def _load_cluster_patch(path: Path) -> list[dict]:
    """
    Read cluster_patch_<group>.csv back into a list of cluster dicts:
      [{"m/z": ..., "Isomer_position": ..., "Aligned_rt_apex": ...,
        "peak count": ..., "patches": [filename, ...]}, ...]
    """
    if not path.exists():
        return []

    df = pd.read_csv(path, header=0)
    # First column is the row-label column (empty header)
    label_col = df.columns[0]
    cluster_cols = [c for c in df.columns if c.startswith("Cluster ")]

    # Build a dict: label -> value for each cluster column
    label_series = df[label_col].fillna("").tolist()

    clusters = []
    for col in cluster_cols:
        values = df[col].tolist()
        row_map = dict(zip(label_series, values))

        patches = []
        for label, val in zip(label_series, values):
            if label == "" and pd.notna(val) and str(val).strip():
                patches.append(str(val).strip())

        clusters.append({
            "col_name": col,
            "m/z": row_map.get("m/z"),
            "Isomer_position": row_map.get("Isomer_position"),
            "Aligned_rt_apex": row_map.get("Aligned_rt_apex"),
            "peak count": row_map.get("peak count"),
            "patches": patches,
        })

    return clusters


def _clusters_to_df(clusters: list[dict]) -> pd.DataFrame:
    """
    Convert a list of cluster dicts back to the transposed DataFrame format
    for writing to the Excel sheet.
    """
    if not clusters:
        return pd.DataFrame()

    max_patches = max(len(c["patches"]) for c in clusters)
    meta_labels = ["m/z", "Isomer_position", "Aligned_rt_apex", "peak count"]
    index_col = meta_labels + [""] * max_patches

    data: dict[str, list] = {"": index_col}
    for cluster in clusters:
        col_values = [
            cluster["m/z"],
            cluster["Isomer_position"],
            cluster["Aligned_rt_apex"],
            cluster["peak count"],
        ]
        patches = cluster["patches"] + [""] * (max_patches - len(cluster["patches"]))
        col_values.extend(patches)
        data[cluster["col_name"]] = col_values

    return pd.DataFrame(data)


# ── Core reclustering logic ───────────────────────────────────────────────────

def recluster_group(group: str, config: type = Config) -> None:
    clustering_dir = config.clustering_dir()
    pixel_dir      = config.pixel_dir()

    alignment_path   = clustering_dir / f"alignment_summary_group_{group}.csv"
    unclustered_path = clustering_dir / f"unclustered_peaks_group_{group}.csv"
    patch_path       = clustering_dir / f"cluster_patch_{group}.csv"
    output_path      = clustering_dir / f"Group_Summary_{group}.xlsx"

    # ── Load input files ──────────────────────────────────────────────────────
    alignment   = pd.read_csv(alignment_path)
    unclustered = pd.read_csv(unclustered_path)
    clusters    = _load_cluster_patch(patch_path)

    height_cols = [c for c in alignment.columns if c.endswith("_height")]
    area_cols   = [c for c in alignment.columns if c.endswith("_area")]
    all_samples = [_sample_name_from_col(c) for c in height_cols]

    alignment["Recluster"] = False

    newly_filled: set[tuple[int, str]] = set()
    matched_peak_ids: set = set()

    has_unclustered = (
        not unclustered.empty
        and "m/z" in unclustered.columns
        and "RT_apex" in unclustered.columns
    )

    # ── Iterate over every blank height cell in the alignment summary ─────────
    if has_unclustered:
        for row_idx, row in alignment.iterrows():
            aligned_rt = row["Aligned_rt_apex"]
            mz         = row["m/z"]
            iso_pos    = row.get("Isomer_position", None)

            for sample in all_samples:
                h_col = _height_col(sample)
                a_col = _area_col(sample)

                height_val = row.get(h_col, np.nan)
                if pd.notna(height_val) and height_val != 0:
                    continue

                candidates = unclustered[
                    (unclustered["m/z"] == mz) &
                    (unclustered["peak_id"].apply(_sample_name_from_peak_id) == sample) &
                    (abs(unclustered["RT_apex"] - aligned_rt) <= RT_TOLERANCE)
                ].copy()

                if candidates.empty:
                    continue

                candidates["_rt_diff"] = abs(candidates["RT_apex"] - aligned_rt)
                best = candidates.loc[candidates["_rt_diff"].idxmin()]

                alignment.at[row_idx, h_col] = best["height"]
                newly_filled.add((row_idx, h_col))

                if a_col in alignment.columns:
                    alignment.at[row_idx, a_col] = best["area"]
                    newly_filled.add((row_idx, a_col))

                alignment.at[row_idx, "Recluster"] = True
                matched_peak_ids.add(best["peak_id"])

                # ── Add PNG to the matching cluster in cluster_patch ───────────
                peak_id  = str(best["peak_id"])
                png_name = f"{peak_id}.png" if not peak_id.endswith(".png") else peak_id

                for cluster in clusters:
                    if cluster["m/z"] == mz and cluster["Isomer_position"] == iso_pos:
                        if png_name not in cluster["patches"]:
                            cluster["patches"].append(png_name)
                            cluster["peak count"] = len(cluster["patches"])
                        break

    # ── Recalculate peak count and Aligned_rt_apex ────────────────────────────
    for row_idx, row in alignment.iterrows():
        if not row["Recluster"]:
            continue

        heights = [row.get(_height_col(s), np.nan) for s in all_samples]
        new_count = sum(1 for h in heights if pd.notna(h) and h != 0)
        alignment.at[row_idx, "peak count"] = new_count

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
        clusters      = clusters,
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
    clusters:     list[dict],
    newly_filled: set[tuple[int, str]],
    height_cols:  list[str],
    area_cols:    list[str],
) -> None:

    wb = Workbook()

    bold_font   = Font(name="Arial", bold=True)
    normal_font = Font(name="Arial", bold=False)
    header_font = Font(name="Arial", bold=True)

    # ── Sheet 1: reclustered summary ──────────────────────────────────────────
    ws1 = wb.active
    ws1.title = "Summary"

    cols = list(alignment.columns)
    if "Recluster" in cols:
        cols.remove("Recluster")
        pc_idx = cols.index("peak count") if "peak count" in cols else len(cols) - 1
        cols.insert(pc_idx + 1, "Recluster")
    alignment = alignment[cols]

    for col_idx, col_name in enumerate(alignment.columns, start=1):
        cell = ws1.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", wrap_text=True)

    col_name_to_excel_col = {name: i + 1 for i, name in enumerate(alignment.columns)}

    for df_row_idx, (_, row) in enumerate(alignment.iterrows()):
        excel_row    = df_row_idx + 2
        orig_df_idx  = alignment.index[df_row_idx]

        for col_name in alignment.columns:
            value = row[col_name]
            if isinstance(value, (np.bool_,)):
                value = bool(value)
            elif isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = float(value) if not np.isnan(value) else None

            excel_col = col_name_to_excel_col[col_name]
            cell = ws1.cell(row=excel_row, column=excel_col, value=value)
            cell.font = bold_font if (orig_df_idx, col_name) in newly_filled else normal_font

    for col_cells in ws1.columns:
        max_len = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
        ws1.column_dimensions[get_column_letter(col_cells[0].column)].width = min(max_len + 2, 40)

    # ── Sheet 2: still-unclustered peaks ─────────────────────────────────────
    ws2 = wb.create_sheet(title="Unclustered")

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
            max_len = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
            ws2.column_dimensions[get_column_letter(col_cells[0].column)].width = min(max_len + 2, 40)
    else:
        ws2.cell(row=1, column=1, value="No unclustered peaks remaining.")

    # ── Sheet 3: cluster PNGs (transposed, Cluster N columns) ─────────────────
    ws3 = wb.create_sheet(title="Cluster PNGs")

    df_patch = _clusters_to_df(clusters)

    if not df_patch.empty:
        meta_rows = {"m/z", "Isomer_position", "Aligned_rt_apex", "peak count"}

        for col_idx, col_name in enumerate(df_patch.columns, start=1):
            cell = ws3.cell(row=1, column=col_idx, value=col_name)
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", wrap_text=True)

        for row_idx, (_, row) in enumerate(df_patch.iterrows(), start=2):
            label = str(row.iloc[0]) if row.iloc[0] else ""
            is_meta = label in meta_rows

            for col_idx, col_name in enumerate(df_patch.columns, start=1):
                value = row[col_name]
                if pd.isna(value) or value == "":
                    value = None
                elif isinstance(value, (np.integer,)):
                    value = int(value)
                elif isinstance(value, (np.floating,)):
                    value = float(value) if not np.isnan(value) else None

                cell = ws3.cell(row=row_idx, column=col_idx, value=value)
                # Bold the metadata label rows; normal for PNG rows
                cell.font = bold_font if is_meta else normal_font

        for col_cells in ws3.columns:
            max_len = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
            ws3.column_dimensions[get_column_letter(col_cells[0].column)].width = min(max_len + 2, 60)
    else:
        ws3.cell(row=1, column=1, value="No cluster patch data available.")

    wb.save(output_path)


# ── Pipeline interface ────────────────────────────────────────────────────────

def process_file_recluster(dirs: dict, group_name: str) -> str:
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


def run_group_checkpoint9(self, dirs: dict, group_name: str) -> None:
    start_time = time.time()
    msg = process_file_recluster(dirs, group_name)
    print(msg)
    elapsed = time.time() - start_time
    print(f"Checkpoint 9 (Post-clustering RT+mass recluster) completed in {elapsed:.2f} seconds!")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recluster unclustered peaks into alignment summary.")
    parser.add_argument("group", help="Group identifier, e.g. Group1")
    parser.add_argument("--base-dir",        default=".",        help="Project base directory")
    parser.add_argument("--output-root",     default="Output",   help="Output root folder name")
    parser.add_argument("--analysis-folder", default="Analysis", help="Analysis folder name")
    args = parser.parse_args()

    Config.BASE_DIR        = Path(args.base_dir)
    Config.OUTPUT_ROOT     = args.output_root
    Config.ANALYSIS_FOLDER = args.analysis_folder
    Config.CURRENT_GROUP   = args.group

    recluster_group(group=args.group, config=Config)