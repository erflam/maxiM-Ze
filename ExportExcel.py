import re
from pathlib import Path
import openpyxl
from openpyxl.styles import Font
import pandas as pd


def _group_sort_key(name: str):
    """
    Sort Group names numerically when possible:
      Group1, Group2, ..., Group10
    Falls back to the full string if no number is found.
    """
    m = re.search(r"(\d+)", str(name))
    return int(m.group(1)) if m else str(name)


def _copy_xlsx_sheet_to_worksheet(
    src_xlsx: Path,
    dest_ws,
    start_row: int = 0,
) -> int:
    """
    Copies all rows (with bold formatting) from the first sheet of src_xlsx
    into dest_ws starting at start_row (0-indexed).

    Returns the number of rows written (excluding the start offset).
    """
    src_wb = openpyxl.load_workbook(src_xlsx)
    src_ws = src_wb.active

    bold_font = Font(bold=True)
    rows_written = 0

    for src_row in src_ws.iter_rows():
        dest_row_idx = start_row + rows_written + 1  # openpyxl is 1-indexed
        for src_cell in src_row:
            dest_cell = dest_ws.cell(row=dest_row_idx, column=src_cell.column, value=src_cell.value)
            # Preserve bold from source cell
            if src_cell.font and src_cell.font.bold:
                dest_cell.font = Font(bold=True)
        rows_written += 1

    src_wb.close()
    return rows_written


def export_all_group_summaries_to_excel(Config) -> Path:
    """
    Config must provide:
      - BASE_DIR (Path)
      - OUTPUT_ROOT (str or Path)
      - ANALYSIS_FOLDER (str)
      - MASS_GROUPS (dict-like; keys are group names, e.g. "Group1", "Group2", ...)

    Writes:
      MassSelectionSummary_<ANALYSIS_FOLDER>.xlsx
    Returns:
      Path to the created Excel file
    """
    output_root = Path(Config.BASE_DIR) / Path(Config.OUTPUT_ROOT) / Config.ANALYSIS_FOLDER
    excel_path = output_root / f"MassSelectionSummary_{Config.ANALYSIS_FOLDER}.xlsx"
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"\n[Excel Export] Saving all group summaries to {excel_path}\n")

    wb = openpyxl.Workbook()
    # Remove the default empty sheet
    wb.remove(wb.active)

    for group_name in sorted(Config.MASS_GROUPS.keys(), key=_group_sort_key):
        cluster_dir = output_root / str(group_name) / "Clustering"

        feature_xlsx = cluster_dir / f"Feature_list_{group_name}.xlsx"
        summary_csv  = cluster_dir / f"alignment_summary_group_{group_name}.csv"
        unresolved_csv  = cluster_dir / f"unresolved_peaks_group_{group_name}.csv"
        unclustered_csv = cluster_dir / f"unclustered_peaks_reclustered_group_{group_name}.csv"

        # Need at least one of the two primary sources
        if not feature_xlsx.exists() and not summary_csv.exists():
            print(f"[!] Skipping {group_name}: neither Feature_list xlsx nor summary CSV found.")
            continue

        # Excel sheet name max length = 31 chars
        sheet_name = str(group_name)[:31]
        ws = wb.create_sheet(title=sheet_name)
        current_row = 0  # 0-indexed row offset for next block

        # ------------------------------------------------------------------
        # PRIMARY BLOCK: Feature_list xlsx (preferred) or fallback to CSV
        # ------------------------------------------------------------------
        if feature_xlsx.exists():
            rows_written = _copy_xlsx_sheet_to_worksheet(feature_xlsx, ws, start_row=current_row)
            print(f"[✔] Sheet '{sheet_name}' → Feature_list ({rows_written} rows) [{feature_xlsx.name}]")
            current_row += rows_written + 2  # +2 for a blank gap row
        else:
            # Fallback: plain CSV written without special formatting
            df_summary = pd.read_csv(summary_csv)
            bold_font = Font(bold=True)
            # Header
            for col_idx, col_name in enumerate(df_summary.columns, start=1):
                cell = ws.cell(row=current_row + 1, column=col_idx, value=col_name)
                cell.font = bold_font
            # Data rows
            for row_offset, (_, row) in enumerate(df_summary.iterrows(), start=1):
                for col_idx, value in enumerate(row, start=1):
                    ws.cell(row=current_row + 1 + row_offset, column=col_idx, value=value)
            rows_written = len(df_summary) + 1  # header + data
            print(f"[✔] Sheet '{sheet_name}' → Summary CSV ({len(df_summary)} rows) [{summary_csv.name}]")
            current_row += rows_written + 2

        # ------------------------------------------------------------------
        # UNRESOLVED PEAKS
        # ------------------------------------------------------------------
        if unresolved_csv.exists():
            df_unresolved = pd.read_csv(unresolved_csv)
            if not df_unresolved.empty:
                # Header
                for col_idx, col_name in enumerate(df_unresolved.columns, start=1):
                    ws.cell(row=current_row + 1, column=col_idx, value=col_name)
                # Data
                for row_offset, (_, row) in enumerate(df_unresolved.iterrows(), start=1):
                    for col_idx, value in enumerate(row, start=1):
                        ws.cell(row=current_row + 1 + row_offset, column=col_idx, value=value)
                rows_written = len(df_unresolved) + 1
                print(f"    ↳ Unresolved peaks added ({len(df_unresolved)} rows)")
                current_row += rows_written + 2
            else:
                print(f"    ↳ Skipped unresolved peaks: file is empty")
        else:
            print(f"    ↳ No unresolved peaks file found")

        # ------------------------------------------------------------------
        # UNCLUSTERED PEAKS
        # Note: if Feature_list xlsx already contains the unclustered section
        # at the bottom (as written by Recluster.py), we skip this block to
        # avoid duplicating it. Only append from CSV if no Feature_list xlsx.
        # ------------------------------------------------------------------
        if not feature_xlsx.exists() and unclustered_csv.exists():
            df_unclustered = pd.read_csv(unclustered_csv)
            if not df_unclustered.empty:
                for col_idx, col_name in enumerate(df_unclustered.columns, start=1):
                    ws.cell(row=current_row + 1, column=col_idx, value=col_name)
                for row_offset, (_, row) in enumerate(df_unclustered.iterrows(), start=1):
                    for col_idx, value in enumerate(row, start=1):
                        ws.cell(row=current_row + 1 + row_offset, column=col_idx, value=value)
                print(f"    ↳ Unclustered peaks added ({len(df_unclustered)} rows)")
                current_row += len(df_unclustered) + 3
            else:
                print(f"    ↳ Skipped unclustered peaks: file is empty")
        elif feature_xlsx.exists():
            print(f"    ↳ Unclustered peaks already included in Feature_list xlsx")
        else:
            print(f"    ↳ No unclustered peaks file found")

    wb.save(excel_path)
    print(f"\n[✔] Excel export complete → {excel_path}")
    return excel_path


def process_export_excel(Config) -> str:
    """
    Pipeline-style entrypoint. Call once after all groups finish.
    """
    excel_path = export_all_group_summaries_to_excel(Config)
    return f"Excel export complete → {excel_path}"