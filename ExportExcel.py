import re
from pathlib import Path
from typing import Optional

import pandas as pd


def _group_sort_key(name: str):
    """
    Sort Group names numerically when possible:
      Group1, Group2, ..., Group10
    Falls back to the full string if no number is found.
    """
    m = re.search(r"(\d+)", str(name))
    return int(m.group(1)) if m else str(name)


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

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        workbook = writer.book
        forced_fmt = workbook.add_format({"bold": True, "font_color": "red"})

        # ---- ORDERED SHEETS HERE ----
        for group_name in sorted(Config.MASS_GROUPS.keys(), key=_group_sort_key):
            cluster_dir = output_root / str(group_name) / "Clustering"

            summary_csv = cluster_dir / f"alignment_summary_group_{group_name}.csv"
            unresolved_csv = cluster_dir / f"unresolved_peaks_group_{group_name}.csv"
            unclustered_csv = cluster_dir / f"unclustered_peaks_reclustered_group_{group_name}.csv"

            if not summary_csv.exists():
                print(f"[!] Skipping {group_name}: summary file not found.")
                continue

            # Excel sheet name max length = 31 chars
            sheet_name = str(group_name)[:31]
            current_row = 0

            # Prefer Feature_list for export (fallback to alignment summary)
            feature_csv = cluster_dir / f"Feature_list_{group_name}.csv"
            chosen_csv = feature_csv if feature_csv.exists() else summary_csv

            df_summary = pd.read_csv(chosen_csv)
            df_summary.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
            print(f"[✔] Sheet '{sheet_name}' → Summary ({len(df_summary)} rows) [{chosen_csv.name}]")

            worksheet = writer.sheets[sheet_name]

            # --- Forced formatting ---
            # Supports either naming scheme:
            #   - New: Forced_files / Forced_any / Forced_n (from our Recluster.py)
            #   - Old: forced_files (your snippet)
            header = list(df_summary.columns)

            forced_files_col = None
            for candidate in ("Forced_files", "forced_files"):
                if candidate in header:
                    forced_files_col = candidate
                    break

            # Optional: highlight a peak_files column if present
            peak_files_col = "peak_files" if "peak_files" in header else None

            if forced_files_col is not None:
                col_forced_files = header.index(forced_files_col)

                # Map sample name -> column index for heights/areas
                sample_height_cols = {c[:-7]: header.index(c) for c in header if c.endswith("_height")}
                sample_area_cols = {c[:-5]: header.index(c) for c in header if c.endswith("_area")}

                def normalize_forced_token(tok: str) -> str:
                    # Tokens might look like:
                    #   {file_base}_mass247.154_peak4  (older style)
                    # Or just:
                    #   {file_base}                    (newer style)
                    m = re.match(r"(.+)_mass\d+(?:\.\d+)?_peak\d+$", tok, flags=re.IGNORECASE)
                    return m.group(1) if m else tok

                for i, row in df_summary.iterrows():
                    forced_files = row.get(forced_files_col, "")
                    if not isinstance(forced_files, str) or not forced_files.strip():
                        continue

                    # Accept separators "," or ";" (Recluster writes ";")
                    raw_tokens = re.split(r"[;,]", forced_files)
                    forced_samples = [normalize_forced_token(x.strip()) for x in raw_tokens if x.strip()]

                    excel_row = current_row + 1 + i  # +1 for header row

                    # Highlight the forced_files cell itself
                    worksheet.write(excel_row, col_forced_files, row.get(forced_files_col, ""), forced_fmt)

                    # Optionally highlight peak_files
                    if peak_files_col is not None:
                        cidx = header.index(peak_files_col)
                        worksheet.write(excel_row, cidx, row.get(peak_files_col, ""), forced_fmt)

                    # Highlight forced sample height/area cells
                    for s in forced_samples:
                        if s in sample_height_cols:
                            cidx = sample_height_cols[s]
                            worksheet.write(excel_row, cidx, row.get(header[cidx], ""), forced_fmt)
                        if s in sample_area_cols:
                            cidx = sample_area_cols[s]
                            worksheet.write(excel_row, cidx, row.get(header[cidx], ""), forced_fmt)

            current_row += len(df_summary) + 2

            # Unresolved peaks
            if unresolved_csv.exists():
                df_unresolved = pd.read_csv(unresolved_csv)
                if not df_unresolved.empty:
                    df_unresolved.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
                    print(f"    ↳ Unresolved peaks added ({len(df_unresolved)} rows)")
                    current_row += len(df_unresolved) + 2
                else:
                    print(f"    ↳ Skipped unresolved peaks: file is empty")
            else:
                print(f"    ↳ No unresolved peaks file found")

            # Unclustered peaks
            if unclustered_csv.exists():
                df_unclustered = pd.read_csv(unclustered_csv)
                if not df_unclustered.empty:
                    df_unclustered.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
                    print(f"    ↳ Unclustered peaks added ({len(df_unclustered)} rows)")
                    current_row += len(df_unclustered) + 2
                else:
                    print(f"    ↳ Skipped unclustered peaks: file is empty")
            else:
                print(f"    ↳ No unclustered peaks file found")

    print(f"\n[✔] Excel export complete → {excel_path}")
    return excel_path


def process_export_excel(Config) -> str:
    """
    Pipeline-style entrypoint. Call once after all groups finish.
    """
    excel_path = export_all_group_summaries_to_excel(Config)
    return f"Excel export complete → {excel_path}"
