import os
import numpy as np
import pandas as pd

# =========================
# User settings
# =========================
DATA_FILE = r"/Users/elizabethflammer/Desktop/maxiMiZe Tests/maxiMiZe Checkpoints/maxiMZe Group 1-30 0304 Test 1/MassSelectionSummary_maxiMZe Group 1-30 0304 Test 1.xlsx"
LIB_FILE  = r"/Users/elizabethflammer/Desktop/Research/MZMine/POS OE Library.csv"

MZ_TOL = 0.0005   # adjustable
RT_TOL = 0.08     # minutes, adjustable

OUTPUT_FILE = os.path.splitext(DATA_FILE)[0] + "_with_library_matches.xlsx"


# =========================
# Helpers
# =========================
def read_library(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        lib = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        lib = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported library file type: {ext}. Use .csv or .xlsx")

    # Normalize expected columns (case-insensitive match)
    col_map = {c.lower().strip(): c for c in lib.columns}
    required = ["adduct", "mz", "rt", "name"]
    missing = [r for r in required if r not in col_map]
    if missing:
        raise ValueError(f"Library file missing required columns: {missing}. Found: {list(lib.columns)}")

    lib = lib.rename(columns={
        col_map["adduct"]: "adduct",
        col_map["mz"]: "mz",
        col_map["rt"]: "rt",
        col_map["name"]: "name",
    })

    lib["mz"] = pd.to_numeric(lib["mz"], errors="coerce")
    lib["rt"] = pd.to_numeric(lib["rt"], errors="coerce")
    lib = lib.dropna(subset=["mz", "rt"]).reset_index(drop=True)
    return lib


def best_match_for_row(row_mz: float, row_rt: float, lib: pd.DataFrame, mz_tol: float, rt_tol: float):
    """
    Returns (adduct, name) for the best match, or (NaN, NaN) if none.
    Best match = smallest |mz diff|, then smallest |rt diff|.
    """
    if pd.isna(row_mz) or pd.isna(row_rt):
        return (np.nan, np.nan)

    mz_diff = (lib["mz"] - row_mz).abs()
    rt_diff = (lib["rt"] - row_rt).abs()

    candidates = lib[(mz_diff <= mz_tol) & (rt_diff <= rt_tol)]
    if candidates.empty:
        return (np.nan, np.nan)

    # rank by mz diff, then rt diff
    c_mz_diff = (candidates["mz"] - row_mz).abs()
    c_rt_diff = (candidates["rt"] - row_rt).abs()
    best_idx = np.lexsort((c_rt_diff.to_numpy(), c_mz_diff.to_numpy()))[0]
    best = candidates.iloc[best_idx]
    return (best["adduct"], best["name"])


# =========================
# Main
# =========================
def main():
    # Read data file sheets
    data_s1 = pd.read_excel(DATA_FILE, sheet_name=0)  # Sheet 1
    data_s2 = pd.read_excel(DATA_FILE, sheet_name=1)  # Sheet 2 (untouched)

    # Validate required columns in Sheet 1
    required_s1 = ["Group", "m/z", "Aligned_rt_apex", "Recluster"]
    missing_s1 = [c for c in required_s1 if c not in data_s1.columns]
    if missing_s1:
        raise ValueError(f"Data file Sheet 1 missing columns: {missing_s1}. Found: {list(data_s1.columns)}")

    # Load library
    lib = read_library(LIB_FILE)

    # Coerce numeric
    data_s1["m/z"] = pd.to_numeric(data_s1["m/z"], errors="coerce")
    data_s1["Aligned_rt_apex"] = pd.to_numeric(data_s1["Aligned_rt_apex"], errors="coerce")

    # Match row-by-row
    matches = data_s1.apply(
        lambda r: best_match_for_row(r["m/z"], r["Aligned_rt_apex"], lib, MZ_TOL, RT_TOL),
        axis=1,
        result_type="expand"
    )
    matches.columns = ["Adduct", "Library Match"]

    # Build output Sheet 1:
    # Col 1: Group -> “Processing Group”
    # Col 2: m/z -> “MZ”
    # Col 3: Adduct
    # Col 4: Aligned_rt_apex -> “Aligned RT”
    # Col 5: Library Match
    # Col 6: Recluster
    # Col 7+: Samples (keep whatever other columns exist in original order, excluding ones already placed)
    out = data_s1.copy()
    out = pd.concat([out, matches], axis=1)

    # Rename
    out = out.rename(columns={
        "Group": "Processing Group",
        "m/z": "MZ",
        "Aligned_rt_apex": "Aligned RT",
    })

    # Reorder columns
    fixed_cols = ["Processing Group", "MZ", "Adduct", "Aligned RT", "Library Match", "Recluster"]
    # "Adduct" and "Library Match" came from matches
    # Keep remaining columns (samples etc.) in original order
    remaining = [c for c in out.columns if c not in fixed_cols]
    out = out[fixed_cols + remaining]

    # Write output workbook
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="Sheet1", index=False)
        data_s2.to_excel(writer, sheet_name="Sheet2", index=False)

    print(f"Done. Wrote: {OUTPUT_FILE}")
    print(f"Used tolerances: MZ_TOL={MZ_TOL}, RT_TOL={RT_TOL} min")


if __name__ == "__main__":
    main()
