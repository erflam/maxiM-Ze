import os
import numpy as np
import pandas as pd

DATA_FILE = r"/Users/elizabethflammer/Desktop/maxiMiZe Tests/maxiMiZe Checkpoints/maxiMZe Group 1-30 0304 Test 1/MassSelectionSummary_maxiMZe Group 1-30 0304 Test 1.xlsx"
LIB_FILE  = r"/Users/elizabethflammer/Desktop/Research/MZMine/POS OE Library Metformin Baseline.csv"

MZ_TOL = 0.0005   # adjustable
RT_TOL = 0.1      # minutes, adjustable

OUTPUT_FILE = os.path.splitext(DATA_FILE)[0] + "_with_library_matches.xlsx"

def read_library(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        lib = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        lib = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported library file type: {ext}. Use .csv or .xlsx")

    # Normalize column names for matching (case-insensitive)
    col_map = {c.lower().strip(): c for c in lib.columns}

    # Required columns
    required = ["mz", "rt", "name"]
    missing = [r for r in required if r not in col_map]
    if missing:
        raise ValueError(
            f"Library file missing required columns: {missing}. Found: {list(lib.columns)}"
        )

    # Optional columns
    has_adduct = "adduct" in col_map
    has_formula = "formula" in col_map

    # Rename to canonical names
    rename_dict = {
        col_map["mz"]: "mz",
        col_map["rt"]: "rt",
        col_map["name"]: "name",
    }
    if has_adduct:
        rename_dict[col_map["adduct"]] = "adduct"
    if has_formula:
        rename_dict[col_map["formula"]] = "formula"

    lib = lib.rename(columns=rename_dict)

    # Ensure optional columns exist (keeps downstream logic simple)
    if "adduct" not in lib.columns:
        lib["adduct"] = np.nan
    if "formula" not in lib.columns:
        lib["formula"] = np.nan

    # Coerce numeric
    lib["mz"] = pd.to_numeric(lib["mz"], errors="coerce")
    lib["rt"] = pd.to_numeric(lib["rt"], errors="coerce")
    lib = lib.dropna(subset=["mz", "rt"]).reset_index(drop=True)

    return lib

def best_match_for_row(
    row_mz: float,
    row_rt: float,
    lib: pd.DataFrame,
    mz_tol: float,
    rt_tol: float,
    top_n_secondary: int = 5
):
    """
    Returns (best_adduct, best_name, best_formula, secondary_matches_str) or NaNs if none.

    Best match = smallest |Δmz|, then smallest |Δrt|.
    Secondary matches = other candidates within tolerance, ranked similarly,
    formatted WITHOUT formulas (only Adduct and/or Name if available).
    """
    if pd.isna(row_mz) or pd.isna(row_rt):
        return (np.nan, np.nan, np.nan, np.nan)

    mz_diff = (lib["mz"] - row_mz).abs()
    rt_diff = (lib["rt"] - row_rt).abs()

    candidates = lib[(mz_diff <= mz_tol) & (rt_diff <= rt_tol)].copy()
    if candidates.empty:
        return (np.nan, np.nan, np.nan, np.nan)

    candidates["dmz"] = (candidates["mz"] - row_mz).abs()
    candidates["drt"] = (candidates["rt"] - row_rt).abs()
    candidates = candidates.sort_values(["dmz", "drt"], ascending=[True, True]).reset_index(drop=True)

    best = candidates.iloc[0]

    best_adduct = best.get("adduct", np.nan)
    if pd.isna(best_adduct) or str(best_adduct).strip() == "":
        best_adduct = np.nan

    best_name = best.get("name", np.nan)

    best_formula = best.get("formula", np.nan)
    if pd.isna(best_formula) or str(best_formula).strip() == "":
        best_formula = np.nan

    # Secondary matches (NO formulas included)
    secondary = candidates.iloc[1:1 + top_n_secondary]
    if secondary.empty:
        secondary_str = np.nan
    else:
        parts = []
        for _, r in secondary.iterrows():
            label_bits = []

            adduct_val = r.get("adduct", np.nan)
            if pd.notna(adduct_val) and str(adduct_val).strip() != "":
                label_bits.append(str(adduct_val).strip())

            name_val = r.get("name", np.nan)
            if pd.notna(name_val) and str(name_val).strip() != "":
                label_bits.append(str(name_val).strip())

            label = " | ".join(label_bits) if label_bits else "(match)"
            parts.append(f"{label} (Δmz={r['dmz']:.6f}, Δrt={r['drt']:.3f})")

        secondary_str = "; ".join(parts)

    return (best_adduct, best_name, best_formula, secondary_str)

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
        lambda r: best_match_for_row(r["m/z"], r["Aligned_rt_apex"], lib, MZ_TOL, RT_TOL, top_n_secondary=5),
        axis=1,
        result_type="expand"
    )
    matches.columns = ["Adduct", "Library Match", "Formula", "Secondary Matches"]

    # Build output Sheet 1
    out = data_s1.copy()
    out = pd.concat([out, matches], axis=1)

    # Rename
    out = out.rename(columns={
        "Group": "Processing Group",
        "m/z": "MZ",
        "Aligned_rt_apex": "Aligned RT",
    })

    # Reorder columns (Formula next to Library Match; Secondary Matches after Formula)
    fixed_cols = [
        "Processing Group",
        "MZ",
        "Adduct",
        "Aligned RT",
        "Library Match",
        "Formula",
        "Secondary Matches",
        "Recluster",
    ]
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