import re
import json
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr
try:
    import cv2  # recommended for blur
except Exception:
    cv2 = None

@dataclass
class ClusterConfig:
    rt_unresolved_threshold_min: float = 0.01
    similarity_threshold: float = 0.7
    resize_hw: Tuple[int, int] = (50, 50)  # (H,W)
    blur_kernel: int = 5                  # odd integer; ignored if cv2 missing
    width_ratio_threshold: float = 0.75   # accept if width ratio >= this (corr fallback)

@dataclass
class PeakComponent:
    peak_id: str
    peak_num: int
    rt_start: float
    rt_apex: float
    rt_end: float
    pixel_start: int
    pixel_end: int
    height: float
    area: float

@dataclass
class PeakInfo:
    peak_id: str
    patch_path: Path
    file_base: str
    group: str
    mass: float
    peak_num: int

    rt_start: float
    rt_apex: float
    rt_end: float
    pixel_start: int
    pixel_end: int
    peak_height: float
    peak_area: float

    image_gray: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((1, 1), dtype=np.uint8))
    shape_profile: Optional[np.ndarray] = field(repr=False, default=None)

    merged: bool = False
    component_peaks: List[PeakComponent] = field(default_factory=list)

    def rt_width(self) -> float:
        return float(self.rt_end - self.rt_start)

class PeakClusterer:
    """
    Uses dirs dict with:
      dirs['patch']       : patch PNGs
      dirs['pixel']       : pixel CSVs
      dirs['clustering']  : outputs
    """

    PATCH_RE = re.compile(
        r"""
        ^(?P<file_base>.+?)              # everything before _mass
        _mass(?P<mass>\d+(?:\.\d+)?)     # float mass
        _Peak(?P<peak_num>\d+)           # int peak number
        _(?P<group>Group\d+)             # GroupX
        \.png$
        """,
        re.VERBOSE,
    )

    REQUIRED_COLUMNS = [
        "peak_num",
        "RT_start", "RT_apex", "RT_end",
        "pixel_start", "pixel_end",
        "height", "area",
    ]

    COLUMN_ALIASES = {
        "pixel_start": ["Pixel_start", "pixelStart", "PixelStart", "pixel_start", "pixel start", "Pixel start"],
        "pixel_end": ["Pixel_end", "pixelEnd", "PixelEnd", "pixel_end", "pixel end", "Pixel end"],
        "peak_num": ["peak_num", "Peak_num", "PeakNum", "peak number", "PeakNumber"],
        "RT_start": ["RT_start", "rt_start", "Rt_start", "RT start"],
        "RT_apex": ["RT_apex", "rt_apex", "Rt_apex", "RT apex"],
        "RT_end": ["RT_end", "rt_end", "Rt_end", "RT end"],
        "height": ["height", "Height", "peak_height", "Peak_height"],
        "area": ["area", "Area", "peak_area", "Peak_area"],
    }

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = list(df.columns)
        rename_map: Dict[str, str] = {}
        alias_to_canon: Dict[str, str] = {}
        for canon, aliases in self.COLUMN_ALIASES.items():
            for a in aliases:
                alias_to_canon[a] = canon
        existing = set(cols)
        for c in cols:
            if c in alias_to_canon:
                canon = alias_to_canon[c]
                if canon not in existing:
                    rename_map[c] = canon
        if rename_map:
            df = df.rename(columns=rename_map)
        return df

    def _assert_required_columns(self, df: pd.DataFrame, path: Path) -> pd.DataFrame:
        df = self._normalize_columns(df)
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Pixel CSV missing required columns {missing} in {path}. "
                f"Found: {list(df.columns)}"
            )
        return df

    def __init__(self, dirs: Dict[str, str | Path], config: Optional[ClusterConfig] = None):
        self.dirs = {k: Path(v) for k, v in dirs.items()}
        self.cfg = config or ClusterConfig()
        for k in ("patch", "pixel", "clustering"):
            if k not in self.dirs:
                raise KeyError(f"dirs must include '{k}'")
        self.dirs["clustering"].mkdir(parents=True, exist_ok=True)

    def run(self, group: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        patch_dir = self.dirs["patch"]
        all_files_for_group = sorted({p.stem.split("_mass")[0] for p in patch_dir.glob(f"*_{group}.png")})

        all_peaks = self._load_all_peaks_for_group(patch_dir, group)
        all_patch_stems = {p.stem for p in patch_dir.glob(f"*_{group}.png")}

        if not all_peaks:
            df_align = pd.DataFrame()
            df_unclustered = pd.DataFrame({"peak_id": sorted(all_patch_stems)})
            df_summary = pd.DataFrame()
            self._write_outputs(group, df_align, df_summary, df_unclustered)
            return df_align, df_summary, df_unclustered

        mass_peaks: Dict[float, Dict[str, List[PeakInfo]]] = {}
        for p in all_peaks:
            mass_peaks.setdefault(p.mass, {}).setdefault(p.file_base, []).append(p)

        for mass, file_map in mass_peaks.items():
            for fb, peaks in list(file_map.items()):
                file_map[fb] = self._merge_unresolved(peaks)

        accepted_rows: List[Dict[str, Any]] = []
        accepted_peak_ids: Set[str] = set()

        for mass, file_map in mass_peaks.items():
            for fb in file_map:
                file_map[fb] = sorted(file_map[fb], key=lambda x: x.rt_apex)

            max_isomers = max((len(v) for v in file_map.values()), default=0)

            full_files = {fb: peaks for fb, peaks in file_map.items() if len(peaks) == max_isomers}
            partial_files = {fb: peaks for fb, peaks in file_map.items() if len(peaks) < max_isomers}

            iso_groups: List[List[PeakInfo]] = [[] for _ in range(max_isomers)]
            for fb, peaks in full_files.items():
                for iso_idx, pk in enumerate(peaks):
                    iso_groups[iso_idx].append(pk)

            for fb, peaks in partial_files.items():
                for pk in peaks:
                    best_idx = self._find_best_isomer_group(pk, iso_groups)
                    if best_idx is not None:
                        iso_groups[best_idx].append(pk)

            for iso_idx, candidates in enumerate(iso_groups):
                if not candidates:
                    continue
                validated = self._validate_isomer_set(candidates)
                if not validated:
                    continue
                aligned = self._compute_aligned_values(validated)
                for pk in validated:
                    accepted_rows.append(self._make_alignment_row(pk, mass, iso_idx + 1, aligned))
                    accepted_peak_ids.add(pk.peak_id)

        df_align = pd.DataFrame(accepted_rows)
        if not df_align.empty:
            df_align.sort_values(["group", "mass", "isomer_position", "file", "rt_apex"], inplace=True)

        df_unclustered = pd.DataFrame({"peak_id": sorted(all_patch_stems - accepted_peak_ids)})
        df_summary = self._build_summary(df_align, group, all_files=all_files_for_group)
        self._write_outputs(group, df_align, df_summary, df_unclustered)
        return df_align, df_summary, df_unclustered

    # ------------------------------------------------------------------
    # Internal helper: build elution-rank map for a pixel CSV + mass
    # ------------------------------------------------------------------

    def _get_mz_candidates(
        self,
        df_pix: pd.DataFrame,
        mass: float,
        mz_tol: float = 0.002,
    ) -> pd.DataFrame:
        """Return rows from df_pix matching *mass* (±mz_tol), sorted by RT_apex.

        If no m/z column is present the entire DataFrame is returned sorted by
        RT_apex so downstream rank logic still works.
        """
        mz_col = next((c for c in ["m/z", "mass", "_mz_key"] if c in df_pix.columns), None)
        if mz_col is not None:
            candidates = df_pix.loc[
                (df_pix[mz_col].astype(float) - mass).abs() <= mz_tol
            ].copy()
        else:
            candidates = df_pix.copy()

        if "RT_apex" in candidates.columns:
            candidates = candidates.sort_values("RT_apex").reset_index(drop=True)

        return candidates

    def _collect_patch_entries_for_group(self, group: str) -> List[Dict[str, Any]]:
        """Collect parsed patch PNG metadata for one group."""
        patch_dir = self.dirs["patch"]
        entries: List[Dict[str, Any]] = []

        for png_path in sorted(patch_dir.glob(f"*_{group}.png")):
            m = self.PATCH_RE.match(png_path.name)
            if not m or m.group("group") != group:
                continue

            entries.append({
                "patch_path": png_path,
                "patch_file": png_path.name,
                "peak_id": png_path.stem,
                "file_base": m.group("file_base"),
                "mass": float(m.group("mass")),
                "filename_peak_number": int(m.group("peak_num")),
                "group": m.group("group"),
            })

        return entries

    def _get_mz_candidates_preserve_index(
        self,
        df_pix: pd.DataFrame,
        mass: float,
        mz_tol: float = 0.002,
    ) -> pd.DataFrame:
        """Return m/z-matched pixel rows sorted by RT_apex while preserving original index."""
        mz_col = next((c for c in ["m/z", "mass", "_mz_key"] if c in df_pix.columns), None)

        if mz_col is not None:
            candidates = df_pix.loc[
                (df_pix[mz_col].astype(float) - mass).abs() <= mz_tol
            ].copy()
        else:
            candidates = df_pix.copy()

        if "RT_apex" in candidates.columns:
            candidates = candidates.sort_values("RT_apex")

        return candidates

    def _build_filename_rank_map(
        self,
        patch_entries: List[Dict[str, Any]],
    ) -> Dict[Tuple[str, float], List[int]]:
        """Map (file_base, mass) -> sorted filename PeakX values."""
        out: Dict[Tuple[str, float], List[int]] = defaultdict(list)

        for e in patch_entries:
            out[(e["file_base"], e["mass"])].append(e["filename_peak_number"])

        return {k: sorted(v) for k, v in out.items()}

    def _write_matched_pixel_csvs(self, group: str) -> None:
        """
        Write one debug CSV per pixel CSV:
          <file_base>_peaks_pix_<group>_matched.csv

        Matching logic:
          - match by m/z first
          - if one pixel row and one patch file for that m/z, direct match
          - if multiple, sort pixel rows by RT_apex and patch files by filename PeakX
          - assign by rank
        """
        pixel_dir = self.dirs["pixel"]
        patch_entries = self._collect_patch_entries_for_group(group)

        if not patch_entries:
            return

        entries_by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for e in patch_entries:
            entries_by_file[e["file_base"]].append(e)

        for file_base, file_entries in entries_by_file.items():
            pixel_csv = pixel_dir / f"{file_base}_peaks_pix_{group}.csv"
            if not pixel_csv.exists():
                continue

            try:
                df_pix = pd.read_csv(pixel_csv)
                df_pix = self._assert_required_columns(df_pix, pixel_csv)
            except Exception:
                continue

            # Add debug columns.
            df_matched = df_pix.copy()
            df_matched["matched_peak_patch_file"] = ""
            df_matched["matched_peak_id"] = ""
            df_matched["filename_peak_number"] = np.nan
            df_matched["match_method"] = ""
            df_matched["match_warning"] = ""

            entries_by_mass: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
            for e in file_entries:
                entries_by_mass[e["mass"]].append(e)

            for mass, mass_entries in entries_by_mass.items():
                patch_sorted = sorted(mass_entries, key=lambda x: x["filename_peak_number"])
                candidates = self._get_mz_candidates_preserve_index(df_matched, mass)

                if candidates.empty:
                    warning = "no_pixel_rows_for_mz"
                    continue

                n_pix = len(candidates)
                n_patch = len(patch_sorted)

                if n_pix == 1 and n_patch == 1:
                    match_method = "single_mz_single_patch"
                else:
                    match_method = "mz_rt_rank_order"

                warning = ""
                if n_pix != n_patch:
                    warning = f"count_mismatch_pixel_rows_{n_pix}_patch_files_{n_patch}"

                # Assign in rank order.
                for rank, patch_entry in enumerate(patch_sorted):
                    if rank >= n_pix:
                        # More patch files than pixel rows. No row to annotate.
                        continue

                    original_idx = candidates.index[rank]

                    existing_peak_id = str(df_matched.at[original_idx, "matched_peak_id"])
                    if existing_peak_id:
                        row_warning = "row_already_matched"
                        if warning:
                            row_warning = warning + ";row_already_matched"
                    else:
                        row_warning = warning

                    df_matched.at[original_idx, "matched_peak_patch_file"] = patch_entry["patch_file"]
                    df_matched.at[original_idx, "matched_peak_id"] = patch_entry["peak_id"]
                    df_matched.at[original_idx, "filename_peak_number"] = patch_entry["filename_peak_number"]
                    df_matched.at[original_idx, "match_method"] = match_method
                    df_matched.at[original_idx, "match_warning"] = row_warning

            out_csv = pixel_dir / f"{file_base}_peaks_pix_{group}_matched.csv"
            df_matched.to_csv(out_csv, index=False)

    def _load_all_peaks_for_group(self, patch_dir: Path, group: str) -> List[PeakInfo]:
        """Load PeakInfo objects for every patch PNG belonging to *group*.

        Matching strategy (fixes peak_num mismatch after slicing/splitting):
          1. Collect all filename peak_nums for each (file_base, mass) pair so we
             know their sorted elution order (rank).
          2. For each PNG, filter the pixel CSV by m/z (±0.002 Da) and sort
             candidates by RT_apex.
          3. Use the filename peak_num's rank within its (file_base, mass) group
             to index into the sorted pixel CSV rows — NOT the raw peak_num value.
        """
        pixel_dir = self.dirs["pixel"]

        # --- Pass 1: collect all (file_base, mass, peak_num) from patch filenames ---
        patch_entries: List[Tuple[Path, str, float, int]] = []  # (path, file_base, mass, peak_num)
        fn_peaks_by_key: Dict[Tuple[str, float], List[int]] = defaultdict(list)

        for png_path in sorted(patch_dir.glob("*.png")):
            m = self.PATCH_RE.match(png_path.name)
            if not m or m.group("group") != group:
                continue
            file_base = m.group("file_base")
            mass = float(m.group("mass"))
            peak_num = int(m.group("peak_num"))
            patch_entries.append((png_path, file_base, mass, peak_num))
            fn_peaks_by_key[(file_base, mass)].append(peak_num)

        # Sort filename peak_nums so rank 0 = earliest-labelled = earliest eluting
        fn_peaks_sorted: Dict[Tuple[str, float], List[int]] = {
            k: sorted(v) for k, v in fn_peaks_by_key.items()
        }

        # --- Pass 2: load pixel CSV rows by elution rank ---
        peaks: List[PeakInfo] = []

        for png_path, file_base, mass, peak_num in patch_entries:
            pixel_csv = pixel_dir / f"{file_base}_peaks_pix_{group}.csv"
            if not pixel_csv.exists():
                continue

            try:
                df_pix = pd.read_csv(pixel_csv)
            except Exception:
                continue

            df_pix = self._assert_required_columns(df_pix, pixel_csv)

            # Filter by m/z and sort by RT_apex
            candidates = self._get_mz_candidates(df_pix, mass)
            if candidates.empty:
                continue

            # Determine elution rank of this filename peak_num
            fn_sorted = fn_peaks_sorted.get((file_base, mass), [peak_num])
            elution_rank = fn_sorted.index(peak_num) if peak_num in fn_sorted else 0

            # Clamp to available rows
            row_idx = min(elution_rank, len(candidates) - 1)
            r = candidates.iloc[row_idx]

            img = self._read_grayscale(png_path)
            prof = self._extract_shape_profile(img)

            peaks.append(
                PeakInfo(
                    peak_id=png_path.stem,
                    patch_path=png_path,
                    file_base=file_base,
                    group=group,
                    mass=mass,
                    peak_num=peak_num,
                    rt_start=float(r["RT_start"]),
                    rt_apex=float(r["RT_apex"]),
                    rt_end=float(r["RT_end"]),
                    pixel_start=int(r["pixel_start"]),
                    pixel_end=int(r["pixel_end"]),
                    peak_height=float(r["height"]),
                    peak_area=float(r["area"]),
                    image_gray=img,
                    shape_profile=prof,
                    merged=False,
                    component_peaks=[],
                )
            )

        return peaks

    def _merge_unresolved(self, peaks: List[PeakInfo]) -> List[PeakInfo]:
        if len(peaks) <= 1:
            return peaks

        peaks_sorted = sorted(peaks, key=lambda x: x.rt_apex)
        groups: List[List[PeakInfo]] = []
        cur = [peaks_sorted[0]]

        for pk in peaks_sorted[1:]:
            if abs(pk.rt_apex - cur[-1].rt_apex) < self.cfg.rt_unresolved_threshold_min:
                cur.append(pk)
            else:
                groups.append(cur)
                cur = [pk]
        groups.append(cur)

        out: List[PeakInfo] = []
        for g in groups:
            if len(g) == 1:
                out.append(g[0])
                continue

            rt_start = min(p.rt_start for p in g)
            rt_end = max(p.rt_end for p in g)
            pixel_start = min(p.pixel_start for p in g)
            pixel_end = max(p.pixel_end for p in g)
            peak_area = float(sum(p.peak_area for p in g))
            peak_height = float(max(p.peak_height for p in g))

            heights = np.array([p.peak_height for p in g], dtype=float)
            apexes = np.array([p.rt_apex for p in g], dtype=float)
            w = heights / heights.sum() if heights.sum() > 0 else np.ones_like(heights) / len(heights)
            rt_apex = float((w * apexes).sum())

            rep = max(g, key=lambda p: p.peak_height)
            merged_id = rep.peak_id + "__MERGED"

            out.append(
                PeakInfo(
                    peak_id=merged_id,
                    patch_path=rep.patch_path,
                    file_base=rep.file_base,
                    group=rep.group,
                    mass=rep.mass,
                    peak_num=rep.peak_num,
                    rt_start=rt_start,
                    rt_apex=rt_apex,
                    rt_end=rt_end,
                    pixel_start=pixel_start,
                    pixel_end=pixel_end,
                    peak_height=peak_height,
                    peak_area=peak_area,
                    image_gray=rep.image_gray,
                    shape_profile=rep.shape_profile,
                    merged=True,
                    component_peaks=[
                        PeakComponent(
                            peak_id=p.peak_id,
                            peak_num=p.peak_num,
                            rt_start=p.rt_start,
                            rt_apex=p.rt_apex,
                            rt_end=p.rt_end,
                            pixel_start=p.pixel_start,
                            pixel_end=p.pixel_end,
                            height=p.peak_height,
                            area=p.peak_area,
                        )
                        for p in sorted(g, key=lambda x: x.rt_apex)
                    ],
                )
            )

        return out

    def _find_best_isomer_group(self, pk: PeakInfo, iso_groups: List[List[PeakInfo]]) -> Optional[int]:
        """
        Assign a peak from a partial file to the best isomer group purely by shape similarity.
        Returns None if no group scores above similarity_threshold, or if shape profile is missing.
        """
        if pk.shape_profile is None:
            return None

        best_idx = None
        best_score = -np.inf

        for idx, group in enumerate(iso_groups):
            if not group:
                continue

            shape_scores = []
            for g in group:
                if g.shape_profile is not None:
                    c = self._safe_corr(pk.shape_profile, g.shape_profile)
                    if not np.isnan(c):
                        shape_scores.append(c)

            if not shape_scores:
                continue

            score = float(np.mean(shape_scores))
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_score < self.cfg.similarity_threshold:
            return None

        return best_idx

    def _validate_isomer_set(self, peaks: List[PeakInfo]) -> List[PeakInfo]:
        if len(peaks) <= 1:
            return peaks

        profiles = [p.shape_profile for p in peaks]
        if any(pr is None for pr in profiles):
            return peaks

        n = len(peaks)
        corr = np.full((n, n), np.nan, dtype=float)
        for i in range(n):
            corr[i, i] = 1.0
            for j in range(i + 1, n):
                c = self._safe_corr(profiles[i], profiles[j])
                corr[i, j] = c
                corr[j, i] = c

        avg = np.nanmean(corr, axis=1)
        ref_idx = int(np.nanargmax(avg))
        ref_prof = profiles[ref_idx]
        ref_width = peaks[ref_idx].rt_width() if peaks[ref_idx].rt_width() > 0 else 1e-9

        validated: List[PeakInfo] = []
        for pk, prof in zip(peaks, profiles):
            c = self._safe_corr(ref_prof, prof)
            if c >= self.cfg.similarity_threshold:
                validated.append(pk)
                continue

            w = pk.rt_width() if pk.rt_width() > 0 else 1e-9
            width_ratio = min(w, ref_width) / max(w, ref_width)
            if width_ratio >= self.cfg.width_ratio_threshold:
                validated.append(pk)

        return validated

    def _safe_corr(self, a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float).ravel()
        b = np.asarray(b, dtype=float).ravel()
        if a.size == 0 or b.size == 0 or a.size != b.size:
            return float("nan")
        if np.allclose(a, a[0]) or np.allclose(b, b[0]):
            return float("nan")
        return float(pearsonr(a, b)[0])

    def _compute_aligned_values(self, peaks: List[PeakInfo]) -> Dict[str, float]:
        return {
            "aligned_rt_start": float(np.mean([p.rt_start for p in peaks])),
            "aligned_rt_apex": float(np.mean([p.rt_apex for p in peaks])),
            "aligned_rt_end": float(np.mean([p.rt_end for p in peaks])),
            "aligned_pixel_start": float(np.mean([p.pixel_start for p in peaks])),
            "aligned_pixel_end": float(np.mean([p.pixel_end for p in peaks])),
        }

    def _make_alignment_row(self, pk: PeakInfo, mass: float, isomer_position: int, aligned: Dict[str, float]) -> Dict[str, Any]:
        return {
            "group": pk.group,
            "mass": float(mass),
            "isomer_position": int(isomer_position),
            "file": pk.file_base,
            "peak_id": pk.peak_id,
            "peak_num": int(pk.peak_num),
            "rt_start": pk.rt_start,
            "rt_apex": pk.rt_apex,
            "rt_end": pk.rt_end,
            "aligned_rt_start": aligned["aligned_rt_start"],
            "aligned_rt_apex": aligned["aligned_rt_apex"],
            "aligned_rt_end": aligned["aligned_rt_end"],
            "pixel_start": int(pk.pixel_start),
            "pixel_end": int(pk.pixel_end),
            "aligned_pixel_start": aligned["aligned_pixel_start"],
            "aligned_pixel_end": aligned["aligned_pixel_end"],
            "height": float(pk.peak_height),
            "area": float(pk.peak_area),
            "merged": bool(pk.merged),
            "component_peaks_json": json.dumps([asdict(c) for c in pk.component_peaks]) if pk.merged else "",
            "patch_path": str(pk.patch_path),
        }

    def _build_summary(self, df_align: pd.DataFrame, group: str, all_files: Optional[List[str]] = None) -> pd.DataFrame:
        if df_align.empty:
            return pd.DataFrame()

        expanded: List[Dict[str, Any]] = []
        for _, r in df_align.iterrows():
            base = {
                "group": r["group"],
                "mass": float(r["mass"]),
                "isomer_position": int(r["isomer_position"]),
                "file": r["file"],
                "aligned_rt_apex": float(r["aligned_rt_apex"]),
            }

            if bool(r.get("merged", False)) and isinstance(r.get("component_peaks_json", ""), str) and r["component_peaks_json"]:
                comps = json.loads(r["component_peaks_json"])
                comps = sorted(comps, key=lambda c: float(c["rt_apex"]))
                for i, c in enumerate(comps, start=1):
                    expanded.append({**base, "component_number": i, "height": float(c["height"]), "area": float(c["area"])})
            else:
                expanded.append({**base, "component_number": 1, "height": float(r["height"]), "area": float(r["area"])})

        df_exp = pd.DataFrame(expanded)

        idx = ["group", "mass", "isomer_position", "aligned_rt_apex", "component_number"]
        h = df_exp.pivot_table(index=idx, columns="file", values="height", aggfunc="first")
        a = df_exp.pivot_table(index=idx, columns="file", values="area", aggfunc="first")
        if all_files:
            h = h.reindex(columns=all_files)
            a = a.reindex(columns=all_files)
        h.columns = [f"{c}_height" for c in h.columns]
        a.columns = [f"{c}_area" for c in a.columns]

        df_wide = pd.concat([h, a], axis=1).reset_index()
        df_wide = df_wide.loc[df_wide["group"] == group].copy()

        height_cols = [c for c in df_wide.columns if c.endswith("_height")]
        area_cols = [c for c in df_wide.columns if c.endswith("_area")]
        group_cols = ["group", "mass", "isomer_position", "aligned_rt_apex"]

        agg_spec = {c: "max" for c in height_cols}
        agg_spec.update({c: "sum" for c in area_cols})
        df_wide = df_wide.groupby(group_cols, as_index=False).agg(agg_spec)

        df_wide["peak_count"] = df_wide[height_cols].notna().sum(axis=1).astype(int)
        df_wide = df_wide.rename(columns={
            "group": "Group", "mass": "m/z", "isomer_position": "Isomer_position",
            "aligned_rt_apex": "Aligned_rt_apex", "peak_count": "peak count",
        })

        meta_cols = ["Group", "m/z", "Isomer_position", "Aligned_rt_apex", "peak count"]
        height_cols2 = sorted([c for c in df_wide.columns if c.endswith("_height")])
        area_cols2 = sorted([c for c in df_wide.columns if c.endswith("_area")])
        df_wide = df_wide[meta_cols + height_cols2 + area_cols2]
        df_wide.sort_values(["m/z", "Isomer_position"], inplace=True)
        return df_wide

    def _build_cluster_patch(self, df_align: pd.DataFrame) -> pd.DataFrame:
        """
        Builds a transposed manifest where each cluster is a column (Cluster 1, Cluster 2, ...),
        and rows are: m/z, Isomer_position, Aligned_rt_apex, peak count, then one row per PNG.

        Layout:
                                Cluster 1                               Cluster 2
        m/z                     86.0964                                 86.0964
        Isomer_position         1                                       2
        Aligned_rt_apex         2.192767442                             2.503651163
        peak count              43                                      43
                                C007_0002_mass86.0964_Peak3_Group3.png  C007_0002_mass86.0964_Peak4_Group3.png
                                C009_0002_mass86.0964_Peak3_Group3.png  C009_0002_mass86.0964_Peak4_Group3.png
        """
        if df_align.empty:
            return pd.DataFrame()

        group_cols = ["group", "mass", "isomer_position", "aligned_rt_apex"]

        clusters: List[Dict] = []
        for key, grp in df_align.groupby(group_cols, sort=True):
            group_val, mass_val, iso_pos, aligned_rt = key
            patch_names = sorted([Path(p).name for p in grp["patch_path"].tolist()])
            clusters.append({
                "m/z": mass_val,
                "Isomer_position": int(iso_pos),
                "Aligned_rt_apex": aligned_rt,
                "peak count": len(patch_names),
                "patches": patch_names,
            })

        col_names = [f"Cluster {i + 1}" for i in range(len(clusters))]
        max_patches = max((len(c["patches"]) for c in clusters), default=0)

        # Row labels: 4 metadata rows + one blank-label row per PNG
        meta_labels = ["m/z", "Isomer_position", "Aligned_rt_apex", "peak count"]
        index_col = meta_labels + [""] * max_patches

        data: Dict[str, List] = {"": index_col}
        for col, cluster in zip(col_names, clusters):
            col_values: List[Any] = [
                cluster["m/z"],
                cluster["Isomer_position"],
                cluster["Aligned_rt_apex"],
                cluster["peak count"],
            ]
            # Pad patch list to max_patches so all columns are the same length
            patches = cluster["patches"] + [""] * (max_patches - len(cluster["patches"]))
            col_values.extend(patches)
            data[col] = col_values

        return pd.DataFrame(data)

    def _read_grayscale(self, png_path: Path) -> np.ndarray:
        with Image.open(png_path) as im:
            im = im.convert("L")
            return np.array(im, dtype=np.uint8)

    def _extract_shape_profile(self, gray: np.ndarray) -> np.ndarray:
        H, W = self.cfg.resize_hw
        im = Image.fromarray(gray).resize((W, H), resample=Image.BILINEAR)
        arr = np.array(im, dtype=np.float32)

        if cv2 is not None and self.cfg.blur_kernel >= 3 and self.cfg.blur_kernel % 2 == 1:
            arr = cv2.GaussianBlur(arr, (self.cfg.blur_kernel, self.cfg.blur_kernel), 0)

        prof = arr.mean(axis=0)  # length W
        mn, mx = float(prof.min()), float(prof.max())
        if mx - mn < 1e-9:
            return np.zeros_like(prof, dtype=np.float32)
        return ((prof - mn) / (mx - mn)).astype(np.float32)

    def _enrich_unclustered(self, df_unclustered: pd.DataFrame, group: str) -> pd.DataFrame:
        """Add m/z, RT_apex, RT_start, RT_end, height, and area columns to unclustered peaks.

        Uses the same robust matching strategy as peak loading:
          1. Match by m/z.
          2. Sort pixel CSV rows by RT_apex.
          3. Sort all patch filenames for the same file_base/mass by filename PeakX.
          4. Use filename PeakX only as an ordering label, not as peak_num lookup.
        """
        if df_unclustered.empty:
            return df_unclustered

        pixel_dir = self.dirs["pixel"]

        all_patch_entries = self._collect_patch_entries_for_group(group)
        rank_map = self._build_filename_rank_map(all_patch_entries)

        patch_by_peak_id = {
            e["peak_id"]: e
            for e in all_patch_entries
        }

        records = []

        for peak_id in df_unclustered["peak_id"]:
            e = patch_by_peak_id.get(peak_id)

            if e is None:
                records.append({
                    "peak_id": peak_id,
                    "m/z": None,
                    "RT_start": None,
                    "RT_apex": None,
                    "RT_end": None,
                    "height": None,
                    "area": None,
                    "match_method": "unparsed_peak_id",
                    "match_warning": "peak_id_not_found_in_patch_files",
                })
                continue

            file_base = e["file_base"]
            mass = e["mass"]
            filename_peak_number = e["filename_peak_number"]

            pixel_csv = pixel_dir / f"{file_base}_peaks_pix_{group}.csv"

            rt_start = rt_apex = rt_end = height = area = None
            match_method = ""
            match_warning = ""

            if not pixel_csv.exists():
                match_warning = "missing_pixel_csv"
            else:
                try:
                    df_pix = pd.read_csv(pixel_csv)
                    df_pix = self._assert_required_columns(df_pix, pixel_csv)

                    candidates = self._get_mz_candidates(df_pix, mass)

                    if candidates.empty:
                        match_warning = "no_pixel_rows_for_mz"
                    else:
                        fn_sorted = rank_map.get((file_base, mass), [filename_peak_number])
                        elution_rank = (
                            fn_sorted.index(filename_peak_number)
                            if filename_peak_number in fn_sorted
                            else 0
                        )

                        if len(candidates) == 1 and len(fn_sorted) == 1:
                            match_method = "single_mz_single_patch"
                        else:
                            match_method = "mz_rt_rank_order"

                        if len(candidates) != len(fn_sorted):
                            match_warning = (
                                f"count_mismatch_pixel_rows_{len(candidates)}"
                                f"_patch_files_{len(fn_sorted)}"
                            )

                        row_idx = min(elution_rank, len(candidates) - 1)
                        r = candidates.iloc[row_idx]

                        rt_start = float(r["RT_start"])
                        rt_apex = float(r["RT_apex"])
                        rt_end = float(r["RT_end"])
                        height = float(r["height"])
                        area = float(r["area"])

                except Exception as ex:
                    match_warning = f"lookup_error:{type(ex).__name__}"

            records.append({
                "peak_id": peak_id,
                "m/z": mass,
                "RT_start": rt_start,
                "RT_apex": rt_apex,
                "RT_end": rt_end,
                "height": height,
                "area": area,
                "filename_peak_number": filename_peak_number,
                "match_method": match_method,
                "match_warning": match_warning,
            })

        return pd.DataFrame(records)

    def _write_outputs(self, group: str, df_align: pd.DataFrame, df_summary: pd.DataFrame,
                       df_unclustered: pd.DataFrame) -> None:
        outdir = self.dirs["clustering"]
        self._write_matched_pixel_csvs(group)
        df_align.to_csv(outdir / "peak_alignment.csv", index=False)
        df_unclustered = self._enrich_unclustered(df_unclustered, group)
        df_unclustered.to_csv(outdir / f"unclustered_peaks_group_{group}.csv", index=False)
        df_summary.to_csv(outdir / f"alignment_summary_group_{group}.csv", index=False)
        df_patch = self._build_cluster_patch(df_align)
        df_patch.to_csv(outdir / f"cluster_patch_{group}.csv", index=False)

def process_file_cluster_peaks(dirs: Dict[str, str | Path], group_name: str, config: Optional[ClusterConfig] = None) -> str:
    """
    Pipeline stage entrypoint.

    Writes:
      - <clustering>/peak_alignment.csv
      - <clustering>/alignment_summary_group_<GroupX>.csv
      - <clustering>/unclustered_peaks_group_<GroupX>.csv
      - <clustering>/cluster_patch_<GroupX>.csv
    """
    t0 = time.time()
    clusterer = PeakClusterer(dirs=dirs, config=config)
    df_align, df_summary, df_unclustered = clusterer.run(group=group_name)
    elapsed = time.time() - t0

    return (
        f"Clustering complete for {group_name}. "
        f"Aligned peaks: {0 if df_align.empty else len(df_align)} rows. "
        f"Summary rows: {0 if df_summary.empty else len(df_summary)}. "
        f"Unclustered patches: {len(df_unclustered)}. "
        f"Elapsed: {elapsed:.2f} s."
    )