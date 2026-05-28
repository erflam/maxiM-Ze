import os
import re
from pathlib import Path
from Config import Config
import pandas as pd
import numpy as np
from numba import njit, prange
from pyteomics import mzml, mzxml


# ── Scan cache (CSR format) ───────────────────────────────────────────────────
#
# Each input file is parsed from XML exactly once per run. The raw scan data is
# stored on disk as a compressed .npz in CSR (Compressed Sparse Row) format:
#
#   scan_nums  : int32[n_scans]       — scan numbers
#   rts        : float32[n_scans]     — retention times
#   offsets    : int64[n_scans + 1]   — offsets[i]:offsets[i+1] = slice for scan i
#   mzs_flat   : float32[total_pts]   — all m/z values concatenated
#   ints_flat  : float32[total_pts]   — all intensity values concatenated
#
# Subsequent groups load this binary cache (~10-50× faster than XML parsing)
# and run the numba CSR extraction directly on the flat arrays.
# ─────────────────────────────────────────────────────────────────────────────

def _scan_cache_path(fp: str) -> Path:
    base = os.path.splitext(os.path.basename(fp))[0]
    cache_dir = Config._analysis_output_root() / "scan_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{base}.npz"


def _save_scan_cache(fp: str, scan_nums, rts, offsets, mzs_flat, ints_flat) -> None:
    np.savez_compressed(
        str(_scan_cache_path(fp)),
        scan_nums=np.array(scan_nums, dtype=np.int32),
        rts=np.array(rts, dtype=np.float32),
        offsets=np.array(offsets, dtype=np.int64),
        mzs_flat=mzs_flat.astype(np.float32),
        ints_flat=ints_flat.astype(np.float32),
    )


def _load_scan_cache(fp: str):
    """Returns (scan_nums, rts, offsets, mzs_flat, ints_flat) or None."""
    path = _scan_cache_path(fp)
    if not path.exists():
        return None
    d = np.load(str(path))
    return d['scan_nums'], d['rts'], d['offsets'], d['mzs_flat'], d['ints_flat']


# ── Numba CSR extraction ──────────────────────────────────────────────────────

@njit(parallel=True)
def _eic_from_csr(rts, offsets, mzs_flat, ints_flat, mass_list, tolerance):
    """
    Extract EIC intensities from CSR scan cache for all masses in parallel.

    Returns a (n_scans, n_masses) float32 matrix.
    Scans are parallelised across CPU cores via prange.
    """
    n_scans = len(rts)
    n_masses = len(mass_list)
    out = np.zeros((n_scans, n_masses), dtype=np.float32)
    for s in prange(n_scans):
        start = offsets[s]
        end = offsets[s + 1]
        for m in range(n_masses):
            target = mass_list[m]
            total = np.float32(0.0)
            for k in range(start, end):
                if abs(mzs_flat[k] - target) <= tolerance:
                    total += ints_flat[k]
            out[s, m] = total
    return out


def _eic_matrix_to_df(scan_nums, rts, intensity_matrix, mass_list):
    """Convert the (n_scans, n_masses) matrix to the long-format DataFrame."""
    rows_s, rows_m = np.where(intensity_matrix > 0)
    if len(rows_s) == 0:
        return pd.DataFrame(columns=['scan', 'rt', 'mass', 'intensity'])
    return pd.DataFrame({
        'scan':      scan_nums[rows_s],
        'rt':        rts[rows_s],
        'mass':      mass_list[rows_m],
        'intensity': intensity_matrix[rows_s, rows_m],
    })


# ── MSFileAnalyzer (unchanged public API) ────────────────────────────────────

class MSFileAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.base_name = os.path.splitext(os.path.basename(file_path))[0]
        self._cached_eic = None

    def get_reader(self):
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext == '.mzxml':
            return mzxml.read(self.file_path, use_index=True, huge_tree=True)
        elif ext == '.mzml':
            return mzml.read(self.file_path, use_index=True, huge_tree=True)
        raise ValueError(f"Unsupported file format: {ext}")

    def get_retention_time(self, scan):
        try:
            rt_val = scan.get('retentionTime')
            if rt_val is not None:
                if isinstance(rt_val, str):
                    m = re.match(r'PT(?P<v>[\d\.]+)S', rt_val)
                    return float(m.group('v')) if m else float(rt_val)
                return float(rt_val)

            scan_list = scan.get('scanList', {}).get('scan', [])
            if scan_list:
                for cv in scan_list[0].get('cvParam', []):
                    if cv.get('accession') == 'MS:1000016':
                        val = cv.get('value')
                        if isinstance(val, str) and val.startswith('PT') and val.endswith('S'):
                            return float(val[2:-1])
                        return float(val)
        except Exception as e:
            print(f"RT extraction warning: {str(e)}")
        raise KeyError("No retention time found")

    @staticmethod
    @njit(parallel=True)
    def fast_eic_extraction_parallel(mzs, ints, mass_list, tolerance):
        results = np.zeros(len(mass_list))
        for i in prange(len(mass_list)):
            mz_target = mass_list[i]
            mask = np.abs(mzs - mz_target) <= tolerance
            results[i] = ints[mask].sum() if mask.any() else 0.0
        return results

    @staticmethod
    @njit
    def fast_eic_extraction(mzs, ints, mass_list, tolerance):
        results = np.zeros(len(mass_list))
        for i, mz_target in enumerate(mass_list):
            mask = np.abs(mzs - mz_target) <= tolerance
            results[i] = ints[mask].sum() if mask.any() else 0.0
        return results

    def _convert_array_dtype(self, arr):
        if arr.dtype.byteorder == '>' or arr.dtype == '>f4':
            return arr.astype(np.float32)
        return arr

    def extract_eic(self):
        if self._cached_eic is not None:
            return self._cached_eic

        records = []
        mass_list = np.array(Config.MASS_LIST, dtype=np.float32)

        try:
            with self.get_reader() as reader:
                for scan in reader:
                    try:
                        rt = self.get_retention_time(scan)
                        scan_num = scan.get('num', None)
                        mzs = self._convert_array_dtype(scan['m/z array'])
                        ints = self._convert_array_dtype(scan['intensity array'])
                        intensities = self.fast_eic_extraction_parallel(mzs, ints, mass_list, Config.MASS_TOLERANCE)
                        for m, i in zip(mass_list, intensities):
                            if i > 0:
                                records.append({'scan': scan_num, 'rt': rt, 'mass': m, 'intensity': i})
                    except KeyError:
                        continue
                    except Exception as e:
                        print(f"Skipping scan due to error: {str(e)}")
                        continue
        except Exception as e:
            print(f"Fatal error processing {self.file_path}: {str(e)}")
            raise

        self._cached_eic = pd.DataFrame(records)
        return self._cached_eic


class MSFileAnalyzerOptimized(MSFileAnalyzer):
    """Optimized version with CSR disk cache — parses XML at most once per run."""

    def _read_and_cache(self):
        """
        Parse the mzML/mzXML file, build CSR arrays, save to disk cache,
        and return (scan_nums, rts, offsets, mzs_flat, ints_flat).
        """
        scan_nums = []
        rts = []
        mzs_chunks = []
        ints_chunks = []
        offsets = [0]

        try:
            with self.get_reader() as reader:
                for scan in reader:
                    try:
                        rt = self.get_retention_time(scan)
                        scan_num = scan.get('num', None)
                        mzs = self._convert_array_dtype(scan['m/z array'])
                        ints = self._convert_array_dtype(scan['intensity array'])

                        scan_nums.append(scan_num)
                        rts.append(rt)
                        mzs_chunks.append(mzs)
                        ints_chunks.append(ints)
                        offsets.append(offsets[-1] + len(mzs))

                    except KeyError:
                        continue
                    except Exception as e:
                        print(f"Skipping scan due to error: {str(e)}")
                        continue
        except Exception as e:
            print(f"Fatal error processing {self.file_path}: {str(e)}")
            raise

        mzs_flat = np.concatenate(mzs_chunks).astype(np.float32) if mzs_chunks else np.empty(0, np.float32)
        ints_flat = np.concatenate(ints_chunks).astype(np.float32) if ints_chunks else np.empty(0, np.float32)

        _save_scan_cache(self.file_path, scan_nums, rts, offsets, mzs_flat, ints_flat)
        return (
            np.array(scan_nums, dtype=np.int32),
            np.array(rts, dtype=np.float32),
            np.array(offsets, dtype=np.int64),
            mzs_flat,
            ints_flat,
        )

    def extract_eic(self):
        if self._cached_eic is not None:
            return self._cached_eic

        # Load CSR cache from disk, or build it from XML on first access
        cached = _load_scan_cache(self.file_path)
        if cached is None:
            scan_nums, rts, offsets, mzs_flat, ints_flat = self._read_and_cache()
        else:
            scan_nums, rts, offsets, mzs_flat, ints_flat = cached

        mass_list = np.array(Config.MASS_LIST, dtype=np.float32)
        tolerance = np.float32(Config.MASS_TOLERANCE)

        intensity_matrix = _eic_from_csr(rts, offsets, mzs_flat, ints_flat, mass_list, tolerance)
        self._cached_eic = _eic_matrix_to_df(scan_nums, rts, intensity_matrix, mass_list)
        return self._cached_eic


# ── Checkpoint 1 (unchanged signature) ───────────────────────────────────────

def process_file_checkpoint1(fp, dirs, group_name):
    """Checkpoint 1: Extract EIC raw CSV with scan numbers, group-specific filename."""
    Config.set_mass_group(group_name)
    try:
        base = os.path.splitext(os.path.basename(fp))[0]
        assert Config.CURRENT_GROUP is not None, "No group selected"
        group_tag = Config.CURRENT_GROUP.replace(" ", "")
        raw_csv = os.path.join(dirs['csv'], f"{base}_EIC_raw_{group_tag}.csv")
        assert os.path.exists(dirs['csv']), f"CSV directory does not exist: {dirs['csv']}"

        if os.path.exists(raw_csv):
            return f"[↷] {base} (raw cached)"

        analyzer = MSFileAnalyzerOptimized(fp)
        df_raw = analyzer.extract_eic()
        df_raw.to_csv(raw_csv, index=False, float_format='%.3f')
        del df_raw
        del analyzer

        return f"[✔] {base} (raw)"
    except Exception as e:
        return f"[!] Error: {os.path.basename(fp)}: {str(e)[:50]}"