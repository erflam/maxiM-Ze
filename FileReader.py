import os
import re
from pathlib import Path
from Config import Config
import pandas as pd
import numpy as np
from numba import njit, prange
from pyteomics import mzml, mzxml


# ── Scan cache (CSR format) ───────────────────────────────────────────────────
# Each input file is parsed from XML once and the raw scan arrays saved as a
# compressed .npz (CSR format) so subsequent groups skip XML parsing.
#
# IMPORTANT: _scan_cache_path uses Config._analysis_output_root(), which is
# only correct in the MAIN process. In spawned workers this path may be wrong
# (workers re-import Config with default class values). The scan cache is
# therefore a best-effort optimisation — if the path is wrong the fallback
# XML read is used transparently.
#
# The RT manifest (scan_nums + rts only) is written to dirs['csv'] by
# process_file_checkpoint1, which always receives the correct dirs dict.
# EICBuilder reads it from the same location — no Config dependency.
# ─────────────────────────────────────────────────────────────────────────────

def _scan_cache_path(fp: str) -> Path:
    base = os.path.splitext(os.path.basename(fp))[0]
    cache_dir = Config._analysis_output_root() / "scan_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{base}.npz"


def _save_scan_cache(fp: str, scan_nums, rts, offsets, mzs_flat, ints_flat) -> None:
    try:
        np.savez_compressed(
            str(_scan_cache_path(fp)),
            scan_nums=np.array(scan_nums, dtype=np.int32),
            rts=np.array(rts, dtype=np.float32),
            offsets=np.array(offsets, dtype=np.int64),
            mzs_flat=mzs_flat.astype(np.float32),
            ints_flat=ints_flat.astype(np.float32),
        )
    except Exception:
        pass  # best-effort; workers may have wrong path


def _load_scan_cache(fp: str):
    """Returns (scan_nums, rts, offsets, mzs_flat, ints_flat) or None."""
    try:
        path = _scan_cache_path(fp)
        if not path.exists():
            return None
        d = np.load(str(path))
        return d['scan_nums'], d['rts'], d['offsets'], d['mzs_flat'], d['ints_flat']
    except Exception:
        return None


def rt_manifest_path(fp: str, csv_dir: str) -> Path:
    """
    RT manifest lives in the group's EIC CSVs directory alongside the raw CSV.
    This path is always correct because csv_dir comes from dirs['csv'] which
    is passed explicitly to every worker — no Config dependency.
    """
    base = os.path.splitext(os.path.basename(fp))[0]
    return Path(csv_dir) / f"{base}_rt.npz"


# ── Numba CSR extraction (checkpoint 1) ──────────────────────────────────────

@njit(parallel=True)
def _eic_from_csr(rts, offsets, mzs_flat, ints_flat, mass_list, tolerance):
    """Extract EIC intensities from CSR scan cache for all masses in parallel.
    Returns (n_scans, n_masses) float32 matrix."""
    n_scans  = len(rts)
    n_masses = len(mass_list)
    out = np.zeros((n_scans, n_masses), dtype=np.float32)
    for s in prange(n_scans):
        start = offsets[s]
        end   = offsets[s + 1]
        for m in range(n_masses):
            target = mass_list[m]
            total  = np.float32(0.0)
            for k in range(start, end):
                if abs(mzs_flat[k] - target) <= tolerance:
                    total += ints_flat[k]
            out[s, m] = total
    return out


def _eic_matrix_to_df(scan_nums, rts, intensity_matrix, mass_list):
    rows_s, rows_m = np.where(intensity_matrix > 0)
    if len(rows_s) == 0:
        return pd.DataFrame(columns=['scan', 'rt', 'mass', 'intensity'])
    return pd.DataFrame({
        'scan':      scan_nums[rows_s],
        'rt':        rts[rows_s],
        'mass':      mass_list[rows_m],
        'intensity': intensity_matrix[rows_s, rows_m],
    })


# ── MSFileAnalyzer ────────────────────────────────────────────────────────────

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
                        mzs  = self._convert_array_dtype(scan['m/z array'])
                        ints = self._convert_array_dtype(scan['intensity array'])
                        intensities = self.fast_eic_extraction_parallel(
                            mzs, ints, mass_list, Config.MASS_TOLERANCE)
                        for m, i in zip(mass_list, intensities):
                            if i > 0:
                                records.append({'scan': scan_num, 'rt': rt,
                                                'mass': m, 'intensity': i})
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
    """
    Optimized EIC extractor with two improvements:

    1. CSR scan cache (.npz) — skips XML parsing on second+ access.
       Written to Config._analysis_output_root()/scan_cache/ which may be
       wrong in spawned workers; falls back to XML read transparently.

    2. RT manifest — scan_nums + rts for ALL scans (including zero-intensity).
       NOT written here; written by process_file_checkpoint1 to dirs['csv']
       where the path is always correct.  Exposed via self.all_scan_nums /
       self.all_rts so the checkpoint function can save it.
    """

    def __init__(self, file_path):
        super().__init__(file_path)
        self.all_scan_nums: list = []   # all scans (including zero-intensity)
        self.all_rts:       list = []   # parallel retention times

    def _read_and_cache(self):
        """Parse file, populate all_scan_nums/all_rts, save CSR cache."""
        scan_nums  = []
        rts        = []
        mzs_chunks = []
        ints_chunks= []
        offsets    = [0]

        try:
            with self.get_reader() as reader:
                for scan in reader:
                    try:
                        rt       = self.get_retention_time(scan)
                        scan_num = scan.get('num', None)
                        mzs  = self._convert_array_dtype(scan['m/z array'])
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

        # Expose full scan list for RT manifest saving by checkpoint 1
        self.all_scan_nums = scan_nums
        self.all_rts       = rts

        mzs_flat  = np.concatenate(mzs_chunks).astype(np.float32)  if mzs_chunks  else np.empty(0, np.float32)
        ints_flat = np.concatenate(ints_chunks).astype(np.float32)  if ints_chunks else np.empty(0, np.float32)

        _save_scan_cache(self.file_path, scan_nums, rts, offsets, mzs_flat, ints_flat)

        return (
            np.array(scan_nums, dtype=np.int32),
            np.array(rts,       dtype=np.float32),
            np.array(offsets,   dtype=np.int64),
            mzs_flat,
            ints_flat,
        )

    def extract_eic(self):
        if self._cached_eic is not None:
            return self._cached_eic

        cached = _load_scan_cache(self.file_path)
        if cached is None:
            scan_nums, rts, offsets, mzs_flat, ints_flat = self._read_and_cache()
        else:
            scan_nums, rts, offsets, mzs_flat, ints_flat = cached
            # Populate all_scan_nums / all_rts from cache so checkpoint 1
            # can still save the RT manifest even on a cache-hit path
            self.all_scan_nums = scan_nums.tolist()
            self.all_rts       = rts.tolist()

        mass_list = np.array(Config.MASS_LIST, dtype=np.float32)
        tolerance = np.float32(Config.MASS_TOLERANCE)

        intensity_matrix  = _eic_from_csr(rts, offsets, mzs_flat, ints_flat, mass_list, tolerance)
        self._cached_eic  = _eic_matrix_to_df(scan_nums, rts, intensity_matrix, mass_list)
        return self._cached_eic


# ── Checkpoint 1 ─────────────────────────────────────────────────────────────

def process_file_checkpoint1(fp, dirs, group_name):
    """Checkpoint 1: Extract EIC raw CSV; also write RT manifest to dirs['csv']."""
    Config.set_mass_group(group_name)
    try:
        base      = os.path.splitext(os.path.basename(fp))[0]
        assert Config.CURRENT_GROUP is not None, "No group selected"
        group_tag = Config.CURRENT_GROUP.replace(" ", "")
        raw_csv   = os.path.join(dirs['csv'], f"{base}_EIC_raw_{group_tag}.csv")
        assert os.path.exists(dirs['csv']), f"CSV directory does not exist: {dirs['csv']}"

        rt_path = rt_manifest_path(fp, dirs['csv'])

        if os.path.exists(raw_csv):
            # Raw CSV cached — ensure RT manifest exists too (first run after upgrade)
            if not rt_path.exists():
                # Try to backfill from scan cache
                cached = _load_scan_cache(fp)
                if cached is not None:
                    sn, rt_arr, *_ = cached
                    np.savez_compressed(str(rt_path),
                                        scan_nums=sn, rts=rt_arr)
            return f"[↷] {base} (raw cached)"

        analyzer = MSFileAnalyzerOptimized(fp)
        df_raw   = analyzer.extract_eic()
        df_raw.to_csv(raw_csv, index=False, float_format='%.4f')

        # Save RT manifest to dirs['csv'] — always the correct path in workers
        if analyzer.all_scan_nums:
            np.savez_compressed(
                str(rt_path),
                scan_nums=np.array(analyzer.all_scan_nums, dtype=np.int32),
                rts=np.array(analyzer.all_rts,       dtype=np.float32),
            )

        del df_raw
        del analyzer
        return f"[✔] {base} (raw)"
    except Exception as e:
        return f"[!] Error: {os.path.basename(fp)}: {str(e)[:50]}"