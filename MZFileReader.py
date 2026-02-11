import os
import re
from Config import Config
import pandas as pd
import numpy as np
from numba import njit, prange
from pyteomics import mzml, mzxml
import time

class MSFileAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.base_name = os.path.splitext(os.path.basename(file_path))[0]
        self._cached_eic = None

    def get_reader(self):
        """Get appropriate reader using pyteomics only."""
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext == '.mzxml':
            return mzxml.read(self.file_path, use_index=True, huge_tree=True)
        elif ext == '.mzml':
            return mzml.read(self.file_path, use_index=True, huge_tree=True)
        raise ValueError(f"Unsupported file format: {ext}")

    def get_retention_time(self, scan):
        """Extract retention time with robust error handling."""
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
        """Numba-accelerated parallel EIC extraction."""
        results = np.zeros(len(mass_list))
        mzs_r = np.round(mzs, 4)
        
        for i in prange(len(mass_list)):
            mz_target = mass_list[i]
            mask = np.abs(mzs_r - mz_target) <= tolerance
            results[i] = ints[mask].sum() if mask.any() else 0.0
        return results

    @staticmethod
    @njit
    def fast_eic_extraction(mzs, ints, mass_list, tolerance):
        """Numba-accelerated EIC extraction."""
        results = np.zeros(len(mass_list))
        mzs_r = np.round(mzs, 4)
        for i, mz_target in enumerate(mass_list):
            mask = np.abs(mzs_r - mz_target) <= tolerance
            results[i] = ints[mask].sum() if mask.any() else 0.0
        return results

    def _convert_array_dtype(self, arr):
        """Convert array dtype to standard float32 if needed."""
        if arr.dtype.byteorder == '>' or arr.dtype == '>f4':
            return arr.astype(np.float32)
        return arr

    def extract_eic(self):
        """Robust EIC extraction with comprehensive error handling."""
        if self._cached_eic is not None:
            return self._cached_eic
            
        records = []
        mass_list = np.array(Config.MASS_LIST, dtype=np.float32)
        
        try:
            with self.get_reader() as reader:
                for scan in reader:
                    try:
                        rt = self.get_retention_time(scan)
                        
                        # Handle dtype conversion for m/z and intensity arrays
                        mzs = self._convert_array_dtype(scan['m/z array'])
                        ints = self._convert_array_dtype(scan['intensity array'])
                        
                        # Use parallel version for large arrays
                        if len(mzs) > 10000:
                            intensities = self.fast_eic_extraction_parallel(
                                mzs, ints, mass_list, Config.MASS_TOLERANCE
                            )
                        else:
                            intensities = self.fast_eic_extraction(
                                mzs, ints, mass_list, Config.MASS_TOLERANCE
                            )
                        
                        # Only add non-zero intensities to reduce memory
                        for m, i in zip(mass_list, intensities):
                            if i > 0:
                                records.append({
                                    'rt': rt,
                                    'mass': m,
                                    'intensity': i
                                })
                        
                    except KeyError as e:
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
    """Optimized version of MSFileAnalyzer with better memory management"""

    def extract_eic(self):
        """Memory-optimized EIC extraction"""
        if self._cached_eic is not None:
            return self._cached_eic

        # Use string version of Config masses for output!
        mass_list = list(Config.MASS_LIST)
        mass_list_str = [f"{m:.4f}" for m in Config.MASS_LIST]
        rt_values = []
        intensity_values = []
        mass_indices = []

        try:
            with self.get_reader() as reader:
                for scan in reader:
                    try:
                        rt = self.get_retention_time(scan)
                        mzs = self._convert_array_dtype(scan['m/z array'])
                        ints = self._convert_array_dtype(scan['intensity array'])
                        # Use parallel version for large arrays
                        if len(mzs) > 10000:
                            intensities = self.fast_eic_extraction_parallel(
                                mzs, ints, np.array(mass_list, dtype=np.float32), Config.MASS_TOLERANCE
                            )
                        else:
                            intensities = self.fast_eic_extraction(
                                mzs, ints, np.array(mass_list, dtype=np.float32), Config.MASS_TOLERANCE
                            )
                        for idx, i in enumerate(intensities):
                            if i > 0:
                                rt_values.append(rt)
                                intensity_values.append(i)
                                mass_indices.append(idx)
                    except KeyError:
                        continue
                    except Exception as e:
                        print(f"Skipping scan due to error: {str(e)}")
                        continue
        except Exception as e:
            print(f"Fatal error processing {self.file_path}: {str(e)}")
            raise

        # HERE: Use list of strings, not the (possibly imprecise) float mass!
        self._cached_eic = pd.DataFrame({
            'rt': rt_values,
            'mass': [mass_list_str[i] for i in mass_indices],
            'intensity': intensity_values
        })
        return self._cached_eic
