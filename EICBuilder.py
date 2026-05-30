from FileReader import MSFileAnalyzer, rt_manifest_path
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, peak_widths, savgol_filter, argrelextrema
from scipy.ndimage import gaussian_filter1d
import plotly.io as pio
import plotly.graph_objects as go
from PIL import Image
import os
from pathlib import Path
from Config import Config
import colorsys

REL_HEIGHT_BY_MASS = {104.1069: 0.985, 187.0964: 0.985, 119.0896: 0.98}
DEFAULT_REL_HEIGHT = 0.99
MIN_WINDOW_MINUTES = 6.0
X_PADDING_MINUTES  = 0.10

SHOULDER_MIN_HEIGHT_FRAC = 0.02
SHOULDER_MIN_NOISE_MULT  = 3.0
SHOULDER_MIN_SEP_MIN     = 0.06
SHOULDER_VALLEY_DROP_FRAC= 0.85
SHOULDER_LOCALMAX_ORDER  = 2

DEBUG_PEAK_FILTER       = False
EXPORT_DEBUG_CSV        = False
BYPASS_CACHE_WHEN_DEBUG = False

NOISE_PEAK_COUNT_THRESHOLD = 15


def _dark_hex_palette(n: int):
    if n <= 0:
        return []
    colors = []
    for i in range(n):
        h = (i / n) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.25)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


def _load_intensity_by_mass(file_path: str, mass_list: list,
                             csv_dir: str, group_tag: str,
                             analyzer: MSFileAnalyzer):
    """
    Build {mass: intensity_array[n_scans]} for every mass.

    Fast path  — RT manifest + raw EIC CSV (written by checkpoint 1 to
                 csv_dir, which is always the correct path in workers).
                 Pure pandas/numpy — no XML parsing, no numba JIT overhead.

    Fallback   — re-read the mzXML/mzML directly (original behaviour).

    Mass matching uses a ±0.001 Da tolerance to handle the 3-4 dp rounding
    difference between Config.MASS_LIST values and CSV-stored values.
    """
    base         = os.path.splitext(os.path.basename(file_path))[0]
    raw_csv_path = Path(csv_dir) / f"{base}_EIC_raw_{group_tag}.csv"
    rt_path      = rt_manifest_path(file_path, csv_dir)

    if rt_path.exists() and raw_csv_path.exists():
        rt_data   = np.load(str(rt_path))
        scan_nums = rt_data['scan_nums']   # int32[n_scans]
        rts       = rt_data['rts']         # float32[n_scans]
        n_scans   = len(rts)

        df_raw = pd.read_csv(raw_csv_path)

        # scan_num → row-index lookup
        scan_to_idx = {int(s): i for i, s in enumerate(scan_nums)}

        intensity_by_mass = {}
        for mass in mass_list:
            # Tolerance-based match handles 3dp vs 4dp precision difference
            sub = df_raw[(df_raw['mass'] - float(mass)).abs() < 0.001]

            arr = np.zeros(n_scans, dtype=np.float32)
            if not sub.empty:
                s_vals = sub['scan'].values
                i_vals = sub['intensity'].values.astype(np.float32)
                idxs   = np.array([scan_to_idx.get(int(s), -1) for s in s_vals])
                valid  = idxs >= 0
                arr[idxs[valid]] = i_vals[valid]

            intensity_by_mass[mass] = arr

        return rts.tolist(), scan_nums.tolist(), intensity_by_mass

    # ── Fallback: read file directly ─────────────────────────────────────────
    print(f"[EICBuilder] RT manifest missing for {os.path.basename(file_path)} — reading from file")
    rt_values   = []
    scan_numbers= []
    ibm_lists   = {mass: [] for mass in mass_list}

    with analyzer.get_reader() as reader:
        for scan in reader:
            try:
                rt       = analyzer.get_retention_time(scan)
                scan_num = scan.get('num', None)
                mzs  = np.asarray(scan['m/z array'],     dtype=np.float32)
                ints = np.asarray(scan['intensity array'], dtype=np.float32)
                for mass in mass_list:
                    mask = np.abs(np.round(mzs, 4) - mass) <= Config.MASS_TOLERANCE
                    ibm_lists[mass].append(float(np.sum(ints[mask])) if np.any(mask) else 0.0)
                rt_values.append(rt)
                scan_numbers.append(scan_num)
            except KeyError:
                continue
            except Exception as e:
                print(f"Error processing scan: {str(e)}")
                continue

    intensity_by_mass = {m: np.array(v, dtype=np.float32) for m, v in ibm_lists.items()}
    return rt_values, scan_numbers, intensity_by_mass


def improved_peak_cutting(intensity_vals_smooth, rt_vals, peaks, width_results, specific_mass):
    MIN_APEX_SEP_MIN = 0.015
    MAX_SEG_WIDTH_MIN= 0.8
    MIN_SEG_WIDTH_MIN= 0.015

    all_minima = set()
    for sigma in [0.5, 1.0, 2.0]:
        smoothed = gaussian_filter1d(intensity_vals_smooth, sigma=sigma)
        for order in [2, 3]:
            all_minima.update(argrelextrema(smoothed, np.less, order=order)[0])

    second_deriv = np.gradient(np.gradient(intensity_vals_smooth))
    all_minima.update(np.where((second_deriv[:-2] < 0) & (second_deriv[2:] > 0))[0] + 1)

    all_minima = np.array(sorted(all_minima))
    if len(all_minima) == 0:
        return peaks, width_results

    rt_delta = float(np.mean(np.diff(rt_vals))) if len(rt_vals) > 1 else 0.0

    def seg_time_ok(l_i, r_i):
        if rt_delta <= 0:
            return True
        wmin = (r_i - l_i) * rt_delta
        return MIN_SEG_WIDTH_MIN <= wmin <= MAX_SEG_WIDTH_MIN

    def peak_quality(apex_idx, left_idx, right_idx):
        if right_idx <= left_idx or apex_idx < left_idx or apex_idx > right_idx:
            return 0.0
        seg     = intensity_vals_smooth[left_idx:right_idx + 1]
        apex_v  = intensity_vals_smooth[apex_idx]
        left_m  = np.min(seg[:apex_idx - left_idx + 1]) if apex_idx > left_idx else apex_v
        right_m = np.min(seg[apex_idx - left_idx:])     if apex_idx < right_idx else apex_v
        prom    = apex_v - max(left_m, right_m)
        lw = seg[:apex_idx - left_idx]
        rw = seg[apex_idx - left_idx + 1:]
        ml = min(len(lw), len(rw))
        if ml >= 2:
            a, b = lw[-ml:], rw[:ml][::-1]
            sym  = 0 if (np.std(a) == 0 or np.std(b) == 0) else max(0, float(np.corrcoef(a, b)[0, 1]))
        else:
            sym = 0
        return prom * (1 + 0.3 * sym)

    rel_height_split = 0.85
    try:
        wrs   = peak_widths(intensity_vals_smooth, peaks, rel_height=rel_height_split)
        vsplit= wrs[0] > 0
    except Exception:
        vsplit= np.zeros(len(peaks), dtype=bool)
        wrs   = tuple(np.zeros(len(peaks)) for _ in range(4))

    new_peaks = []

    for i_pk, pk in enumerate(peaks):
        lip = wrs[2][i_pk] if vsplit[i_pk] else width_results[2][i_pk]
        rip = wrs[3][i_pk] if vsplit[i_pk] else width_results[3][i_pk]
        li  = max(0, int(np.floor(lip)))
        ri  = min(len(intensity_vals_smooth) - 1, int(np.ceil(rip)))

        internal = [v for v in all_minima if li < v < ri]
        if not internal:
            new_peaks.append(pk)
            continue

        sig_valleys = []
        pk_int = intensity_vals_smooth[pk]
        for v in internal:
            v_int = intensity_vals_smooth[v]
            lw    = slice(max(0, v - 3), min(len(intensity_vals_smooth), v + 4))
            lm    = np.max(intensity_vals_smooth[lw])
            lph   = np.max(intensity_vals_smooth[li:v + 1])
            rph   = np.max(intensity_vals_smooth[v:ri + 1])
            ratio = min(lph, rph) / max(lph, rph)
            if v_int <= (0.6 + 0.3 * ratio) * lm:
                if abs(rt_vals[v] - rt_vals[pk]) >= MIN_APEX_SEP_MIN / 2:
                    sig_valleys.append(v)

        if not sig_valleys:
            new_peaks.append(pk)
            continue

        segs = list(zip([li] + sig_valleys, sig_valleys + [ri]))
        cands= []
        for sl, sr in segs:
            if sr - sl < 3 or not seg_time_ok(sl, sr):
                continue
            ai = sl + int(np.argmax(intensity_vals_smooth[sl:sr + 1]))
            q  = peak_quality(ai, sl, sr)
            if q > pk_int * 0.1:
                cands.append((ai, q))

        if len(cands) >= 2:
            cands.sort(key=lambda x: x[1], reverse=True)
            sel = [cands[0][0]]
            for ai, _ in cands[1:]:
                if min(abs(rt_vals[ai] - rt_vals[s]) for s in sel) >= MIN_APEX_SEP_MIN:
                    sel.append(ai)
            new_peaks.extend(sel)
        elif cands:
            new_peaks.append(cands[0][0])
        else:
            new_peaks.append(pk)

    if new_peaks:
        new_peaks = np.unique(np.array(new_peaks, dtype=int))
        try:
            nwr       = peak_widths(intensity_vals_smooth, new_peaks, rel_height=0.99)
            vm        = nwr[0] > 0
            new_peaks = new_peaks[vm]
            nwr       = tuple(a[vm] for a in nwr)
            return new_peaks, nwr
        except Exception as e:
            print(f"Error recalculating widths: {e}")
            return peaks, width_results

    return peaks, width_results


def looks_like_real_peak(y_raw_window: np.ndarray):
    nan = float("nan")
    if y_raw_window is None:
        return True, nan, nan, nan, nan, nan
    if len(y_raw_window) < 7:
        y = np.asarray(y_raw_window, dtype=float)
        apex = float(np.max(y)) if len(y) else 0.0
        return True, nan, nan, apex, float(y[0]) if len(y) else 0.0, float(y[-1]) if len(y) else 0.0
    y = np.asarray(y_raw_window, dtype=float)
    ai = int(np.argmax(y)); apex = float(y[ai])
    le = float(y[0]); re = float(y[-1])
    if apex <= 0: return False, nan, nan, apex, le, re
    left = y[:ai + 1]; right = y[ai:]
    if len(left) < 3 or len(right) < 3: return True, nan, nan, apex, le, re
    if (apex - le) / apex < 0.25: return False, nan, nan, apex, le, re
    if (apex - re) / apex < 0.25: return False, nan, nan, apex, le, re
    ld = np.diff(left);  fu = float(np.mean(ld > 0)) if ld.size else 0.0
    rd = np.diff(right); fd = float(np.mean(rd < 0)) if rd.size else 0.0
    if DEBUG_PEAK_FILTER:
        print(f"[PeakCheck] apex={apex:.2f} start={le:.2f} end={re:.2f} "
              f"frac_up={fu:.2f} frac_down={fd:.2f}")
    return fu >= 0.7 and fd >= 0.7, fu, fd, apex, le, re


def analyze_ms_file_plotly(file_path, output_image_path, file_colors,
                            axis_meta_csv=None, dirs=None):
    """Analyze MS file and generate plotly visualization.

    Parameters
    ----------
    dirs : dict, optional
        Pipeline dirs dict (must include 'csv').  When provided the RT
        manifest fast-path is used; otherwise falls back to file reading.
    """
    group_tag = Config.CURRENT_GROUP.replace(" ", "")

    use_cache = not (BYPASS_CACHE_WHEN_DEBUG and (DEBUG_PEAK_FILTER or EXPORT_DEBUG_CSV))

    if use_cache and os.path.exists(output_image_path):
        base      = os.path.splitext(os.path.basename(file_path))[0]
        peaks_csv = os.path.join(
            os.path.dirname(os.path.dirname(output_image_path)),
            'csv', f"{base}_peaks_{group_tag}.csv")
        if os.path.exists(peaks_csv):
            try:
                return pd.read_csv(peaks_csv).to_dict('records'), [], set()
            except Exception:
                pass

    analyzer = MSFileAnalyzer(file_path)
    base     = analyzer.base_name

    mass_list     = list(Config.MASS_LIST)
    mass_list_str = [f"{m:.4f}" for m in mass_list]

    # Determine csv_dir: prefer dirs['csv'], fall back to deriving from png path
    if dirs is not None and 'csv' in dirs:
        csv_dir = dirs['csv']
    else:
        csv_dir = os.path.dirname(output_image_path).replace(
            os.sep + "EIC PNGs", os.sep + "EIC CSVs")

    rt_values, scan_numbers, intensity_by_mass = _load_intensity_by_mass(
        file_path, mass_list, csv_dir=csv_dir,
        group_tag=group_tag, analyzer=analyzer
    )

    peaks_prefilter_by_mass: dict[str, list] = {s: [] for s in mass_list_str}
    debug_rows       = []
    rt_vals          = np.array(rt_values)
    scan_numbers_arr = np.array(scan_numbers, dtype=object)
    gui_noise_level  = Config.GROUP_NOISE_LEVEL

    def split_shoulders(rt_vals, scan_numbers_arr, intensity_raw, intensity_smooth,
                        li, ri, noise_level, mass_str, base_name):
        x_win = rt_vals[li:ri + 1]; y_raw = intensity_raw[li:ri + 1]
        y_smo = intensity_smooth[li:ri + 1]
        ss = scan_numbers_arr[li]; se = scan_numbers_arr[ri]
        mk = lambda apx, area: {
            'File': base_name, 'm/z': mass_str,
            'RT_start': round(float(x_win[0]), 4),
            'RT_apex':  round(float(rt_vals[apx]), 4),
            'RT_end':   round(float(x_win[-1]), 4),
            'scan_start': ss, 'scan_end': se,
            'Peak Area': round(float(area), 2),
            'height': round(float(intensity_raw[apx]), 2)}

        if len(y_smo) < 7:
            ai = li + int(np.argmax(y_raw))
            return [mk(ai, trapezoid(y_raw, x_win))]

        mi = li + int(np.argmax(y_smo)); mh = float(intensity_raw[mi])
        lms = argrelextrema(y_smo, np.greater, order=SHOULDER_LOCALMAX_ORDER)[0]

        cands = [li + int(lm) for lm in lms
                 if (li + int(lm)) != mi
                 and abs(float(rt_vals[li + int(lm)]) - float(rt_vals[mi])) >= SHOULDER_MIN_SEP_MIN
                 and float(intensity_raw[li + int(lm)]) >= SHOULDER_MIN_HEIGHT_FRAC * mh
                 and float(intensity_raw[li + int(lm)]) >= SHOULDER_MIN_NOISE_MULT * float(noise_level)]

        if not cands:
            return [mk(mi, trapezoid(y_raw, x_win))]

        best = bv = None; bh = -1.0
        for ci in cands:
            lo, hi = min(ci, mi), max(ci, mi)
            if hi - lo < 3: continue
            vi = lo + int(np.argmin(intensity_smooth[lo:hi + 1]))
            ch = float(intensity_raw[ci]); mhh = float(intensity_raw[mi])
            vh = float(intensity_raw[vi])
            sm = min(ch, mhh)
            if sm <= 0 or vh > SHOULDER_VALLEY_DROP_FRAC * sm: continue
            if ch > bh: best = ci; bh = ch; bv = vi

        if best is None or bv is None:
            return [mk(mi, trapezoid(y_raw, x_win))]

        la = li + int(np.argmax(intensity_smooth[li:bv + 1]))
        ra = bv + int(np.argmax(intensity_smooth[bv:ri + 1]))
        al = trapezoid(intensity_raw[li:bv + 1],  rt_vals[li:bv + 1])
        ar = trapezoid(intensity_raw[bv:ri + 1], rt_vals[bv:ri + 1])

        if float(intensity_raw[la]) >= float(intensity_raw[ra]):
            mi2, sh, am, as_ = la, ra, al, ar
        else:
            mi2, sh, am, as_ = ra, la, ar, al

        return [mk(sh, as_), mk(mi2, am)]

    trace_candidates: list[tuple] = []

    for mi, specific_mass in enumerate(mass_list):
        intensity_vals = np.asarray(intensity_by_mass[specific_mass], dtype=np.float32)
        if len(intensity_vals) < 3 or np.max(intensity_vals) == 0:
            continue

        wl = min(7, len(intensity_vals))
        if wl % 2 == 0: wl -= 1
        try:
            ivs = savgol_filter(intensity_vals, window_length=wl, polyorder=2) if wl >= 3 else intensity_vals
        except Exception:
            ivs = intensity_vals

        nl  = np.std(ivs[:min(20, len(ivs))])
        mxi = np.max(ivs)
        mnh = max(mxi * 0.04, nl * 3)

        try:
            peaks, _ = find_peaks(ivs, height=mnh, distance=1)
            if len(peaks) == 0: continue
            rh = REL_HEIGHT_BY_MASS.get(round(float(specific_mass), 4), DEFAULT_REL_HEIGHT)
            wr = peak_widths(ivs, peaks, rel_height=rh)
            vm = wr[0] > 0; peaks = peaks[vm]; wr = tuple(a[vm] for a in wr)
            if len(peaks) == 0: continue
            try:
                peaks, wr = improved_peak_cutting(ivs, rt_vals, peaks, wr, specific_mass)
            except Exception as e:
                print(f"Peak cutting failed for mass {mass_list_str[mi]}: {e}")
        except Exception as e:
            print(f"Error finding peaks for mass {mass_list_str[mi]}: {str(e)}")
            continue

        mz_str = mass_list_str[mi]

        for i, idx in enumerate(peaks):
            try:
                li = max(0, int(np.floor(wr[2][i])))
                ri = min(len(rt_vals) - 1, int(np.ceil(wr[3][i])))
                dur = rt_vals[ri] - rt_vals[li]

                if dur > Config.MAX_PEAK_DURATION:
                    vx = np.where(rt_vals <= rt_vals[li] + Config.MAX_PEAK_DURATION)[0]
                    if vx.size: ri = int(vx[vx >= li].max())
                    else: continue
                    dur = rt_vals[ri] - rt_vals[li]

                if not (0.03 <= dur <= 0.75): continue

                xp = rt_vals[li:ri + 1]; yp = intensity_vals[li:ri + 1]
                is_peak, fu, fd, apex, le, re = looks_like_real_peak(yp)

                debug_rows.append({
                    "File": base, "m/z": float(specific_mass),
                    "scan_start": scan_numbers_arr[li], "scan_end": scan_numbers_arr[ri],
                    "RT_start": float(rt_vals[li]),  "RT_end": float(rt_vals[ri]),
                    "apex_intensity": float(apex) if np.isfinite(apex) else np.nan,
                    "start_intensity": float(le)  if np.isfinite(le)   else np.nan,
                    "end_intensity":   float(re)  if np.isfinite(re)   else np.nan,
                    "frac_up":   float(fu) if np.isfinite(fu) else np.nan,
                    "frac_down": float(fd) if np.isfinite(fd) else np.nan,
                    "passed_filter": bool(is_peak),
                })

                if len(yp) >= 5:
                    al = int(np.argmax(yp))
                    ps = ivs[li + al: ri + 1]
                    cl = None
                    for k in range(1, len(ps) - 1):
                        if ps[k] <= ps[k - 1] and ps[k] < ps[k + 1]:
                            cl = k; break
                    if cl is not None:
                        nri = li + al + cl
                        ah  = ivs[li + al]
                        if ah > 0 and (ivs[nri] / ah) < 0.15:
                            ri = nri; xp = rt_vals[li:ri + 1]; yp = intensity_vals[li:ri + 1]

                recs = split_shoulders(rt_vals, scan_numbers_arr, intensity_vals, ivs,
                                       li, ri, nl, mz_str, base)
                peaks_prefilter_by_mass[mz_str].extend(recs)
                if any(r.get('height', 0) >= gui_noise_level for r in recs):
                    trace_candidates.append((mz_str, xp.copy(), yp.copy(), specific_mass))

            except Exception as e:
                print(f"Error processing peak {i} for mass {mass_list_str[mi]}: {str(e)}")
                continue

    noisy_masses: set[str] = set()
    for mz_str, records in peaks_prefilter_by_mass.items():
        if len(records) > NOISE_PEAK_COUNT_THRESHOLD:
            noisy_masses.add(mz_str)
            print(f"[Noise] m/z {mz_str} detected as noise ({len(records)} peaks in {base})")

    peaks_prefilter: list[dict] = []
    peaks_out:       list[dict] = []
    for mz_str, records in peaks_prefilter_by_mass.items():
        if mz_str in noisy_masses: continue
        peaks_prefilter.extend(records)
        peaks_out.extend([r for r in records if r.get('height', 0) >= gui_noise_level])

    fig = go.Figure()
    mc  = _dark_hex_palette(len(mass_list))
    mcm = {mass_list[i]: mc[i] for i in range(len(mass_list))}
    all_peak_rts: list[float] = []

    for (mz_str, xp, yp, sm) in trace_candidates:
        if mz_str in noisy_masses: continue
        fig.add_trace(go.Scatter(x=xp, y=yp, mode='lines',
                                  line=dict(color=mcm[sm], width=3), showlegend=False))
        all_peak_rts.extend(xp.tolist())

    if not fig.data:
        print(f"[!] No peaks above noise ({gui_noise_level:.0f}) in {base}; skipping.")
        if EXPORT_DEBUG_CSV:
            dc = os.path.join(os.path.dirname(os.path.dirname(output_image_path)),
                              'csv', f"{base}_peak_debug_{group_tag}.csv")
            os.makedirs(os.path.dirname(dc), exist_ok=True)
            pd.DataFrame(debug_rows).to_csv(dc, index=False)
        return peaks_out, peaks_prefilter, noisy_masses

    mnr = float(np.min(rt_vals)); mxr = float(np.max(rt_vals))
    pmn = float(np.min(all_peak_rts)) if all_peak_rts else mnr
    pmx = float(np.max(all_peak_rts)) if all_peak_rts else mxr
    ctr = 0.5 * (pmn + pmx)
    hw  = 0.5 * max(MIN_WINDOW_MINUTES, pmx - pmn)
    x0  = max(ctr - hw - X_PADDING_MINUTES, mnr)
    x1  = min(ctr + hw + X_PADDING_MINUTES, mxr)
    if (x1 - x0) < MIN_WINDOW_MINUTES:
        need = MIN_WINDOW_MINUTES - (x1 - x0)
        ar = min(need, mxr - x1); x1 += ar; need -= ar
        x0 -= min(need, x0 - mnr)

    if axis_meta_csv is not None:
        os.makedirs(os.path.dirname(axis_meta_csv), exist_ok=True)
        pd.DataFrame([{"x0": float(x0), "x1": float(x1),
                       "png_width": 1600, "png_height": 900}]).to_csv(axis_meta_csv, index=False)

    fig.update_xaxes(range=[x0, x1], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_layout(width=1600, height=900, margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    try:
        pio.write_image(fig, output_image_path, format='png', engine='kaleido', scale=4)
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        try:
            rp = output_image_path.replace('.png', '_raw.png')
            pio.write_image(fig, rp, format='png', engine='kaleido', scale=4)
            with Image.open(rp) as img:
                img.save(output_image_path, optimize=True, compress_level=9)
            os.remove(rp)
        except Exception as e2:
            print(f"Fallback image saving also failed: {str(e2)}")

    del fig

    for lst, key in [(peaks_out, 'out'), (peaks_prefilter, 'pre')]:
        if lst:
            df = pd.DataFrame(lst)
            if {'m/z', 'RT_apex', 'Peak Area'}.issubset(df.columns):
                df['RT_apex']   = pd.to_numeric(df['RT_apex'],   errors='coerce')
                df['Peak Area'] = pd.to_numeric(df['Peak Area'], errors='coerce')
                df = df.sort_values(['m/z', 'RT_apex', 'Peak Area'], ascending=[True, True, False])
                df = df.drop_duplicates(subset=['m/z', 'RT_apex'], keep='first')
                if key == 'out':   peaks_out       = df.to_dict('records')
                else:              peaks_prefilter = df.to_dict('records')

    if EXPORT_DEBUG_CSV:
        dc = os.path.join(os.path.dirname(os.path.dirname(output_image_path)),
                          'csv', f"{base}_peak_debug_{group_tag}.csv")
        os.makedirs(os.path.dirname(dc), exist_ok=True)
        pd.DataFrame(debug_rows).to_csv(dc, index=False)
        print(f"[DEBUG] wrote {len(debug_rows)} rows to {dc}")

    return peaks_out, peaks_prefilter, noisy_masses


def process_file_checkpoint2(fp, dirs, file_colors, group_name):
    """Checkpoint 2: Build EIC PNG + peaks CSV (group-specific filename)."""
    Config.set_mass_group(group_name)
    try:
        base      = os.path.splitext(os.path.basename(fp))[0]
        group_tag = Config.CURRENT_GROUP.replace(" ", "")

        png_path            = os.path.join(dirs['png'], f"EIC_{base}_{group_tag}.png")
        peaks_csv           = os.path.join(dirs['csv'], f"{base}_peaks_{group_tag}.csv")
        peaks_prefilter_csv = os.path.join(dirs['csv'], f"{base}_peaks_prefilter_{group_tag}.csv")

        if os.path.exists(png_path) and os.path.exists(peaks_csv):
            return f"[↷] {base} (png+peaks cached)"

        axis_meta_csv = os.path.join(dirs['csv'], f"{base}_axis_{group_tag}.csv")

        # Pass dirs so analyze_ms_file_plotly can locate the RT manifest correctly
        peaks, peaks_prefilter, noisy_masses = analyze_ms_file_plotly(
            fp, png_path, file_colors, axis_meta_csv=axis_meta_csv, dirs=dirs
        )

        if noisy_masses:
            remaining = [m for m in [f"{x:.4f}" for x in Config.MASS_LIST]
                         if m not in noisy_masses]
            if not remaining:
                return f"[–] {base} skipped — all masses detected as noise"

        if peaks_prefilter:
            pd.DataFrame(peaks_prefilter).to_csv(peaks_prefilter_csv, index=False, float_format='%.3f')
        if peaks:
            pd.DataFrame(peaks).to_csv(peaks_csv, index=False, float_format='%.3f')

        n_total = len(peaks_prefilter); n_kept = len(peaks)
        noise_note = f", {len(noisy_masses)} noisy m/z removed" if noisy_masses else ""
        return (f"[✔] {base} (png+peaks) — {n_kept} peaks kept, "
                f"{n_total - n_kept} below noise ({Config.GROUP_NOISE_LEVEL:.0f}){noise_note}")

    except Exception as e:
        return f"[!] Error: {os.path.basename(fp)}: {str(e)[:50]}"