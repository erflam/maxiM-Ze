# Resolving.py
import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image

from Config import Config


# --- Constants (kept exactly the same as your code) ---
MIN_RT_DIFF = 0.095          # min RT separation
VALLEY_DROP_MIN = 0.40       # require at least 40% drop to call two peaks resolved
MIN_PIXEL_SEP = 6            # minimal pixel gap between apices
MARGIN_FRAC = 0.08

KERNEL_3x3 = np.ones((3, 3), np.uint8)


def estimate_plot_bounds(h, w):
    t, l = int(MARGIN_FRAC * h), int(MARGIN_FRAC * w)
    return max(0, t), h - t, max(0, l), w - l


def build_elevation_from_alpha(alpha):
    edges = cv2.Canny(alpha, 0, 0)
    edges = cv2.dilate(edges, KERNEL_3x3, iterations=1)
    dist = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 3)
    normalized = (dist / max(1.0, dist.max()) * 255).astype(np.uint8)
    return 255 - cv2.medianBlur(normalized, 5)


def watershed_labels(alpha_roi, seed_cols_rel):
    h, w = alpha_roi.shape
    if w < 2 or h < 2 or not np.any(seed_cols_rel):
        return np.zeros((h, w), dtype=np.int32)

    elev = build_elevation_from_alpha(alpha_roi)
    markers = np.zeros((h, w), np.int32)

    y_range = slice(h // 4, 3 * h // 4)
    for i, xc in enumerate(seed_cols_rel, 1):
        xc = np.clip(int(xc), 0, w - 1)
        markers[y_range, xc] = i

    color = cv2.cvtColor(elev, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)
    markers[markers < 0] = 0
    return markers


def column_labels(labels):
    h, w = labels.shape
    out = np.zeros(w, dtype=np.int32)

    for x in range(w):
        col = labels[:, x]
        col = col[col > 0]
        if col.size:
            vals, counts = np.unique(col, return_counts=True)
            out[x] = vals[np.argmax(counts)]

    # Fill gaps
    nz = np.nonzero(out)[0]
    if nz.size:
        out[:nz[0]] = out[nz[0]]
        out[nz[-1] + 1:] = out[nz[-1]]
    return out


def find_boundary_or_dip(lbl1d, profile, x0, x1, id1, id2):
    if x0 > x1:
        x0, x1 = x1, x0

    if np.any(lbl1d):
        seg = lbl1d[x0:x1 + 1]
        diff_idx = np.where(np.diff(seg) != 0)[0]
        for k in diff_idx:
            l = seg[k]
            r = seg[k + 1] if k + 1 < seg.size else 0
            if (l == id1 and r == id2) or (l == id2 and r == id1):
                return x0 + k + 1

    # Fallback to profile dip
    s = profile[max(0, x0):min(len(profile), x1 + 1)]
    if len(s) < 3:
        return (x0 + x1) // 2

    d = np.diff(s)
    neg_pos = (d[:-1] < 0) & (d[1:] > 0)
    idxs = np.where(neg_pos)[0]
    return x0 + int(idxs[0] + 1) if idxs.size else x0 + int(np.argmin(s))


def smooth_profile(p):
    # 1D gaussian smoothing for stable valley metrics
    if p.ndim == 1:
        p2 = cv2.GaussianBlur(p.reshape(1, -1).astype(np.float32), (1, 21), 0).ravel()
        return p2
    return p


def valley_metrics(profile_smooth, xL, xR):
    """Return (valley_x, valley_y, left_apex_y, right_apex_y) within [xL,xR]."""
    if xL > xR:
        xL, xR = xR, xL
    xL = max(0, xL)
    xR = min(len(profile_smooth) - 1, xR)

    seg = profile_smooth[xL:xR + 1]
    if seg.size == 0:
        return (xL + xR) // 2, 0.0, 0.0, 0.0

    v_local_idx = int(np.argmin(seg))
    valley_x = xL + v_local_idx
    valley_y = profile_smooth[valley_x]
    left_apex_y = profile_smooth[xL]
    right_apex_y = profile_smooth[xR]
    return valley_x, float(valley_y), float(left_apex_y), float(right_apex_y)


def resolve_isomers_checkpoint3(file_paths, dirs, group_name):
    """
    Checkpoint 3 (Resolving):
    - Reads:  {base}_peaks_{Group}.csv
    - Reads:  EIC_{base}_{Group}.png
    - Writes: {base}_peaks_pix.csv   (in dirs['pixel'])
    """
    Config.set_mass_group(group_name)
    group_tag = Config.CURRENT_GROUP.replace(" ", "")

    for fp in file_paths:
        base = os.path.splitext(os.path.basename(fp))[0]

        peaks_csv = os.path.join(dirs['csv'], f"{base}_peaks_{group_tag}.csv")
        pixel_csv = os.path.join(dirs['pixel'], f"{base}_peaks_pix.csv")
        png_path = os.path.join(dirs['png'], f"EIC_{base}_{group_tag}.png")

        # Skip if already processed
        if os.path.exists(pixel_csv):
            continue

        if not os.path.exists(peaks_csv) or not os.path.exists(png_path):
            print(f"Skipping {base}: missing required files")
            continue

        print(f"Processing {base}...")

        # Load and filter data
        df = pd.read_csv(peaks_csv, usecols=['m/z', 'RT_start', 'RT_apex', 'RT_end', 'Peak Area', 'height'])
        df = df.rename(columns={'Peak Area': 'area'})
        df = df[(df['RT_end'] - df['RT_start']).between(0.05, 0.75)]
        df['height'] = df.get('height', np.nan)

        if df.empty:
            continue

        # Load image once (used only to derive pixel spans; no slicing will be saved)
        im = Image.open(png_path).convert('RGBA')
        W, H = im.size
        A = np.array(im)[..., 3].astype(np.uint8)

        # Calculate RT-pixel conversion factors
        rt_min, rt_max = df['RT_start'].min() - 0.1, df['RT_end'].max() + 0.1
        rt_range = rt_max - rt_min
        width_factor = (W - 1) / rt_range

        def rt_to_px(rt_vals):
            return np.clip(((np.asarray(rt_vals) - rt_min) * width_factor).astype(int), 0, W - 1)

        def px_to_rt(px_vals):
            return rt_min + (np.asarray(px_vals) / (W - 1)) * rt_range

        # Get plot bounds and profile once
        top, bot, left, right = estimate_plot_bounds(H, W)
        alpha_roi = A[top:bot, left:right]
        profile_raw = A[top:bot, left:right].sum(axis=0).astype(np.float32)
        profile = smooth_profile(profile_raw)

        # Group and sort data
        df['_mz_key'] = df['m/z'].astype(str)
        df = df.sort_values(['_mz_key', 'RT_apex']).reset_index(drop=True)

        rows_out = []
        for mz_key, grp in df.groupby('_mz_key', sort=False):
            grp = grp.sort_values('RT_apex').reset_index(drop=True)

            if len(grp) == 1:
                # Single peak - no watershed needed
                row = grp.iloc[0].to_dict()
                ps, pe = rt_to_px([row['RT_start'], row['RT_end']])
                row.update({
                    'pixel_start': int(ps), 'pixel_end': int(pe),
                    'peak_type': 'resolved',
                    'cluster_id': f'{mz_key}_0',
                    'is_cluster_lead': True
                })
                rows_out.append(row)
                continue

            # Pixel calculations
            apex_px = rt_to_px(grp['RT_apex'].values)
            l_px = rt_to_px(grp['RT_start'].values)
            r_px = rt_to_px(grp['RT_end'].values)
            seeds_rel = np.clip(apex_px - left, 0, right - left - 1)

            # Watershed segmentation to constrain spans (not to force cuts)
            labels = watershed_labels(alpha_roi, seeds_rel)
            lbl1d = column_labels(labels)

            # Decide which adjacent pairs are truly resolved
            resolved_pair = []
            for i in range(len(grp) - 1):
                rt_diff = float(grp['RT_apex'].iloc[i + 1] - grp['RT_apex'].iloc[i])
                px_diff = int(abs(apex_px[i + 1] - apex_px[i]))

                r1 = int(np.clip(apex_px[i] - left, 0, right - left - 1))
                r2 = int(np.clip(apex_px[i + 1] - left, 0, right - left - 1))
                rL, rR = (r1, r2) if r1 <= r2 else (r2, r1)
                vx, vy, yL, yR = valley_metrics(profile, rL, rR)
                min_apex_y = max(1.0, min(yL, yR))
                valley_drop = 1.0 - float(vy) / float(min_apex_y)

                is_resolved = (rt_diff >= MIN_RT_DIFF) and (px_diff >= MIN_PIXEL_SEP) and (valley_drop >= VALLEY_DROP_MIN)

                cut = None
                if is_resolved:
                    cut = left + find_boundary_or_dip(lbl1d, profile, rL, rR, i + 1, i + 2)
                    cut = int(np.clip(cut, 0, W - 1))
                resolved_pair.append((is_resolved, cut))

            # Build clusters of co-eluting peaks
            clusters = []
            current = [0]
            for i, (ok, _) in enumerate(resolved_pair):
                if ok:
                    clusters.append(current)
                    current = [i + 1]
                else:
                    current.append(i + 1)
            clusters.append(current)

            # For each cluster, compute span and push rows
            for cidx, members in enumerate(clusters):
                member_idxs = np.array(members, dtype=int)
                ps = int(np.min(l_px[member_idxs]))
                pe = int(np.max(r_px[member_idxs]))

                # Constrain by watershed label extents if available
                w_lefts, w_rights = [], []
                for j in member_idxs:
                    label_id = j + 1
                    mask = (lbl1d == label_id)
                    if np.any(mask):
                        idxs = np.where(mask)[0]
                        w_lefts.append(int(left + idxs[0]))
                        w_rights.append(int(left + idxs[-1]))
                if w_lefts and w_rights:
                    ps = max(ps, min(w_lefts))
                    pe = min(pe, max(w_rights))
                ps = int(np.clip(ps, 0, W - 2))
                pe = int(np.clip(pe, ps + 1, W))

                if len(member_idxs) == 1:
                    i = member_idxs[0]
                    ps_single = max(ps, int(l_px[i]))
                    pe_single = min(pe, int(r_px[i]))
                    row = grp.iloc[i].to_dict()
                    row.update({
                        'RT_start': float(px_to_rt(ps_single)),
                        'RT_end': float(px_to_rt(pe_single)),
                        'pixel_start': int(ps_single),
                        'pixel_end': int(pe_single),
                        'peak_type': 'resolved',
                        'cluster_id': f'{mz_key}_{cidx}',
                        'is_cluster_lead': True
                    })
                    rows_out.append(row)
                else:
                    for k, i in enumerate(member_idxs):
                        row = grp.iloc[i].to_dict()
                        row.update({
                            'RT_start': float(px_to_rt(ps)),
                            'RT_end': float(px_to_rt(pe)),
                            'pixel_start': int(ps),
                            'pixel_end': int(pe),
                            'peak_type': 'coeluting',
                            'cluster_id': f'{mz_key}_{cidx}',
                            'is_cluster_lead': bool(k == 0)
                        })
                        rows_out.append(row)

        # Create output dataframe
        df_out = pd.DataFrame(rows_out)
        df_out['peak_num'] = range(1, len(df_out) + 1)

        # Ensure required columns exist
        required_cols = ['File', 'm/z', 'RT_start', 'RT_apex', 'RT_end', 'area', 'height',
                         'pixel_start', 'pixel_end', 'peak_type', 'cluster_id', 'is_cluster_lead']
        for col in required_cols:
            if col not in df_out.columns:
                df_out[col] = np.nan if col not in ['is_cluster_lead'] else False

        # Only write the pixel CSVs; do NOT slice/save PNGs
        df_out.to_csv(pixel_csv, index=False)

        print(f"[✔] {base}: wrote {len(df_out)} rows to {os.path.basename(pixel_csv)}")

    print('Checkpoint 3 completed (CSV only, no PNG slices).')


def count_peaks_per_file_summary(dirs):
    """
    Count the number of peaks detected in each sample and generate a detailed summary CSV
    including m/z, RT_apex, pixel_start, and pixel_end for each peak,
    sorted by RT_apex (keeping pixel_start and pixel_end matched).
    """
    print("\n[✔] Generating detailed peak count summary (sorted by RT_apex) with m/z...")

    peaks_dir = dirs['pixel']
    summary_file = os.path.join(dirs['counts'], f'peak_count_summary_{Config.CURRENT_GROUP}.csv')

    if not os.path.exists(peaks_dir):
        print(f"[!] Peaks directory not found: {peaks_dir}")
        return

    peak_summaries = []

    for file_name in os.listdir(peaks_dir):
        if not file_name.endswith("_peaks_pix.csv"):
            continue

        sample_name = file_name.replace("_peaks_pix.csv", "")
        peaks_csv_path = os.path.join(peaks_dir, file_name)

        try:
            df = pd.read_csv(peaks_csv_path)

            base_required = ["RT_apex", "pixel_start", "pixel_end"]
            missing = [c for c in base_required if c not in df.columns]
            if missing:
                print(f"[!] Missing columns {missing} in {file_name}, skipping...")
                continue

            has_mz = "m/z" in df.columns
            if not has_mz:
                print(f"[!] Column 'm/z' not found in {file_name}; masses will be blank in summary.")

            df = df.sort_values(by="RT_apex", ascending=True, ignore_index=True)

            num_peaks = len(df)
            sample_data = {"Sample": sample_name, "NumPeaks": num_peaks}

            for i, row in df.iterrows():
                peak_idx = i + 1

                if has_mz and pd.notna(row["m/z"]):
                    try:
                        mz_val = float(row["m/z"])
                        mz_str = f"{mz_val:.4f}"
                    except Exception:
                        mz_str = ""
                else:
                    mz_str = ""

                sample_data[f"Peak {peak_idx} m/z"] = mz_str
                sample_data[f"Peak {peak_idx} RT_apex"] = row["RT_apex"]
                sample_data[f"Peak {peak_idx} pixel_start"] = row["pixel_start"]
                sample_data[f"Peak {peak_idx} pixel_end"] = row["pixel_end"]

            peak_summaries.append(sample_data)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

    if not peak_summaries:
        print("[!] No valid peak files found. Summary not created.")
        return

    summary_df = pd.DataFrame(peak_summaries)
    summary_df.sort_values(by="Sample", inplace=True)
    summary_df.to_csv(summary_file, index=False)

    print(f"[✔] Detailed (RT_apex-sorted) peak summary with m/z saved to: {summary_file}")
