import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image

from Config import Config

def _group_tag(group_name: str) -> str:
    # Your disk filenames use Group1 / Group2 (no spaces)
    return group_name.replace(" ", "")

def _paths_for_file(base: str, dirs: dict, group_name: str):
    tag = _group_tag(group_name)

    peaks_csv = os.path.join(dirs["csv"], f"{base}_peaks_{tag}.csv")
    png_path = os.path.join(dirs["png"], f"EIC_{base}_{tag}.png")

    # NEW: axis metadata produced by EIC builder
    axis_meta_csv = os.path.join(dirs["csv"], f"{base}_axis_{tag}.csv")

    pixel_csv = os.path.join(dirs["pixel"], f"{base}_peaks_pix_{tag}.csv")
    return peaks_csv, png_path, axis_meta_csv, pixel_csv

def process_file_checkpoint3(file_path: str, dirs: dict, group_name: str) -> str:
    """
    Resolving checkpoint:
    Reads:
      - {base}_peaks_{GroupTag}.csv
      - EIC_{base}_{GroupTag}.png
    Writes:
      - {base}_peaks_pix_{GroupTag}.csv
    """
    Config.set_mass_group(group_name)

    base = os.path.splitext(os.path.basename(file_path))[0]
    peaks_csv, png_path, axis_meta_csv, pixel_csv = _paths_for_file(base, dirs, group_name)

    # Parameters
    MIN_RT_DIFF = 0.095
    MIN_PIXEL_SEP = 6
    VALLEY_DROP_MIN = 0.40
    MARGIN_FRAC = 0.08
    VERY_CLOSE_RT = 0.02
    SHOULDER_RT_MAX = 0.0355  # min

    KERNEL_3x3 = np.ones((3, 3), np.uint8)

    # Functions
    def estimate_plot_bounds(h, w):
        t = int(MARGIN_FRAC * h)
        l = int(MARGIN_FRAC * w)
        return max(0, t), h - t, max(0, l), w - l

    def build_elevation_from_alpha(alpha):
        edges = cv2.Canny(alpha, 0, 0)
        edges = cv2.dilate(edges, KERNEL_3x3, iterations=1)
        dist = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 3)
        if dist.max() < 1:
            norm = np.zeros_like(dist, dtype=np.uint8)
        else:
            norm = (dist / dist.max() * 255).astype(np.uint8)
        return 255 - cv2.medianBlur(norm, 5)

    def watershed_labels(alpha_roi, seed_cols_rel):
        h, w = alpha_roi.shape
        if w < 2 or h < 2 or not np.any(seed_cols_rel):
            return np.zeros((h, w), np.int32)

        elev = build_elevation_from_alpha(alpha_roi)
        markers = np.zeros((h, w), np.int32)

        y_range = slice(h // 4, 3 * h // 4)
        for i, xc in enumerate(seed_cols_rel, 1):
            xc = int(np.clip(xc, 0, w - 1))
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
        nz = np.nonzero(out)[0]
        if nz.size:
            out[:nz[0]] = out[nz[0]]
            out[nz[-1] + 1:] = out[nz[-1]]
        return out

    def find_boundary_or_dip(lbl1d, profile, x0, x1, id1, id2):
        n = len(lbl1d)
        if n == 0:
            return 0

        if x0 > x1:
            x0, x1 = x1, x0

        x0 = max(0, min(x0, n - 1))
        x1 = max(0, min(x1, n - 1))

        if x0 == x1:
            return x0

        seg = lbl1d[x0:x1 + 1]
        diff_idx = np.where(np.diff(seg) != 0)[0]
        for k in diff_idx:
            L = seg[k]
            R = seg[k + 1] if k + 1 < seg.size else 0
            if (L == id1 and R == id2) or (L == id2 and R == id1):
                return x0 + k + 1

        s = profile[x0:x1 + 1]
        if len(s) < 3:
            return (x0 + x1) // 2

        d = np.diff(s)
        candidates = np.where((d[:-1] < 0) & (d[1:] > 0))[0]
        if candidates.size:
            return x0 + int(candidates[0] + 1)

        return x0 + int(np.argmin(s))

    def smooth_profile(p):
        if p.ndim == 1:
            return cv2.GaussianBlur(
                p.reshape(1, -1).astype(np.float32),
                (1, 21),
                0
            ).ravel()
        return p

    def valley_metrics(profile_smooth, xL, xR):
        if xL > xR:
            xL, xR = xR, xL
        xL = max(0, xL)
        xR = min(len(profile_smooth) - 1, xR)
        seg = profile_smooth[xL:xR + 1]
        if seg.size == 0:
            return (xL + xR) // 2, 0, 0, 0
        local_min = int(np.argmin(seg))
        vx = xL + local_min
        vy = profile_smooth[vx]
        yL = profile_smooth[xL]
        yR = profile_smooth[xR]
        return vx, float(vy), float(yL), float(yR)

    def is_smaller_shoulder(grp, idx_a, idx_b, shoulder_rt_max):
        """
        True if peak idx_a is a shoulder of idx_b:
          - very close in RT_apex (< shoulder_rt_max minutes)
          - and smaller (height if available for both, else area)
        """
        rt_a = float(grp["RT_apex"].iloc[idx_a])
        rt_b = float(grp["RT_apex"].iloc[idx_b])
        if abs(rt_a - rt_b) >= shoulder_rt_max:
            return False

        ha = grp["height"].iloc[idx_a]
        hb = grp["height"].iloc[idx_b]

        # Use height when both are available
        if pd.notna(ha) and pd.notna(hb):
            return float(ha) < float(hb)

        # Otherwise fall back to area
        return float(grp["area"].iloc[idx_a]) < float(grp["area"].iloc[idx_b])

    # Main Loop
    if os.path.exists(pixel_csv):
        return f"[↷] {base} ({group_name}) pixel csv cached"

    if not (os.path.exists(peaks_csv) and os.path.exists(png_path)):
        return f"[!] Skipping {base} ({group_name}): missing peaks csv or png"

    df = pd.read_csv(
        peaks_csv,
        usecols=["m/z", "RT_start", "RT_apex", "RT_end", "Peak Area", "height"]
    ).rename(columns={"Peak Area": "area"})

    df = df[(df["RT_end"] - df["RT_start"]).between(0.03, 0.75)]
    df["height"] = df.get("height", np.nan)

    if df.empty:
        return f"[!] {base} ({group_name}): no peaks after filtering"

    im = Image.open(png_path).convert("RGBA")
    W, H = im.size
    A = np.array(im)[..., 3].astype(np.uint8)

    axis_meta_csv = os.path.join(dirs["csv"], f"{base}_axis_{_group_tag(group_name)}.csv")

    if os.path.exists(axis_meta_csv):
        meta = pd.read_csv(axis_meta_csv).iloc[0]
        rt_min = float(meta["x0"])
        rt_max = float(meta["x1"])
    else:
        rt_min = df["RT_start"].min() - 0.1
        rt_max = df["RT_end"].max() + 0.1

    rt_range = rt_max - rt_min
    if rt_range <= 0:
        return f"[!] {base} ({group_name}): invalid RT range for pixel mapping"

    width_factor = (W - 1) / rt_range

    def rt_to_px(x):
        return np.clip(((np.asarray(x) - rt_min) * width_factor).astype(int), 0, W - 1)

    def px_to_rt(px):
        return rt_min + (np.asarray(px) / (W - 1)) * rt_range

    top, bot, left, right = estimate_plot_bounds(H, W)
    alpha_roi = A[top:bot, left:right]
    profile = smooth_profile(alpha_roi.sum(axis=0).astype(np.float32))

    df["_mz_key"] = df["m/z"].astype(str)
    df = df.sort_values(["_mz_key", "RT_apex"]).reset_index(drop=True)

    rows_out = []

    for mz_key, grp in df.groupby("_mz_key", sort=False):
        grp = grp.sort_values("RT_apex").reset_index(drop=True)

        if len(grp) == 1:
            row = grp.iloc[0].to_dict()
            ps, pe = rt_to_px([row["RT_start"], row["RT_end"]])
            row.update({
                "RT_start": float(px_to_rt(ps)),
                "RT_end": float(px_to_rt(pe)),
                "Pixel_start": int(ps),
                "Pixel_end": int(pe),
                "peak_type": "resolved",
                "cluster_id": f"{mz_key}_0",
                "is_cluster_lead": True
            })
            rows_out.append(row)
            continue

        apex_px = rt_to_px(grp["RT_apex"].values)
        l_px = rt_to_px(grp["RT_start"].values)
        r_px = rt_to_px(grp["RT_end"].values)

        seeds_rel = np.clip(apex_px - left, 0, right - left - 1)
        labels = watershed_labels(alpha_roi, seeds_rel)
        lbl1d = column_labels(labels)

        resolved_pair = []
        for i in range(len(grp) - 1):
            rt_diff = grp["RT_apex"].iloc[i + 1] - grp["RT_apex"].iloc[i]
            px_diff = abs(apex_px[i + 1] - apex_px[i])

            r1 = int(np.clip(apex_px[i] - left, 0, right - left - 1))
            r2 = int(np.clip(apex_px[i + 1] - left, 0, right - left - 1))
            rL, rR = sorted([r1, r2])

            vx, vy, yL, yR = valley_metrics(profile, rL, rR)
            min_apex_y = max(1.0, min(yL, yR))
            valley_drop = 1 - (vy / min_apex_y)

            no_rt_overlap = grp["RT_start"].iloc[i + 1] >= grp["RT_end"].iloc[i]
            no_pixel_overlap = r_px[i] < l_px[i + 1]

            is_resolved = (
                (rt_diff >= MIN_RT_DIFF)
                and (px_diff >= MIN_PIXEL_SEP)
                and (valley_drop >= VALLEY_DROP_MIN)
                and no_rt_overlap
                and no_pixel_overlap
            )
            resolved_pair.append(is_resolved)

        clusters = []
        current = [0]
        for i, res in enumerate(resolved_pair):
            if res:
                clusters.append(current)
                current = [i + 1]
            else:
                current.append(i + 1)
        clusters.append(current)

        for cidx, members in enumerate(clusters):
            member_idxs = np.array(members)

            if len(member_idxs) == 1:
                i = member_idxs[0]
                ps = int(l_px[i])
                pe = int(r_px[i])

                row = grp.iloc[i].to_dict()
                row.update({
                    "RT_start": float(px_to_rt(ps)),
                    "RT_end": float(px_to_rt(pe)),
                    "Pixel_start": ps,
                    "Pixel_end": pe,
                    "peak_type": "resolved",
                    "cluster_id": f"{mz_key}_{cidx}",
                    "is_cluster_lead": True
                })
                rows_out.append(row)
                continue

            spans = {}
            for idx in member_idxs:
                lbl = idx + 1
                mask = (lbl1d == lbl)
                if np.any(mask):
                    locs = np.where(mask)[0]
                    spans[lbl] = (left + locs[0], left + locs[-1])

            cuts = {}
            for k in range(len(member_idxs) - 1):
                i1 = member_idxs[k]
                i2 = member_idxs[k + 1]
                rt_diff = grp["RT_apex"].iloc[i2] - grp["RT_apex"].iloc[i1]

                if rt_diff < VERY_CLOSE_RT:
                    cut = (apex_px[i1] + apex_px[i2]) // 2
                else:
                    r1 = int(np.clip(apex_px[i1] - left, 0, right - left - 1))
                    r2 = int(np.clip(apex_px[i2] - left, 0, right - left - 1))
                    cut_rel = find_boundary_or_dip(lbl1d, profile, r1, r2, i1 + 1, i2 + 1)
                    cut = left + cut_rel

                lo = min(apex_px[i1], apex_px[i2])
                hi = max(apex_px[i1], apex_px[i2])
                if hi - lo >= 2:
                    cut = max(cut, lo + 1)
                    cut = min(cut, hi - 1)

                cuts[k] = int(np.clip(cut, 0, W - 1))

            for k, idx in enumerate(member_idxs):
                lbl = idx + 1

                if k == 0:
                    ps = int(l_px[idx])
                else:
                    ps = cuts[k - 1]

                if k == len(member_idxs) - 1:
                    pe = int(r_px[idx])
                else:
                    pe = cuts[k]

                if lbl in spans:
                    wL, wR = spans[lbl]
                    ps = max(ps, wL)
                    pe = min(pe, wR)

                ps = int(np.clip(ps, 0, W - 2))
                pe = int(np.clip(pe, ps + 1, W))

                # Determine if THIS member is a shoulder (smaller peak within SHOULDER_RT_MAX of a larger neighbor)
                is_shoulder = False

                # Only compare to immediate neighbors in RT order within this cluster
                if k > 0:
                    # idx is shoulder of previous member?
                    if is_smaller_shoulder(grp, idx, member_idxs[k - 1], SHOULDER_RT_MAX):
                        is_shoulder = True

                if (not is_shoulder) and (k < len(member_idxs) - 1):
                    # idx is shoulder of next member?
                    if is_smaller_shoulder(grp, idx, member_idxs[k + 1], SHOULDER_RT_MAX):
                        is_shoulder = True

                peak_type = "shoulder" if is_shoulder else "coeluting"

                row = grp.iloc[idx].to_dict()
                row.update({
                    "RT_start": float(px_to_rt(ps)),
                    "RT_end": float(px_to_rt(pe)),
                    "Pixel_start": ps,
                    "Pixel_end": pe,
                    "peak_type": peak_type,
                    "cluster_id": f"{mz_key}_{cidx}",
                    "is_cluster_lead": bool(k == 0)
                })
                rows_out.append(row)

    df_out = pd.DataFrame(rows_out)

    df_out = df_out.sort_values(
        by=["RT_apex", "RT_start", "m/z"],
        ascending=[True, True, True],
        kind="mergesort"
    ).reset_index(drop=True)

    df_out["peak_num"] = np.arange(1, len(df_out) + 1)

    df_out.to_csv(pixel_csv, index=False)
    return f"[✔] {base} ({group_name}): wrote {len(df_out)} rows to {os.path.basename(pixel_csv)}"

def count_peaks_per_file_summary(dirs: dict, group_name: str) -> str:
    Config.set_mass_group(group_name)

    print("\n[✔] Generating detailed peak count summary (sorted by RT_apex) with m/z...")

    peaks_dir = dirs["pixel"]
    tag = _group_tag(group_name)

    summary_file = os.path.join(dirs["counts"], f"peak_count_summary_{tag}.csv")

    if not os.path.exists(peaks_dir):
        return f"[!] Peaks directory not found: {peaks_dir}"

    peak_summaries = []

    for file_name in os.listdir(peaks_dir):
        if not file_name.endswith(f"_peaks_pix_{tag}.csv"):
            continue

        sample_name = file_name.replace(f"_peaks_pix_{tag}.csv", "")
        peaks_csv_path = os.path.join(peaks_dir, file_name)

        try:
            df = pd.read_csv(peaks_csv_path)

            base_required = ["RT_apex", "Pixel_start", "Pixel_end"]
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

                if has_mz and pd.notna(row.get("m/z", np.nan)):
                    try:
                        mz_val = float(row["m/z"])
                        mz_str = f"{mz_val:.4f}"
                    except Exception:
                        mz_str = ""
                else:
                    mz_str = ""

                sample_data[f"Peak {peak_idx} m/z"] = mz_str
                sample_data[f"Peak {peak_idx} RT_apex"] = row["RT_apex"]
                sample_data[f"Peak {peak_idx} pixel_start"] = row["Pixel_start"]
                sample_data[f"Peak {peak_idx} pixel_end"] = row["Pixel_end"]

            peak_summaries.append(sample_data)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

    if not peak_summaries:
        return "[!] No valid peak files found. Summary not created."

    summary_df = pd.DataFrame(peak_summaries)
    summary_df.sort_values(by="Sample", inplace=True)
    summary_df.to_csv(summary_file, index=False)

    return f"[✔] Detailed (RT_apex-sorted) peak summary with m/z saved to: {summary_file}"