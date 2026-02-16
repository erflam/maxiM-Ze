import colorsys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go


# --------------------------------------------------------
# Patch path builder (matches current naming)
# --------------------------------------------------------

def _build_patch_path(patch_dir: Path, file_base: str, mass: float, peak_num: int, group: str) -> Path:
    mass_str = f"{mass:.6f}".rstrip("0").rstrip(".")
    return patch_dir / f"{file_base}_mass{mass_str}_Peak{int(peak_num)}_{group}.png"


# --------------------------------------------------------
# Static Composite Visualization
# --------------------------------------------------------

def checkpoint_visualization_static_composite(cls, group_name: str) -> Optional[Path]:

    cluster_dir = Path(cls.dirs["clustering"])
    composite_dir = Path(cls.dirs["composites"])
    composite_dir.mkdir(parents=True, exist_ok=True)

    alignment_file = cluster_dir / "peak_alignment.csv"
    if not alignment_file.exists():
        print("[Visualization] No peak_alignment.csv found.")
        return None

    df = pd.read_csv(alignment_file)

    required = {
        "file",
        "mass",
        "isomer_position",
        "rt_start",
        "rt_apex",
        "rt_end",
        "aligned_rt_apex",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"[Visualization] Missing columns: {missing}")

    all_files = sorted(df["file"].astype(str).unique())
    n_samples = max(len(all_files), 1)

    colors: Dict[str, tuple] = {}
    for i, f in enumerate(all_files):
        hue = i / n_samples
        r, g, b = colorsys.hsv_to_rgb(hue, 0.5, 1.0)
        colors[f] = (r, g, b, 0.5)

    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot peak rectangles
    for f in all_files:
        sub = df[df["file"] == f]
        for _, peak in sub.iterrows():
            width = peak["rt_end"] - peak["rt_start"]
            center = peak["rt_apex"]

            rect = patches.Rectangle(
                (center - width / 2, 0),
                width,
                1.0,
                facecolor=colors[f],
                edgecolor="none",
                alpha=0.5,
            )
            ax.add_patch(rect)

    # Plot aligned cluster centers
    for (mass, pos), grp in df.groupby(["mass", "isomer_position"]):
        rt_mean = grp["aligned_rt_apex"].mean()
        ax.axvline(x=rt_mean, linestyle="--", alpha=0.3)
        ax.text(
            rt_mean,
            1.18,
            f"{mass:.4f}_{rt_mean:.2f} - P{pos}",
            rotation=90,
            va="top",
            ha="right",
            alpha=0.7,
        )

    ax.set_xlabel("Retention Time (min)")
    ax.set_ylabel("Normalized Intensity")
    ax.set_title(f"Composite Peak Visualization - {group_name}")

    legend_elements = [
        patches.Patch(facecolor=colors[f], label=f)
        for f in all_files
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")

    rt_min = df["rt_start"].min()
    rt_max = df["rt_end"].max()
    ax.set_xlim(rt_min - 0.5, rt_max + 0.5)
    ax.set_ylim(0, 1.2)

    out_path = composite_dir / f"composite_visualization_{group_name}.png"

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Visualization] Static composite saved → {out_path}")
    return out_path


# --------------------------------------------------------
# Interactive Patch Overlay Visualization
# --------------------------------------------------------

def checkpoint_visualization_interactive_patches(cls, group_name: str) -> Optional[Path]:

    cluster_dir = Path(cls.dirs["clustering"])
    composite_dir = Path(cls.dirs["composites"])
    composite_dir.mkdir(parents=True, exist_ok=True)

    alignment_file = cluster_dir / "peak_alignment.csv"
    if not alignment_file.exists():
        print("[Visualization] No peak_alignment.csv found.")
        return None

    df = pd.read_csv(alignment_file)

    required = {
        "file",
        "mass",
        "peak_num",
        "isomer_position",
        "rt_start",
        "rt_end",
        "aligned_rt_apex",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"[Visualization] Missing columns: {missing}")

    patch_dir = Path(cls.dirs["patch"])

    fig = go.Figure()

    for _, peak in df.iterrows():
        file_base = peak["file"]
        mass = peak["mass"]
        peak_num = peak["peak_num"]

        aligned_center = peak["aligned_rt_apex"]
        width = peak["rt_end"] - peak["rt_start"]

        patch_path = _build_patch_path(
            patch_dir,
            file_base,
            mass,
            peak_num,
            group_name,  # FIXED
        )

        if not patch_path.exists():
            continue

        rt_start = aligned_center - (width / 2)

        fig.add_layout_image(
            dict(
                source=str(patch_path),
                x=rt_start,
                y=1,
                sizex=width,
                sizey=1,
                xref="x",
                yref="y",
                sizing="stretch",
                opacity=0.75,
                layer="below",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[aligned_center],
                y=[0.5],
                mode="markers",
                marker=dict(size=5, opacity=0),
                showlegend=False,
                hovertemplate=(
                    f"File: {file_base}<br>"
                    f"Mass: {mass:.6f}<br>"
                    f"Peak: {peak_num}<br>"
                    f"Aligned RT: {aligned_center:.3f}<br>"
                    f"Isomer position: {peak['isomer_position']}<br>"
                    "<extra></extra>"
                ),
            )
        )

    rt_min = df["rt_start"].min()
    rt_max = df["rt_end"].max()

    fig.update_layout(
        title=f"Interactive Peak Patches Composite - {group_name}",
        xaxis_title="Retention Time (min)",
        yaxis_title="Normalized Intensity",
        height=900,
        width=1800,
        yaxis_range=[0, 1.2],
        plot_bgcolor="white",
    )

    fig.update_xaxes(range=[rt_min - 0.5, rt_max + 0.5])

    out_html = composite_dir / f"composite_patches_aligned_{group_name}.html"
    fig.write_html(str(out_html), include_plotlyjs=True, full_html=True)

    print(f"[Visualization] Interactive composite saved → {out_html}")
    return out_html


# --------------------------------------------------------
# Pipeline Entry Point
# --------------------------------------------------------

def process_visualizations(cls, group_name: str) -> str:
    out1 = checkpoint_visualization_static_composite(cls, group_name)
    out2 = checkpoint_visualization_interactive_patches(cls, group_name)
    return f"Visualization complete → {out1} ; {out2}"


