#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_diag.py
------------
Quick plots for diagnostics CSV produced by evaluate_full (diag_*.csv).
Modified to compare TWO CSV files (Base vs Improved).

Changes:
- Lines are black, no labels.
- Meas points from BASE only (red).
- Base Sim (Blue), Imp Sim (green).

Usage:
    python plot_diag.py --csv_base diag_base.csv --csv_imp diag_new.csv --outdir diag_compare

Optional:
    --knn_thresh 2.0
    --cyc_good 0.15
    --cyc_warn 0.25
    --topn 50

    --savefmt png|svg
    --figsize W,H
    --grid
    --grid_alpha 0.3
"""
import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


POSSIBLE_COLS = [
    "sup_ps", "prior_l2_ps", "prior_bnd_ps",
    "cyc_sim_ps", "knn_min", "knn_mean_k", "jac_sigma_max", "proxy_floor_ps"
]


def ensure_cols(df: pd.DataFrame, name: str):
    have = [c for c in POSSIBLE_COLS if c in df.columns]
    print(f"[{name}] existing columns: {have}")
    if len(have) == 0:
        print(f"Warning: No expected columns found in {name}. CSV columns: {list(df.columns)}")
    return set(have)


def _parse_figsize(s: str):
    """
    Parse 'W,H' -> (float(W), float(H))
    """
    if s is None:
        return (6.0, 4.0)
    try:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 2:
            raise ValueError
        w = float(parts[0])
        h = float(parts[1])
        if w <= 0 or h <= 0:
            raise ValueError
        return (w, h)
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"--figsize expects 'W,H' like '6,4'. Got: {s}"
        ) from e


def _ext_for_fmt(savefmt: str) -> str:
    savefmt = (savefmt or "png").lower()
    if savefmt not in ("png", "svg"):
        raise ValueError(f"Unsupported --savefmt: {savefmt}. Use png or svg.")
    return "." + savefmt


def _savefig(outpath: str, savefmt: str = "png", dpi: int = 140):
    savefmt = (savefmt or "png").lower()
    if savefmt == "png":
        plt.savefig(outpath, dpi=dpi)
    else:
        plt.savefig(outpath)


def _apply_grid(enable: bool = False, alpha: float = 0.3):
    if enable:
        plt.grid(True, alpha=alpha)


def _split_sim_meas(df):
    if "domain" in df.columns:
        sim = df[df["domain"] == "sim"]
        meas = df[df["domain"] == "meas"]
    else:
        sim = df
        meas = df.iloc[0:0]
    return sim, meas


def scatter_xy_compare(
    df_base: pd.DataFrame,
    df_imp: pd.DataFrame,
    x: str,
    y: str,
    outpath: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    figsize=(6, 4),
    savefmt: str = "png",
    grid: bool = False,
    grid_alpha: float = 0.3,
    dpi: int = 140,
):
    plt.figure(figsize=figsize)

    # Split
    base_sim, base_meas = _split_sim_meas(df_base)
    imp_sim, _ = _split_sim_meas(df_imp)  # Ignore imp meas

    # Plot Base Sim (Blue)
    if len(base_sim) > 0:
        plt.scatter(
            base_sim[x].values, base_sim[y].values,
            s=15, c='tab:blue', alpha=0.5, label='Base (sim)'
        )
    
    # Plot Improved Sim (green)
    if len(imp_sim) > 0:
        plt.scatter(
            imp_sim[x].values, imp_sim[y].values,
            s=15, c='tab:green', alpha=0.5, label='Improved (sim)'
        )

    # Plot Base Meas (red) - Only once
    if len(base_meas) > 0:
        plt.scatter(
            base_meas[x].values, base_meas[y].values,
            s=40, c='red', marker='x', alpha=1.0, linewidths=1.5, label='Meas'
        )

    plt.xlabel(xlabel or x)
    plt.ylabel(ylabel or y)
    if title:
        plt.title(title)
    
    plt.legend()
    _apply_grid(grid, grid_alpha)

    plt.tight_layout()
    _savefig(outpath, savefmt=savefmt, dpi=dpi)
    plt.close()


def scatter_knn_vs_cyc_compare(
    df_base: pd.DataFrame,
    df_imp: pd.DataFrame,
    xcol: str,
    outpath: str,
    xlabel: str,
    good: float = 0.15,
    warn: float = 0.25,
    knn_thresh: float = 2.0,
    figsize=(6, 4),
    savefmt: str = "png",
    grid: bool = False,
    grid_alpha: float = 0.3,
    dpi: int = 140,
):
    plt.figure(figsize=figsize)

    # Split
    base_sim, base_meas = _split_sim_meas(df_base)
    imp_sim, _ = _split_sim_meas(df_imp)

    # 1. Base Sim -> Blue
    if len(base_sim) > 0:
        plt.scatter(
            base_sim[xcol], base_sim["cyc_sim_ps"], 
            s=12, c='tab:blue', alpha=0.5, label="Base (sim)"
        )
    
    # 2. Improved Sim -> green
    if len(imp_sim) > 0:
        plt.scatter(
            imp_sim[xcol], imp_sim["cyc_sim_ps"], 
            s=12, c='tab:green', alpha=0.5, label="Improved (sim)"
        )

    # 3. Base Meas -> red (Only Base)
    if len(base_meas) > 0:
        plt.scatter(
            base_meas[xcol], base_meas["cyc_sim_ps"],
            marker="x", s=40, c='red', alpha=1.0, linewidths=1.5,
            label="Meas"
        )

    # Reference lines (Black, no labels)
    plt.axvline(knn_thresh, linestyle="--", color='k', alpha=0.6)
    plt.axhline(good, linestyle="--", color='k', alpha=0.6)
    plt.axhline(warn, linestyle="--", color='k', alpha=0.6)

    plt.xlabel(xlabel)
    plt.ylabel("cyc_sim per-sample")
    plt.title(f"{xcol} vs cyc_sim (Compare)")

    plt.legend(loc='best', fontsize='small')
    _apply_grid(grid, grid_alpha)

    plt.tight_layout()
    _savefig(outpath, savefmt=savefmt, dpi=dpi)
    plt.close()


def scatter_knn_vs_prior_compare(
    df_base: pd.DataFrame,
    df_imp: pd.DataFrame,
    xcol: str,
    outpath: str,
    xlabel: str,
    knn_thresh: float = 2.0,
    figsize=(6, 4),
    savefmt: str = "png",
    grid: bool = False,
    grid_alpha: float = 0.3,
    dpi: int = 140,
):
    plt.figure(figsize=figsize)

    # Split
    base_sim, base_meas = _split_sim_meas(df_base)
    imp_sim, _ = _split_sim_meas(df_imp)

    # 1. Base Sim -> Blue
    if len(base_sim) > 0:
        plt.scatter(
            base_sim[xcol], base_sim["prior_bnd_ps"], 
            s=12, c='tab:blue', alpha=0.5, label="Base (sim)"
        )
    
    # 2. Improved Sim -> green
    if len(imp_sim) > 0:
        plt.scatter(
            imp_sim[xcol], imp_sim["prior_bnd_ps"], 
            s=12, c='tab:green', alpha=0.5, label="Improved (sim)"
        )

    # 3. Base Meas -> red
    if len(base_meas) > 0:
        plt.scatter(
            base_meas[xcol], base_meas["prior_bnd_ps"],
            marker="x", s=40, c='red', alpha=1.0, linewidths=1.5,
            label="Meas"
        )

    # Reference lines (Black, no labels)
    plt.axvline(knn_thresh, linestyle="--", color='k', alpha=0.6)

    plt.xlabel(xlabel)
    plt.ylabel("prior_bnd_ps per-sample")
    plt.title(f"{xcol} vs prior_bnd_ps (Compare)")

    plt.legend(loc='best', fontsize='small')
    _apply_grid(grid, grid_alpha)

    plt.tight_layout()
    _savefig(outpath, savefmt=savefmt, dpi=dpi)
    plt.close()


def hist_1d_compare(
    base_series: pd.Series,
    imp_series: pd.Series,
    outpath: str,
    bins: int = 60,
    xlabel: str | None = None,
    title: str | None = None,
    figsize=(6, 4),
    savefmt: str = "png",
    grid: bool = False,
    grid_alpha: float = 0.3,
    dpi: int = 140,
):
    plt.figure(figsize=figsize)
    
    # Determine common bin range
    v1 = base_series.dropna().values
    v2 = imp_series.dropna().values
    
    if len(v1) == 0 and len(v2) == 0:
        return # Skip empty

    all_vals = np.concatenate([v1, v2])
    min_v, max_v = np.min(all_vals), np.max(all_vals)
    bin_edges = np.linspace(min_v, max_v, bins)

    plt.hist(v1, bins=bin_edges, alpha=0.5, color='tab:blue', label='Base')
    plt.hist(v2, bins=bin_edges, alpha=0.5, color='tab:green', label='Improved')

    plt.xlabel(xlabel or "")
    if title:
        plt.title(title)
    
    plt.legend()
    _apply_grid(grid, grid_alpha)

    plt.tight_layout()
    _savefig(outpath, savefmt=savefmt, dpi=dpi)
    plt.close()


def save_bad_samples(df, out_path, topn, col, desc):
    if col in df.columns:
        df.sort_values(col, ascending=False).head(topn) \
            .to_csv(out_path, index=False)
        print(f"Saved top {topn} bad ({col}) for {desc} to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    # Changed arguments
    ap.add_argument("--csv_base", type=str, required=True, help="Path to BASE diagnostics CSV")
    ap.add_argument("--csv_imp", type=str, required=True, help="Path to IMPROVED diagnostics CSV")
    
    ap.add_argument("--outdir", type=str, default="diag_compare", help="Output directory for plots")
    ap.add_argument("--knn_thresh", type=float, default=2.0, help="Heuristic OOD threshold for KNN (z-space)")
    ap.add_argument("--cyc_good", type=float, default=0.15, help="Guide line for cyc_sim_ps (good)")
    ap.add_argument("--cyc_warn", type=float, default=0.25, help="Guide line for cyc_sim_ps (warn)")
    ap.add_argument("--topn", type=int, default=50, help="Rows to save for worst samples tables")

    ap.add_argument("--savefmt", type=str, default="png", choices=["png", "svg"],
                    help="Figure save format: png or svg")
    ap.add_argument("--figsize", type=_parse_figsize, default=(6.0, 4.0),
                    help="Matplotlib figsize in inches: 'W,H' e.g. '6,4'")

    ap.add_argument("--grid", action="store_true",
                    help="Enable matplotlib grid for all plots")
    ap.add_argument("--grid_alpha", type=float, default=0.3,
                    help="Grid transparency (alpha)")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load Data
    print(f"Loading Base: {args.csv_base}")
    df_base = pd.read_csv(args.csv_base)
    print(f"Loading Improved: {args.csv_imp}")
    df_imp = pd.read_csv(args.csv_imp)

    have_base = ensure_cols(df_base, "Base")
    have_imp = ensure_cols(df_imp, "Improved")
    
    # Common columns
    have_common = have_base.intersection(have_imp)

    ext = _ext_for_fmt(args.savefmt)

    # --- Scatter: kNN vs cyc_sim ---
    if "knn_min" in have_common and "cyc_sim_ps" in have_common:
        scatter_knn_vs_cyc_compare(
            df_base, df_imp, "knn_min",
            os.path.join(args.outdir, f"compare_knnmin_vs_cyc{ext}"),
            xlabel="kNN distance (z-space)",
            good=args.cyc_good, warn=args.cyc_warn, knn_thresh=args.knn_thresh,
            figsize=args.figsize, savefmt=args.savefmt,
            grid=args.grid, grid_alpha=args.grid_alpha
        )

    if "knn_mean_k" in have_common and "cyc_sim_ps" in have_common:
        scatter_knn_vs_cyc_compare(
            df_base, df_imp, "knn_mean_k",
            os.path.join(args.outdir, f"compare_knnmeank_vs_cyc{ext}"),
            xlabel="mean k-NN distance (k)",
            good=args.cyc_good, warn=args.cyc_warn, knn_thresh=args.knn_thresh,
            figsize=args.figsize, savefmt=args.savefmt,
            grid=args.grid, grid_alpha=args.grid_alpha
        )

    # --- Scatter: kNN vs prior_bnd_ps ---
    if "knn_min" in have_common and "prior_bnd_ps" in have_common:
        scatter_knn_vs_prior_compare(
            df_base, df_imp, "knn_min",
            os.path.join(args.outdir, f"compare_knnmin_vs_priorbnd{ext}"),
            xlabel="kNN distance (z-space)",
            knn_thresh=args.knn_thresh,
            figsize=args.figsize, savefmt=args.savefmt,
            grid=args.grid, grid_alpha=args.grid_alpha
        )

    # --- Scatter: cyc_sim_ps vs jac_sigma_max ---
    if "cyc_sim_ps" in have_common and "jac_sigma_max" in have_common:
        scatter_xy_compare(
            df_base, df_imp, "cyc_sim_ps", "jac_sigma_max",
            os.path.join(args.outdir, f"compare_cyc_vs_jac{ext}"),
            xlabel="cyc_sim_ps",
            ylabel="jac_sigma_max",
            title="cyc_sim_ps vs jac_sigma_max",
            figsize=args.figsize, savefmt=args.savefmt,
            grid=args.grid, grid_alpha=args.grid_alpha
        )

    # --- Histograms ---
    # Helper for repetitive calls
    def run_hist(col, xlabel, title):
        if col in have_common:
            hist_1d_compare(
                df_base[col], df_imp[col],
                os.path.join(args.outdir, f"compare_hist_{col}{ext}"),
                bins=60, xlabel=xlabel, title=title,
                figsize=args.figsize, savefmt=args.savefmt,
                grid=args.grid, grid_alpha=args.grid_alpha
            )

    run_hist("jac_sigma_max", "Jacobian spectral norm", "Jacobian spectral norm Distribution")
    run_hist("prior_bnd_ps", "prior_bnd per-sample", "prior_bnd Distribution")
    run_hist("cyc_sim_ps", "cyc_sim per-sample", "cyc_sim Distribution")
    run_hist("proxy_floor_ps", "proxy_floor per-sample", "proxy_floor Distribution")

    # --- Worst samples tables (Output separate files for base and imp) ---
    # Base
    save_bad_samples(df_base, os.path.join(args.outdir, "bad_samples_base_cyc.csv"), args.topn, "cyc_sim_ps", "Base")
    save_bad_samples(df_base, os.path.join(args.outdir, "bad_samples_base_priorbnd.csv"), args.topn, "prior_bnd_ps", "Base")
    
    # Improved
    save_bad_samples(df_imp, os.path.join(args.outdir, "bad_samples_imp_cyc.csv"), args.topn, "cyc_sim_ps", "Improved")
    save_bad_samples(df_imp, os.path.join(args.outdir, "bad_samples_imp_priorbnd.csv"), args.topn, "prior_bnd_ps", "Improved")

    print(f"[plot_diag] Wrote comparison plots to: {args.outdir}")


if __name__ == "__main__":
    main()