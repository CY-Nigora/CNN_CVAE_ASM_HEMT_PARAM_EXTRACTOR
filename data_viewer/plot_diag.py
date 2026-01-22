#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_diag.py
------------
Quick plots for diagnostics CSV produced by evaluate_full (diag_*.csv).

Usage:
    python plot_diag.py --csv diag_test.csv --outdir diag_plots

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


def ensure_cols(df: pd.DataFrame):
    have = [c for c in POSSIBLE_COLS if c in df.columns]
    print(f"exsisting columns: {have}")
    if len(have) == 0:
        raise ValueError(f"No expected columns found. CSV columns: {list(df.columns)}")
    return have


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
    """
    Save current figure with format control.
    - png: uses dpi
    - svg: vector; dpi is not meaningful
    """
    savefmt = (savefmt or "png").lower()
    if savefmt == "png":
        plt.savefig(outpath, dpi=dpi)
    else:
        plt.savefig(outpath)


def _apply_grid(enable: bool = False, alpha: float = 0.3):
    if enable:
        plt.grid(True, alpha=alpha)


def scatter_xy(
    df: pd.DataFrame,
    x: str,
    y: str,
    outpath: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    vlines=None,
    hlines=None,
    s: int = 12,
    figsize=(6, 4),
    savefmt: str = "png",
    grid: bool = False,
    grid_alpha: float = 0.3,
    dpi: int = 140,
):
    plt.figure(figsize=figsize)
    plt.scatter(df[x].values, df[y].values, s=s, alpha=0.7)

    if vlines:
        for vx in vlines:
            plt.axvline(vx, linestyle="--")
    if hlines:
        for hy in hlines:
            plt.axhline(hy, linestyle="--")

    plt.xlabel(xlabel or x)
    plt.ylabel(ylabel or y)
    if title:
        plt.title(title)

    _apply_grid(grid, grid_alpha)

    plt.tight_layout()
    _savefig(outpath, savefmt=savefmt, dpi=dpi)
    plt.close()


# === overlay meas points on kNN→cyc scatter ===
def scatter_knn_vs_cyc_with_meas(
    df: pd.DataFrame,
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

    # 分域
    if "domain" in df.columns:
        sim = df[df["domain"] == "sim"]
        meas = df[df["domain"] == "meas"]
    else:
        sim = df
        meas = df.iloc[0:0]  # empty

    # SIM 点
    if xcol in sim.columns and "cyc_sim_ps" in sim.columns and len(sim):
        plt.scatter(sim[xcol].values, sim["cyc_sim_ps"].values, s=12, alpha=0.7, label="sim")

    # MEAS 点
    if xcol in meas.columns and "cyc_sim_ps" in meas.columns and len(meas):
        plt.scatter(
            meas[xcol].values,
            meas["cyc_sim_ps"].values,
            marker="x",
            s=36,
            alpha=0.9,
            label="meas",
        )

    # 参考线
    plt.axvline(knn_thresh, linestyle="--")
    plt.axhline(good, linestyle="--")
    plt.axhline(warn, linestyle="--")

    plt.xlabel(xlabel)
    plt.ylabel("cyc_sim per-sample  ")
    plt.title(f"{xcol} vs cyc_sim")

    if len(meas) or (len(sim) and len(meas)):
        plt.legend()

    _apply_grid(grid, grid_alpha)

    plt.tight_layout()
    _savefig(outpath, savefmt=savefmt, dpi=dpi)
    plt.close()


def scatter_knn_vs_prior_with_meas(
    df: pd.DataFrame,
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

    # 分域
    if "domain" in df.columns:
        sim = df[df["domain"] == "sim"]
        meas = df[df["domain"] == "meas"]
    else:
        sim = df
        meas = df.iloc[0:0]  # empty

    # SIM 点
    if xcol in sim.columns and "prior_bnd_ps" in sim.columns and len(sim):
        plt.scatter(sim[xcol].values, sim["prior_bnd_ps"].values, s=12, alpha=0.7, label="sim")

    # MEAS 点
    if xcol in meas.columns and "prior_bnd_ps" in meas.columns and len(meas):
        plt.scatter(
            meas[xcol].values,
            meas["prior_bnd_ps"].values,
            marker="x",
            s=36,
            alpha=0.9,
            label="meas",
        )

    # 参考线
    plt.axvline(knn_thresh, linestyle="--")

    plt.xlabel(xlabel)
    plt.ylabel("prior_bnd_ps per-sample  ")
    plt.title(f"{xcol} vs prior_bnd_ps")

    if len(meas) or (len(sim) and len(meas)):
        plt.legend()

    _apply_grid(grid, grid_alpha)

    plt.tight_layout()
    _savefig(outpath, savefmt=savefmt, dpi=dpi)
    plt.close()


def hist_1d(
    series: pd.Series,
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
    plt.hist(series.dropna().values, bins=bins)
    plt.xlabel(xlabel or "")
    if title:
        plt.title(title)

    _apply_grid(grid, grid_alpha)

    plt.tight_layout()
    _savefig(outpath, savefmt=savefmt, dpi=dpi)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="diag_test.csv", help="Path to diagnostics CSV")
    ap.add_argument("--outdir", type=str, default="diag_plots", help="Output directory for plots")
    ap.add_argument("--knn_thresh", type=float, default=2.0, help="Heuristic OOD threshold for KNN (z-space)")
    ap.add_argument("--cyc_good", type=float, default=0.15, help="Guide line for cyc_sim_ps (good)")
    ap.add_argument("--cyc_warn", type=float, default=0.25, help="Guide line for cyc_sim_ps (warn)")
    ap.add_argument("--topn", type=int, default=50, help="Rows to save for worst samples tables")

    # NEW: save format + figure size
    ap.add_argument("--savefmt", type=str, default="png", choices=["png", "svg"],
                    help="Figure save format: png or svg")
    ap.add_argument("--figsize", type=_parse_figsize, default=(6.0, 4.0),
                    help="Matplotlib figsize in inches: 'W,H' e.g. '6,4'")

    # NEW: global grid switch + alpha
    ap.add_argument("--grid", action="store_true",
                    help="Enable matplotlib grid for all plots")
    ap.add_argument("--grid_alpha", type=float, default=0.3,
                    help="Grid transparency (alpha)")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    have = ensure_cols(df)

    ext = _ext_for_fmt(args.savefmt)

    # Scatter: kNN vs cyc_sim
    if "knn_min" in have and "cyc_sim_ps" in have:
        scatter_knn_vs_cyc_with_meas(
            df, "knn_min",
            os.path.join(args.outdir, f"scatter_knnmin_vs_cyc{ext}"),
            xlabel="kNN distance to proxy Y_train_norm (z-space)",
            good=args.cyc_good, warn=args.cyc_warn, knn_thresh=args.knn_thresh,
            figsize=args.figsize, savefmt=args.savefmt,
            grid=args.grid, grid_alpha=args.grid_alpha
        )

    if "knn_mean_k" in have and "cyc_sim_ps" in have:
        scatter_knn_vs_cyc_with_meas(
            df, "knn_mean_k",
            os.path.join(args.outdir, f"scatter_knnmeank_vs_cyc{ext}"),
            xlabel="mean k-NN distance (k)",
            good=args.cyc_good, warn=args.cyc_warn, knn_thresh=args.knn_thresh,
            figsize=args.figsize, savefmt=args.savefmt,
            grid=args.grid, grid_alpha=args.grid_alpha
        )

    # Scatter: kNN vs prior_bnd_ps
    if "knn_min" in have and "prior_bnd_ps" in have:
        scatter_knn_vs_prior_with_meas(
            df, "knn_min",
            os.path.join(args.outdir, f"scatter_knnmin_vs_priorbnd{ext}"),
            xlabel="kNN distance to proxy Y_train_norm (z-space)",
            knn_thresh=args.knn_thresh,
            figsize=args.figsize, savefmt=args.savefmt,
            grid=args.grid, grid_alpha=args.grid_alpha
        )

    # Scatter: cyc_sim_ps vs jac_sigma_max
    if "cyc_sim_ps" in have and "jac_sigma_max" in have:
        scatter_xy(
            df, "cyc_sim_ps", "jac_sigma_max",
            os.path.join(args.outdir, f"scatter_cyc_sim_ps_vs_jac_sigma_max{ext}"),
            xlabel="cyc_sim_ps",
            ylabel="jac_sigma_max per-sample",
            title="cyc_sim_ps vs jac_sigma_max",
            figsize=args.figsize, savefmt=args.savefmt,
            grid=args.grid, grid_alpha=args.grid_alpha
        )

    # Histograms
    if "jac_sigma_max" in have:
        hist_1d(
            df["jac_sigma_max"],
            os.path.join(args.outdir, f"hist_jac_sigma_max{ext}"),
            bins=60, xlabel="Jacobian spectral norm", title="Jacobian spectral norm (proxy g)",
            figsize=args.figsize, savefmt=args.savefmt,
            grid=args.grid, grid_alpha=args.grid_alpha
        )

    if "prior_bnd_ps" in have:
        hist_1d(
            df["prior_bnd_ps"],
            os.path.join(args.outdir, f"hist_prior_bnd_ps{ext}"),
            bins=60, xlabel="prior_bnd per-sample", title="prior_bnd per-sample",
            figsize=args.figsize, savefmt=args.savefmt,
            grid=args.grid, grid_alpha=args.grid_alpha
        )

    if "cyc_sim_ps" in have:
        hist_1d(
            df["cyc_sim_ps"],
            os.path.join(args.outdir, f"hist_cyc_sim_ps{ext}"),
            bins=60, xlabel="cyc_sim per-sample", title="cyc_sim per-sample",
            figsize=args.figsize, savefmt=args.savefmt,
            grid=args.grid, grid_alpha=args.grid_alpha
        )

    if "proxy_floor_ps" in have:
        hist_1d(
            df["proxy_floor_ps"],
            os.path.join(args.outdir, f"hist_proxy_floor_ps{ext}"),
            bins=60, xlabel="proxy_floor per-sample", title="proxy_floor per-sample",
            figsize=args.figsize, savefmt=args.savefmt,
            grid=args.grid, grid_alpha=args.grid_alpha
        )

    # Worst samples tables
    if "cyc_sim_ps" in have:
        df.sort_values("cyc_sim_ps", ascending=False).head(args.topn) \
            .to_csv(os.path.join(args.outdir, "bad_samples_top_by_cyc.csv"), index=False)

    if "prior_bnd_ps" in have:
        df.sort_values("prior_bnd_ps", ascending=False).head(args.topn) \
            .to_csv(os.path.join(args.outdir, "bad_samples_top_by_priorbnd.csv"), index=False)

    # OOD by kNN threshold
    if "knn_min" in have:
        df[df["knn_min"] >= args.knn_thresh] \
            .to_csv(os.path.join(args.outdir, "samples_ood_by_knn.csv"), index=False)

    print(f"[plot_diag] Wrote plots to: {args.outdir} (format={args.savefmt}, figsize={args.figsize}, grid={args.grid})")


if __name__ == "__main__":
    main()
