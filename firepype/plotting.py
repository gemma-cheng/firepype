# firepype/plotting.py
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from .calibration import (
    find_arc_peaks_1d,
    solve_dispersion_from_arc1d,
)
from .utils import ensure_dir

def plot_arc_trace_on_raw(
    arc_img: np.ndarray,
    center_col: int,
    *,
    ap: int = 5,
    bg_in: int = 8,
    bg_out: int = 18,
    half: int = 1,
    row_frac: Tuple[float, float] = (0.35, 0.85),
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """
    Purpose:
        Overlay the extraction geometry on raw arc image:
        - Display selected central column and its neighbours
        - Shade extraction aperture and background side-bands
        - Mark row band for median spatial profiles
    Inputs:
        arc_img: 2D array (rows x cols) of arc image
        center_col: Central column index to draw aperture/background
        ap: Half-width of extraction aperture in columns (default 5)
        bg_in: Inner offset (columns) from center to background band (default 8)
        bg_out: Outer offset (columns) from center to background band (default 18)
        half: Include columns in [center_col - half, center_col + half] (default 1)
        row_frac: Fractional row range for band visualisation (default (0.35, 0.85))
        title: Optional plot title (default None -> auto-generated)
        save_path: Optional path to save plot (default None: no save)
        show: Display figure (default False)
    Returns:
        None
        Saves and/or shows figure of extraction geometry
    """

    img = np.asarray(arc_img, float)
    nrows, ncols = img.shape
    r0 = int(min(row_frac) * nrows)
    r1 = int(max(row_frac) * nrows)

    cols = [center_col + dc for dc in range(-half, half + 1)]
    cols = [c for c in cols if 0 <= c < ncols] or [min(max(center_col, 0), ncols - 1)]

    fig, ax = plt.subplots(figsize=(7, 6))
    vmin, vmax = np.nanpercentile(img, [5, 99])
    ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")

    # Row band
    ax.hlines([r0, r1 - 1], 0, ncols - 1, colors="cyan", linestyles="--", lw=1.1,
        alpha=0.7)

    # Aperture blocks
    for c in cols:
        lo = max(0, c - ap)
        hi = min(ncols - 1, c + ap)
        ax.axvspan(lo, hi, color="lime", alpha=0.18)
    for c in cols:
        ax.axvline(c, color="lime", lw=0.9, alpha=0.75)

    # Background bands around the central column
    c0 = min(max(center_col, 0), ncols - 1)
    ax.axvspan(max(0, c0 - bg_out), max(0, c0 - bg_in), color="orange", alpha=0.18)
    ax.axvspan(min(ncols - 1, c0 + bg_in), min(ncols - 1, c0 + bg_out),
        color="orange", alpha=0.18)

    ax.set_xlim(0, ncols - 1)
    ax.set_ylim(0, nrows - 1)
    ax.set_xlabel("Column (dispersion)")
    ax.set_ylabel("Row (spatial)")
    ax.set_title(title or f"ARC trace overlay (center_col={center_col}, half={half}, ap={ap})")
    fig.tight_layout()

    if save_path:
        ensure_dir(Path(save_path).parent.as_posix())
        fig.savefig(save_path, dpi=140)
    if show:
        plt.show()
    plt.close(fig)


def plot_arc_1d_with_line_labels(
    arc1d: np.ndarray,
    wl_range: Tuple[float, float],
    ref_lines_um: np.ndarray,
    *,
    anchors: Sequence[tuple[int, float]] | None = None,
    solver_deg: int = 3,
    solver_max_sep: float = 0.012,
    arc_col: int | None = None,
    title_tag: str = "",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """
    Purpose:
        Plot high-pass filtered 1D arc profile with labeled wavelengths of
        detected arc features. Dispersion solution is solved using the provided
        line list and optional anchors, then used to annotate peaks
    Inputs:
        arc1d: 1D arc signal (array-like)
        wl_range: Target wavelength span for solution in microns (wl_lo, wl_hi)
        ref_lines_um: Reference line wavelengths in microns
        anchors: Optional list of (pixel_index, wavelength_um) anchor pairs to guide
                 dispersion solution (default None)
        solver_deg: Polynomial degree for dispersion fit (default 3)
        solver_max_sep: Maximum separation (microns) for matching lines (default 0.012)
        arc_col: Optional column index associated with 1D extraction, for title
        title_tag: Extra text appended to the title (default "")
        save_path: Optional path to save the plot (default None: no save)
        show: Display figure (default False)
    Returns:
        None
        Saves and/or shows figure with 1D arc profile and labeled peaks
    """

    y = np.asarray(arc1d, float)
    n = y.size

    base = gaussian_filter1d(y, sigma=15, mode="nearest")
    sm = gaussian_filter1d(y - base, sigma=0.8, mode="nearest")
    xpix = np.arange(n)

    # Peaks for visualisation
    pk, _ = find_arc_peaks_1d(y, sigma_lo=15, sigma_hi=0.8)

    # Solve dispersion for labelling
    wl_sol = solve_dispersion_from_arc1d(
        y,
        wl_range=wl_range,
        ref_lines_um=np.asarray(ref_lines_um, float),
        deg=solver_deg,
        anchors=anchors,
        max_sep=solver_max_sep,
        verbose=False,
    )

    # Map matched peaks into wavelengths for labels
    yvals = np.interp(pk, xpix, sm) if pk.size > 0 else np.array([], float)
    wl_m = wl_sol[pk] if pk.size > 0 else np.array([], float)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xpix, sm, color="0.2", lw=0.9, label="Arc 1D (smoothed)")

    if pk.size > 0:
        ax.vlines(pk, ymin=np.nanmin(sm), ymax=np.nanmax(sm) * 0.85, colors="C3",
            linestyles=":", alpha=0.6)
        ax.scatter(pk, yvals, s=20, color="C1", zorder=3, label="Peaks")
        dy = 0.05 * (np.nanmax(sm) - np.nanmin(sm))
        for p, w, yy in zip(pk, wl_m, yvals):
            ax.text(p, yy + dy, f"{w:.4f}", color="C1", fontsize=8, rotation=90,
                ha="center", va="bottom")

    ttl = f"Extracted ARC 1D (col={arc_col if arc_col is not None else 'n/a'})"
    if title_tag:
        ttl += f" — {title_tag}"
    ax.set_title(ttl)
    ax.set_xlabel("Row (pixel)")
    ax.set_ylabel("Arc counts (arb.)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    if save_path:
        ensure_dir(Path(save_path).parent.as_posix())
        fig.savefig(save_path, dpi=140)
    if show:
        plt.show()
    plt.close(fig)


def plot_1d_spectrum(
    wl_um: np.ndarray,
    flux: np.ndarray,
    title: str,
    save_path: str | Path | None,
    *,
    xlabel: str = "Wavelength (um)",
    ylabel: str = "Flux (arb)",
    show: bool = False,
):
    """
    Purpose:
        Plot simple 1D spectrum, with optional save/show
    Inputs:
        wl_um: 1D wavelengths in microns
        flux: 1D flux values aligned with wl_um
        title: Plot title string
        save_path: Optional path to save plot (default None: no save)
        xlabel: X-axis label (default "Wavelength (um)")
        ylabel: Y-axis label (default "Flux (arb)")
        show: Display figure (default False)
    Returns:
        None
        Saves and/or shows the spectrum plot
    """

    wl = np.asarray(wl_um, float)
    fx = np.asarray(flux, float)
    m = np.isfinite(wl) & np.isfinite(fx)
    if np.count_nonzero(m) < 5:
        return
    wl = wl[m]
    fx = fx[m]
    idx = np.argsort(wl)
    wl = wl[idx]
    fx = fx[idx]
    plt.figure(figsize=(7, 5))
    plt.plot(wl, fx, lw=1.0, color="C3")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        ensure_dir(Path(save_path).parent.as_posix())
        plt.savefig(save_path, dpi=140)
    if show:
        plt.show()
    plt.close()
