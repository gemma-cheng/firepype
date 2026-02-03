# firepype/extraction.py
from __future__ import annotations

from typing import Tuple

import numpy as np


def extract_with_local_bg(
    img: np.ndarray,
    center_col: int,
    *,
    ap: int = 5,
    bg_in: int = 8,
    bg_out: int = 18,
) -> np.ndarray:
    """
    Purpose:
        Extract 1D spectrum by median-collapsing a small aperture around
        center_col and subtracting local background estimated from side bands
    Inputs:
        img: 2D image array with shape (rows, cols)
        center_col: Central column index of object trace
        ap: Half-width (in columns) of extraction aperture (default 5)
        bg_in: Inner offset (in columns) from center to background band (default 8)
        bg_out: Outer offset (in columns) from center to background band (default 18)
    Returns:
        np.ndarray:
            Extracted 1D spectrum (length = number of rows), as float
    """

    nrows, ncols = img.shape
    lo = max(0, center_col - ap)
    hi = min(ncols, center_col + ap + 1)
    bg_left = img[:, max(0, center_col - bg_out) : max(0, center_col - bg_in)]
    bg_right = img[:, min(ncols, center_col + bg_in) : min(ncols, center_col + bg_out)]

    if bg_left.size == 0 and bg_right.size == 0:
        bg = np.zeros(nrows, dtype=img.dtype)
    else:
        if bg_left.size and bg_right.size:
            bg_all = np.concatenate([bg_left, bg_right], axis=1)
        else:
            bg_all = bg_left if bg_left.size else bg_right
        bg = np.median(bg_all, axis=1)
    spec = np.median(img[:, lo:hi], axis=1) - bg

    return spec.astype(float)


def extract_cols_median(
    img: np.ndarray,
    center_col: int,
    *,
    half: int = 1,
    ap: int = 5,
    bg_in: int = 8,
    bg_out: int = 18,
) -> np.ndarray:
    """
    Purpose:
        Extract spectra from multiple adjacent columns around center_col using
        extract_with_local_bg, then median-combine spectra
    Inputs:
        img: 2D image array (rows, cols)
        center_col: Central column index
        half: Include columns in [center_col - half, center_col + half] (default 1)
        ap: Half-width of extraction aperture in columns (default 5)
        bg_in: Inner offset for background window (default 8)
        bg_out: Outer offset for background window (default 18)
    Returns:
        np.ndarray:
            Median of stacked extractions across selected columns (1D array)
    """

    ncols = img.shape[1]
    cols = [center_col + dc for dc in range(-half, half + 1)]
    cols = [c for c in cols if 0 <= c < ncols]
    stacks = [
        extract_with_local_bg(img, c, ap=ap, bg_in=bg_in, bg_out=bg_out) for c in cols
    ]

    return np.median(np.vstack(stacks), axis=0).astype(float)


def extract_cols_median_with_err(
    img: np.ndarray,
    center_col: int,
    *,
    half: int = 1,
    ap: int = 5,
    bg_in: int = 8,
    bg_out: int = 18,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Purpose:
        Extract median-combined 1D spectrum across a footprint centered at
        center_col with local background subtraction, and estimate per-row
        1-sigma errors from background MAD propagated to the median
    Inputs:
        img: 2D image array (rows, cols)
        center_col: Central column index
        half: Include columns in [center_col - half, center_col + half] (default 1)
        ap: Half-width of extraction aperture in columns (default 5)
        bg_in: Inner offset for background window (default 8)
        bg_out: Outer offset for background window (default 18)
    Returns:
        tuple:
            - flux_1d (np.ndarray): Extracted 1D flux versus row
            - sigma_1d (np.ndarray): Estimated per-row 1-sigma uncertainties
    """

    nrows, ncols = img.shape
    cols = [center_col + dc for dc in range(-half, half + 1)]
    cols = [c for c in cols if 0 <= c < ncols]

    specs = []
    sigmas = []

    for c in cols:
        lo = max(0, c - ap)
        hi = min(ncols, c + ap + 1)

        bg_left = img[:, max(0, c - bg_out) : max(0, c - bg_in)]
        bg_right = img[:, min(ncols, c + bg_in) : min(ncols, c + bg_out)]

        if bg_left.size == 0 and bg_right.size == 0:
            bg_med = np.zeros(nrows, dtype=float)
            bg_std = np.zeros(nrows, dtype=float)
        else:
            if bg_left.size and bg_right.size:
                bg_all = np.concatenate([bg_left, bg_right], axis=1)
            else:
                bg_all = bg_left if bg_left.size else bg_right
            bg_med = np.median(bg_all, axis=1).astype(float)
            mad = np.median(np.abs(bg_all - bg_med[:, None]), axis=1) + 1e-12
            bg_std = 1.4826 * mad

        sub = img[:, lo:hi] - bg_med[:, None]
        spec = np.median(sub, axis=1).astype(float)
        specs.append(spec)

        n_eff = max(1, hi - lo)
        sigma_row = np.sqrt(np.pi / 2.0) * bg_std / np.sqrt(n_eff)
        sigmas.append(sigma_row)

    spec_stack = np.vstack(specs)
    sigma_stack = np.vstack(sigmas)
    flux_1d = np.median(spec_stack, axis=0)
    M = max(1, len(cols))
    # Approximate variance of the median of M Gaussians by scaling
    sigma_1d = np.sqrt(np.pi / 2.0) * np.median(sigma_stack**2, axis=0) ** 0.5 / np.sqrt(
        M
    )

    return flux_1d.astype(float), sigma_1d.astype(float)


def estimate_negative_scale(
    img: np.ndarray,
    pos_col: int,
    neg_col: int,
    *,
    ap: int = 5,
    row_bg_frac: Tuple[float, float] = (0.0, 0.2),
    row_bg_frac_hi: Tuple[float, float] = (0.8, 1.0),
) -> float:
    """
    Purpose:
        Provide simple estimator for the negative-beam scale g using outer-row
        bands as sky. Primarily for reference/tests - the main pipeline should
        prefer detection.estimate_negative_scale_robust
    Inputs:
        img: 2D image array (rows, cols)
        pos_col: Column index of positive object trace
        neg_col: Column index of negative object trace
        ap: Half-width of extraction aperture in columns (default 5)
        row_bg_frac: Fractional row range for lower background band (default (0.0, 0.2))
        row_bg_frac_hi: Fractional row range for upper background band (default (0.8, 1.0))
    Returns:
        float:
            Estimated negative beam scale g
    """

    nrows = img.shape[0]
    pos = np.median(img[:, max(0, pos_col - ap) : pos_col + ap + 1], axis=1)
    neg = np.median(img[:, max(0, neg_col - ap) : neg_col + ap + 1], axis=1)

    r0a = int(min(row_bg_frac) * nrows)
    r1a = int(max(row_bg_frac) * nrows)
    r0b = int(min(row_bg_frac_hi) * nrows)
    r1b = int(max(row_bg_frac_hi) * nrows)
    rows_bg = np.r_[r0a:r1a, r0b:r1b]

    num = float(np.sum(pos[rows_bg] * neg[rows_bg]))
    den = float(np.sum(neg[rows_bg] ** 2) + 1e-12)

    return num / den
