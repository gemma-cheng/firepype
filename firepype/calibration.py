# firepype/calibration.py
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import numpy.linalg as npl
from numpy.polynomial import chebyshev as cheb
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment

from .utils import cheb_design_matrix, robust_weights


def find_arc_peaks_1d(
    arc1d: np.ndarray,
    *,
    min_prom_frac: float = 0.008,
    sigma_lo: float = 15.0,
    sigma_hi: float = 0.8,
    distance: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Purpose:
        Detect positive and negative peaks in 1D arc profile by removing broad
        baseline with a Gaussian low-pass filter, lightly smoothing residual
        (high-pass), and using adaptive prominence threshold
    Inputs:
        arc1d: 1D array-like arc signal
        min_prom_frac: Minimum prominence as fraction of robust range (default 0.008)
        sigma_lo: Sigma of low-pass Gaussian for baseline (default 15.0)
        sigma_hi: Sigma of Gaussian smoothing for residual (default 0.8)
        distance: Minimum pixel separation between peaks (default 3)
    Returns:
        tuple:
            - pk (np.ndarray): Indices of candidate peaks (merged pos/neg), with
              edge-trimming to avoid boundary artifacts
            - sm (np.ndarray): High-pass smoothed profile used for detection
    """

    y = np.asarray(arc1d, float)
    n = y.size
    base = gaussian_filter1d(y, sigma=sigma_lo, mode="nearest")
    sm = gaussian_filter1d(y - base, sigma=sigma_hi, mode="nearest")
    p1, p99 = np.percentile(sm, [1, 99])
    mad = np.median(np.abs(sm - np.median(sm))) + 1e-12
    noise = 1.4826 * mad
    prom = max(1.5 * noise, min_prom_frac * (p99 - p1))
    pk_pos, _ = find_peaks(sm, prominence=float(max(prom, 1e-6)), distance=distance)
    pk_neg, _ = find_peaks(-sm, prominence=float(max(prom, 1e-6)), distance=distance)
    pk = np.unique(np.r_[pk_pos, pk_neg])
    pk = pk[(pk > 3) & (pk < n - 3)]

    return pk.astype(int), sm


def match_peaks_to_refs(
    px_peaks: Iterable[int],
    ref_lines_um: np.ndarray,
    wl_lo: float,
    wl_hi: float,
    *,
    max_sep: float = 0.012,
    deg_seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Purpose:
        Coarsely match detected pixel peaks to reference wavelength list within
        given wavelength range by:
          - Seeding linear pixel-to-wavelength mapping
          - Building sparse cost matrix with candidate matches near seed
          - Solving aassignment problem and pruning outliers via light fit
    Inputs:
        px_peaks: Iterable of detected peak pixel indices
        ref_lines_um: 1D array of reference line wavelengths in microns
        wl_lo: Lower wavelength bound (microns)
        wl_hi: Upper wavelength bound (microns)
        max_sep: Maximum allowed separation (microns) for accepted matches (default 0.012)
        deg_seed: Degree of Chebyshev fit used only for outlier pruning (default 1)
    Returns:
        tuple:
            - px_m (np.ndarray): Matched pixel indices (int), sorted and monotonic in wavelength
            - wl_m (np.ndarray): Corresponding matched wavelengths (float)
            Empty arrays are returned if insufficient matches found
    """

    px = np.asarray(px_peaks, float)
    refs = np.asarray(ref_lines_um, float)
    refs = refs[(refs >= wl_lo) & (refs <= wl_hi)]

    if px.size < 8 or refs.size < 8:
        return np.array([], int), np.array([], float)

    x = (px - px.min()) / max(px.max() - px.min(), 1.0)
    wl_seed = wl_lo + x * (wl_hi - wl_lo)

    rows, cols, costs = [], [], []

    for i, wl_s in enumerate(wl_seed):
        j0 = np.searchsorted(refs, wl_s)
        for j in (j0 - 4, j0 - 3, j0 - 2, j0 - 1, j0, j0 + 1, j0 + 2, j0 + 3, j0 + 4):
            if 0 <= j < refs.size:
                d = abs(refs[j] - wl_s)
                if d <= 3 * max_sep:
                    rows.append(i)
                    cols.append(j)
                    costs.append(d)
    if not rows:
        return np.array([], int), np.array([], float)

    C = np.full((px.size, refs.size), 1e3, float)
    C[rows, cols] = costs
    r_idx, c_idx = linear_sum_assignment(C)
    ok = C[r_idx, c_idx] <= max_sep
    r_idx = r_idx[ok]
    c_idx = c_idx[ok]

    if r_idx.size < 6:
        return np.array([], int), np.array([], float)

    px_m = px[r_idx].astype(int)
    wl_m = refs[c_idx].astype(float)
    order = np.argsort(px_m)
    px_m = px_m[order]
    wl_m = wl_m[order]
    good = np.concatenate(([True], np.diff(wl_m) > 0))
    px_m = px_m[good]
    wl_m = wl_m[good]

    # Light sanity fit (not used directly; only for outlier pruning)
    n = int(px.max()) + 2
    x_full = np.linspace(-1.0, 1.0, n)
    x_m = np.interp(px_m, np.arange(x_full.size), x_full)
    coef = cheb.chebfit(x_m, wl_m, deg=deg_seed)
    wl_fit = cheb.chebval(x_m, coef)
    res = wl_m - wl_fit
    s = 1.4826 * np.median(np.abs(res - np.median(res)) + 1e-12)
    keep = np.abs(res) <= max(2.5 * s, max_sep)

    return px_m[keep], wl_m[keep]


def solve_dispersion_from_arc1d(
    arc1d: np.ndarray,
    *,
    wl_range: tuple[float, float],
    ref_lines_um: np.ndarray,
    deg: int = 3,
    anchors: Sequence[tuple[int, float]] | None = None,
    max_sep: float = 0.012,
    verbose: bool = True,
    anchor_weight: float = 12.0,
    enforce_anchor_order: bool = False,
) -> np.ndarray:
    """
    Purpose:
        Fit robust Chebyshev pixel --> wavelength dispersion for 1D arc:
          - Detect peaks and match to reference wavelengths
          - Perform robust IRLS Chebyshev fit (degree deg)
          - Align endpoints to given wl_range
          - Optionally add weighted anchor "pseudo-observations" and enforce
            anchor ordering with light smoothing
    Inputs:
        arc1d: 1D arc spectrum
        wl_range: Tuple (wl_lo, wl_hi) microns for target coverage
        ref_lines_um: 1D array of reference wavelengths (microns)
        deg: Polynomial degree for Chebyshev fit (default 3)
        anchors: Optional list of (pixel_index, wavelength_um) anchors (default None)
        max_sep: Max separation (microns) for matching peaks to refs (default 0.012)
        verbose: Print anchor RMS diagnostics (default True)
        anchor_weight: Multiplicative weight for anchor observations in IRLS (default 12.0)
        enforce_anchor_order: Enforce monotonic constraints at anchors and
                              apply light smoothing (default False)
    Returns:
        np.ndarray:
            Wavelength array (microns) of same length as arc1d, strictly increasing
            and clipped/shifted to the range wl_range
    """

    y = np.asarray(arc1d, float)
    n = y.size
    wl_lo, wl_hi = float(wl_range[0]), float(wl_range[1])

    # Peaks and matching
    pk, _ = find_arc_peaks_1d(y)

    if pk.size < 8:
        raise RuntimeError("Too few arc peaks found for dispersion fit")

    px_m, wl_m = match_peaks_to_refs(
        pk, ref_lines_um, wl_lo, wl_hi, max_sep=max_sep, deg_seed=1
    )

    if px_m.size < max(8, deg + 3):
        px_m, wl_m = match_peaks_to_refs(
            pk, ref_lines_um, wl_lo, wl_hi, max_sep=1.3 * max_sep, deg_seed=1
        )

    if px_m.size < max(8, deg + 3):
        raise RuntimeError("Insufficient matched lines for dispersion fit")

    # Map pixels to Chebyshev domain
    x_full = np.linspace(-1.0, 1.0, n)
    x_m = np.interp(px_m, np.arange(x_full.size), x_full)

    # Build fit arrays and base weights
    x_fit = x_m.copy()
    y_fit = wl_m.astype(float).copy()
    w0 = np.ones_like(y_fit)

    # Add optional anchors as weighted pseudo-observations
    if anchors:
        for p, w in anchors:
            p_i = int(p)
            w_v = float(w)
            if 0 <= p_i < n and wl_lo <= w_v <= wl_hi:
                x_fit = np.r_[x_fit, np.interp(p_i, np.arange(x_full.size), x_full)]
                y_fit = np.r_[y_fit, w_v]
                w0 = np.r_[w0, float(anchor_weight)]

    # Robust IRLS fit
    X = cheb_design_matrix(x_fit, deg)
    w = np.ones_like(y_fit)

    for _ in range(12):
        w_tot = w * w0
        WX = X * w_tot[:, None]
        Wy = y_fit * w_tot
        coef, *_ = npl.lstsq(WX, Wy, rcond=None)
        res = y_fit - X.dot(coef)
        w_new = robust_weights(res, c=4.685)
        if np.allclose(w, w_new, atol=1e-3):
            break
        w = w_new

    wl = cheb.chebval(x_full, coef)

    # Endpoint alignment to wl_range
    span_fit = wl[-1] - wl[0]
    span_tar = wl_hi - wl_lo

    if abs(span_fit - span_tar) / max(span_tar, 1e-12) > 0.002:
        a = span_tar / (span_fit + 1e-12)
        b = wl_lo - a * wl[0]
        wl = a * wl + b
    else:
        wl = wl + (wl_lo - wl[0])

    # Enforce monotonicity
    for i in range(1, n):
        if wl[i] <= wl[i - 1]:
            wl[i] = wl[i - 1] + 1e-9
    wl[0], wl[-1] = max(wl[0], wl_lo), min(wl[-1], wl_hi)

    # Optional anchor order enforcement and light smoothing
    if enforce_anchor_order and anchors:
        anc = [
            (int(p), float(w))
            for (p, w) in anchors
            if 0 <= int(p) < n and wl_lo <= float(w) <= wl_hi
        ]
        anc.sort(key=lambda t: t[0])

        for p_a, w_a in anc:
            wl[: p_a + 1] = np.minimum(wl[: p_a + 1], w_a)

            for i in range(1, p_a + 1):
                if wl[i] <= wl[i - 1]:
                    wl[i] = wl[i - 1] + 1e-9

        for p_a, w_a in reversed(anc):
            wl[p_a:] = np.maximum(wl[p_a:], w_a)

            for i in range(p_a + 1, n):
                if wl[i] <= wl[i - 1]:
                    wl[i] = wl[i - 1] + 1e-9

        wl = np.clip(wl, wl_lo, wl_hi)
        wl = gaussian_filter1d(wl, sigma=1.2, mode="nearest")

        for i in range(1, n):
            if wl[i] <= wl[i - 1]:
                wl[i] = wl[i - 1] + 1e-9
        wl[0], wl[-1] = max(wl[0], wl_lo), min(wl[-1], wl_hi)

    if verbose and anchors:
        diffs = []

        for p, w_anchor in anchors:
            p_i = int(p)
            w_v = float(w_anchor)
            if 0 <= p_i < n and wl_lo <= w_v <= wl_hi:
                diffs.append(wl[p_i] - w_v)

        if diffs:
            rms_nm = np.sqrt(np.mean(np.square(diffs))) * 1e3
            mx_nm = np.max(np.abs(diffs)) * 1e3
            print(
                f"[DISPERSION] anchors: RMS={rms_nm:.2f} nm, "
                f"max|Δ|={mx_nm:.2f} nm over {len(diffs)} anchors"
            )

    return wl


def global_wavecal_from_arc1d(
    arc1d: np.ndarray,
    wl_range: tuple[float, float],
    ref_lines_um: np.ndarray,
    anchors_global: Sequence[tuple[int, float]] | None = None,
    *,
    tol_init: float = 0.020,
    tol_refine: float = 0.015,
    verbose: bool = True,
) -> np.ndarray:
    """
    Purpose:
        Compatibility wrapper around solve_dispersion_from_arc1d, solving a
        degree-3 robust dispersion with optional global anchors
    Inputs:
        arc1d: 1D arc spectrum
        wl_range: Tuple (wl_lo, wl_hi) microns for target coverage
        ref_lines_um: 1D array of reference wavelengths (microns)
        anchors_global: Optional list of (pixel_index, wavelength_um) anchors
        verbose: Print diagnostics (default True)
    Returns:
        np.ndarray:
            Wavelength array (microns) corresponding to each pixel in arc1d
    """

    return solve_dispersion_from_arc1d(
        arc1d,
        wl_range=wl_range,
        ref_lines_um=np.asarray(ref_lines_um, float),
        deg=3,
        anchors=anchors_global,
        max_sep=0.012,
        verbose=verbose,
    )


def average_wavecal_across_cols(
    arc_img: np.ndarray,
    center_col: int,
    *,
    half: int,
    ref_lines_um: np.ndarray,
    wl_range: tuple[float, float] = None,
    anchors: Sequence[tuple[int, float]] | None = None,
    deg: int = 3,
    max_sep: float = 0.012,
) -> np.ndarray:
    """
    Purpose:
        Compute averaged wavelength-per-pixel solution across small
        footprint of columns centered on center_col. Each column is extracted
        with local background, solved independently, then averaged and aligned
        to the given wl_range
    Inputs:
        arc_img: 2D arc image array (rows x cols)
        center_col: Central column index around which to average solutions
        half: Use columns in [center_col - half, center_col + half]
        ref_lines_um: 1D array of reference wavelengths (microns)
        wl_range: Tuple (wl_lo, wl_hi) microns for target coverage (required)
        anchors: Optional anchor list passed to the solver (default None)
        deg: Chebyshev degree for per-column dispersion fits (default 3)
        max_sep: Max separation for peak-ref matching (default 0.012)
    Returns:
        np.ndarray:
            Averaged wavelength solution (microns) on the native pixel grid,
            strictly increasing and clipped to wl_range
    Raises:
        ValueError: If wl_range or ref_lines_um are missing/empty
    """

    if wl_range is None:
        raise ValueError("average_wavecal_across_cols: wl_range must be provided")

    if ref_lines_um is None or len(ref_lines_um) == 0:
        raise ValueError("average_wavecal_across_cols: ref_lines_um must be provided")

    ncols = arc_img.shape[1]
    cols = [center_col + dc for dc in range(-half, half + 1)]
    cols = [c for c in cols if 0 <= c < ncols]
    wl_list = []

    for c in cols:
        a1d = extract_with_local_bg_simple(arc_img, c)
        wl_c = solve_dispersion_from_arc1d(
            a1d,
            wl_range=wl_range,
            ref_lines_um=np.asarray(ref_lines_um, float),
            deg=deg,
            anchors=anchors,
            max_sep=max_sep,
            verbose=False,
        )

        if wl_c[0] > wl_c[-1]:
            wl_c = wl_c[::-1]
        wl_list.append(wl_c)

    wl_avg = np.nanmean(np.vstack(wl_list), axis=0)

    if wl_avg[0] > wl_avg[-1]:
        wl_avg = wl_avg[::-1]

    wl_min, wl_max = wl_range
    span_fit = wl_avg[-1] - wl_avg[0]
    span_tar = wl_max - wl_min

    if abs(span_fit - span_tar) / max(span_tar, 1e-12) > 0.002:
        a = span_tar / (span_fit + 1e-12)
        b = wl_min - a * wl_avg[0]
        wl_avg = a * wl_avg + b

    else:
        wl_avg = wl_avg + (wl_min - wl_avg[0])
    wl_avg = np.clip(wl_avg, wl_min, wl_max)

    return wl_avg


def extract_with_local_bg_simple(
    img: np.ndarray,
    center_col: int,
    *,
    ap: int = 5,
    bg_in: int = 8,
    bg_out: int = 18,
) -> np.ndarray:
    """
    Purpose:
        Minimal local background subtraction used by average_wavecal_across_cols:
        median-collapses small aperture around center_col and subtracts
        background estimated from side bands
    Inputs:
        img: 2D image array (rows, cols)
        center_col: Central column index for extraction
        ap: Half-width of extraction aperture in columns (default 5)
        bg_in: Inner offset for background windows (default 8)
        bg_out: Outer offset for background windows (default 18)
    Returns:
        np.ndarray:
            Extracted per-row 1D profile after local background subtraction
    """

    nrows, ncols = img.shape
    lo = max(0, center_col - ap)
    hi = min(ncols, center_col + ap + 1)
    bg_left = img[:, max(0, center_col - bg_out) : max(0, center_col - bg_in)]
    bg_right = img[:, min(ncols, center_col + bg_in) : min(ncols, center_col + bg_out)]

    if bg_left.size == 0 and bg_right.size == 0:
        bg = np.zeros(nrows, dtype=float)

    else:
        bg = np.median(
            np.concatenate([bg_left, bg_right], axis=1), axis=1
        )

    spec = np.median(img[:, lo:hi], axis=1) - bg

    return spec


# -------------------- Interp-edge mask (for coaddition step) --------------------
def mask_interp_edge_artifacts(
    grid_wl: np.ndarray,
    wl_native: np.ndarray,
    f_native: np.ndarray,
    err_native: np.ndarray,
    *,
    min_span_px: int = 10,
    pad_bins: int = 8,
    min_keep_bins: int = 24,
) -> np.ndarray:
    """
    Purpose:
        Create a boolean mask on the coadd grid to exclude bins near interpolation
        edges, keeping only interior segments where the native spectrum spans a
        sufficiently long contiguous region. This reduces spurious edge artifacts
    Inputs:
        grid_wl: 1D target wavelength grid for coaddition
        wl_native: 1D native wavelengths where the spectrum is defined
        f_native: 1D native flux (unused for masking, kept for API symmetry)
        err_native: 1D native errors (unused for masking, kept for API symmetry)
        min_span_px: Minimum contiguous native span (pixels) to consider (default 10)
        pad_bins: Bins to exclude from each side of a contiguous span (default 8)
        min_keep_bins: Minimum interior bins to retain after padding (default 24)
    Returns:
        np.ndarray:
            Boolean mask on grid_wl where True indicates bins to keep
            False near edges or outside the native coverage
    """
    grid_wl = np.asarray(grid_wl, float)
    wl_native = np.asarray(wl_native, float)
    if wl_native.size < max(3, min_span_px) or not np.any(np.isfinite(wl_native)):
        return np.zeros_like(grid_wl, dtype=bool)
    wmin = np.nanmin(wl_native)
    wmax = np.nanmax(wl_native)
    inside = np.isfinite(grid_wl) & (grid_wl > wmin) & (grid_wl < wmax)

    keep = np.zeros_like(inside, dtype=bool)
    n = inside.size
    i = 0
    while i < n:
        if not inside[i]:
            i += 1
            continue
        j = i
        while j < n and inside[j]:
            j += 1
        ii = i + pad_bins
        jj = j - pad_bins
        if jj - ii >= min_keep_bins:
            keep[ii:jj] = True
        i = j
    return keep
