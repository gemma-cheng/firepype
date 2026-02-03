# firepype/telluric.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy import signal as sps


def _ensure_dir(p: str | Path):
    """
    Purpose:
        Ensure that directory exists (create parents if needed)
    Inputs:
        p: Directory path to create
    Returns:
        None
    """

    Path(p).mkdir(parents=True, exist_ok=True)


def _load_fits_primary(path: str | Path):
    """
    Purpose:
        Load primary HDU data and header from FITS file
    Inputs:
        path: Path to FITS file
    Returns:
        tuple:
            - data (np.ndarray): Primary HDU image data
            - header (fits.Header): FITS header from the primary HDU
    """

    with fits.open(str(path)) as hdul:
        return np.asarray(hdul[0].data), hdul[0].header.copy()


def _load_table_spectrum(path: str | Path):
    """
    Purpose:
        Load spectrum from FITS table with columns wavelength_um, flux, and optional flux_err
    Inputs:
        path: Path to FITS file with binary table in HDU 1
    Returns:
        tuple:
            - wl (np.ndarray): Wavelengths in microns
            - fx (np.ndarray): Flux array
            - err (np.ndarray | None): Flux error array if present, else None
            - header (fits.Header): Primary header
    """

    with fits.open(str(path)) as hdul:
        hdr = hdul[0].header.copy()
        tab = hdul[1].data
        wl = np.asarray(tab["wavelength_um"], float)
        fx = np.asarray(tab["flux"], float)
        err = np.asarray(tab["flux_err"], float) if ("flux_err" in tab.names) else None

    return wl, fx, err, hdr


def _write_spectrum_with_err(
    path: str | Path,
    wl_um: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray | None,
    base_header: fits.Header | None,
    history: list[str] | None = None,
):
    """
    Purpose:
        Write spectrum (and optional errors) to 2-HDU FITS file:
        - Primary HDU with (optional) header and history
        - Binary table HDU with wavelength_um, flux, and optional flux_err
    Inputs:
        path: Output FITS path
        wl_um: 1D array of wavelengths in microns
        flux: 1D array of flux values
        err: 1D array of flux errors or None
        base_header: FITS header to attach to the primary HDU (optional)
        history: List of strings appended as HISTORY cards (optional)
    Returns:
        None
    """

    cols = [
        fits.Column(name="wavelength_um", array=np.asarray(wl_um, float), format="D"),
        fits.Column(name="flux", array=np.asarray(flux, np.float32), format="E"),
    ]

    if err is not None:
        cols.append(
            fits.Column(name="flux_err", array=np.asarray(err, np.float32), format="E")
        )

    hdu_tab = fits.BinTableHDU.from_columns(cols)
    hdu_prim = fits.PrimaryHDU(header=base_header.copy() if base_header else fits.Header())

    if history:
        for h in history:
            hdu_prim.header["HISTORY"] = str(h).encode("ascii", "ignore").decode("ascii")

    hdul = fits.HDUList([hdu_prim, hdu_tab])
    _ensure_dir(Path(path).parent)
    hdul.writeto(str(path), overwrite=True)


def _orient_to_increasing(wl, fx):
    """
    Purpose:
        Ensure wavelengths increase with index; if decreasing, reverse both arrays
    Inputs:
        wl: 1D array-like wavelengths
        fx: 1D array-like flux aligned with wl
    Returns:
        tuple:
            - wl_out (np.ndarray): Wavelength array in increasing order
            - fx_out (np.ndarray): Flux array reoriented to match wl_out
    """

    wl = np.asarray(wl, float)
    fx = np.asarray(fx, float)

    return (wl[::-1], fx[::-1]) if (wl.size >= 2 and wl[0] > wl[-1]) else (wl, fx)


def _assert_monotonic_and_align(wl, fx):
    """
    Purpose:
        Clean and align wavelength and flux arrays:
        - Remove non-finite entries
        - Sort by wavelength
        - Enforce strictly increasing wavelengths (drop ties)
    Inputs:
        wl: 1D array-like wavelengths
        fx: 1D array-like flux aligned with wl
    Returns:
        tuple:
            - wl_out (np.ndarray): Strictly increasing wavelengths
            - fx_out (np.ndarray): Flux values aligned to wl_out
    """

    wl = np.asarray(wl, float)
    fx = np.asarray(fx, float)
    m = np.isfinite(wl) & np.isfinite(fx)
    wl = wl[m]
    fx = fx[m]

    if wl.size == 0:
        return wl, fx

    idx = np.argsort(wl)
    wl = wl[idx]
    fx = fx[idx]

    if wl.size >= 2:
        good = np.concatenate(([True], np.diff(wl) > 0))
        wl = wl[good]
        fx = fx[good]
    return wl, fx


def _plot_1d(wl, fx, title, path_png, xlabel="Wavelength (um)", ylabel="Flux", show=False):
    """
    Purpose:
        Plot 1D spectrum and save to a PNG file
    Inputs:
        wl: 1D array-like wavelengths
        fx: 1D array-like flux
        title: Plot title
        path_png: Output path
        xlabel: X-axis label (default: 'Wavelength (um)')
        ylabel: Y-axis label (default: 'Flux')
        show: Display plot
    Returns:
        None

    """

    wl = np.asarray(wl, float)
    fx = np.asarray(fx, float)
    m = np.isfinite(wl) & np.isfinite(fx)

    if m.sum() < 5:
        return

    wl = wl[m]
    fx = fx[m]

    plt.figure(figsize=(8, 5))
    plt.plot(wl, fx, lw=1.0, color="C3")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _ensure_dir(Path(path_png).parent)
    plt.savefig(path_png, dpi=140)

    if show:
        plt.show()
    plt.close()


def _cheb_design_matrix(x, deg):
    """
    Purpose:
        Build Chebyshev design matrix T_k(x) for k=0..deg on scaled inputs
    Inputs:
        x: 1D array-like input coordinates (not necessarily scaled)
        deg: Non-negative integer polynomial degree
    Returns:
        np.ndarray:
            Matrix of shape (N, deg+1) where column k is T_k(x)
    """

    x = np.asarray(x, float)
    X = np.ones((x.size, deg + 1), float)

    if deg >= 1:
        X[:, 1] = x

    for k in range(2, deg + 1):
        X[:, k] = 2.0 * x * X[:, k - 1] - X[:, k - 2]

    return X


def _robust_weights(res, c=4.685):
    """
    Purpose:
        Compute Tukey's biweight robust regression weights from residuals
    Inputs:
        res: 1D array-like residuals
        c: Tuning constant controlling downweighting (default 4.685)
    Returns:
        np.ndarray:
            Weights in [0,1] of same shape as res
    """

    r = np.asarray(res, float)
    s = np.nanmedian(np.abs(r - np.nanmedian(r))) * 1.4826 + 1e-12
    u = r / (c * s)
    w = (1 - u**2)
    w[(np.abs(u) >= 1) | ~np.isfinite(w)] = 0.0

    return w**2


def _find_arc_peaks_1d(arc1d, min_prom_frac=0.008, sigma_lo=15, sigma_hi=0.8, distance=3):
    """
    Purpose:
        Detect positive and negative peaks in 1D arc profile via baseline removal
        and adaptive prominence thresholding
    Inputs:
        arc1d: 1D array-like arc signal
        min_prom_frac: Minimum prominence as fraction of robust range (default 0.008)
        sigma_lo: Sigma of low-pass Gaussian for baseline (default 15)
        sigma_hi: Sigma for smoothing residual (default 0.8)
        distance: Minimum peak spacing in samples (default 3)
    Returns:
        tuple:
            - pk (np.ndarray): Indices of candidate peaks (merged pos/neg), edge-trimmed
            - sm (np.ndarray): High-pass–smoothed signal used for detection
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


def _match_peaks_to_refs(px_peaks, ref_lines_um, wl_lo, wl_hi, max_sep=0.012):
    """
    Purpose:
        Match detected pixel peaks to reference wavelengths using seeded linear
        mapping and Hungarian assignment with maximum separation constraint
    Inputs:
        px_peaks: 1D array of peak pixel indices
        ref_lines_um: 1D array of reference line wavelengths in microns
        wl_lo: Lower bound of wavelength range (microns)
        wl_hi: Upper bound of wavelength range (microns)
        max_sep: Maximum allowed match separation in microns (default 0.012)
    Returns:
        tuple:
            - px_m (np.ndarray): Matched pixel indices (sorted, strictly increasing in λ)
            - wl_m (np.ndarray): Corresponding matched reference wavelengths
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

    from scipy.optimize import linear_sum_assignment

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

    return px_m[good], wl_m[good]


def _solve_dispersion_from_arc1d(arc1d, wl_range, ref_lines_um, deg=3):
    """
    Purpose:
        Solve wavelength solution from 1D arc spectrum by detecting peaks,
        matching to reference lines, and fitting Chebyshev polynomial with robust weights
    Inputs:
        arc1d: 1D array-like arc spectrum
        wl_range: (wl_lo, wl_hi) tuple in microns for target wavelength coverage
        ref_lines_um: 1D array of reference lines in microns
        deg: Polynomial degree for Chebyshev fit (default 3)
    Returns:
        np.ndarray:
            Wavelength array (microns) of same length as arc1d, strictly increasing
            and clipped to [wl_lo, wl_hi]
    Raises:
        RuntimeError: If too few peaks or matched lines are found
    """

    y = np.asarray(arc1d, float)
    n = y.size
    wl_lo, wl_hi = float(wl_range[0]), float(wl_range[1])
    pk, _ = _find_arc_peaks_1d(y)

    if pk.size < 8:
        raise RuntimeError("Too few arc peaks")

    px_m, wl_m = _match_peaks_to_refs(pk, ref_lines_um, wl_lo, wl_hi, max_sep=0.012)

    if px_m.size < max(8, deg + 3):
        raise RuntimeError("Insufficient matched lines")

    x_full = np.linspace(-1.0, 1.0, n)
    x_m = np.interp(px_m, np.arange(n), x_full)
    X = _cheb_design_matrix(x_m, deg)
    w = np.ones_like(wl_m)

    for _ in range(12):
        coef, *_ = npl.lstsq(X * w[:, None], wl_m * w, rcond=None)
        res = wl_m - X.dot(coef)
        w_new = _robust_weights(res)

        if np.allclose(w, w_new, atol=1e-3):
            break

        w = w_new

    wl = np.polynomial.chebyshev.chebval(x_full, coef)
    span_fit = wl[-1] - wl[0]
    span_tar = wl_hi - wl_lo

    if abs(span_fit - span_tar) / max(span_tar, 1e-12) > 0.002:
        a = span_tar / (span_fit + 1e-12)
        b = wl_lo - a * wl[0]
        wl = a * wl + b

    else:
        wl += (wl_lo - wl[0])

    for i in range(1, n):
        if wl[i] <= wl[i - 1]:
            wl[i] = wl[i - 1] + 1e-9

    return np.clip(wl, wl_lo, wl_hi)


def _extract_with_local_bg(img, center_col, ap=7, bg_in=12, bg_out=26):
    """
    Purpose:
        Extract 1D spectrum by median-collapsing columns around center column,
        subtracting local background estimated from side bands
    Inputs:
        img: 2D array image (rows x cols)
        center_col: Central column index for extraction
        ap: Half-width of extraction aperture in columns (default 7)
        bg_in: Inner offset (columns) from center to begin background windows (default 12)
        bg_out: Outer offset (columns) from center to end background windows (default 26)
    Returns:
        np.ndarray:
            1D extracted spectrum per row after background subtraction
    """

    nrows, ncols = img.shape
    lo = max(0, center_col - ap)
    hi = min(ncols, center_col + ap + 1)
    bg_left = img[:, max(0, center_col - bg_out) : max(0, center_col - bg_in)]
    bg_right = img[:, min(ncols, center_col + bg_in) : min(ncols, center_col + bg_out)]

    if bg_left.size == 0 and bg_right.size == 0:
        bg = np.zeros(nrows, dtype=img.dtype)

    else:
        bg = np.median(np.concatenate([bg_left, bg_right], axis=1), axis=1)

    return np.median(img[:, lo:hi], axis=1) - bg


def _extract_cols_median_with_err(img, center_col, half=1, ap=7, bg_in=12, bg_out=26):
    """
    Purpose:
        Extract 1D spectrum by median-combining multiple adjacent columns around
        center_col with local background subtraction, and estimate per-row errors
    Inputs:
        img: 2D array image (rows x cols)
        center_col: Central column index
        half: Include columns [center_col-half, center_col+half] (default 1)
        ap: Half-width of per-column extraction aperture (default 7)
        bg_in: Inner offset of background windows (default 12)
        bg_out: Outer offset of background windows (default 26)
    Returns:
        tuple:
            - flux_1d (np.ndarray): Extracted 1D flux (rows)
            - sigma_1d (np.ndarray): Estimated 1-sigma errors per row
    """

    nrows, ncols = img.shape
    cols = [c for c in range(center_col - half, center_col + half + 1) if 0 <= c < ncols]
    specs, sigmas = [], []

    for c in cols:
        lo = max(0, c - ap)
        hi = min(ncols, c + ap + 1)
        bg_left = img[:, max(0, c - bg_out) : max(0, c - bg_in)]
        bg_right = img[:, min(ncols, c + bg_in) : min(ncols, c + bg_out)]

        if bg_left.size == 0 and bg_right.size == 0:
            bg_med = np.zeros(nrows, dtype=float)
            bg_std = np.zeros(nrows, dtype=float)

        else:
            bg_all = (
                np.concatenate([bg_left, bg_right], axis=1)
                if (bg_left.size and bg_right.size)
                else (bg_left if bg_left.size else bg_right)
            )

            bg_med = np.median(bg_all, axis=1).astype(float)
            mad = np.median(np.abs(bg_all - bg_med[:, None]), axis=1) + 1e-12
            bg_std = 1.4826 * mad

        sub = img[:, lo:hi] - bg_med[:, None]
        spec = np.median(sub, axis=1).astype(float)
        specs.append(spec)
        n_eff = max(1, hi - lo)
        sigma_row = np.sqrt(np.pi / 2) * bg_std / np.sqrt(n_eff)
        sigmas.append(sigma_row)

    spec_stack = np.vstack(specs)
    sigma_stack = np.vstack(sigmas)
    flux_1d = np.median(spec_stack, axis=0)
    M = max(1, len(cols))
    sigma_1d = np.sqrt(np.pi / 2) * np.median(sigma_stack**2, axis=0) ** 0.5 / np.sqrt(M)

    return flux_1d.astype(float), sigma_1d.astype(float)


def _detect_slit_edges(data, x_hint=(900, 1300), hint_expand=150, row_frac=(0.35, 0.85)):
    """
    Purpose:
        Detect left/right slit edges from median spatial profile using gradient peaks
        in restricted column window and row band
    Inputs:
        data: 2D array image (rows x cols)
        x_hint: Tuple of approximate slit bounds (lo, hi) in columns
        hint_expand: Additional search half-width to expand around x_hint (default 150)
        row_frac: Fractional row range (lo, hi) for band extraction (default (0.35, 0.85))
    Returns:
        tuple:
            - left_edge (int): Detected left edge column index
            - right_edge (int): Detected right edge column index
            - sm (np.ndarray): Smoothed high-pass profile used
            - g (np.ndarray): Smoothed gradient used for peak finding
            - (lo, hi) (tuple[int,int]): Effective search column range
            - (r0, r1) (tuple[int,int]): Row band used for the profile
    """

    nrows, ncols = data.shape
    r0 = int(min(row_frac) * nrows)
    r1 = int(max(row_frac) * nrows)
    band = data[r0:r1, :]
    lo_global, hi_global = int(0.03 * ncols), int(0.97 * ncols)
    xmin, xmax = max(lo_global, min(x_hint)), min(hi_global, max(x_hint))
    lo = max(lo_global, xmin - hint_expand)
    hi = min(hi_global, xmax + hint_expand)
    prof = np.median(band, axis=0).astype(float)
    base = gaussian_filter1d(prof, 120, mode="nearest")
    hp = prof - base
    sm = gaussian_filter1d(hp, 3, mode="nearest")
    g = gaussian_filter1d(np.gradient(sm), 2.2, mode="nearest")
    prom = np.percentile(np.abs(g[lo:hi]), 70)
    L_idx, _ = find_peaks(g[lo:hi], prominence=float(max(prom, 1e-6)), distance=12)
    R_idx, _ = find_peaks(-g[lo:hi], prominence=float(max(prom, 1e-6)), distance=12)

    if L_idx.size == 0 or R_idx.size == 0:
        return lo, hi, sm, g, (lo, hi), (r0, r1)

    L = lo + L_idx
    R = lo + R_idx
    best = (-1e9, lo, hi)

    for l in L:
        for r in R[R > l + 8]:
            interior = np.median(sm[l:r])
            left_bg = np.median(sm[max(lo, l - 60) : l])
            right_bg = np.median(sm[r : min(hi, r + 60)])
            score = abs(interior - 0.5 * (left_bg + right_bg))

            if score > best[0]:
                best = (score, l, r)

    _, le, re = best

    return int(le), int(re), sm, g, (lo, hi), (r0, r1)


def _detect_objects_in_slit(data, left_edge, right_edge, row_frac=(0.40, 0.80)):
    """
    Purpose:
        Detect bright and dark object centroids within the slit by analyzing
        median column profile between edges
    Inputs:
        data: 2D array image (rows x cols)
        left_edge: Left slit edge column index
        right_edge: Right slit edge column index
        row_frac: Fractional row limits for band extraction (default (0.40, 0.80))
    Returns:
        tuple:
            - obj_pos_abs (int): Column of brightest object peak within slit
            - obj_neg_abs (int): Column of darkest trough within slit
            - prof (np.ndarray): Smoothed median profile used for detection
            - (r0, r1) (tuple[int,int]): Row band used
    """

    nrows, _ = data.shape
    r0 = int(min(row_frac) * nrows)
    r1 = int(max(row_frac) * nrows)
    mid_lo = left_edge + int(0.20 * (right_edge - left_edge))
    mid_hi = right_edge - int(0.20 * (right_edge - left_edge))
    band = data[r0:r1, mid_lo : mid_hi + 1]
    prof = gaussian_filter1d(np.median(band, axis=0).astype(float), 4.0)
    pos_rel = int(np.argmax(prof))
    neg_rel = int(np.argmin(prof))
    obj_pos_abs = mid_lo + pos_rel
    obj_neg_abs = mid_lo + neg_rel

    return int(obj_pos_abs), int(obj_neg_abs), prof, (r0, r1)


def _find_arc_trace_col_strong(
    arc_img, approx_col=None, search_half=240, x_hint=(900, 1300), row_frac=(0.35, 0.85)
):
    """
    Purpose:
        Find strong arc column (trace position) by peak-searching median
        spatial profile, optionally near approximate column
    Inputs:
        arc_img: 2D arc image array
        approx_col: Optional approximate column index around which to search
        search_half: Half-width of search window around approx_col (default 240)
        x_hint: Tuple giving broader column region for initial profile (default (900,1300))
        row_frac: Fractional row band for profile median (default (0.35,0.85))
    Returns:
        int:
            Selected column index corresponding to strongest arc-like feature
    """

    img = np.asarray(arc_img, float)
    nrows, ncols = img.shape
    lo_x = int(max(0, min(x_hint)))
    hi_x = int(min(ncols, max(x_hint)))
    r0 = int(min(row_frac) * nrows)
    r1 = int(max(row_frac) * nrows)
    band = img[r0:r1, lo_x:hi_x]
    prof = np.median(band, axis=0).astype(float)
    base = gaussian_filter1d(prof, 65, mode="nearest")
    hp = prof - base
    sm = gaussian_filter1d(hp, 3, mode="nearest")
    prom = max(np.nanpercentile(np.abs(sm), 98) * 0.6, 10.0)
    cand_idx, _ = find_peaks(sm, prominence=float(prom), distance=10)
    cand_cols = (lo_x + cand_idx).astype(int)

    if approx_col is not None and cand_cols.size:
        cand_cols = cand_cols[np.abs(cand_cols - int(approx_col)) <= int(search_half)]

    if cand_cols.size == 0:
        if approx_col is None:
            return int(lo_x + int(np.argmax(sm)))

        lo = max(0, int(approx_col) - int(search_half))
        hi = min(ncols, int(approx_col) + int(search_half) + 1)
        j = int(np.argmax(sm[(lo - lo_x) : (hi - lo_x)]))

        return int(lo + j)

    win = 5
    scores = []

    for c in cand_cols:
        j = int(np.clip(c - lo_x, 0, band.shape[1] - 1))
        j0 = max(0, j - win)
        j1 = min(band.shape[1], j + win + 1)
        scores.append(float(np.nanmedian(band[:, j0:j1], axis=0).max()))

    return int(cand_cols[int(np.argmax(scores))])


def _average_wavecal_across_cols(arc_img, center_col, half=1, ref_lines_um=None, wl_range=(0.83, 2.45)):
    """
    Purpose:
        Compute averaged wavelength solution across a few neighboring columns
        around chosen arc column by solving per-column and averaging
    Inputs:
        arc_img: 2D arc image array
        center_col: Central column index to calibrate around
        half: Use columns [center_col-half, center_col+half] (default 1)
        ref_lines_um: 1D array of reference line wavelengths in microns (required)
        wl_range: Tuple (wl_lo, wl_hi) microns for target coverage (default (0.83, 2.45))
    Returns:
        np.ndarray:
            Averaged wavelength solution (microns) on native pixel grid,
            clipped and oriented to increase
    Raises:
        ValueError: If ref_lines_um is None
    """

    if ref_lines_um is None:
        raise ValueError("ref_lines_um required")

    ncols = arc_img.shape[1]
    cols = [c for c in range(center_col - half, center_col + half + 1) if 0 <= c < ncols]
    wl_list = []

    for c in cols:
        a1d = _extract_with_local_bg(arc_img, c, ap=5, bg_in=8, bg_out=18)
        wl_c = _solve_dispersion_from_arc1d(
            a1d, wl_range=wl_range, ref_lines_um=np.asarray(ref_lines_um, float), deg=3
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

    return np.clip(wl_avg, wl_min, wl_max)


def _load_vega_model(path: str | Path):
    """
    Purpose:
        Load Vega model spectrum from a two-column text file (wavelength, flux)
        Auto-convert wavelengths to microns if in Å (divide by 1e4)
    Inputs:
        path: Path-like to text file
    Returns:
        tuple:
            - wl (np.ndarray): Wavelengths in microns
            - fx (np.ndarray): Flux values, sorted by wavelength and finite
    """

    arr = np.genfromtxt(path, dtype=float)
    wl = np.asarray(arr[:, 0], float)
    fx = np.asarray(arr[:, 1], float)

    if np.nanmedian(wl) > 50:
        wl = wl / 1e4  # Å -> µm

    m = np.isfinite(wl) & np.isfinite(fx)
    wl = wl[m]
    fx = fx[m]
    idx = np.argsort(wl)

    return wl[idx], fx[idx]


def _gaussian_broaden_to_R(wl_um, fx, R_target=2000.0, oversample=2):
    """
    Purpose:
        Convolve spectrum with Gaussian in log-wavelength space to reach target
        resolving power R
    Inputs:
        wl_um: 1D wavelengths in microns
        fx: 1D flux array aligned with wl_um
        R_target: Target resolving power (default 2000)
        oversample: Oversampling factor for log grid (default 2)
    Returns:
        tuple:
            - wl_out (np.ndarray): Output wavelengths (microns) on log grid back-converted
            - fx_out (np.ndarray): Broadened flux array
    """

    wl = np.asarray(wl_um, float)
    fx = np.asarray(fx, float)

    if wl.size < 10:
        return wl, fx

    logw = np.log(wl)
    step = np.median(np.diff(logw)) / max(1, oversample)
    grid = np.arange(logw.min(), logw.max() + step / 2, step)
    f = np.interp(np.exp(grid), wl, fx, left=np.nan, right=np.nan)
    m = np.isfinite(f)

    if m.sum() < 10:
        return wl, fx

    if (~m).any():
        f[~m] = np.interp(np.where(~m)[0], np.where(m)[0], f[m])

    sigma_pix = (1.0 / (2.355 * R_target)) / step

    if sigma_pix < 0.3:
        return wl, fx

    g = sps.windows.gaussian(M=int(8 * sigma_pix) | 1, std=sigma_pix)
    g /= g.sum()
    fb = np.convolve(f, g, mode="same")

    return np.exp(grid), fb


def _mask_a0v_intrinsic_lines(wl_um):
    """
    Purpose:
        Mask regions around intrinsic A0V stellar lines where ratios are unreliable
    Inputs:
        wl_um: 1D array of wavelengths in microns
    Returns:
        np.ndarray:
            Boolean mask array where True indicates keep (not masked)
    """

    lines = [0.9546, 1.0049, 1.0941, 1.2821, 1.513, 1.641, 1.681, 1.736, 2.1661, 2.281, 2.625, 2.874]
    wl = np.asarray(wl_um, float)
    m = np.ones_like(wl, dtype=bool)

    for w0 in lines:
        m &= ~(np.abs(wl - w0) < 0.006)

    return m


def _mask_deep_telluric_regions(wl_um):
    """
    Purpose:
        Mask very deep telluric absorption regions where correction is unreliable
    Inputs:
        wl_um: 1D array of wavelengths in microns
    Returns:
        np.ndarray:
            Boolean mask array where True indicates keep (not masked)
    """

    wl = np.asarray(wl_um, float)
    m = np.ones_like(wl, dtype=bool)

    for a, b in [(1.35, 1.42), (1.80, 1.95), (0.90, 0.94)]:
        m &= ~((wl >= a) & (wl <= b))

    return m


def _band_segments(wl):
    """
    Purpose:
        Define J, H, K band boolean masks on provided wavelength grid
    Inputs:
        wl: 1D array of wavelengths in microns
    Returns:
        list[np.ndarray]:
            List of boolean masks [J, H, K] selecting each band's wavelength range
    """

    wl = np.asarray(wl, float)
    J = (wl >= 0.98) & (wl <= 1.31)
    H = (wl >= 1.52) & (wl <= 1.74)
    K = (wl >= 2.06) & (wl <= 2.33)

    return [J, H, K]


def _build_T_per_band_dense(
    wl_std,
    fx_std,
    wl_vega,
    fx_vega,
    R_fire=2000.0,
    deg_cont=2,
    min_run=60,
    smooth_sigma=0.7,
    clip_lo=0.06,
    clip_hi=1.12,
):
    """
    Purpose:
        Build dense telluric transmission on standard star wavelength grid
        by ratioing to broadened Vega model, fitting smooth continuum per band,
        and smoothing only within contiguous supported segments
    Inputs:
        wl_std: 1D wavelengths (microns) of standard star spectrum
        fx_std: 1D flux of standard star aligned with wl_std
        wl_vega: 1D Vega model wavelengths (microns)
        fx_vega: 1D Vega model flux
        R_fire: Target resolution for broadening Vega (default 2000)
        deg_cont: Degree of Chebyshev continuum fit per band (default 2)
        min_run: Minimum contiguous run length to accept for smoothing (default 60)
        smooth_sigma: Sigma for Gaussian smoothing in ln T space (default 0.7)
        clip_lo: Lower clip for T values (default 0.06)
        clip_hi: Upper clip for T values (default 1.12)
    Returns:
        np.ndarray:
            Array on wl_std grid; NaN outside supported segments,
            clipped to [clip_lo, clip_hi]
    """

    wl_std = np.asarray(wl_std, float)
    fx_std = np.asarray(fx_std, float)

    wvb, fvb = _gaussian_broaden_to_R(wl_vega, fx_vega, R_target=R_fire, oversample=2)
    fvb_i = np.interp(wl_std, wvb, fvb, left=np.nan, right=np.nan)

    def cheb_fit(x, y, deg):
        """
        Purpose:
            Fit Chebyshev polynomial to (x, y) using robust reweighting
        Inputs:
            x: 1D array of x-values
            y: 1D array of y-values
            deg: Polynomial degree
        Returns:
            Callable:
                Function P(z) that evaluates fitted polynomial at z
        """

        x = np.asarray(x, float)
        y = np.asarray(y, float)

        if x.size < deg + 3:
            m = np.nanmedian(y) if y.size else 1.0
            return lambda z: np.full_like(np.asarray(z, float), m)

        t = (x - x.min()) / max(x.ptp(), 1e-12) * 2 - 1
        X = _cheb_design_matrix(t, deg)
        w = np.ones_like(y)

        for _ in range(10):
            coef, *_ = npl.lstsq(X * w[:, None], y * w, rcond=None)
            res = y - X.dot(coef)
            w_new = _robust_weights(res)

            if np.allclose(w, w_new, atol=1e-3):
                break

            w = w_new

        def P(z):
            z = np.asarray(z, float)
            tz = (z - x.min()) / max(x.ptp(), 1e-12) * 2 - 1

            return _cheb_design_matrix(tz, deg).dot(coef)

        return P

    T = np.full_like(wl_std, np.nan, dtype=float)

    for mband in _band_segments(wl_std):
        base = mband & np.isfinite(fx_std) & np.isfinite(fvb_i) & (fvb_i > 0)

        if base.sum() < min_run:
            continue

        mfit = base & _mask_deep_telluric_regions(wl_std) & _mask_a0v_intrinsic_lines(wl_std)

        if mfit.sum() < min_run:
            mfit = base & _mask_deep_telluric_regions(wl_std)

        if mfit.sum() < max(30, deg_cont + 3):
            mfit = base

        ratio = fx_std[mfit] / np.maximum(fvb_i[mfit], 1e-20)
        keep = np.isfinite(ratio) & (ratio > 0)

        if keep.sum() < max(30, deg_cont + 3):
            continue

        w_fit = wl_std[mfit][keep]
        r_fit = ratio[keep]
        P = cheb_fit(w_fit, r_fit, deg_cont)

        idx_band = np.where(mband)[0]
        idx_base = idx_band[base[idx_band]]

        if idx_base.size == 0:
            continue

        cont = P(wl_std[idx_base])
        rawT = fx_std[idx_base] / np.maximum(fvb_i[idx_base] * cont, 1e-20)

        runs = []
        i = 0

        while i < idx_base.size:
            j = i

            while j + 1 < idx_base.size and (idx_base[j + 1] == idx_base[j] + 1):
                j += 1

            if (j - i + 1) >= min_run:
                runs.append((i, j))
            i = j + 1

        for a, b in runs:
            seg_idx = idx_base[a : b + 1]
            vals = np.clip(rawT[a : b + 1], 1e-4, 5.0)
            v = np.log(np.clip(vals, clip_lo, clip_hi))
            v = gaussian_filter1d(v, sigma=smooth_sigma, mode="nearest")
            T[seg_idx] = np.exp(v)

    T = np.where(np.isfinite(T), np.clip(T, clip_lo, clip_hi), np.nan)

    return T


def apply_telluric_correction(
    science_fits: str | Path,
    raw_dir: str | Path,
    std_a_id: int,
    std_b_id: int,
    arc_path: str | Path,
    ref_list_path: str | Path,
    vega_model_path: str | Path,
    out_dir: str | Path,
    *,
    qa_dir: str | Path | None = None,
    wl_range: Tuple[float, float] = (0.83, 2.45),
    slit_x_hint: Tuple[int, int] = (900, 1300),
    row_fraction: Tuple[float, float] = (0.35, 0.85),
    std_ap: int = 9,
    std_bg_in: int = 16,
    std_bg_out: int = 32,
    R_fire: float = 2000.0,
    T_min: float = 0.2,
    T_max: float = 1.2,
    show_plots: bool = False,
) -> Path:
    """
    Purpose:
        Apply telluric correction to science spectrum using A0V standard and Vega model:
          - Load science spectrum and standard frames (A/B) and arc
          - Detect slit edges and object position in standard, choose extraction column
          - Solve wavelength solution from arc and average across nearby columns
          - Extract standard spectrum (POS-only) with tuned aperture/background
          - Build telluric transmission on standard grid (per band)
          - Interpolate T to science wavelength grid and apply within valid bounds
          - Save corrected spectrum and QA plots
    Inputs:
        science_fits: Path to input science spectrum FITS with table
                      (wavelength_um, flux[, flux_err])
        raw_dir: Directory containing raw standard frames (e.g., fire_####.fits)
        std_a_id: Integer ID for the A frame (e.g., 1234 for fire_1234.fits)
        std_b_id: Integer ID for the B frame (e.g., 1235 for fire_1235.fits)
        arc_path: Path to arc FITS (2D)
        ref_list_path: Path to reference line list (text; units parsed to microns)
        vega_model_path: Path to Vega model text file; wavelengths auto-converted to microns
        out_dir: Output directory for corrected FITS and QA
        qa_dir: Optional directory for QA plots (default: out_dir/qa)
        wl_range: Wavelength range (microns) for wavecal solution (default (0.83, 2.45))
        slit_x_hint: Approximate slit column bounds for edge detection (default (900, 1300))
        row_fraction: Fractional row range for profiles (default (0.35, 0.85))
        std_ap: Base half-aperture (columns) for standard extraction (default 9)
        std_bg_in: Base inner background offset (columns) (default 16)
        std_bg_out: Base outer background offset (columns) (default 32)
        R_fire: Resolving power used to broaden Vega (default 2000)
        T_min: Minimum allowed T when applying correction (default 0.2)
        T_max: Maximum allowed T when applying correction (default 1.2)
        show_plots: Show QA plots
    Returns:
        Path:
            Path to the output telluric-corrected FITS table saved
    """

    out_dir = Path(out_dir)
    qa_dir = Path(qa_dir) if qa_dir is not None else (out_dir / "qa")
    _ensure_dir(out_dir)
    _ensure_dir(qa_dir)

    # Load science
    sci_wl, sci_fx, sci_err, sci_hdr = _load_table_spectrum(science_fits)
    sci_wl, sci_fx = _orient_to_increasing(sci_wl, sci_fx)
    sci_wl, sci_fx = _assert_monotonic_and_align(sci_wl, sci_fx)
    if sci_err is not None:
        sci_err = np.asarray(sci_err, float)

        if sci_err.size == sci_wl.size:
            _, sci_err = _orient_to_increasing(sci_wl, sci_err)
            _, sci_err = _assert_monotonic_and_align(sci_wl, sci_err)

        else:
            idx_err = np.linspace(0, 1, sci_err.size)
            idx_wl = np.linspace(0, 1, sci_wl.size)
            sci_err = np.interp(idx_wl, idx_err, sci_err).astype(float)

    _plot_1d(sci_wl, sci_fx, "Science coadd (input)", qa_dir / "science_input.png", show=show_plots)

    # Load ARC + lines + standard frames
    arc_data, _ = _load_fits_primary(arc_path)
    ref_lines = _load_line_list_to_microns(ref_list_path)

    def build_path(num): return Path(raw_dir) / f"fire_{num:04d}.fits"

    stdA, _ = _load_fits_primary(build_path(std_a_id))
    stdB, _ = _load_fits_primary(build_path(std_b_id))

    # Slit and object positions on standard
    std_sub = stdA - stdB
    le, re, *_ = _detect_slit_edges(std_sub, x_hint=slit_x_hint, hint_expand=250, row_frac=row_fraction)

    if (re - le) < 25:
        le, re = slit_x_hint

    obj_pos_a, obj_neg_a, _, _ = _detect_objects_in_slit(stdA, le, re, row_frac=(0.40, 0.80))

    def column_brightness(img, col, ap=5):
        """
        Purpose:
            Compute median brightness metric in local vertical band around column
        Inputs:
            img: 2D image array
            col: Column index
            ap: Half-width of local column window (default 5)
        Returns:
            float:
                Median brightness within row band used
        """

        nrows = img.shape[0]
        r0 = int(0.45 * nrows)
        r1 = int(0.75 * nrows)
        c0 = max(0, col - ap)
        c1 = min(img.shape[1], col + ap + 1)

        return float(np.nanmedian(img[r0:r1, c0:c1]))

    bp = column_brightness(stdA, obj_pos_a, ap=5) if le <= obj_pos_a <= re else -np.inf
    bn = column_brightness(stdA, obj_neg_a, ap=5) if le <= obj_neg_a <= re else -np.inf

    if (not np.isfinite(bp) and not np.isfinite(bn)) or abs(obj_pos_a - obj_neg_a) < 8:
        band_rows = stdA[int(row_fraction[0] * stdA.shape[0]) : int(row_fraction[1] * stdA.shape[0]), le:re]
        prof = np.median(band_rows, axis=0).astype(float)
        arc_col_std = le + int(np.argmax(gaussian_filter1d(prof, 3.0)))

    else:
        arc_col_std = obj_pos_a if bp >= bn else obj_neg_a

    band_rows = stdA[
        int(row_fraction[0] * stdA.shape[0]) : int(row_fraction[1] * stdA.shape[0]),
        max(le, arc_col_std - 8) : min(re, arc_col_std + 9),
    ]

    prof_local = np.median(band_rows, axis=0).astype(float)
    arc_col_std = max(le, arc_col_std - 8) + int(np.argmax(gaussian_filter1d(prof_local, 2.0)))

    # Wavecal and POS-only extraction on standard
    wl_std_pix = _average_wavecal_across_cols(arc_data, arc_col_std, half=1, ref_lines_um=ref_lines, wl_range=wl_range)

    def extract_std_tuned(img, col, ap0, bgi0, bgo0):
        """
        Purpose:
            Extract robust standard star spectrum by trying multiple aperture
            and background settings and selecting a viable result
        Inputs:
            img: 2D image array
            col: Central column index
            ap0: Base half-aperture in columns
            bgi0: Base inner background offset
            bgo0: Base outer background offset
        Returns:
            tuple:
                - flux (np.ndarray): Extracted 1D flux
                - err (np.ndarray): Estimated per-row 1-sigma errors
        """

        ap_list = [ap0, ap0 + 2, ap0 + 4, ap0 + 6]

        bg_list = [
            (bgi0, bgo0),
            (max(bgi0 + 4, 16), max(bgo0 + 8, 32)),
            (max(bgi0 + 8, 22), max(bgo0 + 14, 42)),
        ]

        best = (-1e30, None, None, (ap0, bgi0, bgo0))

        for ap in ap_list:
            for bgi, bgo in bg_list:
                fx, er = _extract_cols_median_with_err(img, col, half=1, ap=ap, bg_in=bgi, bg_out=bgo)
                med = float(np.nanmedian(fx))

                if med > 0:
                    return fx, er

                if med > best[0]:
                    best = (med, fx, er, (ap, bgi, bgo))

        _, fx, er, _ = best

        return fx, er

    std_flux, std_err = extract_std_tuned(stdA, arc_col_std, std_ap, std_bg_in, std_bg_out)

    # Fallbacks if needed
    if float(np.nanmedian(std_flux)) <= 0 and le <= obj_neg_a <= re and obj_neg_a != arc_col_std:
        alt_fx, alt_er = extract_std_tuned(stdA, obj_neg_a, std_ap, std_bg_in, std_bg_out)
        if float(np.nanmedian(alt_fx)) > float(np.nanmedian(std_flux)):
            std_flux, std_err = alt_fx, alt_er
            arc_col_std = obj_neg_a

    if std_flux is not None and std_flux.size > 11:
        std_flux = gaussian_filter1d(std_flux, sigma=0.5, mode="nearest")

    wl_std_pix, std_flux = _assert_monotonic_and_align(*_orient_to_increasing(wl_std_pix, std_flux))

    if std_err is not None:
        _, std_err = _assert_monotonic_and_align(*_orient_to_increasing(wl_std_pix, std_err))

    # Load Vega model
    v_wl, v_fx = _load_vega_model(vega_model_path)

    if np.nanmedian(v_fx) <= 0:
        v_fx = np.abs(v_fx) + 1e-6

    # Trim ends
    Kstd = 6

    if wl_std_pix.size > 2 * Kstd:
        wl_std = wl_std_pix[Kstd:-Kstd]
        std_fx = std_flux[Kstd:-Kstd]

    else:
        wl_std, std_fx = wl_std_pix, std_flux

    # Build T on native standard grid
    T_std = _build_T_per_band_dense(
        wl_std,
        std_fx,
        v_wl,
        v_fx,
        R_fire=R_fire,
        deg_cont=2,
        min_run=60,
        smooth_sigma=0.7,
        clip_lo=0.06,
        clip_hi=1.12,
    )

    def dense_support_mask(w, y, min_run=60):
        """
        Purpose:
            Identify indices where y is finite/positive in contiguous runs of at least min_run
        Inputs:
            w: 1D x-values (unused except for length consistency)
            y: 1D values to test for support
            min_run: Minimum length of contiguous run to keep (default 60)
        Returns:
            np.ndarray:
                Boolean mask of same shape as y indicating supported indices
        """

        w = np.asarray(w, float)
        y = np.asarray(y, float)
        m = np.isfinite(w) & np.isfinite(y) & (y > 0)
        keep = np.zeros_like(m, bool)
        i = 0
        n = m.size

        while i < n:
            if not m[i]:
                i += 1
                continue
            j = i

            while j < n and m[j]:
                j += 1

            if (j - i) >= min_run:
                keep[i:j] = True

            i = j

        return keep

    m_dense = dense_support_mask(wl_std, T_std, min_run=60)
    T_on_grid = np.full_like(sci_wl, np.nan, float)

    if m_dense.sum() >= 60:
        wmin, wmax = wl_std[m_dense].min(), wl_std[m_dense].max()
        inside = (sci_wl >= wmin) & (sci_wl <= wmax)

        if inside.any():
            T_on_grid[inside] = np.interp(sci_wl[inside], wl_std[m_dense], T_std[m_dense])

    # Apply T
    T_safe = np.where(
        np.isfinite(T_on_grid) & (T_on_grid >= T_min) & (T_on_grid <= T_max),
        T_on_grid,
        np.nan,
    )

    fx_corr = np.divide(
        sci_fx,
        T_safe,
        out=np.full_like(sci_fx, np.nan, dtype=float),
        where=np.isfinite(T_safe),
    )

    err_corr = None

    if sci_err is not None:
        err_corr = np.divide(
            sci_err,
            T_safe,
            out=np.full_like(sci_err, np.nan, dtype=float),
            where=np.isfinite(T_safe),
        )

    # QA plots
    _plot_1d(sci_wl, sci_fx, "Science coadd (input)", qa_dir / "science_input.png", show=show_plots)
    mT = np.isfinite(T_on_grid)

    if mT.sum() >= 10:
        _plot_1d(
            sci_wl[mT],
            T_on_grid[mT],
            "Telluric T(λ) — science grid",
            qa_dir / "T_on_science_grid.png",
            xlabel="Wavelength (um)",
            ylabel="Transmission",
            show=False,
        )
    m_corr = np.isfinite(fx_corr)

    if m_corr.sum() >= 10:
        _plot_1d(
            sci_wl[m_corr],
            fx_corr[m_corr],
            "Science (telluric-corrected, masked)",
            qa_dir / "science_telluric_corrected.png",
            show=show_plots,
        )

    # Save corrected FITS
    out_fits = out_dir / f"telluric_corrected_{Path(science_fits).stem}.fits"

    hist = [
        "Telluric: POS-only standard, band-wise scaling, deep-gap masking",
        f"Vega broadened to R~{R_fire}, smoothing within contiguous segments only",
        f"Applied only where {T_min}<=T<={T_max} and within overlap",
    ]

    _write_spectrum_with_err(out_fits, sci_wl, fx_corr, err_corr, base_header=sci_hdr, history=hist)

    return out_fits


def _load_line_list_to_microns(path: str | Path) -> np.ndarray:
    """
    Purpose:
        Load reference line list from text, parsing units to microns (um)
        Accepts tokens with explicit units (um, µm, nm, A/Ang/angstroms) or bare numbers
        with heuristic unit inference
    Inputs:
        path: Path-like to text file containing line wavelengths
    Returns:
        np.ndarray:
            Sorted array of wavelengths in microns (float), finite values only
    """

    waves = []

    with open(path, "r") as f:
        for raw in f:
            s = raw.strip()

            if not s or s.startswith("#"):
                continue

            parts = s.replace(",", " ").split()
            val = None
            unit = None

            for tok in parts:
                tl = tok.lower()
                try:
                    if tl.endswith(("um", "µm", "micron", "microns")):
                        unit = "um"
                        num = "".join(ch for ch in tok if (ch.isdigit() or ch in ".-+eE"))
                        val = float(num)
                        break

                    if tl.endswith("nm"):
                        unit = "nm"
                        val = float(tok[:-2])
                        break

                    if tl.endswith(("a", "ang", "angstrom", "angstroms")):
                        unit = "A"
                        num = "".join(ch for ch in tok if (ch.isdigit() or ch in ".-+eE"))
                        val = float(num)
                        break

                    val = float(tok)
                    unit = None
                    break

                except Exception:
                    continue

            if val is None:
                continue

            if unit is None:
                if val > 1000:
                    unit = "A"
                elif 400 <= val <= 5000:
                    unit = "nm"
                else:
                    unit = "um"

            waves.append(val if unit == "um" else val / 1000.0 if unit == "nm" else val / 1e4)

    arr = np.array(waves, dtype=float)
    arr = arr[np.isfinite(arr)]

    return np.sort(arr)
