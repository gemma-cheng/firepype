# firepype/utils.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d


def ensure_dir(p: str | Path):
    """
    Purpose:
        Create a directory (and all missing parents) if it does not already exist
    Inputs:
        p: Path-like string or Path to the directory to create
    Returns:
        None
    """

    Path(p).mkdir(parents=True, exist_ok=True)


def sg_window_len(n_points: int, preferred: int, min_odd: int = 5) -> int:
    """
    Purpose:
        Compute odd window length for filtering/smoothing that:
        - Does not exceed n_points
        - Is at least min_odd
        - Is as close as possible to preferred
        - Is odd (decremented by 1 if even)
    Inputs:
        n_points: Total number of points available (upper bound for window)
        preferred: Preferred window length
        min_odd: Minimum allowed odd window length (default 5)
    Returns:
        int:
            An odd window length satisfying the constraints (>=3 if possible)
    """

    if n_points <= 0:
        return 0

    w = min(preferred, n_points)

    if w % 2 == 0:
        w -= 1

    w = max(w, min_odd)

    if w > n_points:
        w = n_points if (n_points % 2 == 1) else (n_points - 1)

    return max(w, 3)


def cheb_design_matrix(x: np.ndarray, deg: int) -> np.ndarray:
    """
    Purpose:
        Build Chebyshev T_k(x) design matrix up to degree for inputs x in [-1, 1]
    Inputs:
        x: 1D array of input values (ideally scaled to [-1, 1])
        deg: Non-negative integer degree of polynomial basis
    Returns:
        np.ndarray:
            Matrix of shape (len(x), deg+1) with columns [T0(x), T1(x), ..., T_deg(x)]
    """

    x = np.asarray(x, float)
    X = np.ones((x.size, deg + 1), float)

    if deg >= 1:
        X[:, 1] = x

    for k in range(2, deg + 1):
        X[:, k] = 2.0 * x * X[:, k - 1] - X[:, k - 2]

    return X


def robust_weights(residuals: np.ndarray, c: float = 4.685) -> np.ndarray:
    """
    Purpose:
        Compute Tukey's biweight (squared) robust regression weights from residuals.
        Points with |u| >= 1 (u = r/(c*s)) receive zero weight
    Inputs:
        residuals: 1D array of residual values
        c: Tuning constant controlling downweighting (default 4.685)
    Returns:
        np.ndarray:
            Weights in [0, 1], same shape as residuals
    """

    r = np.asarray(residuals, float)
    s = np.nanmedian(np.abs(r - np.nanmedian(r))) * 1.4826 + 1e-12
    u = r / (c * s)
    w = (1 - u**2)
    w[(np.abs(u) >= 1) | ~np.isfinite(w)] = 0.0

    return w**2


def orient_to_increasing(wl: np.ndarray, fx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Purpose:
        Ensure wavelength array increases with index. If decreasing, reverse both
        wavelength and flux arrays to maintain alignment
    Inputs:
        wl: 1D array of wavelengths
        fx: 1D array of flux values aligned with wl
    Returns:
        tuple:
            - wl_out (np.ndarray): Wavelengths in increasing order
            - fx_out (np.ndarray): Flux values reoriented to match wl_out
    """

    wl = np.asarray(wl, float)
    fx = np.asarray(fx, float)

    if wl.size >= 2 and wl[0] > wl[-1]:
        return wl[::-1], fx[::-1]

    return wl, fx


def assert_monotonic_and_align(
    wl: np.ndarray,
    fx: np.ndarray,
    name: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Purpose:
        Clean and align wavelength and flux arrays by:
        - Dropping non-finite entries
        - Sorting by wavelength
        - Enforcing strictly increasing wavelengths (removing non-increasing steps)
        Raises if monotonicity cannot be achieved
    Inputs:
        wl: 1D array of wavelengths
        fx: 1D array of flux values aligned with wl
        name: Optional label used in error messages
    Returns:
        tuple:
            - wl_out (np.ndarray): Strictly increasing wavelengths
            - fx_out (np.ndarray): Flux aligned to wl_out
    Raises:
        RuntimeError: If wavelengths are not strictly increasing after alignment
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

    if wl.size >= 2 and not np.all(np.diff(wl) > 0):
        raise RuntimeError(f"{name}: wl not strictly increasing after alignment")

    return wl, fx


# -------------------- Interpolation and mask helpers --------------------
def mask_interp_edge_artifacts(
    grid_wl: np.ndarray,
    wl_native: np.ndarray,
    f_native: np.ndarray | None,
    err_native: np.ndarray | None,
    *,
    min_span_px: int = 10,
    pad_bins: int = 8,
    min_keep_bins: int = 24,
) -> np.ndarray:
    """
    Purpose:
        Build a boolean mask (True=keep) on the coadd grid that retains only bins
        well-supported by native data
        Rules:
          1) Keep bins strictly inside native wavelength span
          2) Trim pad_bins bins from both ends of each inside segment
          3) Drop any inside segment shorter than min_keep_bins
          4) If native span < min_span_px, keep nothing
    Inputs:
        grid_wl: 1D coadd wavelength grid
        wl_native: 1D native wavelengths where data exist.
        f_native: Native flux (unused; present for API symmetry)
        err_native: Native errors (unused; present for API symmetry)
        min_span_px: Minimum contiguous native length (pixels) to consider (default 10)
        pad_bins: Padding to exclude at both ends of each inside segment (default 8)
        min_keep_bins: Minimum interior length to keep after padding (default 24)
    Returns:
        np.ndarray:
            Boolean mask on grid_wl where True indicates bins to keep
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


def clean_bool_runs(mask: np.ndarray, min_run: int = 24) -> np.ndarray:
    """
    Purpose:
        Remove short True segments from a boolean mask. Any contiguous run of True
        values shorter than min_run is set to False
    Inputs:
        mask: Boolean array
        min_run: Minimum run-length of True values to retain (default 24)
    Returns:
        np.ndarray:
            Cleaned boolean mask
    """
    m = np.asarray(mask, bool).copy()
    n = m.size
    i = 0
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i
        while j < n and m[j]:
            j += 1
        if (j - i) < min_run:
            m[i:j] = False
        i = j
    return m


# -------------------- Sky-line helper --------------------
def skyline_mask_from_1d(y: np.ndarray, sigma_hi: float = 3.0, win: int = 51) -> np.ndarray:
    """
    Purpose:
        Identify strong skyline-like deviations in a 1D vector using high-pass
        filtering and a robust MAD-based threshold
    Inputs:
        y: 1D array of values
        sigma_hi: Threshold in Gaussian sigmas for detection (default 3.0)
        win: Window parameter guiding the smoothing scale (default 51)
    Returns:
        np.ndarray:
            Boolean mask of same length as y where True indicates a skyline-like outlier
    """
    y = np.asarray(y, float)
    if y.size < 20:
        return np.zeros_like(y, dtype=bool)
    base = gaussian_filter1d(y, max(5, win // 4), mode="nearest")
    hp = y - base
    mad = np.nanmedian(np.abs(hp - np.nanmedian(hp))) + 1e-12
    thr = sigma_hi * 1.4826 * mad
    return np.abs(hp) > thr


# -------------------- Line list loader --------------------
def load_line_list_to_microns(path: str | Path) -> np.ndarray:
    """
    Purpose:
        Load a line list file of wavelengths with optional units and convert to microns.
        Supported units: um/µm/micron(s), nm, A/Angstrom(s). Bare numbers are
        heuristically interpreted based on magnitude
    Inputs:
        path: Path to the text file containing wavelengths, one or more tokens per line
              Lines starting with '#' are ignored; commas are allowed
    Returns:
        np.ndarray:
            Sorted 1D float array of wavelengths in microns
    """
    waves: list[float] = []
    with open(path, "r") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.replace(",", " ").split()
            val = None
            unit = None
            for tok in parts:
                t = tok.strip()
                tl = t.lower()
                try:
                    if tl.endswith(("um", "µm", "micron", "microns")):
                        unit = "um"
                        num = "".join(ch for ch in t if (ch.isdigit() or ch in ".-+eE"))
                        val = float(num)
                        break
                    if tl.endswith("nm"):
                        unit = "nm"
                        val = float(t[:-2])
                        break
                    if tl.endswith(("a", "ang", "angs", "angstrom", "angstroms")):
                        unit = "A"
                        num = "".join(ch for ch in t if (ch.isdigit() or ch in ".-+eE"))
                        val = float(num)
                        break
                    # bare number
                    val = float(t)
                    unit = None
                    break
                except Exception:
                    continue
            if val is None:
                continue
            if unit is None:
                # heuristic unit inference
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
