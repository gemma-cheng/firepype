# firepype/detection.py
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .utils import skyline_mask_from_1d


def detect_slit_edges(
    data: np.ndarray,
    x_hint: Tuple[int, int] | None = None,
    hint_expand: int = 150,
    row_frac: Tuple[float, float] = (0.35, 0.85),
    debug: bool = False,
):
    """
    Purpose:
        Detect left and right slit edges in a 2D frame by forming median spatial
        profile over row band, high-pass filtering, and locating opposing gradient
        peaks indicative of slit boundaries. Uses optional x-range hint and
        adaptive recentring to focus search
    Inputs:
        data: 2D image array (rows x cols)
        x_hint: Optional tuple (xmin, xmax) of expected slit horizontal span (columns)
        hint_expand: Columns to expand around x_hint for search window (default 150)
        row_frac: Fractional row limits (lo, hi) used to build profile (default (0.35, 0.85))
        debug: Print diagnostic information
    Returns:
        tuple:
            - left_edge (int): Detected left slit edge column index
            - right_edge (int): Detected right slit edge column index
            - sm (np.ndarray): Smoothed, high-pass profile used for detection (per column)
            - g (np.ndarray): Smoothed gradient of sm used to find edge peaks
            - (lo, hi) (tuple[int, int]): Final column search window used
            - (r0, r1) (tuple[int, int]): Row band used to form median profile
    """

    nrows, ncols = data.shape
    r0 = int(max(0, min(row_frac[0], row_frac[1])) * nrows)
    r1 = int(min(1.0, max(row_frac[0], row_frac[1])) * nrows)
    band_rows = data[r0:r1, :]

    lo_global = int(0.03 * ncols)
    hi_global = int(0.97 * ncols)

    if x_hint is not None and len(x_hint) == 2:
        xmin = max(lo_global, int(min(x_hint)))
        xmax = min(hi_global, int(max(x_hint)))
        lo = max(lo_global, xmin - hint_expand)
        hi = min(hi_global, xmax + hint_expand)

    else:
        lo, hi = lo_global, hi_global

    prof_raw = np.median(band_rows, axis=0).astype(float)

    def band_recenter(profile, target_width_px):
        base = gaussian_filter1d(profile, 150, mode="nearest")
        hp = profile - base
        hp = gaussian_filter1d(hp, 3, mode="nearest")
        w = max(40, int(target_width_px))
        score = np.convolve(np.abs(hp), np.ones(w, float), mode="same")
        c = int(np.argmax(score))
        lo2 = max(lo_global, c - w // 2)
        hi2 = min(hi_global, c + w // 2)

        if hi2 - lo2 < 30:
            lo2 = max(lo_global, c - 30)
            hi2 = min(hi_global, c + 30)

        return lo2, hi2

    SLIT_TARGET_WIDTH_FRAC = 0.28

    if (hi - lo) > 0.5 * (hi_global - lo_global):
        lo, hi = band_recenter(prof_raw, SLIT_TARGET_WIDTH_FRAC * ncols)

    best = None

    for base_sig, sm_sig, grad_sig in [(120, 3.0, 2.2), (80, 2.2, 1.8)]:
        base_lo = gaussian_filter1d(prof_raw, base_sig, mode="nearest")
        hp = prof_raw - base_lo
        sm = gaussian_filter1d(hp, sm_sig, mode="nearest")
        g = gaussian_filter1d(np.gradient(sm), grad_sig, mode="nearest")

        def find_pairs(lo_i, hi_i, prom_pct=70, dist=12):
            if hi_i <= lo_i + 10:
                return []

            prom = np.percentile(np.abs(g[lo_i:hi_i]), prom_pct)
            L_idx, _ = find_peaks(
                g[lo_i:hi_i], prominence=float(max(prom, 1e-6)), distance=dist
            )

            R_idx, _ = find_peaks(
                -g[lo_i:hi_i], prominence=float(max(prom, 1e-6)), distance=dist
            )

            L = lo_i + L_idx
            R = lo_i + R_idx

            def score_pair(l, r):
                if r - l < 12:
                    return -np.inf

                interior = np.median(sm[l:r])
                left_bg = np.median(sm[max(lo_i, l - 60) : l])
                right_bg = np.median(sm[r : min(hi_i, r + 60)])
                bg = 0.5 * (left_bg + right_bg)
                contrast = abs(interior - bg)
                var_interior = np.median(np.abs(sm[l:r] - interior))
                steep_left = abs(g[l]) if 0 <= l < ncols else 0.0
                steep_right = abs(g[r]) if 0 <= r < ncols else 0.0

                return contrast + 0.5 * var_interior + 0.2 * (steep_left + steep_right)

            cand = []

            for l in L:
                for r in R[R > l + 8]:
                    cand.append((score_pair(int(l), int(r)), int(l), int(r)))

            cand.sort(reverse=True)

            return cand

        cand = find_pairs(lo, hi, prom_pct=70, dist=12)

        if not cand:
            cand = find_pairs(lo, hi, prom_pct=60, dist=8)

        if cand:
            chosen = cand[0]
            best = (chosen[1], chosen[2], sm, g, (lo, hi), (r0, r1))
            break

    if best is None:
        lo2, hi2 = band_recenter(prof_raw, SLIT_TARGET_WIDTH_FRAC * ncols)
        base_lo = gaussian_filter1d(prof_raw, 120, mode="nearest")
        hp = prof_raw - base_lo
        sm = gaussian_filter1d(hp, 3, mode="nearest")
        g = gaussian_filter1d(np.gradient(sm), 2.2, mode="nearest")
        window = 40
        Lcand = np.argmax(
            np.abs(g[max(lo_global, lo2 - window) : min(hi_global, lo2 + window)])
        )

        Rcand = np.argmax(
            np.abs(g[max(lo_global, hi2 - window) : min(hi_global, hi2 + window)])
        )

        left_edge = max(lo_global, lo2 - window) + int(Lcand)
        right_edge = max(lo_global, hi2 - window) + int(Rcand)

        if left_edge >= right_edge:
            left_edge, right_edge = lo2, hi2

        best = (left_edge, right_edge, sm, g, (lo2, hi2), (r0, r1))

    left_edge, right_edge, sm, g, (lo, hi), (r0, r1) = best

    if debug:
        print(
            f"[detect_slit_edges] rows={r0}:{r1}, final {left_edge}:{right_edge} "
            f"(W={right_edge-left_edge})"
        )

    return left_edge, right_edge, sm, g, (lo, hi), (r0, r1)


def detect_objects_in_slit(
    data: np.ndarray,
    left_edge: int,
    right_edge: int,
    row_frac: Tuple[float, float] = (0.40, 0.80),
    min_sep_frac: float = 0.06,
    edge_pad_frac: float = 0.04,
    debug: bool = False,
):
    """
    Purpose:
        Find approximate column positions of positive and negative object
        traces within the slit by analyzing smoothed median spatial profile,
        enforcing minimum separation and padding from slit edges
    Inputs:
        data: 2D image array (rows x cols)
        left_edge: Left slit edge column index
        right_edge: Right slit edge column index
        row_frac: Fractional row band used to compute median profile (default (0.40, 0.80))
        min_sep_frac: Minimum separation between pos/neg in units of slit width (default 0.06)
        edge_pad_frac: Padding from edges as fraction of slit width (default 0.04)
        debug: If True, print diagnostics
    Returns:
        tuple:
            - obj_pos_abs (int): Column of positive object peak
            - obj_neg_abs (int): Column of negative object trough
            - prof (np.ndarray): Smoothed median profile used for detection
            - (r0, r1) (tuple[int,int]): Row band used to build profile
    """

    nrows, _ = data.shape
    r0 = int(max(0, min(row_frac[0], row_frac[1])) * nrows)
    r1 = int(min(1.0, max(row_frac[0], row_frac[1])) * nrows)

    mid_lo = left_edge + int(0.20 * (right_edge - left_edge))
    mid_hi = right_edge - int(0.20 * (right_edge - left_edge))
    mid_lo = max(left_edge, mid_lo)
    mid_hi = min(right_edge, mid_hi)

    band = data[r0:r1, mid_lo : mid_hi + 1]
    prof = np.median(band, axis=0).astype(float)
    prof = gaussian_filter1d(prof, sigma=4.0)

    pos_rel = int(np.argmax(prof))
    neg_rel = int(np.argmin(prof))
    obj_pos_abs = mid_lo + pos_rel
    obj_neg_abs = mid_lo + neg_rel

    pad = max(2, int(edge_pad_frac * (right_edge - left_edge)))
    min_sep = max(3, int(min_sep_frac * (right_edge - left_edge)))

    def clamp_to_mid(x):
        return min(max(x, mid_lo + pad), mid_hi - pad)

    obj_pos_abs = clamp_to_mid(obj_pos_abs)
    obj_neg_abs = clamp_to_mid(obj_neg_abs)

    if abs(obj_pos_abs - obj_neg_abs) < min_sep:
        exc_lo = max(mid_lo, min(obj_pos_abs, obj_neg_abs) - min_sep // 2)
        exc_hi = min(mid_hi, max(obj_pos_abs, obj_neg_abs) + min_sep // 2)
        mask = np.ones_like(prof, dtype=bool)
        mask[(exc_lo - mid_lo) : (exc_hi - mid_lo + 1)] = False

        if prof[pos_rel] >= -prof[neg_rel]:
            cand = np.where(mask, prof, np.inf)
            neg_rel2 = int(np.argmin(cand))
            obj_neg_abs = clamp_to_mid(mid_lo + neg_rel2)

        else:
            cand = np.where(mask, -prof, np.inf)
            pos_rel2 = int(np.argmin(cand))
            obj_pos_abs = clamp_to_mid(mid_lo + pos_rel2)

    if obj_pos_abs == obj_neg_abs:
        order = np.argsort(prof)
        order_pos = np.argsort(-prof)

        for j in order:
            cand = mid_lo + int(j)

            if (
                abs(cand - obj_pos_abs) >= min_sep
                and (mid_lo + pad) <= cand <= (mid_hi - pad)
            ):
                obj_neg_abs = cand
                break

        for j in order_pos:
            cand = mid_lo + int(j)
            if (
                abs(cand - obj_neg_abs) >= min_sep
                and (mid_lo + pad) <= cand <= (mid_hi - pad)
            ):
                obj_pos_abs = cand
                break

    if debug:
        print(
            f"[detect_objects_in_slit] rows={r0}:{r1} POS={obj_pos_abs} "
            f"NEG={obj_neg_abs} sep={abs(obj_pos_abs - obj_neg_abs)}"
        )

    return obj_pos_abs, obj_neg_abs, prof, (r0, r1)


def find_arc_trace_col_strong(
    arc_img: np.ndarray,
    approx_col: int | None = None,
    *,
    search_half: int = 240,
    min_sep: int = 12,
    x_hint: Tuple[int, int] | None = None,
    row_frac: Tuple[float, float] = (0.35, 0.85),
    debug_print: bool = True,
) -> int:
    """
    Purpose:
        Select strong arc column for tracing by analyzing row-banded median
        spatial profile of arc image, high-pass filtering, and scoring peak
        candidates near optional approximate column
    Inputs:
        arc_img: 2D arc image array (rows x cols)
        approx_col: Optional approximate column index to bias selection near
        search_half: Half-width of search window around approx_col (default 240)
        min_sep: Minimum spacing between candidate peaks in pixels (default 12)
        x_hint: Optional (xmin, xmax) limit for analysis within columns
        row_frac: Fractional row band used to compute profile (default (0.35, 0.85))
        debug_print: Print chosen candidate details
    Returns:
        int:
            Selected column index best suited for arc tracing
    """

    img = np.asarray(arc_img, float)
    nrows, ncols = img.shape

    lo_x = int(max(0, (min(x_hint) if x_hint else 0)))
    hi_x = int(min(ncols, (max(x_hint) if x_hint else ncols)))

    r0 = int(min(row_frac) * nrows)
    r1 = int(max(row_frac) * nrows)
    band = img[r0:r1, lo_x:hi_x]
    prof = np.median(band, axis=0).astype(float)

    base = gaussian_filter1d(prof, 65, mode="nearest")
    hp = prof - base
    sm = gaussian_filter1d(hp, 3, mode="nearest")

    prom = max(np.nanpercentile(np.abs(sm), 98) * 0.6, 10.0)
    cand_idx, _ = find_peaks(
        sm, prominence=float(prom), distance=int(max(min_sep, 8))
    )

    if cand_idx.size == 0:
        prom2 = max(np.nanpercentile(np.abs(sm), 95) * 0.4, 5.0)
        cand_idx, _ = find_peaks(
            sm, prominence=float(prom2), distance=int(max(min_sep, 6))
        )

    cand_cols = (lo_x + cand_idx).astype(int)

    if approx_col is not None and cand_cols.size:
        cand_cols = cand_cols[
            np.abs(cand_cols - int(approx_col)) <= int(search_half)
        ]

    if cand_cols.size == 0:
        if approx_col is not None:
            lo = max(0, int(approx_col) - int(search_half))
            hi = min(ncols, int(approx_col) + int(search_half) + 1)

        else:
            lo, hi = lo_x, hi_x
        j = int(np.argmax(sm[(lo - lo_x) : (hi - lo_x)]))
        best_col = lo + j

        if debug_print:
            print(f"[arc-col] fallback peak at col {best_col}")

        return int(best_col)

    win = 5
    scores = []

    for c in cand_cols:
        j = int(np.clip(c - lo_x, 0, band.shape[1] - 1))
        j0 = max(0, j - win)
        j1 = min(band.shape[1], j + win + 1)
        scores.append(float(np.nanmedian(band[:, j0:j1], axis=0).max()))

    best_col = int(cand_cols[int(np.argmax(scores))])

    if debug_print:
        print(f"[arc-col] candidates={cand_cols.tolist()} -> chosen {best_col}")

    return best_col


def estimate_parity(
    img: np.ndarray, pos_col: int, neg_col: int, row_bands=((0.40, 0.60), (0.60, 0.80))
) -> int:
    """
    Purpose:
        Estimate AB parity from two row bands by comparing median brightness
        around positive and negative column positions
    Inputs:
        img: 2D image array (rows x cols)
        pos_col: Column index of the positive trace
        neg_col: Column index of the negative trace
        row_bands: Iterable of (lo, hi) fractional row bands to evaluate
    Returns:
        int:
            +1 if the positive is brighter in aggregate; otherwise -1
    """

    def one_band(rf):
        r0 = int(min(rf) * img.shape[0])
        r1 = int(max(rf) * img.shape[0])
        cpos0 = max(0, pos_col - 2)
        cpos1 = min(img.shape[1], pos_col + 3)
        cneg0 = max(0, neg_col - 2)
        cneg1 = min(img.shape[1], neg_col + 3)
        v_pos = float(np.nanmedian(img[r0:r1, cpos0:cpos1]))
        v_neg = float(np.nanmedian(img[r0:r1, cneg0:cneg1]))

        return +1 if v_pos >= v_neg else -1

    s = sum(one_band(b) for b in row_bands)

    return +1 if s >= 0 else -1


def estimate_negative_scale_robust(
    img: np.ndarray,
    pos_col: int,
    neg_col: int,
    ap: int = 5,
    row_exclude_frac: Tuple[float, float] = (0.40, 0.80),
    g_limits: Tuple[float, float] = (0.1, 10.0),
) -> float:
    """
    Purpose:
        Robustly estimate negative-beam scale g such that pos ≈ g * neg,
        using only sky rows (excluding a central band) and masking rows with
        strong skylines detected in either beam
    Inputs:
        img: 2D image array (rows, cols)
        pos_col: Column index of positive trace
        neg_col: Column index of negative trace
        ap: Half-width of per-column extraction aperture (default 5)
        row_exclude_frac: Fractional row band to exclude as containing object light (default (0.40, 0.80))
        g_limits: Tuple of (min, max) bounds to clamp final g (default (0.1, 10.0))
    Returns:
        float:
            Robust estimate of g, clamped to g_limits
    """

    nrows = img.shape[0]
    pos = np.median(img[:, max(0, pos_col - ap) : pos_col + ap + 1], axis=1).astype(
        float
    )

    neg = np.median(img[:, max(0, neg_col - ap) : neg_col + ap + 1], axis=1).astype(
        float
    )

    r0e = int(min(row_exclude_frac) * nrows)
    r1e = int(max(row_exclude_frac) * nrows)
    sky_rows = np.r_[0:r0e, r1e:nrows]

    sky_bad = skyline_mask_from_1d(pos, sigma_hi=3.5) | skyline_mask_from_1d(
        neg, sigma_hi=3.5
    )

    m = np.ones(nrows, dtype=bool)
    m[sky_rows] = True
    m &= ~sky_bad
    m &= np.isfinite(pos) & np.isfinite(neg) & (np.abs(neg) > 1e-9)

    if np.count_nonzero(m) < 40:
        m = np.isfinite(pos) & np.isfinite(neg) & (np.abs(neg) > 1e-9)
        center = np.zeros(nrows, dtype=bool)
        center[r0e:r1e] = True
        m &= ~center

    if np.count_nonzero(m) < 20:
        m = np.isfinite(pos) & np.isfinite(neg) & (np.abs(neg) > 1e-9)

    ratios = pos[m] / neg[m]
    med = np.nanmedian(ratios)
    mad = np.nanmedian(np.abs(ratios - med)) + 1e-12
    keep = np.abs(ratios - med) <= 3.5 * 1.4826 * mad
    ratios_t = ratios[keep] if np.any(keep) else ratios
    g = float(np.nanmedian(ratios_t))
    g = float(np.clip(g, g_limits[0], g_limits[1]))

    return g


def refine_neg_column_local(
    img: np.ndarray, pos_col: int, neg_col_init: int, *, search_half: int = 10, ap: int = 5
) -> int:
    """
    Purpose:
        Refine negative-column position by local search within +/- search_half
        columns to minimise median-absolute-deviation (MAD) of residual
        pos - g * neg, where g is robustly re-estimated at each candidate
    Inputs:
        img: 2D image array (rows, cols)
        pos_col: Column index of positive trace
        neg_col_init: Initial column index for negative trace
        search_half: Half-width of integer search window (default 10)
        ap: Half-width of per-column extraction aperture (default 5)
    Returns:
        int:
            Best-fit negative column index that minimizes residual MAD
    """

    ncols = img.shape[1]
    candidates = range(
        max(0, neg_col_init - search_half), min(ncols, neg_col_init + search_half + 1)
    )

    best_col, best_score = neg_col_init, -np.inf

    for cneg in candidates:
        g = estimate_negative_scale_robust(img, pos_col, cneg, ap=ap)
        pos = np.median(
            img[:, max(0, pos_col - ap) : pos_col + ap + 1], axis=1
        )

        neg = np.median(
            img[:, max(0, cneg - ap) : cneg + ap + 1], axis=1
        )

        resid = pos - g * neg
        mad = np.nanmedian(np.abs(resid - np.nanmedian(resid))) + 1e-12
        score = -mad

        if score > best_score:
            best_score, best_col = score, cneg

    return int(best_col)
