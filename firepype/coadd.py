# firepype/coadd.py
from __future__ import annotations

import numpy as np


class CoaddAccumulator:
    """
    Purpose:
        Maintain inverse-variance coaddition accumulator on a fixed wavelength
        grid. Add multiple spectra (already interpolated to this grid) along
        with their per-bin uncertainties and an optional validity mask; to
        accumulate weighted sums and producecco-added spectrum and errors
    Inputs:
        grid_wl (at init): 1D array of target wavelength grid in microns (or
            arbitrary units) onto which all spectra are interpolated
    Returns:
        CoaddAccumulator instance
    """

    def __init__(self, grid_wl: np.ndarray):
        """
        Purpose:
            Initialise coaddition accumulator for a specified wavelength grid
        Inputs:
            grid_wl: 1D array of wavelengths defining the coadd grid
        Returns:
            None
        """

        self.grid = np.asarray(grid_wl, float)
        self.flux_sum = np.zeros_like(self.grid, dtype=float)
        self.weight_sum = np.zeros_like(self.grid, dtype=float)

    def add_spectrum(
        self,
        flux: np.ndarray,
        err: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> None:
        """
        Purpose:
            Add single spectrum to accumulator. Spectrum should already be
            interpolated onto the accumulator's grid (self.grid). Accumulation is
            performed using inverse-variance weights
        Inputs:
            flux: 1D array of flux values on self.grid (same shape as grid)
            err: 1D array of 1-sigma uncertainties on self.grid (same shape)
            mask: Optional boolean mask (same shape) where True indicates a bin
                  should be used. If provided, it is combined with finite/positive
                  error masking
        Returns:
            None
        Raises:
            ValueError: If flux/err (or mask, when provided) shapes do not match
                        the grid shape
        """

        f = np.asarray(flux, float)
        e = np.asarray(err, float)

        if f.shape != self.grid.shape or e.shape != self.grid.shape:
            raise ValueError("flux/err must match grid shape")

        m = np.isfinite(f) & np.isfinite(e) & (e > 0)

        if mask is not None:
            mask = np.asarray(mask, bool)

            if mask.shape != self.grid.shape:
                raise ValueError("mask must match grid shape")

            m &= mask

        if not np.any(m):
            return

        w = 1.0 / np.maximum(e[m] ** 2, 1e-20)
        self.flux_sum[m] += w * f[m]
        self.weight_sum[m] += w


    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Purpose:
            Compute final coadded flux and error arrays from accumulated
            weighted sums
        Inputs:
            None
        Returns:
            tuple:
                - flux (np.ndarray): Weighted-mean flux on self.grid; NaN where no
                  samples contributed (weight == 0)
                - err (np.ndarray): 1-sigma uncertainty as sqrt(1/weight); NaN where
                  weight == 0
                - used (np.ndarray): Boolean mask where weight > 0 (bins used)
        """

        used = self.weight_sum > 0
        flux = np.full_like(self.grid, np.nan, dtype=float)
        err = np.full_like(self.grid, np.nan, dtype=float)
        flux[used] = self.flux_sum[used] / self.weight_sum[used]
        err[used] = np.sqrt(1.0 / self.weight_sum[used])

        return flux, err, used
