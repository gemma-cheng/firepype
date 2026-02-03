# firepype/io.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from astropy.io import fits


def parse_id_list(spec: str) -> List[int]:
    """
    Parse a frame-ID specification like "1-4, 8, 10-7" into a sorted, unique
    list of integers: [1, 2, 3, 4, 7, 8, 9, 10].

    - Ranges are inclusive.
    - Ranges may be descending (e.g., "10-7").
    - Whitespace is ignored.
    - Duplicates are removed (e.g., "1-4,3-6" -> 1..6).
    """
    ids: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            a_str, b_str = token.split("-", 1)
            a = int(a_str.strip())
            b = int(b_str.strip())
            lo, hi = (a, b) if a <= b else (b, a)
            ids.update(range(lo, hi + 1))
        else:
            ids.add(int(token))
    return sorted(ids)
    

def pairs_from_ids(ids: Iterable[int]) -> List[Tuple[int, int]]:
    """
    Purpose:
        Form consecutive AB pairs from a list of frame IDs. If count is odd,
        last frame is dropped to maintain pairing
    Inputs:
        ids: Iterable of frame IDs (ints). Order doesn't matter
    Returns:
        list[tuple[int, int]]:
            List of (A_id, B_id) pairs formed consecutively after sorting
            e.g. [1,2,3,4] -> [(1,2),(3,4)]
    """

    ids_sorted = sorted(int(x) for x in ids)
    if len(ids_sorted) < 2:
        return []
    if len(ids_sorted) % 2 != 0:
        # Drop the last ID to maintain pairs
        ids_sorted = ids_sorted[:-1]
    return [(ids_sorted[i], ids_sorted[i + 1]) for i in range(0, len(ids_sorted), 2)]


def build_fire_path(
    base_dir: str | Path,
    num: int,
    prefix: str = "fire_",
    pad: int = 4,
    ext: str = ".fits",
) -> str:
    """
    Purpose:
        Build standard FIRE file path from components
        e.g. build_fire_path('/data/raw', 23) -> '/data/raw/fire_0023.fits
    Inputs:
        base_dir: Base directory containing the files
        num: Integer ID to be zero-padded
        prefix: Filename prefix (default 'fire_')
        pad: Zero-padding width for the ID (default 4)
        ext: File extension, including dot (default '.fits')
    Returns:
        The constructed file path as a POSIX string
    """

    base = Path(base_dir)
    return (base / f"{prefix}{num:0{pad}d}{ext}").as_posix()


def load_fits(path: str | Path):
    """
    Purpose:
        Load primary HDU data and header from a FITS file with validation
    Inputs:
        path: Path to the FITS file
    Returns:
        tuple:
            - data (np.ndarray): Primary HDU data as float32
            - header (fits.Header): Copy of the primary header
    Raises:
        FileNotFoundError: If FITS file does not exist
        ValueError: If FITS file has no primary data
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"FITS file not found: {p}")
    with fits.open(p.as_posix()) as hdul:
        data = hdul[0].data
        header = hdul[0].header.copy()
    if data is None:
        raise ValueError(f"No primary data in FITS file: {p}")
    return np.asarray(data, dtype=np.float32), header


def save_fits(path: str | Path, data, header):
    """
    Purpose:
        Save array and header as a primary FITS HDU. Parent directories
        are created if needed
    Inputs:
        path: Output file path
        data: Numpy array to write as primary image
        header: fits.Header to attach to the primary HDU
    Returns:
        Saves FITS file
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(p.as_posix(), data, header, overwrite=True)


def write_coadd_fits_with_header(
    path: str | Path,
    wl_um: np.ndarray,
    flux: np.ndarray,
    base_header: fits.Header,
    frames_used: List[int],
    spectra_used_count: int,
    wl_range: Tuple[float, float],
    grid_step: float,
    extra_history: List[str] | None = None,
):
    """
    Purpose:
        Write co-added spectrum to 2-HDU FITS file:
          - Primary HDU copies base_header and is augmented with metadata
          - Binary table HDU contains columns wavelength_um (D) and flux (E)
    Inputs:
        path: Output FITS path
        wl_um: 1D array of wavelengths in microns
        flux: 1D array of flux values
        base_header: Header to copy into the primary HDU
        frames_used: List of frame IDs that contributed to the coadd
        spectra_used_count: Number of spectra accumulated in the coadd
        wl_range: Tuple (wl_lo, wl_hi) in microns for metadata
        grid_step: Wavelength step (microns) of the coadd grid
        extra_history: Optional list of strings to append as HISTORY
    Returns:
        Saves FITS file with header metadata and table
    """

    wl = np.asarray(wl_um, float)
    fx = np.asarray(flux, np.float32)

    cols = [
        fits.Column(name="wavelength_um", array=wl, format="D"),
        fits.Column(name="flux", array=fx, format="E"),
    ]

    table_hdu = fits.BinTableHDU.from_columns(cols)
    table_hdu.header["TTYPE1"] = "wavelength_um"
    table_hdu.header["TTYPE2"] = "flux"
    table_hdu.header["BUNIT"] = ("arb", "Flux units (median counts arbitrary)")

    prim_hdu = fits.PrimaryHDU(header=base_header.copy())
    prim = prim_hdu.header
    prim["PIPESTEP"] = ("COADD", "This file is a co-added 1D spectrum")
    prim["COADDS"] = (int(spectra_used_count), "Number of spectra accumulated in coadd")
    if frames_used:
        prim["COADF1"] = (str(frames_used[0]), "First raw frame used")
        prim["COADFN"] = (str(frames_used[-1]), "Last raw frame used")
        prim["COADLST"] = (
            ",".join(map(str, frames_used))[:68],
            "List of frames (truncated)",
        )

    prim["WLRANGE"] = (
        f"{wl_range[0]:.3f}-{wl_range[1]:.3f}",
        "Wavelength range (micron)",
    )

    prim["DLAM"] = (float(grid_step), "Coadd wavelength step (micron)")

    if extra_history:
        for h in extra_history:
            s = str(h).encode("ascii", "replace").decode("ascii")
            s = s.replace("µm", "um").replace("μm", "um")
            prim["HISTORY"] = s

    hdul = fits.HDUList([prim_hdu, table_hdu])
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    hdul.writeto(p.as_posix(), overwrite=True)


def write_spectrum_with_err(
    path: str | Path,
    wl_um: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray | None,
    base_header: fits.Header,
    extra_history: List[str] | None = None,
):
    """
    Purpose:
        Write spectrum with optional errors to 2-HDU FITS file:
          - Primary HDU carries base_header (and optional HISTORY)
          - Binary table HDU includes wavelength_um, flux, and optional flux_err
    Inputs:
        path: Output FITS path
        wl_um: 1D array of wavelengths in microns
        flux: 1D array of flux values
        err: Optional 1D array of flux errors (same length as flux)
        base_header: Header to copy into primary HDU
        extra_history: Optional list of strings to append as HISTORY
    Returns:
        Saves FITS file with spectrum and error table
    """

    wl = np.asarray(wl_um, float)
    fx = np.asarray(flux, np.float32)

    cols = [
        fits.Column(name="wavelength_um", array=wl, format="D"),
        fits.Column(name="flux", array=fx, format="E"),
    ]

    if err is not None:
        ee = np.asarray(err, np.float32)
        cols.append(fits.Column(name="flux_err", array=ee, format="E"))

    table_hdu = fits.BinTableHDU.from_columns(cols)
    prim_hdu = fits.PrimaryHDU(header=base_header.copy())
    if extra_history:
        for h in extra_history:
            s = str(h).encode("ascii", "ignore").decode("ascii")
            prim_hdu.header["HISTORY"] = s

    hdul = fits.HDUList([prim_hdu, table_hdu])
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    hdul.writeto(p.as_posix(), overwrite=True)
