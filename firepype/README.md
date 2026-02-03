# firepype

Python pipeline for Magellan/FIRE Prism-mode data. Experimental; validated on a limited dataset. Verify outputs against established pipelines (e.g. [FireHose_v2](https://github.com/jgagneastro/FireHose_v2/)).

## Features

- Slit edge and object detection with robust heuristics
- Arc 1D extraction and line matching
- Robust Chebyshev dispersion solution with optional anchors
- Parity-aware A–B/B–A differencing and robust negative-beam scaling
- Footprint median extraction with error estimates
- Interpolation-edge masking and inverse-variance coaddition
- Telluric correction:
  - POS-only standard extraction
  - Band-wise scaling with deep-gap masking
  - Vega model broadened to the instrument resolution
  - Transmission (T) smoothing only inside dense contiguous regions
- Optional QA plots: arc overlays, labeled arc 1D, final coadd, telluric T(λ), corrected spectra

## Installation

Installation (PyPi):

```python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install firepype
```

Alternatively, install from source:

```python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
git clone https://github.com/gemma-cheng/firepype
cd firepype
pip install -e .[dev]
```

Requirements:
- Python 3.10–3.12
- numpy ≥ 1.22, scipy ≥ 1.9, astropy ≥ 5.2, matplotlib ≥ 3.6


## Quickstart

Minimal end-to-end reduction:

```firepype \
  --raw-dir /path/to/raw \
  --out-dir ./out \
  --arc /path/to/raw/fire_0123.fits \
  --ref-list /path/to/ref/line_list.lst \
  --spec "1-4, 10-7"
```

Telluric/response only:
```firepype-telluric \
  --standard ./out/standard_extracted.fits \
  --out-dir ./out/telluric \
  --stype A0V --plot
```

Outputs
- ./output/qa/ … if QA enabled
- Coadded spectrum FITS: wavelength_um, flux
- Telluric and response FITS in out/telluric/


## Basic Tutorials

Basic usage tutorials can be found in the [`tutorials`](./tutorials/) directory


## Notes on telluric method

- Standard extraction: POS-only column, tuned aperture/background; alternative beam used if POS median is non-positive
- Wavecal: average across a small footprint around the chosen standard column, using ARC + line list
- Vega model: broadened to instrument resolution (R ~ 6000 by default) in log-λ space
- Continuum: robust Chebyshev fit to the standard/model ratio within each band (J/H/K), excluding deep telluric gaps and known A0V intrinsic lines; fit is used to normalize before deriving T
- Transmission T(λ): computed and lightly smoothed only within dense contiguous support (prevents spreading across gaps)
- Application: science flux and errors are divided by T within overlap where T_min ≤ T ≤ T_max; elsewhere, values are left as NaN to avoid artifacts


## Limitations and validation

- Tested on a limited set of FIRE Prism-mode observations; results are not guaranteed.
- Validate outputs against established pipelines (e.g. [FireHose_v2](https://github.com/jgagneastro/FireHose_v2/)):
  - Wavelength RMS per region
  - Sky residuals around OH lines
  - Merged-order continuity
  - S/N consistency


## License

MIT (see [`LICENSE`](./LICENSE)).


## File Structure

- `firepype/`
  - `__init__.py` — package API (exposes `run_ab_pairs`, `apply_telluric_correction`)
  - `config.py` — dataclasses for configuration
  - `io.py` — FITS I/O, path builders, ID pairing
  - `utils.py` — math helpers, masks, line-list loader
  - `calibration.py` — peak detection, line matching, dispersion solver
  - `detection.py` — slit/object detection, parity, negative scaling
  - `extraction.py` — 1D extraction routines
  - `coadd.py` — coaddition accumulator
  - `plotting.py` — optional QA plotting
  - `pipeline.py` — high-level AB-pair orchestration
  - `telluric.py` — telluric correction API
  - `cli.py` — command-line interface (pipeline + telluric subcommand if enabled)
- `tests/` — minimal tests for core functionality
- `tutorials/` — basic walk-through of how to use `firepype`
- `pyproject.toml` — packaging configuration
- `README.md` — this file
- `LICENSE`
