# firepype/cli.py
from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig, RunSpec
from .pipeline import run_ab_pairs


def main():
    """
    Purpose:
        Command-line entry point for the FIRE AB-pair NIR reduction pipeline.
        Parses CLI arguments, constructs PipelineConfig (applying optional
        overrides), and executes AB-pair processing to produce co-added
        spectrum and QA outputs
    Inputs:
        None (arguments are read from sys.argv via argparse)
    Returns:
        None
            Runs the pipeline and writes outputs to disk. Exits with argparse
            error messages on invalid usage
    CLI options:
        --raw-dir         Directory with raw FITS files (required)
        --out-dir         Output directory for products and QA (required)
        --arc             Path to ARC frame FITS (required)
        --ref-list        Path to reference line list file (required)
        --spec            Frame spec string for AB pairing (required)
        --no-qa           Disable QA plots (optional flag)
        --ap-radius       Extraction aperture half-width in pixels (optional)
        --bg-in           Inner background offset in pixels (optional)
        --bg-out          Outer background offset in pixels (optional)
        --footprint-half  Half-size (columns) for footprint median combine (optional)
        --wl-min          Wavelength range minimum in microns (optional)
        --wl-max          Wavelength range maximum in microns (optional)
        --grid-step       Wavelength grid step for coadd in microns (optional)
        --verbose         Enable verbose diagnostic prints (optional)
    """

    p = argparse.ArgumentParser(
        prog="firepype",
        description="FIRE AB-pair NIR reduction pipeline",
    )

    p.add_argument(
        "--raw-dir",
        required=True,
        help="Directory containing raw FITS files (e.g., /data/raw)",
    )

    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for products and QA (e.g., ./out)",
    )

    p.add_argument(
        "--arc",
        required=True,
        help="Path to the ARC frame FITS (e.g., /data/raw/fire_XXXX.fits)",
    )

    p.add_argument(
        "--ref-list",
        required=True,
        help="Path to the line list file (e.g., /data/ref/line_list.lst)",
    )

    p.add_argument(
        "--spec",
        required=True,
        help='Frame spec string for AB pairing, e.g. "XXXX-XXXX" or "XX-XX,XX-XX"',
    )

    p.add_argument(
        "--no-qa",
        action="store_true",
        help="Disable QA plots",
    )

    # Optional default overrides
    p.add_argument(
        "--ap-radius", type=int, default=None, help="Extraction aperture half-width (px)"
    )

    p.add_argument(
        "--bg-in", type=int, default=None, help="Inner background offset (px)"
    )

    p.add_argument(
        "--bg-out", type=int, default=None, help="Outer background offset (px)"
    )

    p.add_argument(
        "--footprint-half",
        type=int,
        default=None,
        help="Half-size (in columns) for footprint median combine",
    )

    p.add_argument(
        "--wl-min", type=float, default=None, help="Wavelength range min (micron)"
    )
    p.add_argument(
        "--wl-max", type=float, default=None, help="Wavelength range max (micron)"
    )
    p.add_argument(
        "--grid-step", type=float, default=None, help="Grid step for coadd (micron)"
    )
    p.add_argument(
        "--verbose", action="store_true", help="Verbose detection/diagnostic prints"
    )

    args = p.parse_args()

    # Build config with defaults, then override if provided
    cfg = PipelineConfig(
        run=RunSpec(
            raw_dir=Path(args.raw_dir),
            out_dir=Path(args.out_dir),
            arc_path=Path(args.arc),
            ref_list_path=Path(args.ref_list),
            user_spec=args.spec,
        )
    )

    cfg.qa.save_figs = not args.no_qa
    cfg.qa.verbose = bool(args.verbose)
    cfg.qa.out_dir = Path(args.out_dir)
    cfg.qa.qa_dir = Path(args.out_dir) / "qa"

    if args.ap_radius is not None:
        cfg.extraction.ap_radius = int(args.ap_radius)

    if args.bg_in is not None:
        cfg.extraction.bg_in = int(args.bg_in)

    if args.bg_out is not None:
        cfg.extraction.bg_out = int(args.bg_out)

    if args.footprint_half is not None:
        cfg.extraction.footprint_half = int(args.footprint_half)

    if args.wl_min is not None or args.wl_max is not None:
        wl_lo = args.wl_min if args.wl_min is not None else cfg.wavecal.wl_range[0]
        wl_hi = args.wl_max if args.wl_max is not None else cfg.wavecal.wl_range[1]
        cfg.wavecal.wl_range = (float(wl_lo), float(wl_hi))

    if args.grid_step is not None:
        cfg.wavecal.grid_step = float(args.grid_step)

    run_ab_pairs(cfg)

def main_telluric() -> None:
    """
    Telluric/response CLI.

    Minimal workflow:
      - Read a standard-star exposure (or coadd) FITS.
      - Optionally read a reference spectrum (A0V) to derive response.
      - Fit/derive telluric transmission and instrument response.
      - Write out telluric.fits and response.fits (or as configured).
    """
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="firepype-telluric",
        description="Telluric + instrument response from a standard star",
    )
    
    parser.add_argument(
        "--standard",
        required=True,
        help="Path to standard star FITS (2D or extracted 1D).",
    )
    
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for telluric/response products.",
    )
    
    parser.add_argument(
        "--ref-spectrum",
        default=None,
        help="Optional path to reference spectrum for the standard "
             "(e.g., A0V template). If omitted, use built-in or empirical.",
    )
    
    parser.add_argument(
        "--stype",
        default="A0V",
        help='Standard star spectral type (default: "A0V").',
    )
    
    parser.add_argument(
        "--airmass",
        type=float,
        default=None,
        help="Airmass to use for telluric modeling (override header if given).",
    )
    
    parser.add_argument(
        "--wl-min",
        type=float,
        default=None,
        help="Minimum wavelength (micron) to include.",
    )
    
    parser.add_argument(
        "--wl-max",
        type=float,
        default=None,
        help="Maximum wavelength (micron) to include.",
    )
    
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save diagnostic plots to out-dir.",
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )

    args = parser.parse_args()

    std_path = Path(args.standard).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    ref_path = (
        None
        if args.ref_spectrum is None
        else Path(args.ref_spectrum).expanduser().resolve()
    )

    # Basic validations
    if not std_path.exists():
        print(f"error: standard file not found: {std_path}", file=sys.stderr)
        sys.exit(2)
    
    if ref_path is not None and not ref_path.exists():
        print(f"error: reference spectrum not found: {ref_path}", file=sys.stderr)
        sys.exit(2)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Default output filenames
    telluric_out = out_dir / "telluric.fits"
    response_out = out_dir / "response.fits"
    qa_dir = out_dir / "qa"

    if not args.overwrite:
        for p in (telluric_out, response_out):
            if p.exists():
                print(f"error: output exists (use --overwrite): {p}", file=sys.stderr)
                sys.exit(2)

    # Import and run the actual logic
    try:
        from .telluric import run_telluric
    except Exception as e:
        print(f"error: could not import telluric runner: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        run_telluric(
            standard_fits=std_path,
            stype=args.stype,
            out_telluric=telluric_out,
            out_response=response_out,
            ref_spectrum=ref_path,
            airmass=args.airmass,
            wl_min=args.wl_min,
            wl_max=args.wl_max,
            save_plots=args.plot,
            qa_dir=qa_dir,
            overwrite=args.overwrite,
        )
    
    except Exception as e:
        print(f"error: telluric processing failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"wrote: {telluric_out}")
    print(f"wrote: {response_out}")
    if args.plot:
        print(f"qa:    {qa_dir}")

if __name__ == "__main__":
    main()
