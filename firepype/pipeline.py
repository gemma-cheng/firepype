# firepype/pipeline.py
from __future__ import annotations

from pathlib import Path
import numpy as np

from .config import PipelineConfig
from .io import (
    parse_id_list,
    pairs_from_ids,
    build_fire_path,
    load_fits,
    save_fits,
    write_spectrum_with_err,
)

from .utils import (
    load_line_list_to_microns,
    assert_monotonic_and_align,
    orient_to_increasing,
    clean_bool_runs,
)

from .detection import (
    detect_slit_edges,
    detect_objects_in_slit,
    find_arc_trace_col_strong,
    estimate_parity,
    refine_neg_column_local,
    estimate_negative_scale_robust,
)

from .extraction import extract_with_local_bg, extract_cols_median_with_err
from .calibration import average_wavecal_across_cols, mask_interp_edge_artifacts
from .coadd import CoaddAccumulator
from .plotting import (
    plot_arc_trace_on_raw,
    plot_arc_1d_with_line_labels,
    plot_1d_spectrum,
)


def run_ab_pairs(cfg: PipelineConfig):
    """
    Purpose:
        Execute AB-pair near-IR reduction pipeline to produce co-added 1D spectrum.
        For each AB pair:
          - Load A and B frames, form A-B and B-A subtractions
          - Detect slit edges and object positions on subtracted frame
          - Choose strong arc column in the arc image near the object
          - Solve wavelength solution across nearby columns
          - Determine parity and extract positive/negative spectra with errors
          - Refine negative-object column and estimate scale factor
          - Combine positive and scaled-negative spectra
          - Reorient, clean, interpolate to grid, mask edges
          - Accumulate into a variance-weighted coadd
        Writes intermediate subtractions, optional QA plots, and a final coadd FITS
    Inputs:
        cfg: PipelineConfig
            Configuration object containing:
            - run: paths, user_spec (frame IDs), out_dir, arc_path, ref_list_path
            - slit: slit_x_hint, slit_hint_expand, row_fraction
            - extraction: ap_radius, bg_in, bg_out, footprint_half, row_fraction
            - wavecal: wl_range, grid_step, deg, max_sep, band_anchors_global
            - qa: qa_dir, save_figs, verbose
    Returns:
        tuple:
            - coadd_grid (np.ndarray): Common wavelength grid in microns
            - coadd_flux (np.ndarray): Co-added flux on coadd_grid
            - coadd_err (np.ndarray): 1-sigma uncertainties on coadd_grid
    Raises:
        RuntimeError:
            If no AB pairs can be formed from cfg.run.user_spec
    """

    # Parse frame IDs and build AB pairs
    ids = parse_id_list(cfg.run.user_spec)
    pairs = pairs_from_ids(ids)
    if not pairs:
        raise RuntimeError("No AB pairs formed. Provide an even number of frames in user_spec")

    # Prepare grid and accumulator
    wl_lo, wl_hi = cfg.wavecal.wl_range
    coadd_grid = np.arange(wl_lo, wl_hi + cfg.wavecal.grid_step / 2.0, cfg.wavecal.grid_step)
    coadd = CoaddAccumulator(coadd_grid)

    # Load calibration data
    refs = load_line_list_to_microns(str(cfg.run.ref_list_path))
    arc_data, _ = load_fits(str(cfg.run.arc_path))

    base_header = None
    spectra_used = 0

    # Ensure output dirs
    Path(cfg.run.out_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.qa.qa_dir).mkdir(parents=True, exist_ok=True)

    # Process AB pairs
    for (A_id, B_id) in pairs:
        A_path = build_fire_path(str(cfg.run.raw_dir), A_id)
        B_path = build_fire_path(str(cfg.run.raw_dir), B_id)

        if base_header is None:
            _, hdr0 = load_fits(A_path)
            base_header = hdr0

        A, hdrA = load_fits(A_path)
        B, hdrB = load_fits(B_path)

        # AB and BA subtractions
        A_sub = A - B
        B_sub = B - A

        # Save subtracted frames
        save_fits(Path(cfg.run.out_dir) / f"A_sub_{A_id}-{B_id}.fits", A_sub, hdrA)
        save_fits(Path(cfg.run.out_dir) / f"B_sub_{A_id}-{B_id}.fits", B_sub, hdrB)

        # Process A and B differences
        for tag, data_sub, hdr in [("A", A_sub, hdrA), ("B", B_sub, hdrB)]:

            # Slit edges and object positions
            left_edge, right_edge, *_ = detect_slit_edges(
                data_sub,
                x_hint=cfg.slit.slit_x_hint,
                hint_expand=cfg.slit.slit_hint_expand,
                row_frac=cfg.slit.row_fraction,
                debug=cfg.qa.verbose,
            )

            obj_pos_abs, obj_neg_abs, _, _ = detect_objects_in_slit(
                data_sub,
                left_edge,
                right_edge,
                row_frac=(0.40, 0.80),
                debug=cfg.qa.verbose,
            )

            # Choose strong arc column near the positive object
            arc_col = find_arc_trace_col_strong(
                arc_data,
                approx_col=obj_pos_abs,
                search_half=240,
                x_hint=cfg.slit.slit_x_hint,
                row_frac=cfg.extraction.row_fraction,
                debug_print=cfg.qa.verbose,
            )

            # Optional QA plots
            if cfg.qa.save_figs:
                plot_arc_trace_on_raw(
                    arc_data,
                    center_col=arc_col,
                    ap=int(cfg.extraction.ap_radius),
                    bg_in=int(cfg.extraction.bg_in),
                    bg_out=int(cfg.extraction.bg_out),
                    half=int(cfg.extraction.footprint_half),
                    row_frac=cfg.extraction.row_fraction,
                    title=f"ARC trace — {tag}_sub {A_id}-{B_id} (ARC col {arc_col})",
                    save_path=Path(cfg.qa.qa_dir)
                    / f"arc_trace_AUTO_{tag}_{A_id}-{B_id}_col{arc_col}.pdf",
                    show=False,
                )

                arc_1d = extract_with_local_bg(
                    arc_data,
                    arc_col,
                    ap=cfg.extraction.ap_radius,
                    bg_in=cfg.extraction.bg_in,
                    bg_out=cfg.extraction.bg_out,
                )

                plot_arc_1d_with_line_labels(
                    arc_1d,
                    wl_range=cfg.wavecal.wl_range,
                    ref_lines_um=refs,
                    anchors=cfg.wavecal.band_anchors_global,
                    solver_deg=cfg.wavecal.deg,
                    solver_max_sep=cfg.wavecal.max_sep,
                    arc_col=arc_col,
                    title_tag=f"{tag}_sub {A_id}-{B_id}",
                    save_path=Path(cfg.qa.qa_dir)
                    / f"arc_1d_with_labels_{tag}_{A_id}-{B_id}_col{arc_col}.pdf",
                    show=False,
                )

            # Wavecal
            wavelengths_per_pixel = average_wavecal_across_cols(
                arc_data,
                arc_col,
                half=int(cfg.extraction.footprint_half),
                ref_lines_um=refs,
                wl_range=cfg.wavecal.wl_range,
                anchors=cfg.wavecal.band_anchors_global,
                deg=cfg.wavecal.deg,
                max_sep=cfg.wavecal.max_sep,
            )

            # Parity and extraction
            par = estimate_parity(data_sub, arc_col, obj_neg_abs)
            data_sub_par = data_sub if par >= 0 else (-data_sub)

            pos_spec, pos_err = extract_cols_median_with_err(
                data_sub_par,
                arc_col,
                half=int(cfg.extraction.footprint_half),
                ap=int(cfg.extraction.ap_radius),
                bg_in=int(cfg.extraction.bg_in),
                bg_out=int(cfg.extraction.bg_out),
            )

            neg_spec, neg_err = extract_cols_median_with_err(
                data_sub_par,
                obj_neg_abs,
                half=int(cfg.extraction.footprint_half),
                ap=int(cfg.extraction.ap_radius),
                bg_in=int(cfg.extraction.bg_in),
                bg_out=int(cfg.extraction.bg_out),
            )

            # Refine negative column and scale
            obj_neg_ref = refine_neg_column_local(
                data_sub_par,
                arc_col,
                obj_neg_abs,
                search_half=8,
                ap=int(cfg.extraction.ap_radius),
            )

            g = estimate_negative_scale_robust(
                data_sub_par,
                arc_col,
                obj_neg_ref,
                ap=int(cfg.extraction.ap_radius),
                row_exclude_frac=(0.40, 0.80),
                g_limits=(0.1, 10.0),
            )

            # Combine and orient
            sci_combined = pos_spec - g * neg_spec
            sci_err = np.sqrt(pos_err**2 + (g * neg_err) ** 2)

            wavelengths_per_pixel, sci_combined = orient_to_increasing(
                wavelengths_per_pixel, sci_combined
            )

            _, sci_err = orient_to_increasing(wavelengths_per_pixel, sci_err)

            wl_m, sci_m = assert_monotonic_and_align(
                wavelengths_per_pixel, sci_combined, name=f"{tag}_{A_id}-{B_id}"
            )

            wl_m, err_m = assert_monotonic_and_align(
                wavelengths_per_pixel, sci_err, name=f"{tag}_ERR_{A_id}-{B_id}"
            )

            # Interpolate to coadd grid and mask edges
            if wl_m.size >= 5:
                K = 6
                if wl_m.size > 2 * K:
                    wl_trim = wl_m[K:-K]
                    sci_trim = sci_m[K:-K]
                    err_trim = err_m[K:-K]
                else:
                    wl_trim, sci_trim, err_trim = wl_m, sci_m, err_m

                f_interp = np.interp(
                    coadd_grid, wl_trim, sci_trim, left=np.nan, right=np.nan
                )

                e_interp = np.interp(
                    coadd_grid, wl_trim, err_trim, left=np.nan, right=np.nan
                )

                edge_mask = mask_interp_edge_artifacts(
                    coadd_grid,
                    wl_trim,
                    sci_trim,
                    err_trim,
                    min_span_px=10,
                    pad_bins=6,
                    min_keep_bins=18,
                )

                m_valid = (
                    edge_mask
                    & np.isfinite(f_interp)
                    & np.isfinite(e_interp)
                    & (e_interp > 0)
                )

                m_valid = clean_bool_runs(m_valid, min_run=18)

                if np.count_nonzero(m_valid) >= 5:
                    coadd.add_spectrum(f_interp, e_interp, mask=m_valid)
                    spectra_used += 1

    # Finalize coadd
    coadd_flux, coadd_err, coadd_mask = coadd.finalize()

    out_fits = (
        Path(cfg.run.out_dir)
        / f"coadd_spectrum_{cfg.run.user_spec.replace(',','-').replace(' ','')}.fits"
    )

    history = [
        f"Frames: {','.join(map(str, parse_id_list(cfg.run.user_spec)))[:60]}",
        f"WL_RANGE={cfg.wavecal.wl_range}, GRID_STEP={cfg.wavecal.grid_step}",
        "Wavecal: robust peak matching + Chebyshev fit; endpoint alignment; anchors",
        "Interpolation edges masked; small islands removed",
    ]

    write_spectrum_with_err(
        out_fits,
        coadd_grid.astype(float),
        coadd_flux.astype(np.float32),
        coadd_err.astype(np.float32),
        base_header=base_header if base_header is not None else None,
        extra_history=history,
    )

    # Optional final plot
    if cfg.qa.save_figs:
        mask = coadd_mask & np.isfinite(coadd_err) & (coadd_err > 0)
        pdf = (
            Path(cfg.qa.qa_dir)
            / f"coadd_{cfg.run.user_spec.replace(',','-').replace(' ','')}.pdf"
        )

        plot_1d_spectrum(
            coadd_grid[mask],
            coadd_flux[mask],
            "Co-added spectrum",
            pdf,
            xlabel="Wavelength (um)",
            ylabel="Flux (arb.)",
            show=False,
        )

    return coadd_grid, coadd_flux, coadd_err
