# tests/test_utils.py
import numpy as np

from firepype.utils import (
    cheb_design_matrix,
    robust_weights,
    orient_to_increasing,
    assert_monotonic_and_align,
    clean_bool_runs,
    mask_interp_edge_artifacts,
    load_line_list_to_microns,
)


def test_orient_and_align():
    wl = np.array([2.0, 1.0, 0.5])
    fx = np.array([10.0, 11.0, 12.0])
    wl2, fx2 = orient_to_increasing(wl, fx)
    assert wl2[0] < wl2[-1]
    wls, fxs = assert_monotonic_and_align(wl2, fx2)
    assert np.all(np.diff(wls) > 0)
    assert fxs.shape == wls.shape


def test_cheb_design_matrix():
    x = np.linspace(-1, 1, 5)
    X = cheb_design_matrix(x, deg=3)
    assert X.shape == (5, 4)
    # T0 == 1
    assert np.allclose(X[:, 0], 1.0)


def test_robust_weights_basic():
    r = np.array([0.0, 0.1, 10.0])
    w = robust_weights(r)
    assert w[0] > w[-1]


def test_mask_interp_edge_artifacts():
    grid = np.linspace(0.0, 10.0, 101)
    native = np.linspace(2.0, 8.0, 51)
    keep = mask_interp_edge_artifacts(grid, native, None, None, pad_bins=5, min_keep_bins=10)
    assert keep.sum() > 0
    assert keep[:5].sum() == 0
    assert keep[-5:].sum() == 0


def test_line_list_loader(tmp_path):
    p = tmp_path / "lines.lst"
    p.write_text("# test\n1.00 um\n500 nm\n10000 A\n")
    arr = load_line_list_to_microns(p)
    # Should be 1.0, 0.5, 1.0 in microns sorted -> [0.5, 1.0, 1.0]
    assert np.allclose(arr, np.array([0.5, 1.0, 1.0]))
