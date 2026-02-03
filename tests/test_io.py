# tests/test_io.py
import numpy as np
from pathlib import Path
from astropy.io import fits

from firepype.io import (
    parse_id_list,
    pairs_from_ids,
    build_fire_path,
    save_fits,
    load_fits,
)


def test_parse_and_pairs():
    ids = parse_id_list("1-4, 8, 10-7")
    assert ids == [1, 2, 3, 4, 7, 8, 9, 10]
    pairs = pairs_from_ids(ids)
    # last item dropped for odd count; here it's even so we have 4 pairs
    assert pairs == [(1, 2), (3, 4), (7, 8), (9, 10)]


def test_build_fire_path():
    p = build_fire_path("/data/raw", 97)
    assert p.endswith("/data/raw/fire_0097.fits")


def test_fits_roundtrip(tmp_path: Path):
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    hdr = fits.Header()
    hdr["TEST"] = True
    f = tmp_path / "x.fits"
    save_fits(f, data, hdr)
    data2, hdr2 = load_fits(f)
    assert np.allclose(data2, data)
    assert hdr2["TEST"] is True
