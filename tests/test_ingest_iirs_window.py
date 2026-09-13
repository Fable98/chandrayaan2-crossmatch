"""Unit tests for memory-safe windowed IIRS loading in ingest_and_prepare.

The full 250x5574x256 IIRS cube (~1.4GB float32 + PCA temporaries) OOM-kills
small hosts. These tests pin the window math that lets the pipeline read only
the pixels covering the region/large-AOI box.

Heavy pipeline deps (geopandas/pandas/pyproj/...) are stubbed: the helpers
under test only need rasterio's Window (installed) plus stdlib.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "data_preprocessing_pipeline" / "scripts"
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Stub heavy modules BEFORE importing the script under test.
_STUBBED_MODULES = [
    "select_triplets",
    "lunar_pipeline",
    "lunar_pipeline.ingest",
    "lunar_pipeline.sensors",
    "lunar_pipeline.illumination",
    "pyproj",
]
for _name in _STUBBED_MODULES:
    if _name not in sys.modules:
        sys.modules[_name] = MagicMock()


@pytest.fixture(scope="module", autouse=True)
def _cleanup_stubbed_modules():
    yield
    for _name in _STUBBED_MODULES:
        sys.modules.pop(_name, None)


import ingest_and_prepare as iap  # noqa: E402

# Real PDS4 footprints of the SIH DATA triplet (see triplets.json entry 2).
IIRS_FP = {
    "west_lon": 336.484986,
    "east_lon": 337.156567,
    "south_lat": -15.529913,
    "north_lat": -0.236739,
}
OHRC_FP = {
    "west_lon": 336.484646,
    "east_lon": 336.589455,
    "south_lat": -3.416904,
    "north_lat": -2.576048,
}
TMC_FP = {
    "west_lon": 336.227537,
    "east_lon": 339.161649,
    "south_lat": -5.057164,
    "north_lat": 43.131132,
}
IIRS_W, IIRS_H = 250, 5574


def _meta(fp):
    m = MagicMock()
    m.footprint = dict(fp)
    return m


def test_large_aoi_lonlat_matches_production_values():
    box = iap._large_aoi_lonlat(_meta(OHRC_FP), _meta(TMC_FP), _meta(IIRS_FP))
    lw, le, ls, ln = box
    assert lw == pytest.approx(336.484986)  # max(IIRS.w, TMC.w)
    assert le == pytest.approx(337.156567)  # min(IIRS.e, TMC.e)
    assert ls == pytest.approx(-3.326476, abs=1e-4)  # OHRC center - 0.33
    assert ln == pytest.approx(-2.666476, abs=1e-4)  # OHRC center + 0.33


def test_window_covers_need_box_with_pad():
    need = (336.50, -3.30, 336.58, -2.70)
    win, (ww, ws, we, wn) = iap.iirs_window_for_lonlat(IIRS_FP, need, IIRS_W, IIRS_H)
    # Window lon/lat must contain the need box.
    assert ww <= need[0] and we >= need[2]
    assert ws <= need[1] and wn >= need[3]
    # Pixels stay inside the raster.
    assert 0 <= win.col_off and win.col_off + win.width <= IIRS_W
    assert 0 <= win.row_off and win.row_off + win.height <= IIRS_H
    assert win.width > 0 and win.height > 0


def test_window_is_small_fraction_of_full_frame():
    box = iap._large_aoi_lonlat(_meta(OHRC_FP), _meta(TMC_FP), _meta(IIRS_FP))
    need = (box[0], box[2], box[1], box[3])
    win, _ = iap.iirs_window_for_lonlat(IIRS_FP, need, IIRS_W, IIRS_H)
    frac = (win.width * win.height) / (IIRS_W * IIRS_H)
    assert frac < 0.06, f"window is {frac:.3%} of full frame (OOM risk)"
    # Sanity on absolute size for this triplet: the large AOI spans nearly
    # the full IIRS swath width (~250 cols) but only ~0.66 deg of latitude
    # (~250 rows of 5574) -> ~65MB float32 instead of ~1.4GB.
    assert win.width <= IIRS_W
    assert win.height <= 400


def test_window_roundtrip_pixel_mapping():
    """Window bounds invert the north-up linear mapping exactly."""
    need = (336.50, -3.30, 336.58, -2.70)
    win, (ww, ws, we, wn) = iap.iirs_window_for_lonlat(
        IIRS_FP, need, IIRS_W, IIRS_H, pad=0
    )
    span_lon = IIRS_FP["east_lon"] - IIRS_FP["west_lon"]
    span_lat = IIRS_FP["north_lat"] - IIRS_FP["south_lat"]
    assert ww == pytest.approx(IIRS_FP["west_lon"] + win.col_off / IIRS_W * span_lon)
    assert we == pytest.approx(IIRS_FP["west_lon"] + (win.col_off + win.width) / IIRS_W * span_lon)
    assert wn == pytest.approx(IIRS_FP["north_lat"] - win.row_off / IIRS_H * span_lat)
    assert ws == pytest.approx(
        IIRS_FP["north_lat"] - (win.row_off + win.height) / IIRS_H * span_lat
    )


def test_window_rejects_non_intersecting_box():
    with pytest.raises(ValueError, match="does not intersect"):
        iap.iirs_window_for_lonlat(IIRS_FP, (10.0, 10.0, 11.0, 11.0), IIRS_W, IIRS_H)


def test_window_rejects_degenerate_footprint():
    bad = dict(IIRS_FP, east_lon=IIRS_FP["west_lon"])
    with pytest.raises(ValueError, match="Degenerate"):
        iap.iirs_window_for_lonlat(bad, (336.5, -3.3, 336.6, -2.7), IIRS_W, IIRS_H)


# --- in-place PCA (sensors._pca_bands, loaded bypassing package __init__) ---


def _load_real_sensors():
    import importlib.util

    path = (
        REPO_ROOT
        / "data_preprocessing_pipeline"
        / "lunar_pipeline"
        / "sensors.py"
    )
    spec = importlib.util.spec_from_file_location("iirs_sensors_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pca_in_place_matches_copy():
    import numpy as _np

    sensors = _load_real_sensors()
    rng = _np.random.default_rng(1)
    arr = (rng.random((8, 30, 25)).astype("float32") * 100 + 500)
    want = sensors.iirs_reduce(arr.copy(), mode="pca", n_components=1)
    got = sensors.iirs_reduce(arr, mode="pca", n_components=1, in_place=True)
    _np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-4)
    assert got.shape == (1, 30, 25)
    # Default path leaves the input untouched (shared-lib contract).
    untouched = (rng.random((8, 30, 25)).astype("float32") * 100 + 500)
    snapshot = untouched.copy()
    sensors.iirs_reduce(untouched, mode="pca", n_components=1)
    _np.testing.assert_array_equal(untouched, snapshot)


# --- _read_window_bounded ----------------------------------------------------

_RWIN_TF = None  # built per-test from rasterio.transform


def _write_synth_tif(path, w=200, h=160, smooth=False):
    import numpy as _np
    import rasterio
    from rasterio.transform import from_bounds as _fb

    if smooth:
        # Low-frequency terrain-like signal: the honest case for comparing two
        # resampling paths (white noise is adversarial to any interpolator).
        yy, xx = _np.mgrid[0:h, 0:w].astype("float64")
        arr = (128 + 60 * _np.sin(xx / 17.0) * _np.cos(yy / 23.0)).astype("uint8")
    else:
        rng = _np.random.default_rng(0)
        arr = (rng.random((h, w)) * 255).astype("uint8")
    tf = _fb(0.0, 0.0, float(w), float(h), w, h)
    with rasterio.open(
        str(path), "w", driver="GTiff", width=w, height=h, count=1,
        dtype="uint8", transform=tf,
    ) as dst:
        dst.write(arr, 1)
    return arr, tf


def test_bounded_read_small_window_is_bit_identical(tmp_path):
    import numpy as _np
    import rasterio
    from rasterio.windows import Window as _W

    src_arr, _ = _write_synth_tif(tmp_path / "s.tif")
    with rasterio.open(tmp_path / "s.tif") as src:
        win = _W(10, 20, 50, 40)
        got, tf = iap._read_window_bounded(src, win)
        want = src.read(1, window=win)
    _np.testing.assert_array_equal(got, want)
    assert got.shape == (40, 50)
    # Transform matches the native window transform exactly.
    assert tuple(tf) == tuple(rasterio.windows.transform(win, src.transform))


def test_bounded_read_huge_window_stays_small_and_close(tmp_path, monkeypatch):
    import numpy as _np
    import rasterio
    from rasterio.windows import Window as _W
    from rasterio.enums import Resampling as _R

    src_arr, _ = _write_synth_tif(tmp_path / "big.tif", w=300, h=240, smooth=True)
    monkeypatch.setattr(iap, "_STRIPE_BYTES", 4096)  # force many stripes
    monkeypatch.setattr(iap, "_READ_WINDOW_LONG_SIDE_CAP", 64)
    with rasterio.open(tmp_path / "big.tif") as src:
        full = _W(0, 0, src.width, src.height)
        got, tf = iap._read_window_bounded(src, full)
        # Aspect-preserving: 300x240 capped at long side 64 -> 64x51.
        want = src.read(1, window=full, out_shape=(51, 64), resampling=_R.bilinear)
    assert got.shape == (51, 64)
    # Striped path is an approximation: close on smooth signal, exact geometry.
    diff = _np.abs(got.astype("int16") - want.astype("int16"))
    assert diff.max() <= 6
    assert diff.mean() <= 1.5
    # Scaled transform maps array pixels back onto the full window.
    assert tf.a == pytest.approx(src.transform.a * 300 / 64)
    assert abs(tf.e) == pytest.approx(abs(src.transform.e) * 240 / 51)
