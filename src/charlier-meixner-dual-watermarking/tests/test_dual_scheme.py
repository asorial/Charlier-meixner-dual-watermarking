"""
Unit tests of the dual scheme that do not need the image dataset.

    python -m pytest tests/test_dual_scheme.py
"""
import contextlib
import io
import os
import sys

import numpy as np
import pytest
import imageio.v3 as iio

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dual_scheme.dual_watermarking_scheme import DualWatermarkingScheme  # noqa: E402
from helpers.orthonornal_matrix import orthmatrix  # noqa: E402
from helpers.utils import arnold_transform, binarize_watermark, unbinarize_watermark  # noqa: E402

KEY = "test-key"


def quiet(f, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*args, **kwargs)


@pytest.fixture(scope="module")
def cover():
    rng = np.random.default_rng(1)
    y, x = np.mgrid[0:512, 0:512]
    base = np.stack([120 + 60 * np.sin(x / 40), 110 + 50 * np.cos(y / 55), 100 + 40 * np.sin((x + y) / 70)], -1)
    return np.clip(base + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def logo():
    return iio.imread(os.path.join(ROOT, "src", "static", "watermark", "qr_asl.png"))


@pytest.fixture(scope="module", params=[0, 1, 2, 3], ids=["C I", "C II", "M I", "M II"])
def marked(request, cover, logo):
    A = orthmatrix(request.param)
    scheme = DualWatermarkingScheme(KEY, 150, 62, 0.58, 0.24, A, A)
    bits = binarize_watermark(arnold_transform(logo, 27).reshape(-1))
    return scheme, quiet(scheme.insert, cover, bits)


def test_arnold_period():
    w = np.arange(64 * 64).reshape(64, 64)
    assert np.array_equal(arnold_transform(arnold_transform(w, 27), 37), w)


def test_imperceptibility(cover, marked):
    _, wm = marked
    mse = np.mean((cover.astype(float) - wm.astype(float)) ** 2)
    assert 10 * np.log10(255 ** 2 / mse) > 35


def test_robust_watermark_without_attack(marked, logo):
    scheme, wm = marked
    raw = quiet(scheme.extraction, wm.copy(), "test.png", save_tampered=False)
    rec = arnold_transform(np.array(unbinarize_watermark([int(b) for b in raw])).reshape(64, 64), 37)
    assert np.mean(rec != logo) < 0.02


def test_no_false_alarm(marked):
    scheme, wm = marked
    assert scheme.authenticate(wm).sum() == 0


def test_wrong_key_rejects(marked):
    _, wm = marked
    other = DualWatermarkingScheme("another-key", 150, 62, 0.58, 0.24, orthmatrix(0), orthmatrix(0))
    assert other.authenticate(wm).mean() > 0.99


def test_single_pixel_change_is_localized(marked):
    scheme, wm = marked
    t = wm.copy()
    t[100, 205, 0] ^= 1                      # LSB of the red channel, block (12, 25)
    tamper_map = scheme.authenticate(t)
    assert tamper_map[12, 25] and tamper_map.sum() == 1


def test_block_relocation_is_detected(marked):
    scheme, wm = marked
    t = wm.copy()
    t[0:8, 0:8] = wm[8:16, 8:16]             # move one block: its tag is bound to the block index
    assert scheme.authenticate(t)[0, 0]
