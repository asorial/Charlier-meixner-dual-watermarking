"""
Implementación vectorizada (NumPy) del esquema dual de src/dual_scheme/dual_watermarking_scheme.py,
para barridos de parámetros. Produce exactamente la misma imagen marcada y los mismos bits que la
clase del repositorio (se comprueba en check_equivalence()).

Capa robusta: momento en zigzag `coef` de cada bloque 8x8 del canal azul, Dither Modulation con
paso `delta` (d0 = -delta/4, d1 = +delta/4), reconstrucción con redondeo y recorte a [0, 255].
Capa frágil: HMAC-SHA256 de 16 bits por bloque (métodos de la clase del repositorio).
"""
import os, sys, io, hmac, hashlib
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dual_scheme.dual_watermarking_scheme import DualWatermarkingScheme
from helpers.utils import matrix_zig_zag, arnold_transform
from helpers.metrics import Metrics
from attacks.noises import gaussian_noise, salt_pepper_noise

KEY, P, X0 = "N;fpr-y7", 0.58, 0.24
CH = 2
ZZ = matrix_zig_zag().reshape(-1)          # vac[k] = bloque.flat[ZZ[k]]
_METR = Metrics()


def to_blocks(ch):
    return ch.reshape(64, 8, 64, 8).transpose(0, 2, 1, 3).reshape(4096, 8, 8)


def from_blocks(b):
    return b.reshape(64, 64, 8, 8).transpose(0, 2, 1, 3).reshape(512, 512)


def _q(x, delta):
    return delta * np.round(x / delta)      # np.round = round-half-even, como round() de Python


def dm_embed(x, bits, delta):
    d = np.where(bits == 0, -delta / 4, delta / 4)
    return _q(x + d, delta) - d


def dm_extract(x, delta):
    e0 = np.abs(dm_embed(x, np.zeros_like(x, dtype=int), delta) - x)
    e1 = np.abs(dm_embed(x, np.ones_like(x, dtype=int), delta) - x)
    return (e1 < e0).astype(np.uint8)       # empate -> 0 (np.argmin)


class FastScheme:
    def __init__(self, A, delta=150, coef=62, tag_bits=16, overflow_safe=False):
        self.A, self.delta, self.coef = np.asarray(A, float), float(delta), int(coef)
        self.overflow_safe = overflow_safe
        self.pos = ZZ[self.coef]
        self.r, self.c = divmod(self.pos, 8)
        self._ref = DualWatermarkingScheme(KEY, delta, coef, P, X0, self.A, self.A, tag_bits)

    def insert(self, cover, bits):
        out = cover.copy()
        B = to_blocks(cover[..., CH].astype(float))
        M = self.A.T @ B @ self.A                                      # A^T B A (BLAS por lotes)
        x = M[:, self.r, self.c].copy()
        y = dm_embed(x, bits, self.delta)
        M[:, self.r, self.c] = y
        R = self.A @ M @ self.A.T                                      # A M A^T
        if self.overflow_safe:
            # Variante: si tras redondear y recortar a [0,255] el bit se decodificaría mal o con un
            # margen menor que Delta/8 respecto de la frontera de decisión, usar el punto de la misma
            # subred (y -/+ Delta) más cercano a x que cumpla ambas condiciones.
            def ok(Rc, yc):
                v = np.einsum("j,bjk,k->b", self.A[:, self.r], np.clip(np.rint(Rc), 0, 255),
                              self.A[:, self.c])
                margin = self.delta / 4 - np.abs(v - dm_embed(v, bits, self.delta))
                return (dm_extract(v, self.delta) == bits) & (margin >= self.delta / 8)
            best_R = R.copy()
            best_cost = np.where(ok(R, y), np.abs(y - x), np.inf)
            for s_ in (-1, 1):
                M2 = M.copy(); M2[:, self.r, self.c] = y + s_ * self.delta
                R2 = self.A @ M2 @ self.A.T
                cost = np.where(ok(R2, y + s_ * self.delta), np.abs(y + s_ * self.delta - x), np.inf)
                take = cost < best_cost
                best_R[take], best_cost[take] = R2[take], cost[take]
            R = best_R
        out[..., CH] = from_blocks(np.clip(np.rint(R), 0, 255)).astype(np.uint8)
        self._write_tags(out)                                          # capa frágil (HMAC)
        return out

    @staticmethod
    def _overflow(R):
        return (R.max(axis=(1, 2)) > 255.5) | (R.min(axis=(1, 2)) < -0.5)

    def _write_tags(self, out):
        """Equivalente vectorizado de write_tag(block_tag(k, bloque)) para los 4096 bloques."""
        s = self._ref
        blocks = out.reshape(64, 8, 64, 8, 3).transpose(0, 2, 1, 3, 4).reshape(4096, 8, 8, 3)
        canon = blocks.copy()
        ys, xs = zip(*s.carriers)
        canon[:, ys, xs, CH] &= 0xFE
        canon = np.ascontiguousarray(canon)
        L = s.tag_bits
        tags = np.empty((4096, L), np.uint8)
        for k in range(4096):
            d = hmac.new(s.kappa, k.to_bytes(4, "big") + canon[k].tobytes(), hashlib.sha256).digest()
            tags[k] = np.unpackbits(np.frombuffer(d[:(L + 7) // 8], np.uint8))[:L]
        canon[:, ys, xs, CH] |= tags
        out[...] = canon.reshape(64, 64, 8, 8, 3).transpose(0, 2, 1, 3, 4).reshape(512, 512, 3)

    def extract_bits(self, img):
        B = to_blocks(img[..., CH].astype(float))
        a, c = self.A[:, self.r], self.A[:, self.c]
        v = np.einsum("j,bjk,k->b", a, B, c)                           # solo el coeficiente usado
        return dm_extract(v, self.delta)


# --------------------------------------------------------------------------- marca y métricas
_ARNOLD_INV = None


def arnold_inverse_perm():
    global _ARNOLD_INV
    if _ARNOLD_INV is None:
        idx = np.arange(4096).reshape(64, 64)
        _ARNOLD_INV = arnold_transform(idx, 37).reshape(-1)
    return _ARNOLD_INV


def recover_watermark(bits):
    """bits (4096) -> marca 64x64 en {0,255} tras Arnold inversa (37 iteraciones), como extract.py."""
    img = (bits.astype(np.uint8) * 255)
    return img[arnold_inverse_perm()].reshape(64, 64)


def ber_ncc(wm_ref, wm_ext):
    """Igual que Metrics.calculate_ber_ncc para marcas binarias {0,255} (Otsu no las altera)."""
    a, b = wm_ref.astype(np.float64).ravel(), wm_ext.astype(np.float64).ravel()
    ber = float(np.mean(a != b))
    if np.all(a == a[0]) and np.all(b == b[0]):
        return ber, 1.0 if a[0] == b[0] else 0.0
    if np.all(b == b[0]):
        return ber, 0.0
    return ber, float(np.corrcoef(a, b)[0, 1])


def ber_ncc_repo(wm_ref, wm_ext):
    return _METR.calculate_ber_ncc(wm_ref, wm_ext)


def psnr(a, b):
    return _METR.psnr(a, b)


def ssim_paper(a, b):
    return float(_METR.ssim(a, b))


def ssim_standard(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


# --------------------------------------------------------------------------- ataques (del repositorio)
def attack(img, kind, level, seed):
    if kind == "none":
        return img
    np.random.seed(seed)
    if kind == "gaussian":
        return gaussian_noise(img.copy(), percent_noise=level)
    if kind == "salt_pepper":
        return np.clip(salt_pepper_noise(img.copy(), density=level), 0, 255).astype(np.uint8)
    if kind == "jpeg":
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=int(level))
        return np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    raise ValueError(kind)


def check_equivalence(cover, bits_str, A, delta=150, coef=62):
    import contextlib
    ref = DualWatermarkingScheme(KEY, delta, coef, P, X0, A, A)
    with contextlib.redirect_stdout(io.StringIO()):
        w_ref = ref.insert(cover, bits_str)
        b_ref = ref.extraction(w_ref.copy(), "eq.bmp", save_tampered=False)
    fs = FastScheme(A, delta, coef)
    w = fs.insert(cover, np.array([int(b) for b in bits_str]))
    b = "".join(map(str, fs.extract_bits(w)))
    return bool(np.array_equal(w, w_ref)), b == b_ref
