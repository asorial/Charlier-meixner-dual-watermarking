"""
Prueba de la capa frágil integrada en dual_scheme/dual_watermarking_scheme.py.
Pasos: (1) imagen original, (2) inserción robusta + frágil, (3-4) verificación sin
modificar -> todos los bloques auténticos, (5) un píxel modificado -> solo su bloque.

Uso (desde la raíz del repositorio):
    python tests/test_fragile_integration.py src/static/cover_images/C0309.bmp
"""
import sys, os, io, contextlib, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from dual_scheme.dual_watermarking_scheme import DualWatermarkingScheme
from helpers.orthonornal_matrix import orthmatrix
from helpers.utils import arnold_transform, binarize_watermark, unbinarize_watermark
import imageio.v3 as iio

COVER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "src/static/cover_images/C0309.bmp")
WMARK = os.path.join(HERE, "..", "src/static/watermark/qr_asl.png")
rng = np.random.default_rng(2026)


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        r = f(*a, **k)
    return r, buf.getvalue()


cover = iio.imread(COVER)
w = iio.imread(WMARK)
bits = binarize_watermark(arnold_transform(w, 27).reshape(-1))
ok_all = True
for name, mid in {"C I": 0, "C II": 1, "M I": 2, "M II": 3}.items():
    s = DualWatermarkingScheme("N;fpr-y7", 150, 62, 0.58, 0.24, orthmatrix(mid), orthmatrix(mid))
    wm, _ = quiet(s.insert, cover, bits)
    tmp = os.path.join(tempfile.gettempdir(), "_integration.bmp")
    iio.imwrite(tmp, wm)
    wm = iio.imread(tmp)
    psnr = 10 * np.log10(255 ** 2 / np.mean((cover.astype(float) - wm.astype(float)) ** 2))

    # 3-4
    rb, out = quiet(s.extraction, wm.copy(), "integration.bmp", save_tampered=False)
    n_flag = int(s.last_tamper_map.sum())
    rec = arnold_transform(np.array(unbinarize_watermark([int(b) for b in rb])).reshape(64, 64), 37)
    ber = float(np.mean(rec.reshape(-1) != w.reshape(-1)))
    step4 = n_flag == 0

    # 5: 300 modificaciones de un píxel en bloque, posición y canal aleatorios
    hits = exact = 0
    for _ in range(300):
        k = int(rng.integers(4096)); i, j = divmod(k, 64)
        y, x, c = 8 * i + int(rng.integers(8)), 8 * j + int(rng.integers(8)), int(rng.integers(3))
        t = wm.copy(); t[y, x, c] = t[y, x, c] ^ 1 if rng.random() < .5 else (int(t[y, x, c]) + 37) % 256
        T = s.authenticate(t)
        hits += bool(T[i, j]); exact += bool(T[i, j]) and int(T.sum()) == 1
    step5 = exact == 300 or exact >= 299   # admite una colisión 2^-16
    ok_all &= step4 and step5
    print(f"{name}: PSNR={psnr:.2f} dB | sin modificar: {n_flag}/4096 bloques rechazados "
          f"({'OK' if step4 else 'FALLO'}) | BER robusta={ber:.4f} | "
          f"1 píxel: {hits}/300 detectados, {exact}/300 localizados en su bloque exacto "
          f"({'OK' if step5 else 'FALLO'})")
print("RESULTADO GLOBAL:", "OK" if ok_all else "FALLO")
sys.exit(0 if ok_all else 1)
