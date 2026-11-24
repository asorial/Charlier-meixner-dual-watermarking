"""
Experimento 4 - Complejidad computacional y tiempos de ejecución.

Mide, para C I, C II, M I, M II y DCT (mismo procedimiento; solo cambia la matriz A):
  * insert       inserción completa (capa robusta + capa frágil HMAC)
  * extract      extracción de la marca robusta (4096 bits) + Arnold inversa (marca 64x64)
  * verify       verificación de autenticidad (mapa de bloques manipulados)
  * memoria      pico de memoria de cada operación (tracemalloc, ejecuciones aparte)
en dos implementaciones que producen exactamente la misma imagen y los mismos bits:
  * reference    la clase del repositorio (src/dual_scheme/dual_watermarking_scheme.py), bucle por bloque
  * vectorized   NumPy por lotes (experiments/exp2_parameter_sensitivity/fastwm.py)

Además verifica empíricamente la complejidad:
  * escalado con el tamaño de bloque N (4, 8, 16, 32) en una imagen 512x512: transformada directa e
    inversa de todos los bloques, coste teórico O((HW/N^2) N^3) = O(HW N);
  * escalado con el tamaño de imagen (256^2 ... 2048^2, N = 8): inserción, extracción y verificación,
    coste teórico O(HW).

Repeticiones: 1 ejecución de calentamiento descartada + R ejecuciones por imagen; se informa de la
mediana y del rango intercuartílico (IQR).

Uso (desde la raíz del repositorio, con el mismo entorno virtual del proyecto):
    python experiments/exp4_complexity/run_exp4.py                 # completo (~10-15 min)
    python experiments/exp4_complexity/run_exp4.py --images 3 --reps 3 --quick   # prueba rápida
Para resultados comparables conviene cerrar otras aplicaciones y tener el portátil enchufado.
"""
import argparse, contextlib, csv, io, json, os, platform, sys, time, tracemalloc
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "exp2_parameter_sensitivity"))

import fastwm as F
from matrices import charlier_matrix, meixner_matrix
from dual_scheme.dual_watermarking_scheme import DualWatermarkingScheme
from helpers.blocks_class_3d import BlocksImageChs
from helpers.orthonornal_matrix import orthmatrix
from helpers.utils import binarize_watermark, arnold_transform, unbinarize_watermark
import imageio.v3 as iio
import hashlib, hmac
import cv2

KEY, DELTA, COEF, P, X0 = "N;fpr-y7", 150, 62, 0.58, 0.24
CONFIGS = ["C I", "C II", "M I", "M II", "DCT"]


def dct_matrix(N=8):
    x, n = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    A = np.cos(np.pi * (2 * x + 1) * n / (2 * N)) * np.sqrt(2 / N)
    A[:, 0] /= np.sqrt(2)
    return A


MATS = {"C I": orthmatrix(0), "C II": orthmatrix(1), "M I": orthmatrix(2), "M II": orthmatrix(3),
        "DCT": dct_matrix()}


# ------------------------------------------------------------------ implementación de referencia
def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def ref_insert(s, cover, bits_str):
    return quiet(s.insert, cover, bits_str)          # incluye la barra de progreso (redirigida)


def ref_extract(s, img):
    """Bucle de DualWatermarkingScheme.extraction() para la marca robusta + Arnold inversa."""
    blocks = BlocksImageChs(img)
    bits = "".join(s.extract_robust_block(b) for b in blocks.block_range_ch(s.CH))[:4096]
    wm = np.array(unbinarize_watermark([int(b) for b in bits])).reshape(64, 64)
    return arnold_transform(wm, 37)


def ref_verify(s, img):
    return s.authenticate(img)


# ------------------------------------------------------------------ implementación vectorizada
def vec_insert(fs, cover, bits):
    return fs.insert(cover, bits)


def vec_extract(fs, img):
    return F.recover_watermark(fs.extract_bits(img))


def vec_verify(fs, img):
    s = fs._ref
    H, W = img.shape[:2]
    blocks = img.reshape(H // 8, 8, W // 8, 8, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 8, 8, 3)
    ys, xs = zip(*s.carriers)
    lsb = (blocks[:, ys, xs, s.CH] & 1).astype(np.uint8)
    canon = blocks.copy(); canon[:, ys, xs, s.CH] &= 0xFE; canon = np.ascontiguousarray(canon)
    L = s.tag_bits; nbytes = (L + 7) // 8
    tags = np.empty((len(canon), L), np.uint8)
    for k in range(len(canon)):
        d = hmac.new(s.kappa, k.to_bytes(4, "big") + canon[k].tobytes(), hashlib.sha256).digest()
        tags[k] = np.unpackbits(np.frombuffer(d[:nbytes], np.uint8))[:L]
    return np.any(tags != lsb, axis=1).reshape(H // 8, W // 8)


# ------------------------------------------------------------------ utilidades de medida
def timeit(fn, *a):
    t = time.perf_counter(); r = fn(*a); return time.perf_counter() - t, r


def peak_mem(fn, *a):
    tracemalloc.start(); tracemalloc.reset_peak()
    fn(*a)
    peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    return peak / 2 ** 20


def max_rss_mb():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 2 ** 20 if sys.platform == "darwin" else r / 2 ** 10
    except Exception:
        try:
            import psutil; return psutil.Process().memory_info().peak_wset / 2 ** 20
        except Exception:
            return float("nan")


def system_info():
    info = {"os": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "logical_cpus": os.cpu_count(), "processor": platform.processor()}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip(); break
    except OSError:
        pass
    try:
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1)
    except (ValueError, OSError, AttributeError):
        try:
            import psutil; info["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
        except Exception:
            pass
    vers = {}
    for mod in ["numpy", "scipy", "cv2", "imageio", "mpmath", "PIL", "skimage"]:
        try:
            m = __import__(mod); vers[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    info["packages"] = vers
    try:
        cfg = io.StringIO()
        with contextlib.redirect_stdout(cfg):
            np.show_config()
        txt = cfg.getvalue().lower()
        info["blas"] = next((b for b in ["openblas", "mkl", "accelerate", "blis"] if b in txt), "unknown")
    except Exception:
        pass
    info["threads_env"] = {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]}
    return info


# ------------------------------------------------------------------ experimentos
def bench_operations(covers, names, bits_str, bits, reps, rows, mem_rows):
    for cfg in CONFIGS:
        A = MATS[cfg]
        s = DualWatermarkingScheme(KEY, DELTA, COEF, P, X0, A, A)
        fs = F.FastScheme(A, DELTA, COEF)
        for name, cover in zip(names, covers):
            wm_ref = ref_insert(s, cover, bits_str); wm_vec = vec_insert(fs, cover, bits)
            assert np.array_equal(wm_ref, wm_vec), "las dos implementaciones difieren"
            ops = {("reference", "insert"): (ref_insert, s, cover, bits_str),
                   ("reference", "extract"): (ref_extract, s, wm_ref),
                   ("reference", "verify"): (ref_verify, s, wm_ref),
                   ("vectorized", "insert"): (vec_insert, fs, cover, bits),
                   ("vectorized", "extract"): (vec_extract, fs, wm_vec),
                   ("vectorized", "verify"): (vec_verify, fs, wm_vec)}
            for (impl, op), (fn, *args) in ops.items():
                fn(*args)                                        # calentamiento
                for r in range(reps):
                    dt, _ = timeit(fn, *args)
                    rows.append(dict(config=cfg, image=name, impl=impl, op=op, rep=r, seconds=dt))
                mem_rows.append(dict(config=cfg, image=name, impl=impl, op=op, peak_mb=peak_mem(fn, *args)))
            print(f"  {cfg:5s} {name}", flush=True)


def transform_all_blocks(img_ch, A, N):
    H, W = img_ch.shape
    B = img_ch.reshape(H // N, N, W // N, N).transpose(0, 2, 1, 3).reshape(-1, N, N)
    M = A.T @ B @ A                                              # A^T B A (BLAS por lotes)
    return A @ M @ A.T                                           # A M A^T


def transform_loop(img_ch, A, N):
    H, W = img_ch.shape
    out = np.empty_like(img_ch)
    for i in range(0, H, N):
        for j in range(0, W, N):
            b = img_ch[i:i + N, j:j + N]
            out[i:i + N, j:j + N] = A @ (A.T @ b @ A) @ A.T
    return out


def bench_block_size(cover, reps, rows):
    ch = cover[..., 2].astype(float)
    for N in [4, 8, 16, 32]:
        mats = {"Charlier": charlier_matrix(7e-4, N), "Meixner": meixner_matrix(4.1e-5, 8e-4, N),
                "DCT": dct_matrix(N)}
        for fam, A in mats.items():
            for impl, fn in [("vectorized", transform_all_blocks), ("loop", transform_loop)]:
                fn(ch, A, N)
                for r in range(reps):
                    dt, _ = timeit(fn, ch, A, N)
                    rows.append(dict(N=N, family=fam, impl=impl, rep=r, seconds=dt,
                                     blocks=(512 // N) ** 2, flops=4 * 512 * 512 * N))
        print(f"  N = {N}", flush=True)


def bench_image_size(cover, bits_str, bits, reps, rows):
    A = MATS["C I"]
    s = DualWatermarkingScheme(KEY, DELTA, COEF, P, X0, A, A)
    for S in [256, 512, 1024, 2048]:
        img = cv2.resize(cover, (S, S), interpolation=cv2.INTER_CUBIC)
        nb = (S // 8) ** 2
        b_str = (bits_str * (nb // 4096 + 1))[:nb]               # un bit por bloque
        wm = ref_insert(s, img, b_str)
        for op, fn, args in [("insert", ref_insert, (s, img, b_str)),
                             ("extract", lambda s_, im: "".join(s_.extract_robust_block(b) for b in
                                                               BlocksImageChs(im).block_range_ch(s_.CH)), (s, wm)),
                             ("verify", ref_verify, (s, wm))]:
            for r in range(reps):
                dt, _ = timeit(fn, *args)
                rows.append(dict(size=S, pixels=S * S, blocks=nb, impl="reference", op=op, rep=r, seconds=dt))
        print(f"  {S}x{S}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=10, help="número de imágenes (orden alfabético)")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="omite el escalado (prueba rápida)")
    ap.add_argument("--label", default="", help="etiqueta de la máquina para los ficheros de salida")
    ap.add_argument("--only", choices=["blocksize"], help="repite solo una parte")
    a = ap.parse_args()
    res = os.path.join(HERE, "results" + (f"_{a.label}" if a.label else "")); os.makedirs(res, exist_ok=True)

    cdir = os.path.join(ROOT, "src", "static", "cover_images")
    names = sorted(f for f in os.listdir(cdir) if f.lower().endswith(".bmp"))[:a.images]
    covers = [iio.imread(os.path.join(cdir, n))[..., :3] for n in names]
    wm = iio.imread(os.path.join(ROOT, "src", "static", "watermark", "qr_asl.png"))
    bits_str = binarize_watermark(arnold_transform(wm, 27).reshape(-1))
    bits = np.array([int(b) for b in bits_str])

    info = system_info(); info.update(images=names, reps=a.reps, delta=DELTA, coef=COEF)
    json.dump(info, open(os.path.join(res, "system_info.json"), "w"), indent=2)
    print(json.dumps({k: info[k] for k in ["os", "cpu_model", "logical_cpus", "ram_gb", "python"] if k in info}))

    t0 = time.time()
    if a.only == "blocksize":
        rb = []; bench_block_size(covers[0], a.reps, rb); write(res, "exp4_block_size.csv", rb); return
    rows, mem = [], []
    print("Operaciones por configuración:"); bench_operations(covers, names, bits_str, bits, a.reps, rows, mem)
    write(res, "exp4_operations.csv", rows); write(res, "exp4_memory.csv", mem)
    if not a.quick:
        rb, ri = [], []
        print("Escalado con el tamaño de bloque:"); bench_block_size(covers[0], a.reps, rb)
        write(res, "exp4_block_size.csv", rb)
        print("Escalado con el tamaño de imagen:"); bench_image_size(covers[0], bits_str, bits, max(3, a.reps // 3), ri)
        write(res, "exp4_image_size.csv", ri)
    info["max_rss_mb"] = max_rss_mb(); info["total_seconds"] = round(time.time() - t0, 1)
    json.dump(info, open(os.path.join(res, "system_info.json"), "w"), indent=2)
    print(f"Hecho en {info['total_seconds']} s -> {res}")


def write(res, name, rows):
    with open(os.path.join(res, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
