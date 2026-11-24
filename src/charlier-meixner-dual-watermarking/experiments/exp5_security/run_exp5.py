"""
Experimento 5 - Seguridad y sensibilidad a las claves (análisis de amenazas realista).

Modelo: el algoritmo es público (principio de Kerckhoffs). Secretos posibles:
  * K    clave de la capa frágil; kappa = SHA-256(K) es la clave del HMAC-SHA256 truncado a 16 bits.
  * eta  iteraciones de Arnold de la marca robusta (periodo 64 -> 63 valores no triviales).
  * Delta, coeficiente, matriz A: parámetros de la capa robusta (no son claves criptográficas).
La capa robusta no usa K: cualquiera que conozca (eta, Delta, coeficiente) puede leerla o alterarla.

Pruebas (24 imágenes, C I salvo indicación; FastScheme = implementación bit-idéntica a la clase):
  T1  clave correcta: bloques rechazados y BER robusta.
  T2  claves frágiles incorrectas: 100 claves aleatorias, las claves a un carácter de K y kappa con
      un bit cambiado -> % de bloques rechazados en una imagen auténtica.
  T3  extracción robusta con K, eta, Delta, coeficiente o matriz incorrectos -> BER, NCC.
  T4  búsqueda exhaustiva de eta por un atacante que no conoce la marca: puntuación de estructura
      (fracción de píxeles vecinos iguales) de las 63 candidatas; ¿la máxima es la correcta?
  T5  falsificación: bloque modificado + etiqueta aleatoria de 16 bits, 2^20 intentos -> tasa de
      aceptación (IC 95 % de Clopper-Pearson) frente a 2^-16.
  T6  ataques a la capa frágil: inversión de LSB portadores, LSB aleatorios en toda la imagen,
      copy-move de bloques completos (con su etiqueta), collage entre imágenes marcadas con la misma
      clave (alineado a la rejilla y desalineado).
  T7  eliminación/sobrescritura de la marca robusta por un atacante que conoce el algoritmo:
      llevar el coeficiente a la frontera de decisión, invertir bits al azar, sobrescribir con otra
      marca; y, como referencia ciega, mediana 3x3 -> PSNR del ataque, BER, % bloques rechazados.

Uso (desde la raíz del repositorio):
    python experiments/exp5_security/run_exp5.py [--covers DIR]
"""
import argparse, hashlib, hmac, json, os, string, sys, time
import numpy as np
from scipy import ndimage, stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "exp2_parameter_sensitivity"))
import fastwm as F
from matrices import charlier_matrix
from helpers.orthonornal_matrix import orthmatrix
from helpers.utils import binarize_watermark, arnold_transform
from helpers.dat import DAT
import imageio.v3 as iio

KEY, DELTA, COEF, ETA = "N;fpr-y7", 150, 62, 27
RNG = np.random.default_rng(20261120)
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)


def dct_matrix(N=8):
    x, n = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    A = np.cos(np.pi * (2 * x + 1) * n / (2 * N)) * np.sqrt(2 / N); A[:, 0] /= np.sqrt(2)
    return A


# ------------------------------------------------------------------ Arnold: potencias de la permutación
P1 = DAT().dat2(np.arange(4096).reshape(64, 64)).reshape(-1)      # out[i] = in[P1[i]]


def arnold_perm(n):
    cur = np.arange(4096)
    for _ in range(n % 64):
        cur = cur[P1]
    return cur


INV = {e: arnold_perm(64 - e) for e in range(1, 64)}               # inversa de eta iteraciones


def decode(bits, eta):
    return (bits.astype(np.uint8) * 255)[INV[eta]].reshape(64, 64)


def structure_score(img):
    """Fracción de pares de vecinos (horizontal y vertical) iguales: ~0.5 para ruido, alta para un QR."""
    b = img > 127
    return float((np.mean(b[:, 1:] == b[:, :-1]) + np.mean(b[1:, :] == b[:-1, :])) / 2)


# ------------------------------------------------------------------ capa frágil vectorizada (clave arbitraria)
CARRIERS = [(i // 8, i % 8) for i in range(16)]
YS, XS = zip(*CARRIERS)


def blocks_of(img):
    return img.reshape(64, 8, 64, 8, 3).transpose(0, 2, 1, 3, 4).reshape(4096, 8, 8, 3)


def tags(kappa, blocks, idx=None):
    canon = blocks.copy(); canon[:, YS, XS, 2] &= 0xFE; canon = np.ascontiguousarray(canon)
    idx = range(len(canon)) if idx is None else idx
    out = np.empty((len(canon), 16), np.uint8)
    for j, k in enumerate(idx):
        d = hmac.new(kappa, int(k).to_bytes(4, "big") + canon[j].tobytes(), hashlib.sha256).digest()
        out[j] = np.unpackbits(np.frombuffer(d[:2], np.uint8))
    return out


def verify(img, kappa):
    b = blocks_of(img)
    return np.any(tags(kappa, b) != (b[:, YS, XS, 2] & 1), axis=1)   # True = rechazado


def kappa_of(key):
    return hashlib.sha256(key.encode("utf-8")).digest()


def psnr(a, b):
    return F.psnr(a, b)


def ber_ncc(ref, ext):
    return F.ber_ncc(ref, ext)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--covers", default=os.path.join(ROOT, "src", "static", "cover_images"))
    ap.add_argument("--images", type=int, default=24)
    ap.add_argument("--forgeries", type=int, default=2 ** 20)
    a = ap.parse_args()
    names = sorted(f for f in os.listdir(a.covers) if f.lower().endswith(".bmp"))[:a.images]
    covers = [iio.imread(os.path.join(a.covers, n))[..., :3] for n in names]
    wm = iio.imread(os.path.join(ROOT, "src", "static", "watermark", "qr_asl.png"))
    bits = np.array([int(b) for b in binarize_watermark(arnold_transform(wm, ETA).reshape(-1))])
    A = orthmatrix(0)
    fs = F.FastScheme(A, DELTA, COEF)
    kap = kappa_of(KEY)
    marked = [fs.insert(c, bits) for c in covers]
    out, t0 = {"images": names, "key": KEY, "eta": ETA, "delta": DELTA, "coef": COEF}, time.time()

    # T1 -------------------------------------------------------------------------------------
    t1 = [(verify(w, kap).mean(), *ber_ncc(wm, decode(fs.extract_bits(w), ETA))) for w in marked]
    out["T1_correct_key"] = dict(rejected_blocks=float(np.mean([x[0] for x in t1])),
                                 ber=float(np.mean([x[1] for x in t1])), ncc=float(np.mean([x[2] for x in t1])))
    print("T1", out["T1_correct_key"], flush=True)

    # T2 -------------------------------------------------------------------------------------
    alphabet = string.ascii_letters + string.digits + string.punctuation
    rand_keys = ["".join(RNG.choice(list(alphabet), 8)) for _ in range(100)]
    one_char = []
    for i in range(len(KEY)):
        c = KEY[i]; nxt = alphabet[(alphabet.index(c) + 1) % len(alphabet)]
        one_char.append(KEY[:i] + nxt + KEY[i + 1:])
    flipped_kappas = []
    for bit in [0, 77, 128, 255]:
        k = bytearray(kap); k[bit // 8] ^= 1 << (bit % 8); flipped_kappas.append(bytes(k))
    sub = marked[:6]                                              # 6 imágenes x claves
    rej = lambda kp: [float(verify(w, kp).mean()) for w in sub]
    r_rand = np.array([rej(kappa_of(k)) for k in rand_keys])
    r_one = np.array([rej(kappa_of(k)) for k in one_char])
    r_bit = np.array([rej(k) for k in flipped_kappas])
    out["T2_wrong_fragile_key"] = dict(
        random_keys=dict(n_keys=len(rand_keys), n_images=len(sub), mean_rejected=float(r_rand.mean()),
                         min_rejected=float(r_rand.min())),
        one_character_keys=dict(keys=one_char, mean_rejected=float(r_one.mean()), min_rejected=float(r_one.min())),
        one_bit_kappa=dict(bits=[0, 77, 128, 255], mean_rejected=float(r_bit.mean()), min_rejected=float(r_bit.min())),
        expected_rejected=1 - 2 ** -16)
    print("T2", {k: (v["mean_rejected"] if isinstance(v, dict) else v) for k, v in out["T2_wrong_fragile_key"].items()}, flush=True)

    # T3 -------------------------------------------------------------------------------------
    rows = []
    def robust(label, group, fs_ext, eta=ETA):
        r = [ber_ncc(wm, decode(fs_ext.extract_bits(w), eta)) for w in marked]
        rows.append(dict(group=group, setting=label, ber=float(np.mean([x[0] for x in r])),
                         ber_sd=float(np.std([x[0] for x in r], ddof=1)), ncc=float(np.mean([x[1] for x in r]))))
    robust("correct parameters", "reference", fs)
    robust("wrong fragile key K (robust layer does not use K)", "K", fs)
    for e in [26, 28, 1, 37, 63]:
        robust(f"Arnold eta = {e}", "eta", fs, eta=e)
    for d in [75, 140, 148, 152, 160, 300]:
        robust(f"Delta = {d}", "Delta", F.FastScheme(A, d, COEF))
    for k in [61, 63, 53, 0]:
        robust(f"coefficient {k}", "coef", F.FastScheme(A, DELTA, k))
    for lab, M in [("C II matrix", orthmatrix(1)), ("M I matrix", orthmatrix(2)), ("M II matrix", orthmatrix(3)),
                   ("Charlier a x0.1", charlier_matrix(7e-5)), ("Charlier a x10", charlier_matrix(7e-3)),
                   ("DCT", dct_matrix())]:
        robust(lab, "matrix", F.FastScheme(M, DELTA, COEF))
    out["T3_robust_parameters"] = rows
    for r in rows:
        print("T3", r["setting"], round(r["ber"], 4), round(r["ncc"], 3), flush=True)

    # T4 -------------------------------------------------------------------------------------
    t4 = []
    for name, w in zip(names, marked):
        for cond, img in [("none", w), ("gaussian 0.10", F.attack(w, "gaussian", 0.10, 7)),
                          ("salt_pepper 0.05", F.attack(w, "salt_pepper", 0.05, 7))]:
            bts = fs.extract_bits(img)
            for e in range(1, 64):
                dec = decode(bts, e)
                b_, n_ = ber_ncc(wm, dec)
                t4.append(dict(image=name, attack=cond, eta=e, structure=structure_score(dec), ber=b_, ncc=n_))
    import pandas as pd
    t4 = pd.DataFrame(t4); t4.to_csv(os.path.join(RES, "exp5_arnold_bruteforce.csv"), index=False)
    best = t4.loc[t4.groupby(["image", "attack"]).structure.idxmax()]
    out["T4_arnold_bruteforce"] = dict(
        candidates=63, success_rate_by_attack=best.groupby("attack").apply(lambda g: float((g.eta == ETA).mean())).to_dict(),
        structure_correct_mean=float(t4[t4.eta == ETA].structure.mean()),
        structure_wrong_mean=float(t4[t4.eta != ETA].structure.mean()),
        structure_wrong_max=float(t4[t4.eta != ETA].structure.max()))
    print("T4", out["T4_arnold_bruteforce"], flush=True)

    # T5 -------------------------------------------------------------------------------------
    n = a.forgeries; w = marked[0]; b = blocks_of(w)
    ks = RNG.integers(0, 4096, n)
    forged = b[ks].copy()
    forged[:, 4, 4, 0] ^= 1                                       # modificación del contenido (fuera de portadores)
    rnd_tags = RNG.integers(0, 2, (n, 16), dtype=np.uint8)
    forged[:, YS, XS, 2] = (forged[:, YS, XS, 2] & 0xFE) | rnd_tags
    t_true = tags(kap, forged, ks)
    acc = int(np.sum(np.all(t_true == rnd_tags, axis=1)))
    ci = stats.beta.ppf([0.025, 0.975], [acc, acc + 1], [n - acc + 1, n - acc]) if acc else [0.0, 1 - 0.025 ** (1 / n)]
    out["T5_forgery"] = dict(trials=n, accepted=acc, rate=acc / n, ci95=[float(ci[0]), float(ci[1])],
                             theoretical=2 ** -16, expected_accepted=n * 2 ** -16)
    print("T5", out["T5_forgery"], flush=True)

    # T6 -------------------------------------------------------------------------------------
    t6 = {}
    w0, w1 = marked[0], marked[1]
    x = w0.copy(); bx = blocks_of(x); bx[:, YS, XS, 2] ^= 1
    x = bx.reshape(64, 64, 8, 8, 3).transpose(0, 2, 1, 3, 4).reshape(512, 512, 3)
    t6["flip_all_carrier_lsbs"] = float(verify(x, kap).mean())
    x = (w0 & 0xFE) | RNG.integers(0, 2, w0.shape, dtype=np.uint8)
    t6["random_lsb_plane_all_pixels"] = float(verify(x, kap).mean())
    det_cm, det_al, det_mis = [], [], []
    for i in range(len(marked)):
        src, dst = marked[i], marked[(i + 1) % len(marked)]
        y0, x0 = RNG.integers(0, 448, 2) // 8 * 8                  # región 64x64 alineada a la rejilla
        ys, xs = RNG.integers(0, 448, 2) // 8 * 8
        c = dst.copy(); c[y0:y0 + 64, x0:x0 + 64] = dst[ys:ys + 64, xs:xs + 64]   # copy-move (misma imagen)
        det_cm.append(verify(c, kap).reshape(64, 64)[y0 // 8:y0 // 8 + 8, x0 // 8:x0 // 8 + 8].mean())
        c = dst.copy(); c[y0:y0 + 64, x0:x0 + 64] = src[y0:y0 + 64, x0:x0 + 64]   # collage, misma posición
        det_al.append(verify(c, kap).reshape(64, 64)[y0 // 8:y0 // 8 + 8, x0 // 8:x0 // 8 + 8].mean())
        c = dst.copy(); c[y0:y0 + 64, x0:x0 + 64] = src[ys:ys + 64, xs:xs + 64]   # collage, otra posición
        det_mis.append(verify(c, kap).reshape(64, 64)[y0 // 8:y0 // 8 + 8, x0 // 8:x0 // 8 + 8].mean())
    t6["copy_move_same_image_detected"] = float(np.mean(det_cm))
    t6["collage_other_image_same_key_same_position_detected"] = float(np.mean(det_al))
    t6["collage_other_image_same_key_other_position_detected"] = float(np.mean(det_mis))
    out["T6_fragile_attacks"] = t6
    print("T6", t6, flush=True)

    # T7 -------------------------------------------------------------------------------------
    rows7 = []
    zz = F.ZZ[COEF]; r_, c_ = divmod(zz, 8)
    other_wm = (RNG.integers(0, 2, 4096)).astype(int)
    for w in marked:
        B = F.to_blocks(w[..., 2].astype(float)); M = A.T @ B @ A; v = M[:, r_, c_].copy()
        def rebuild(newv):
            M2 = M.copy(); M2[:, r_, c_] = newv
            R = A @ M2 @ A.T; o = w.copy(); o[..., 2] = F.from_blocks(np.clip(np.rint(R), 0, 255)).astype(np.uint8)
            return o
        # (a) frontera de decisión: punto medio entre las dos subredes + pequeño ruido
        q0 = F.dm_embed(v, np.zeros_like(v, int), DELTA); q1 = F.dm_embed(v, np.ones_like(v, int), DELTA)
        att_a = rebuild((q0 + q1) / 2 + RNG.uniform(-1, 1, v.shape))
        # (b) invertir cada bit con probabilidad 1/2 (marca aleatoria con los parámetros públicos)
        flip = RNG.integers(0, 2, v.shape).astype(bool)
        cur = F.dm_extract(v, DELTA)
        att_b = rebuild(np.where(flip, F.dm_embed(v, 1 - cur.astype(int), DELTA), v))
        # (c) sobrescritura con la marca de un atacante
        att_c = rebuild(F.dm_embed(v, other_wm, DELTA))
        # (d) referencia ciega: mediana 3x3 por canal
        att_d = ndimage.median_filter(w, size=(3, 3, 1))
        for lab, img in [("move coefficient to decision boundary", att_a), ("flip each bit with p = 1/2", att_b),
                         ("overwrite with attacker watermark", att_c), ("median filter 3x3 (blind)", att_d)]:
            bts = fs.extract_bits(img)
            b_, n_ = ber_ncc(wm, decode(bts, ETA))
            rows7.append(dict(attack=lab, psnr_vs_marked=psnr(w, img), ber=b_, ncc=n_,
                              attacker_wm_ber=float(np.mean(bts != other_wm)) if "overwrite" in lab else np.nan,
                              rejected_blocks=float(verify(img, kap).mean())))
    t7 = pd.DataFrame(rows7).groupby("attack", sort=False).mean().reset_index()
    out["T7_removal"] = t7.to_dict(orient="records")
    print(t7.round(4).to_string(), flush=True)

    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(RES, "exp5_results.json"), "w"), indent=2, default=float)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "exp5_robust_parameters.csv"), index=False)
    t7.to_csv(os.path.join(RES, "exp5_removal.csv"), index=False)
    print("done", out["seconds"], "s")


if __name__ == "__main__":
    main()
