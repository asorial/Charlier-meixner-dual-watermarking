"""
Experimento 6 - Tiempo de ejecución frente a la intensidad del ruido (punto 8 de la revisión).

Resolución fija (512x512, 24 imágenes de src/static/cover_images) y configuración C I (el coste no
depende de la matriz: Experimento 4). Para cada imagen:

  1. Inserción (capa robusta + capa frágil) sobre la imagen original. El ruido se añade DESPUÉS de
     insertar, así que la intensidad del ruido no puede afectar a este tiempo: se mide aparte como
     referencia horizontal.
  2. Ruido gaussiano (p = sigma / std(I)) y sal y pimienta (densidad d) a intensidades crecientes,
     más el caso sin ataque. Cada imagen atacada se genera con una semilla fija, de modo que las
     repeticiones miden la variabilidad del tiempo y no la del ataque.
  3. Para cada condición y repetición se cronometran POR SEPARADO:
       read        lectura de la imagen atacada desde PNG (E/S, no forma parte del algoritmo)
       attack      generación del ataque (no forma parte del algoritmo)
       extract     extracción robusta: 4096 coeficientes + Arnold inversa (algoritmo)
       verify      verificación frágil: HMAC de los 4096 bloques y mapa de manipulación (algoritmo)
       metrics     BER y NCC de la marca recuperada (evaluación, no forma parte del algoritmo)
     en las dos implementaciones (referencia = clase del repositorio, vectorizada = fastwm.py), y
     además la función de extremo a extremo del repositorio DualWatermarkingScheme.extraction():
       e2e_nosave  extraction(save_tampered=False): extracción + verificación + barra de progreso
       e2e_save    extraction(save_tampered=True): además dibuja un rectángulo por bloque marcado y
                   escribe la imagen marcada en disco (informe, coste dependiente de los datos).
  4. Control de salidas anticipadas: una pasada instrumentada (sin cronometrar) cuenta cuántos
     bloques se transforman y cuántos HMAC se calculan en cada condición.

Medida: time.perf_counter_ns, recolector de basura desactivado durante cada medida, una ejecución
de calentamiento por operación y condiciones en orden aleatorio (semilla fija) para que la deriva
térmica o de frecuencia del procesador no se confunda con la intensidad del ruido.

Uso (desde la raíz del repositorio):
    python experiments/exp6_runtime_noise/run_exp6.py                    # completo (~25 min)
    python experiments/exp6_runtime_noise/run_exp6.py --images 2 --reps 2   # prueba rápida
    python experiments/exp6_runtime_noise/run_exp6.py --label portatil   # resultados en results_portatil/
"""
import argparse, contextlib, csv, gc, io, json, os, random, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in [os.path.join(ROOT, "src"), os.path.join(ROOT, "experiments", "exp2_parameter_sensitivity"),
          os.path.join(ROOT, "experiments", "exp4_complexity")]:
    sys.path.insert(0, p)

import imageio.v3 as iio
import fastwm as F
import run_exp4 as E4                     # mismas funciones de medida que el Experimento 4
from dual_scheme.dual_watermarking_scheme import DualWatermarkingScheme
from helpers.utils import binarize_watermark, arnold_transform

KEY, DELTA, COEF, P, X0 = E4.KEY, E4.DELTA, E4.COEF, E4.P, E4.X0
CONFIG = "C I"
LEVELS = {"gaussian": [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50],
          "salt_pepper": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]}


def timed(fn, *a, **k):
    gc.disable()
    t = time.perf_counter_ns(); r = fn(*a, **k); dt = time.perf_counter_ns() - t
    gc.enable()
    return dt * 1e-9, r


def counted_pass(s, img):
    """Pasada sin cronometrar que cuenta transformadas de bloque y HMAC (control de salidas anticipadas)."""
    n = {"robust": 0, "hmac": 0}
    rob, tag = s.extract_robust_block, s.block_tag

    def rob_c(b):
        n["robust"] += 1; return rob(b)

    def tag_c(k, b):
        n["hmac"] += 1; return tag(k, b)
    s.extract_robust_block, s.block_tag = rob_c, tag_c
    try:
        wm = E4.ref_extract(s, img); tm = E4.ref_verify(s, img)
    finally:
        del s.extract_robust_block, s.block_tag              # vuelve a los métodos de la clase
    return n, wm, tm


def metrics(wm_true, wm_rec, tamper):
    ber, ncc = F.ber_ncc(wm_true, wm_rec)
    return ber, ncc, float(tamper.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=24)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--label", default="")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    res = os.path.join(HERE, "results" + (f"_{a.label}" if a.label else "")); os.makedirs(res, exist_ok=True)

    cdir = os.path.join(ROOT, "src", "static", "cover_images")
    names = sorted(f for f in os.listdir(cdir) if f.lower().endswith(".bmp"))[:a.images]
    wm_img = iio.imread(os.path.join(ROOT, "src", "static", "watermark", "qr_asl.png"))
    bits_str = binarize_watermark(arnold_transform(wm_img, 27).reshape(-1))
    bits = np.array([int(b) for b in bits_str])
    wm_true = F.recover_watermark(bits)

    A = E4.MATS[CONFIG]
    s = DualWatermarkingScheme(KEY, DELTA, COEF, P, X0, A, A)
    fs = F.FastScheme(A, DELTA, COEF)
    conds = [("none", 0.0)] + [(k, l) for k in LEVELS for l in LEVELS[k]]

    info = E4.system_info()
    info.update(images=names, reps=a.reps, config=CONFIG, delta=DELTA, coef=COEF, levels=LEVELS,
                timer="time.perf_counter_ns", gc="disabled during each timed call", seed=a.seed)
    json.dump(info, open(os.path.join(res, "system_info.json"), "w"), indent=2)
    print(json.dumps({k: info[k] for k in ["os", "cpu_model", "logical_cpus", "ram_gb", "python"] if k in info}),
          flush=True)

    tmp = tempfile.mkdtemp(prefix="exp6_")
    cwd0 = os.getcwd(); os.chdir(tmp)                   # extraction(save_tampered=True) escribe aquí
    rng = random.Random(a.seed)
    trials, cond_rows, ins_rows = [], [], []
    t_start = time.time()
    try:
        for i, name in enumerate(names):
            cover = np.ascontiguousarray(iio.imread(os.path.join(cdir, name))[..., :3])
            # ---------------------------------------------------------------- inserción (sin ruido)
            wm_ref = E4.ref_insert(s, cover, bits_str); wm_vec = fs.insert(cover, bits)
            assert np.array_equal(wm_ref, wm_vec), "las dos implementaciones difieren"
            for impl, fn, args in [("reference", E4.ref_insert, (s, cover, bits_str)),
                                   ("vectorized", fs.insert, (cover, bits))]:
                fn(*args)
                for r in range(a.reps):
                    dt, _ = timed(fn, *args)
                    ins_rows.append(dict(image=name, impl=impl, rep=r, seconds=dt))

            # ---------------------------------------------------------------- imágenes atacadas
            att, paths, seeds = {}, {}, {}
            for j, (kind, lev) in enumerate(conds):
                seeds[(kind, lev)] = 100003 * i + 101 * j + 7
                img = np.ascontiguousarray(F.attack(wm_ref, kind, lev, seeds[(kind, lev)]), dtype=np.uint8)
                att[(kind, lev)] = img
                paths[(kind, lev)] = os.path.join(tmp, f"att_{j}.png"); iio.imwrite(paths[(kind, lev)], img)
                n, wr, tr = counted_pass(s, img)
                wv, tv = E4.vec_extract(fs, img), E4.vec_verify(fs, img)
                assert np.array_equal(wr, wv) and np.array_equal(tr, tv), "referencia y vectorizada difieren"
                ber, ncc, flagged = metrics(wm_true, wv, tv)
                changed = (img != wm_ref).any(axis=2).reshape(64, 8, 64, 8).any(axis=(1, 3)).mean()
                cond_rows.append(dict(image=name, kind=kind, level=lev, blocks=tv.size,
                                      blocks_transformed=n["robust"], hmac_computed=n["hmac"],
                                      changed_blocks=float(changed), flagged_blocks=flagged, ber=ber, ncc=ncc,
                                      changed_pixels=float((img != wm_ref).any(axis=2).mean())))

            # ---------------------------------------------------------------- calentamiento
            img0 = att[("none", 0.0)]
            for fn, args in [(E4.ref_extract, (s, img0)), (E4.ref_verify, (s, img0)),
                             (E4.vec_extract, (fs, img0)), (E4.vec_verify, (fs, img0))]:
                fn(*args)
            E4.quiet(s.extraction, img0, name, save_tampered=True)

            # ---------------------------------------------------------------- medidas, orden aleatorio
            order = [(c, r) for c in conds for r in range(a.reps)]; rng.shuffle(order)
            for (kind, lev), r in order:
                img = att[(kind, lev)]
                gc.collect()
                row = dict(image=name, kind=kind, level=lev, rep=r)
                ops = [("io", "read", iio.imread, (paths[(kind, lev)],), {}),
                       ("attack", "attack", F.attack, (wm_ref, kind, lev, seeds[(kind, lev)]), {}),
                       ("reference", "extract", E4.ref_extract, (s, img), {}),
                       ("reference", "verify", E4.ref_verify, (s, img), {}),
                       ("vectorized", "extract", E4.vec_extract, (fs, img), {}),
                       ("vectorized", "verify", E4.vec_verify, (fs, img), {}),
                       ("reference", "e2e_nosave", E4.quiet, (s.extraction, img, name), {"save_tampered": False}),
                       ("reference", "e2e_save", E4.quiet, (s.extraction, img, name), {"save_tampered": True})]
                out = {}
                for impl, op, fn, args, kw in ops:
                    dt, out[(impl, op)] = timed(fn, *args, **kw)
                    trials.append(dict(row, impl=impl, op=op, seconds=dt))
                rec, tm = out[("vectorized", "extract")], out[("vectorized", "verify")]
                dt, _ = timed(metrics, wm_true, rec, tm)
                trials.append(dict(row, impl="evaluation", op="metrics", seconds=dt))
            el = time.time() - t_start
            print(f"  {i + 1:2d}/{len(names)} {name}  ({el:.0f} s, ~{el / (i + 1) * (len(names) - i - 1):.0f} s restantes)",
                  flush=True)
    finally:
        os.chdir(cwd0)

    E4.write(res, "exp6_trials.csv", trials)
    E4.write(res, "exp6_conditions.csv", cond_rows)
    E4.write(res, "exp6_insert.csv", ins_rows)
    info["total_seconds"] = round(time.time() - t_start, 1); info["max_rss_mb"] = E4.max_rss_mb()
    json.dump(info, open(os.path.join(res, "system_info.json"), "w"), indent=2)
    print(f"Hecho en {info['total_seconds']} s -> {res}")


if __name__ == "__main__":
    main()
