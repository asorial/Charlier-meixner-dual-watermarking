"""
Tablas, figuras (PDF) y cifras del Experimento 6 a partir de results[_<label>]/.

Uso (desde la raíz del repositorio):
    python experiments/exp6_runtime_noise/plot_exp6.py [--label portatil]
Salida:
    figures[_<label>]/exp6_runtime_noise.pdf   tiempo (mediana e IQR) frente a la intensidad del ruido
    results[_<label>]/exp6_runtime_table.tex   desglose del coste por componente
    results[_<label>]/exp6_summary.csv         mediana, cuartiles y n por operación y condición
    results[_<label>]/exp6_change.csv          cambio relativo (sin ataque -> ruido máximo), IC 95 % bootstrap
    results[_<label>]/exp6_numbers.json        cifras citadas en el texto
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--label", default=""); a = ap.parse_args()
suf = f"_{a.label}" if a.label else ""
RES, FIG = os.path.join(HERE, "results" + suf), os.path.join(HERE, "figures" + suf)
os.makedirs(FIG, exist_ok=True)
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "xtick.color": "#52514e",
    "ytick.color": "#52514e", "grid.color": "#e4e3df", "grid.linewidth": 0.5})
MUTED = "#52514e"
C_EXT, C_VER, C_FLAG, C_BER = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
KINDS = {"gaussian": "Gaussian noise", "salt_pepper": "Salt-and-pepper noise"}
XLAB = {"gaussian": r"Noise level $p$ ($\sigma = p\,\mathrm{std}(I)$)", "salt_pepper": "Noise density $d$"}
RNG = np.random.default_rng(1)

t = pd.read_csv(os.path.join(RES, "exp6_trials.csv"))
c = pd.read_csv(os.path.join(RES, "exp6_conditions.csv"))
ins = pd.read_csv(os.path.join(RES, "exp6_insert.csv"))
info = json.load(open(os.path.join(RES, "system_info.json")))
t["ms"] = t.seconds * 1e3; ins["ms"] = ins.seconds * 1e3
N_IMG, REPS = t.image.nunique(), t.rep.nunique()


def levels(kind):
    """Condición sin ataque (nivel 0) seguida de los niveles del tipo de ruido."""
    return [0.0] + sorted(t[t.kind == kind].level.unique())


def cond(df, kind, lev):
    return df[(df.kind == "none")] if lev == 0 else df[(df.kind == kind) & np.isclose(df.level, lev)]


def q(x):
    return np.median(x), np.percentile(x, 25), np.percentile(x, 75)


# ------------------------------------------------------------------ resumen por condición
rows = []
for (impl, op), g in t.groupby(["impl", "op"]):
    for kind in KINDS:
        for lev in levels(kind):
            x = cond(g, kind, lev).ms.values
            m, q1, q3 = q(x)
            rows.append(dict(impl=impl, op=op, kind=kind, level=lev, median_ms=m, q1_ms=q1, q3_ms=q3,
                             iqr_ms=q3 - q1, n=len(x)))
summ = pd.DataFrame(rows); summ.to_csv(os.path.join(RES, "exp6_summary.csv"), index=False)


# ------------------------------------------------------------------ cambio relativo con IC bootstrap
def per_image_median(g, kind, lev):
    return cond(g, kind, lev).groupby("image").ms.median()


def change(g, kind, lev_hi, B=5000):
    """Cambio relativo de la mediana (pareado por imagen) entre sin ataque y el nivel lev_hi."""
    base, hi = per_image_median(g, kind, 0.0), per_image_median(g, kind, lev_hi)
    r = (hi / base).values - 1
    boot = np.array([np.median(RNG.choice(r, len(r))) for _ in range(B)])
    return float(np.median(r)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def slope_vs_flagged(g, B=5000):
    """Pendiente de la mediana por imagen frente a la fracción de bloques marcados (sal y pimienta),
    relativa a la mediana sin ataque: cambio relativo del tiempo al pasar de 0 % a 100 % de bloques marcados."""
    d = []
    for lev in levels("salt_pepper"):
        mi = per_image_median(g, "salt_pepper", lev)
        fl = cond(c, "salt_pepper", lev).set_index("image").flagged_blocks
        d.append(pd.DataFrame({"t": mi, "f": fl.reindex(mi.index)}))
    d = pd.concat(d).reset_index().rename(columns={"index": "image"})
    imgs = d.image.unique(); base = per_image_median(g, "salt_pepper", 0.0)

    def fit(sel):
        dd = pd.concat([d[d.image == i] for i in sel])
        b = np.polyfit(dd.f, dd.t, 1)[0]
        return b / base.reindex(sel).median()
    est = fit(imgs)
    boot = np.array([fit(RNG.choice(imgs, len(imgs))) for _ in range(1000)])
    return float(est), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


OPS = [("io", "read"), ("attack", "attack"), ("reference", "extract"), ("vectorized", "extract"),
       ("reference", "verify"), ("vectorized", "verify"), ("evaluation", "metrics"),
       ("reference", "e2e_nosave"), ("reference", "e2e_save")]
ch = []
for impl, op in OPS:
    g = t[(t.impl == impl) & (t.op == op)]
    for kind in KINDS:
        hi = max(levels(kind))
        if op == "attack":
            continue
        m, lo, up = change(g, kind, hi)
        dev = summ[(summ.impl == impl) & (summ.op == op) & (summ.kind == kind)].median_ms
        ch.append(dict(impl=impl, op=op, kind=kind, level_hi=hi, rel_change=m, ci_low=lo, ci_high=up,
                       max_abs_dev_levels=float((dev / dev.iloc[0] - 1).abs().max())))
chg = pd.DataFrame(ch); chg.to_csv(os.path.join(RES, "exp6_change.csv"), index=False)
slopes = {f"{impl}_{op}": slope_vs_flagged(t[(t.impl == impl) & (t.op == op)])
          for impl, op in [("reference", "verify"), ("vectorized", "verify"), ("reference", "extract"),
                           ("vectorized", "extract"), ("reference", "e2e_save")]}

# ------------------------------------------------------------------ figura
fig, axes = plt.subplots(3, 2, figsize=(6.6, 6.4), sharex="col",
                         gridspec_kw=dict(height_ratios=[1, 1, 0.8]))
for j, kind in enumerate(KINDS):
    L = levels(kind); x = np.arange(len(L))
    for i, impl in enumerate(["reference", "vectorized"]):
        ax = axes[i, j]
        e = ins[ins.impl == impl].ms
        m, q1, q3 = q(e)
        ax.axhspan(q1, q3, color="#e4e3df", lw=0, zorder=0)
        ax.axhline(m, color=MUTED, lw=1.0, ls="--", zorder=1)
        for op, col, mk in [("extract", C_EXT, "o"), ("verify", C_VER, "s")]:
            s = summ[(summ.impl == impl) & (summ.op == op) & (summ.kind == kind)].set_index("level").loc[L]
            ax.fill_between(x, s.q1_ms, s.q3_ms, color=col, alpha=0.2, lw=0, zorder=2)
            ax.plot(x, s.median_ms, color=col, marker=mk, ms=3.6, lw=1.4, zorder=3)
        ax.set_ylim(0, m * 1.18 if impl == "reference" else max(m, 0) * 1.25)
        ax.grid(axis="y")
        if j == 0:
            ax.set_ylabel(("Reference" if impl == "reference" else "Vectorized") + "\nimplementation (ms)")
        if i == 0:
            ax.set_title(f"({'ab'[j]}) {KINDS[kind]}", loc="left")
    ax = axes[2, j]
    for col_, colr, mk in [("flagged_blocks", C_FLAG, "^"), ("ber", C_BER, "D")]:
        s = c[c.kind.isin(["none", kind])].copy()
        s.loc[s.kind == "none", "level"] = 0.0
        g = s.groupby("level")[col_].agg(["median", lambda v: v.quantile(.25), lambda v: v.quantile(.75)]).loc[L]
        ax.fill_between(x, g.iloc[:, 1], g.iloc[:, 2], color=colr, alpha=0.2, lw=0)
        ax.plot(x, g["median"], color=colr, marker=mk, ms=3.6, lw=1.4)
    ax.set_ylim(-0.03, 1.05); ax.grid(axis="y")
    if j == 0:
        ax.set_ylabel("Fraction")
    ax.set_xticks(x); ax.set_xticklabels(["0"] + [f"{v:g}" for v in L[1:]], rotation=45, ha="right")
    ax.set_xlabel(XLAB[kind])
hs = [Line2D([], [], color=C_EXT, marker="o", ms=3.6, lw=1.4, label="Robust extraction"),
      Line2D([], [], color=C_VER, marker="s", ms=3.6, lw=1.4, label="Fragile verification"),
      Line2D([], [], color=MUTED, ls="--", lw=1.0, label="Embedding (before the attack)"),
      Line2D([], [], color=C_FLAG, marker="^", ms=3.6, lw=1.4, label="Flagged blocks"),
      Line2D([], [], color=C_BER, marker="D", ms=3.6, lw=1.4, label="BER of the robust watermark")]
fig.tight_layout(rect=(0, 0.075, 1, 1), h_pad=0.6)
fig.legend(handles=hs, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.savefig(os.path.join(FIG, "exp6_runtime_noise.pdf"), bbox_inches="tight")
plt.close(fig)


# ------------------------------------------------------------------ tabla
def cell(impl, op, kind, lev):
    r = summ[(summ.impl == impl) & (summ.op == op) & (summ.kind == kind) & np.isclose(summ.level, lev)].iloc[0]
    nd = 1 if r.median_ms >= 10 else 2
    return f"{r.median_ms:.{nd}f} ({r.iqr_ms:.{nd}f})"


def dcell(impl, op, kind):
    r = chg[(chg.impl == impl) & (chg.op == op) & (chg.kind == kind)].iloc[0]
    nd = 0 if max(abs(r.rel_change), abs(r.ci_low), abs(r.ci_high)) >= 0.1 else 1   # >= 10 %: sin decimales
    f = lambda v: ("0" + ".0" * nd if abs(100 * v) < 0.5 * 10 ** -nd else f"{100 * v:+.{nd}f}").replace("-", "$-$")
    return f"{f(r.rel_change)} [{f(r.ci_low)}, {f(r.ci_high)}]"


gmax, smax = max(levels("gaussian")), max(levels("salt_pepper"))
ROWS = [("Algorithm, reference implementation (one block at a time)",
         [("Embedding", "reference", "insert"), ("Robust extraction", "reference", "extract"),
          ("Fragile verification", "reference", "verify")]),
        ("Algorithm, vectorized implementation",
         [("Embedding", "vectorized", "insert"), ("Robust extraction", "vectorized", "extract"),
          ("Fragile verification", "vectorized", "verify")]),
        ("Not included in the algorithm", [("Reading the image (PNG)", "io", "read"),
                                           ("Generating the attack", "attack", "attack"),
                                           ("BER and NCC", "evaluation", "metrics"),
                                           (r"\texttt{extraction()}", "reference", "e2e_nosave"),
                                           (r"\texttt{extraction()} + report", "reference", "e2e_save")])]
L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{Execution time (ms) of each step at fixed resolution ($512\times512$, C\,I): median "
     f"(interquartile range) over {N_IMG} images $\\times$ {REPS} repetitions. "
     r"Change: relative change of the median time between no attack and the strongest noise level "
     f"(Gaussian $p={gmax:g}$, salt-and-pepper $d={smax:g}$), paired by image, with a 95\\% bootstrap "
     r"confidence interval. Embedding is performed before the attack and does not depend on it. "
     r"\texttt{extraction()} is the end-to-end function of the implementation (robust extraction, "
     r"verification and progress output); the report marks the flagged blocks and writes the "
     r"resulting image to disk.}",
     r"\label{tab:runtime_noise}", r"\footnotesize", r"\setlength{\tabcolsep}{2pt}",
     r"\begin{tabular}{lccccc}", r"\toprule",
     r" & No attack & Gaussian & S\&P & \multicolumn{2}{c}{Change (\%)} \\", r"\cmidrule(lr){5-6}",
     f" &  & $p={gmax:g}$ & $d={smax:g}$ & Gaussian & S\\&P \\\\", r"\midrule"]
for block, items in ROWS:
    L.append(r"\multicolumn{6}{l}{\emph{" + block + r"}} \\")
    for lab, impl, op in items:
        if op == "insert":
            e = ins[ins.impl == impl].ms; m, q1, q3 = q(e); nd = 1 if m >= 10 else 2
            L.append(f"{lab} & {m:.{nd}f} ({q3 - q1:.{nd}f}) & \\multicolumn{{4}}{{l}}{{(before the attack)}} \\\\")
            continue
        if op == "attack":
            L.append(f"{lab} & -- & {cell(impl, op, 'gaussian', gmax)} & "
                     f"{cell(impl, op, 'salt_pepper', smax)} & -- & -- \\\\")
            continue
        L.append(f"{lab} & {cell(impl, op, 'gaussian', 0.0)} & {cell(impl, op, 'gaussian', gmax)} & "
                 f"{cell(impl, op, 'salt_pepper', smax)} & "
                 f"{dcell(impl, op, 'gaussian')} & {dcell(impl, op, 'salt_pepper')} \\\\")
    L.append(r"\bottomrule" if block == ROWS[-1][0] else r"\midrule")
L += [r"\end{tabular}", r"\end{table*}"]
open(os.path.join(RES, "exp6_runtime_table.tex"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ cifras para el texto
num = dict(n_images=N_IMG, reps=REPS, cpu=info.get("cpu_model"), python=info.get("python"),
           blocks_transformed=sorted(c.blocks_transformed.unique().tolist()),
           hmac_computed=sorted(c.hmac_computed.unique().tolist()),
           flagged_range_sp=[float(cond(c, "salt_pepper", min(levels("salt_pepper")[1:])).flagged_blocks.median()),
                             float(cond(c, "salt_pepper", smax).flagged_blocks.median())],
           flagged_min_gauss=float(c[c.kind == "gaussian"].flagged_blocks.min()),
           flagged_equals_changed=bool(np.allclose(c.flagged_blocks, c.changed_blocks, atol=1e-3)),
           ber_none=float(cond(c, "gaussian", 0).ber.median()), ber_g_max=float(cond(c, "gaussian", gmax).ber.median()),
           ber_sp_max=float(cond(c, "salt_pepper", smax).ber.median()),
           insert_ms={k: float(ins[ins.impl == k].ms.median()) for k in ["reference", "vectorized"]},
           max_abs_dev=chg.set_index(["impl", "op", "kind"]).max_abs_dev_levels.round(4).to_dict(),
           slope_vs_flagged=slopes)
num["max_abs_dev"] = {" ".join(k): v for k, v in num["max_abs_dev"].items()}
json.dump(num, open(os.path.join(RES, "exp6_numbers.json"), "w"), indent=2)
print(json.dumps(num, indent=1))
print(chg.round(4).to_string())
