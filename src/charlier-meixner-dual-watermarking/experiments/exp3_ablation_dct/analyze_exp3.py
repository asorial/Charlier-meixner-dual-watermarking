"""
Análisis estadístico, tablas (LaTeX/CSV) y figuras (PDF) del Experimento 3.

  * Media, desviación típica (n-1) e IC del 95 % (t de Student) de PSNR, SSIM, BER y NCC.
  * Comparaciones pareadas por imagen (Wilcoxon de rangos con signo, bilateral, zero_method="wilcox")
    de C I, C II, M I y M II frente a la DCT; corrección de Holm
    dentro de cada familia (una por inserción). Tamaño del efecto: correlación
    biserial de rangos pareada r_rb = (W+ - W-)/(W+ + W-). Prueba de Friedman entre las 5 variantes.
  * La variante A = I del CSV de ensayos (si existe) se excluye del análisis.
  * Tasa de éxito: proporción de imágenes con BER <= 0.10 (marca utilizable).
  * Supuestos: diferencias pareadas por imagen (mismas imágenes y ataques con la misma semilla);
    se informa de la proporción de diferencias no normales (Shapiro-Wilk) para justificar la prueba
    no paramétrica; con todas las diferencias nulas la prueba no es aplicable (p = 1).

Uso (desde la raíz del repositorio):
    python experiments/exp3_ablation_dct/analyze_exp3.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "xtick.color": "#52514e",
    "ytick.color": "#52514e", "grid.color": "#e4e3df", "grid.linewidth": 0.5})
MUTED = "#52514e"
ORDER = ["C I", "C II", "M I", "M II", "DCT"]
STY = {"C I": ("#2a78d6", "o", "-"), "C II": ("#eb6834", "s", "-"), "M I": ("#1baf7a", "^", "-"),
       "M II": ("#eda100", "D", "-"), "DCT": ("#e34948", "v", "--")}
TEX = {"C I": r"C\,I", "C II": r"C\,II", "M I": r"M\,I", "M II": r"M\,II", "DCT": "DCT"}
ATT = ["gaussian", "salt_pepper", "median", "mean", "jpeg", "rotation"]
ATT_NAME = {"gaussian": "Gaussian noise", "salt_pepper": "Salt & pepper", "median": "Median filter",
            "mean": "Mean filter", "jpeg": "JPEG compression", "rotation": "Rotation"}
XLAB = {"gaussian": r"$p$ ($\sigma = p\,\mathrm{std}(I)$)", "salt_pepper": "Density",
        "median": "Window size", "mean": "Window size", "jpeg": "Quality factor",
        "rotation": "Angle (degrees, log scale)"}
REP = [("gaussian", 0.10), ("salt_pepper", 0.05), ("median", 3), ("mean", 3), ("jpeg", 100),
       ("jpeg", 90), ("rotation", 1)]
REP_NAME = {("gaussian", 0.10): "Gaussian, $p=0.10$", ("salt_pepper", 0.05): "S\\&P, $d=0.05$",
            ("median", 3): "Median 3$\\times$3", ("mean", 3): "Mean 3$\\times$3",
            ("jpeg", 100): "JPEG, $Q=100$", ("jpeg", 90): "JPEG, $Q=90$",
            ("rotation", 1): "Rotation 1$^\\circ$"}

d = pd.read_csv(os.path.join(RES, "exp3_trials.csv"))
d = d[d.variant.isin(ORDER)].copy()            # se excluye A = I
N_IMG = d.image.nunique()


def ci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); s = x.std(ddof=1) if n > 1 else 0.0
    h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n) if n > 1 else 0.0
    return m, s, m - h, m + h


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0.0
    for i, k in enumerate(o):
        run = max(run, min(1.0, (n - i) * p[k])); adj[k] = run
    return adj


def wilcoxon_pair(a, b):
    diff = np.asarray(a, float) - np.asarray(b, float)
    nz = diff[diff != 0]
    if len(nz) == 0:
        return 1.0, 0.0, 0.0, 0
    res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    r = stats.rankdata(np.abs(nz)); wp, wm = r[nz > 0].sum(), r[nz < 0].sum()
    return float(res.pvalue), float((wp - wm) / (wp + wm)), float(np.median(diff)), len(nz)


# ------------------------------------------------------------------ resúmenes con IC
rows = []
for (emb, var, att, lev), g in d.groupby(["embedding", "variant", "attack", "level"]):
    for met in ["ber", "ncc"]:
        m, s, lo, hi = ci(g[met]); rows.append(dict(embedding=emb, variant=var, attack=att, level=lev,
                                                    metric=met, mean=m, sd=s, ci_low=lo, ci_high=hi, n=len(g)))
summ = pd.DataFrame(rows); summ.to_csv(os.path.join(RES, "exp3_summary_ci.csv"), index=False)
succ = d.assign(ok=d.ber <= 0.10).groupby(["embedding", "variant", "attack", "level"]).ok.mean().unstack("variant")
succ.to_csv(os.path.join(RES, "exp3_success_rate_ber_le_0.10.csv"))

imp = d[d.attack == "none"]
rows = []
for (emb, var), g in imp.groupby(["embedding", "variant"]):
    r = dict(embedding=emb, variant=var)
    for met in ["psnr", "ssim", "ssim_paper"]:
        m, s, lo, hi = ci(g[met]); r.update({f"{met}_mean": m, f"{met}_sd": s, f"{met}_ci_low": lo, f"{met}_ci_high": hi})
    rows.append(r)
impt = pd.DataFrame(rows); impt.to_csv(os.path.join(RES, "exp3_imperceptibility.csv"), index=False)

# ------------------------------------------------------------------ pruebas pareadas
tests = []
for emb in ["original", "safe"]:
    for ref, cands in [("DCT", ["C I", "C II", "M I", "M II"])]:
        fam = []
        for (att, lev), g in d[d.embedding == emb].groupby(["attack", "level"]):
            piv = g.pivot(index="image", columns="variant", values="ber")
            for c in cands:
                p, rrb, md, nnz = wilcoxon_pair(piv[c], piv[ref])
                diffs = (piv[c] - piv[ref]).values
                sw = stats.shapiro(diffs).pvalue if np.ptp(diffs) > 0 and len(diffs) >= 3 else np.nan
                fam.append(dict(embedding=emb, reference=ref, variant=c, attack=att, level=lev, p=p,
                                r_rb=rrb, median_diff=md, n_nonzero=nnz, shapiro_p=sw,
                                mean_var=piv[c].mean(), mean_ref=piv[ref].mean()))
        padj = holm([t["p"] for t in fam])
        for t, pa in zip(fam, padj):
            t["p_holm"] = pa
            t["verdict"] = ("better" if t["median_diff"] < 0 or (t["median_diff"] == 0 and t["r_rb"] < 0) else "worse") \
                if pa < 0.05 else "n.s."
        tests += fam
tests = pd.DataFrame(tests); tests.to_csv(os.path.join(RES, "exp3_wilcoxon.csv"), index=False)

fr = []
for (emb, att, lev), g in d.groupby(["embedding", "attack", "level"]):
    piv = g.pivot(index="image", columns="variant", values="ber")[ORDER]
    try:
        st = stats.friedmanchisquare(*[piv[c] for c in ORDER]); fr.append(dict(embedding=emb, attack=att, level=lev, chi2=st.statistic, p=st.pvalue))
    except ValueError:
        fr.append(dict(embedding=emb, attack=att, level=lev, chi2=np.nan, p=1.0))
pd.DataFrame(fr).to_csv(os.path.join(RES, "exp3_friedman.csv"), index=False)
nonnormal = (tests.shapiro_p < 0.05).mean()

# ------------------------------------------------------------------ tabla 1: imperceptibilidad
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Imperceptibility of the five transforms under identical embedding ($\Delta=150$, "
     f"coefficient 62, blue channel, {N_IMG} images: 50 Caltech Birds and 50 NRC). Mean $\\pm$ s.d. "
     r"[95\% confidence interval]. SSIM: standard local SSIM. Overflow-safe: the reconstruction "
     r"point of the same dither sub-lattice is used when clipping to $[0,255]$ would flip the bit or "
     r"leave it closer than $\Delta/8$ to the decision boundary.}", r"\label{tab:ablation_imperceptibility}",
     r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{llcc}", r"\toprule", r"Embedding & Transform & PSNR (dB) & SSIM \\", r"\midrule"]
for emb, lab in [("original", "Original"), ("safe", "Overflow-safe")]:
    for i, v in enumerate(ORDER):
        r = impt[(impt.embedding == emb) & (impt.variant == v)].iloc[0]
        L.append(f"{lab if i == 0 else ''} & {TEX[v]} & ${r.psnr_mean:.2f} \\pm {r.psnr_sd:.2f}$ "
                 f"[{r.psnr_ci_low:.2f}, {r.psnr_ci_high:.2f}] & ${r.ssim_mean:.4f} \\pm {r.ssim_sd:.4f}$ "
                 f"[{r.ssim_ci_low:.4f}, {r.ssim_ci_high:.4f}] \\\\")
    L.append(r"\midrule" if emb == "original" else r"\bottomrule")
L += [r"\end{tabular}", r"\end{table}"]
open(os.path.join(RES, "exp3_imperceptibility_table.tex"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ tabla 2: comparación estadística
L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{BER (mean $\pm$ s.d. over " + f"{N_IMG}" + r" images) at representative attack levels. "
     r"Paired two-sided Wilcoxon signed-rank tests against the DCT with Holm correction "
     r"(all variants, attacks and levels of each embedding): $^{\blacktriangle}$ significantly lower BER "
     r"than DCT, $^{\triangledown}$ significantly higher, no mark: not significant ($\alpha=0.05$). "
     r"Best mean per row in bold. $^{\dagger}$ Mean BER $>0.2$: the watermark is not recoverable, so a "
     r"significant difference between two such values has no practical meaning.}", r"\label{tab:ablation_statistics}", r"\footnotesize",
     r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{l" + "c" * len(ORDER) + "}", r"\toprule",
     "Attack & " + " & ".join(TEX[v] for v in ORDER) + r" \\", r"\midrule"]
for emb, lab in [("original", "Original embedding"), ("safe", "Overflow-safe embedding")]:
    L.append(r"\multicolumn{%d}{l}{\emph{%s}} \\" % (len(ORDER) + 1, lab))
    for i, (att, lev) in enumerate(REP):
        g = summ[(summ.embedding == emb) & (summ.attack == att) & np.isclose(summ.level, lev) & (summ.metric == "ber")]
        best = g["mean"].min(); cells = []
        for v in ORDER:
            r = g[g.variant == v].iloc[0]
            s = f"{r['mean']:.3f} \\pm {r['sd']:.3f}"
            s = f"\\mathbf{{{s}}}" if np.isclose(r["mean"], best) else s
            mark = ""
            if v != "DCT":
                t = tests[(tests.embedding == emb) & (tests.reference == "DCT") & (tests.variant == v)
                          & (tests.attack == att) & np.isclose(tests.level, lev)].iloc[0]
                mark = {"better": r"\blacktriangle", "worse": r"\triangledown", "n.s.": ""}[t.verdict]
            sup = mark + (r"\dagger" if r["mean"] > 0.2 else "")
            cells.append(f"${s}^{{{sup}}}$" if sup else f"${s}$")
        L.append(f"{REP_NAME[(att, lev)]} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule" if emb == "original" else r"\bottomrule")
L += [r"\end{tabular}", r"\end{table*}"]
open(os.path.join(RES, "exp3_statistics_table.tex"), "w").write("\n".join(L) + "\n")


# ------------------------------------------------------------------ tabla 3: tasa de éxito
L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{Success rate: percentage of the " + f"{N_IMG}" + r" images whose watermark is recovered "
     r"with BER $\leq 0.10$, at representative attack levels. Best value per row in bold.}",
     r"\label{tab:ablation_success}", r"\small", r"\setlength{\tabcolsep}{4pt}",
     r"\begin{tabular}{ll" + "c" * len(ORDER) + "}", r"\toprule",
     "Embedding & Attack & " + " & ".join(TEX[v] for v in ORDER) + r" \\", r"\midrule"]
for emb, lab in [("original", "Original"), ("safe", "Overflow-safe")]:
    for i, (att, lev) in enumerate([("none", 0)] + REP):
        r = succ.loc[(emb, att, lev)][ORDER] * 100
        cells = [f"\\textbf{{{v:.0f}}}" if np.isclose(v, r.max()) and r.max() > 0 else f"{v:.0f}" for v in r]
        name = "No attack" if att == "none" else REP_NAME[(att, lev)]
        L.append(f"{lab if i == 0 else ''} & {name} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule" if emb == "original" else r"\bottomrule")
L += [r"\end{tabular}", r"\end{table*}"]
open(os.path.join(RES, "exp3_success_table.tex"), "w").write("\n".join(L) + "\n")


# ------------------------------------------------------------------ figuras
def curves(metric, emb, fname):
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.5))
    for ax, att in zip(axes.flat, ATT):
        for v in ORDER:
            c, mk, ls = STY[v]
            g = summ[(summ.embedding == emb) & (summ.variant == v) & (summ.metric == metric)
                     & (summ.attack == att)].sort_values("level")
            ax.plot(g.level, g["mean"], color=c, marker=mk, ms=3.2, lw=1.3, ls=ls)
            ax.fill_between(g.level, g.ci_low, g.ci_high, color=c, alpha=0.15, lw=0)
        if att == "jpeg":
            ax.invert_xaxis()
        if att == "rotation":
            ax.set_xscale("log"); ax.set_xticks([0.5, 1, 2, 5, 10, 45]); ax.set_xticklabels(["0.5", "1", "2", "5", "10", "45"])
            ax.minorticks_off()
        if att in ("median", "mean"):
            ax.set_xticks([3, 5, 7])
        ax.set_title(ATT_NAME[att], loc="left"); ax.set_xlabel(XLAB[att], fontsize=8); ax.grid(axis="y")
        if metric == "ber":
            ax.set_ylim(0, 0.8)
            ax.axhline(0.2, color=MUTED, lw=0.6, ls=(0, (1, 2)))
        else:
            ax.set_ylim(-0.05, 1.02)
    for ax in axes[:, 0]:
        ax.set_ylabel("BER" if metric == "ber" else "NCC")
    hs = [Line2D([], [], color=STY[v][0], marker=STY[v][1], ls=STY[v][2], ms=3.5, lw=1.3,
                 label=v) for v in ORDER]
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(handles=hs, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(os.path.join(FIG, fname), bbox_inches="tight")
    plt.close(fig)


for emb in ["original", "safe"]:
    curves("ber", emb, f"exp3_ber_{emb}.pdf")
    curves("ncc", emb, f"exp3_ncc_{emb}.pdf")

print(f"images={N_IMG}; share of non-normal paired differences (Shapiro p<0.05): {nonnormal:.2f}")
print(impt[["embedding", "variant", "psnr_mean", "ssim_mean"]].round(4).to_string(index=False))
