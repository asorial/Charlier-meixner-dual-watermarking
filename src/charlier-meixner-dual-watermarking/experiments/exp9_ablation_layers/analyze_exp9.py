"""
Análisis del Experimento 9: tablas LaTeX, CSV con diferencias pareadas e IC del 95 %, y figuras PDF.

Uso (desde la raíz del repositorio, después de run_exp9.py):
    python experiments/exp9_ablation_layers/analyze_exp9.py
"""
import os, json
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
VAR = ["C I", "C II", "M I", "M II", "DCT", "Identity"]
MOM = VAR[:4]
STY = {"C I": ("#2a78d6", "o", "-"), "C II": ("#eb6834", "s", "-"), "M I": ("#1baf7a", "^", "-"),
       "M II": ("#eda100", "D", "-"), "DCT": ("#e34948", "v", "--"), "Identity": (MUTED, "P", ":")}
TEX = {"C I": r"C\,I", "C II": r"C\,II", "M I": r"M\,I", "M II": r"M\,II", "DCT": "DCT", "Identity": "Identity",
       "all": "--"}
REP = [("none", 0.0), ("gaussian", 0.10), ("salt_pepper", 0.05), ("median", 3), ("mean", 3), ("jpeg", 100),
       ("jpeg", 90), ("rotation", 1)]
REP_NAME = {("none", 0.0): "No attack", ("gaussian", 0.10): "Gaussian, $p=0.10$",
            ("salt_pepper", 0.05): "S\\&P, $d=0.05$", ("median", 3): "Median $3\\times3$",
            ("mean", 3): "Mean $3\\times3$", ("jpeg", 100): "JPEG, $Q=100$", ("jpeg", 90): "JPEG, $Q=90$",
            ("rotation", 1): "Rotation $1^\\circ$"}
ATT_NAME = {"gaussian": "Gaussian noise", "salt_pepper": "Salt & pepper", "median": "Median filter",
            "mean": "Mean filter", "jpeg": "JPEG compression", "rotation": "Rotation"}
XLAB = {"gaussian": r"$p$ ($\sigma = p\,\mathrm{std}(I)$)", "salt_pepper": "Density", "median": "Window size",
        "mean": "Window size", "jpeg": "Quality factor", "rotation": "Angle (degrees, log scale)"}


def ci_mean(x):
    x = np.asarray(x, float); n = len(x); m = x.mean()
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return m, m - h, m + h


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0.0
    for i, k in enumerate(o):
        run = max(run, min(1.0, (n - i) * p[k])); adj[k] = run
    return adj


def wil(d):
    d = np.asarray(d, float)
    return 1.0 if np.all(d == 0) else float(stats.wilcoxon(d, zero_method="wilcox").pvalue)


def cond(df, kind, lev):
    return df[(df.attack == kind) & np.isclose(df.level, lev)]


def fmt_ci(m, lo, hi, nd=3, signed=True):
    f = (lambda v: f"{v:+.{nd}f}") if signed else (lambda v: f"{v:.{nd}f}")
    g = lambda v: f(v).replace("-", "$-$")
    return f"{g(m)} [{g(lo)}, {g(hi)}]"


out = {}
# ================================================================== (A) transformadas
t = pd.read_csv(os.path.join(RES, "exp9_transforms.csv"))
N = t.image.nunique()
q = t[t.attack == "none"].set_index(["variant", "image"])[["psnr", "ssim"]]
tim = pd.read_csv(os.path.join(RES, "exp9_timing.csv"))
tmed = tim.groupby(["variant", "impl", "op"]).ms.median()

# diferencias pareadas por imagen (variante de momentos menos referencia), todas las condiciones
rows = []
for ref in ["DCT", "Identity"]:
    fam = []
    for (kind, lev), g in t.groupby(["attack", "level"]):
        for metric in ["ber", "ncc"]:
            piv = g.pivot(index="image", columns="variant", values=metric)
            for v in MOM:
                d = (piv[v] - piv[ref]).values
                m, lo, hi = ci_mean(d)
                fam.append(dict(reference=ref, variant=v, attack=kind, level=lev, metric=metric, mean_diff=m,
                                ci_low=lo, ci_high=hi, p=wil(d), mean_var=piv[v].mean(), mean_ref=piv[ref].mean()))
    for metric in ["psnr", "ssim"]:
        piv = q[metric].unstack(0)
        for v in MOM:
            d = (piv[v] - piv[ref]).values; m, lo, hi = ci_mean(d)
            fam.append(dict(reference=ref, variant=v, attack="none", level=0.0, metric=metric, mean_diff=m,
                            ci_low=lo, ci_high=hi, p=wil(d), mean_var=piv[v].mean(), mean_ref=piv[ref].mean()))
    pa = holm([r["p"] for r in fam])
    for r, x in zip(fam, pa):
        r["p_holm"] = x
    rows += fam
diffA = pd.DataFrame(rows); diffA.to_csv(os.path.join(RES, "exp9_transforms_paired_differences.csv"), index=False)

summA = []
for (v, kind, lev), g in t.groupby(["variant", "attack", "level"]):
    for metric in ["ber", "ncc"]:
        m, lo, hi = ci_mean(g[metric]); summA.append(dict(variant=v, attack=kind, level=lev, metric=metric,
                                                          mean=m, ci_low=lo, ci_high=hi))
summA = pd.DataFrame(summA); summA.to_csv(os.path.join(RES, "exp9_transforms_summary.csv"), index=False)

# ---- tabla A1: calidad y tiempos
L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{Six variants of the same algorithm that differ only in the $8\times8$ block transform "
     f"({N} images, $\\Delta=150$, coefficient 62, blue channel, same fragile layer). PSNR and local SSIM "
     r"between cover and watermarked image: mean $\pm$ half-width of the 95\% confidence interval. BER without attack: bits lost "
     r"by embedding alone (rounding and clipping). Times: median per $512\times512$ image, vectorized "
     r"implementation (all images, 3 repetitions) and reference implementation (24 images, 3 repetitions).}",
     r"\label{tab:ablation6_quality}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
     r"\begin{tabular}{lcccccccc}", r"\toprule",
     r" & & & & \multicolumn{2}{c}{Vectorized (ms)} & \multicolumn{3}{c}{Reference (ms)} \\",
     r"\cmidrule(lr){5-6}\cmidrule(lr){7-9}",
     r"Transform & PSNR (dB) & SSIM & BER, no attack & Embed & Extract & Embed & Extract & Verify \\", r"\midrule"]
for v in VAR:
    pm = ci_mean(q.loc[v].psnr); sm = ci_mean(q.loc[v].ssim); b0 = cond(t[t.variant == v], "none", 0).ber
    L.append(" & ".join([TEX[v], f"{pm[0]:.2f}$\\pm${pm[2] - pm[0]:.2f}", f"{sm[0]:.4f}$\\pm${sm[2] - sm[0]:.4f}",
                         f"{b0.mean():.4f}",
                         f"{tmed[(v, 'vectorized', 'insert dual')]:.1f}", f"{tmed[(v, 'vectorized', 'extract robust')]:.2f}",
                         f"{tmed[(v, 'reference', 'insert dual')]:.0f}", f"{tmed[(v, 'reference', 'extract robust')]:.0f}",
                         f"{tmed[(v, 'reference', 'verify fragile')]:.0f}"]) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open(os.path.join(RES, "exp9_ablation6_quality_table.tex"), "w").write("\n".join(L) + "\n")

# ---- tabla A2: BER media por variante y diferencias pareadas (momentos - DCT, momentos - identidad)
L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{BER of the robust watermark at representative attack levels (mean over " + f"{N}" +
     r" images; same attack realization for all variants). Right: paired difference "
     r"(moment transform minus reference), averaged over C\,I, C\,II, M\,I and M\,II, $\pm$ the largest "
     r"half-width of their 95\% confidence intervals; negative values favour the moments. "
     r"$^{\ast}$ all four differences significant (Wilcoxon signed-rank, Holm, $\alpha=0.05$).}",
     r"\label{tab:ablation6_ber}", r"\footnotesize", r"\setlength{\tabcolsep}{2pt}",
     r"\begin{tabular}{lcccccccc}", r"\toprule",
     r" & \multicolumn{6}{c}{Mean BER} & \multicolumn{2}{c}{Moments minus reference} \\",
     r"\cmidrule(lr){2-7}\cmidrule(lr){8-9}",
     "Attack & " + " & ".join(TEX[v] for v in VAR) + r" & $-$DCT & $-$Identity \\", r"\midrule"]
for kind, lev in REP:
    g = cond(t, kind, lev); means = g.groupby("variant").ber.mean()
    best = means.min()
    cells = [(f"\\textbf{{{means[v]:.3f}}}" if np.isclose(means[v], best) else f"{means[v]:.3f}") for v in VAR]
    dd = []
    for ref in ["DCT", "Identity"]:
        d = diffA[(diffA.reference == ref) & (diffA.metric == "ber") & (diffA.attack == kind) & np.isclose(diffA.level, lev)]
        sig = "$^{\\ast}$" if (d.p_holm < 0.05).all() else ""
        f = lambda v: f"{v:+.3f}".replace("-", "$-$")
        hw = float(((d.ci_high - d.ci_low) / 2).max())
        dd.append(f"{f(d.mean_diff.mean())}$\\pm${hw:.3f}{sig}")
    L.append(f"{REP_NAME[(kind, lev)]} & " + " & ".join(cells + dd) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open(os.path.join(RES, "exp9_ablation6_ber_table.tex"), "w").write("\n".join(L) + "\n")


# ---- figuras A: curvas BER y NCC
def curves(metric, fname):
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.5))
    for ax, att in zip(axes.flat, ATT_NAME):
        for v in VAR:
            c, mk, ls = STY[v]
            g = summA[(summA.variant == v) & (summA.metric == metric) & (summA.attack == att)].sort_values("level")
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
            ax.set_ylim(0, 0.8); ax.axhline(0.2, color=MUTED, lw=0.6, ls=(0, (1, 2)))
        else:
            ax.set_ylim(-0.05, 1.02)
    for ax in axes[:, 0]:
        ax.set_ylabel("BER" if metric == "ber" else "NCC")
    hs = [Line2D([], [], color=STY[v][0], marker=STY[v][1], ls=STY[v][2], ms=3.5, lw=1.3, label=v) for v in VAR]
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(handles=hs, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(os.path.join(FIG, fname), bbox_inches="tight"); plt.close(fig)


curves("ber", "exp9_ablation6_ber.pdf")
curves("ncc", "exp9_ablation6_ncc.pdf")

# ================================================================== (B) capas
rb = pd.read_csv(os.path.join(RES, "exp9_layers_robust.csv"))
au = pd.read_csv(os.path.join(RES, "exp9_layers_auth.csv"))
ql = pd.read_csv(os.path.join(RES, "exp9_layers_quality.csv"))
NB = ql.image.nunique()

inter = []
for cfg in MOM:
    # efecto de la capa frágil sobre la robusta: dual - solo robusta, por imagen y condición
    r = rb[rb.config == cfg].pivot_table(index=["image", "attack", "level"], columns="variant", values="ber")
    d = (r["dual"] - r["robust only"])
    rn = rb[rb.config == cfg].pivot_table(index=["image", "attack", "level"], columns="variant", values="ncc")
    dn = (rn["dual"] - rn["robust only"])
    by_cond = d.groupby(level=["attack", "level"]).agg(lambda s: ci_mean(s.values))
    worst = max(by_cond, key=lambda x: max(abs(x[1]), abs(x[2])))
    # efecto de la capa robusta sobre la autenticación: dual - solo frágil, por imagen
    a_d = au[(au.config == cfg) & (au.variant == "dual")].set_index(["image", "tamper", "fraction", "rep"])
    a_f = au[(au.variant == "fragile only")].set_index(["image", "tamper", "fraction", "rep"])
    f1d = (a_d.F1 - a_f.F1.reindex(a_d.index)).groupby(level="image").mean()
    qd = ql[(ql.config == cfg)].set_index(["variant", "image"])
    qf = ql[ql.variant == "fragile only"].set_index("image")
    psnr_cost = ci_mean((qd.loc["robust only"].psnr - qd.loc["dual"].psnr).values)
    inter.append(dict(config=cfg,
                      ber_diff_mean_all=float(d.mean()), ber_diff_max_abs=float(d.abs().max()),
                      ncc_diff_max_abs=float(dn.abs().max()),
                      images_conditions_with_diff=int((d != 0).sum()), n_pairs=int(d.size),
                      worst_condition_ci=[float(x) for x in worst],
                      bit_errors_no_attack_robust=int(qd.loc["robust only"].bit_errors_no_attack.sum()),
                      bit_errors_no_attack_dual=int(qd.loc["dual"].bit_errors_no_attack.sum()),
                      f1_diff=list(map(float, ci_mean(f1d.values))),
                      psnr_cost_fragile_in_dual=list(map(float, psnr_cost)),
                      psnr_cost_robust_in_dual=list(map(float, ci_mean((qf.psnr.reindex(qd.loc["dual"].index) - qd.loc["dual"].psnr).values)))))
out["interaction"] = inter


def auth_stats(g):
    TP, FP, FN = g.TP.sum(), g.FP.sum(), g.FN.sum()
    return dict(precision=TP / (TP + FP) if TP + FP else np.nan, recall=TP / (TP + FN), FP=int(FP), FN=int(FN),
                F1_mean=g.F1.mean(), detected=g.detected_image.mean(), n=len(g))


L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{Interaction between the two layers (" + f"{NB}" + r" images). Robust only: watermark "
     r"embedded without the fragile tags; fragile only: tags written on the cover (independent of the "
     r"moment matrix); dual: complete scheme. PSNR and SSIM between cover and output. BER of the robust "
     r"watermark with the same attack realizations. Intact: blocks flagged by the verification of the "
     r"untouched output (all images). Tampered: copy-move, splicing and object removal of 1--20\% of the "
     r"image at known positions (" + f"{au[au.variant=='fragile only'].shape[0]}" + r" per variant); block-level "
     r"precision (pooled), mean $F_1$ and missed blocks (out of " + f"{int(au[au.variant=='fragile only'].gt_blocks.sum()):,}".replace(",", "\\,") + r"). Embedding time: median per image (vectorized). "
     r"n/a: the layer is absent.}", r"\label{tab:layer_interaction}", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
     r"\begin{tabular}{llcccccccccc}", r"\toprule",
     r" & & \multicolumn{2}{c}{Distortion} & \multicolumn{3}{c}{Robust BER} & \multicolumn{4}{c}{Authentication} & Time (ms) \\",
     r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(lr){8-11}\cmidrule(lr){12-12}",
     r"Config. & Layers & PSNR & SSIM & None & G 0.10 & S\&P 0.05 & Intact & Prec. & $F_1$ & Missed & Embed \\",
     r"\midrule"]
tvec = tim[tim.impl == "vectorized"].groupby(["variant", "op"]).ms.median()
fr = auth_stats(au[au.variant == "fragile only"]); qf = ql[ql.variant == "fragile only"]
for cfg in MOM:
    for var in ["robust only", "fragile only", "dual"]:
        qq = qf if var == "fragile only" else ql[(ql.config == cfg) & (ql.variant == var)]
        cells = [TEX[cfg] if var == "robust only" else "", {"robust only": "Robust", "fragile only": "Fragile",
                                                              "dual": "Dual"}[var], f"{qq.psnr.mean():.2f}", f"{qq.ssim.mean():.4f}"]
        if var == "fragile only":
            cells += ["n/a"] * 3
        else:
            g = rb[(rb.config == cfg) & (rb.variant == var)]
            cells += [f"{cond(g, k, l).ber.mean():.4f}" for k, l in [("none", 0), ("gaussian", 0.10), ("salt_pepper", 0.05)]]
        if var == "robust only":
            cells += ["n/a"] * 4
        else:
            s = fr if var == "fragile only" else auth_stats(au[(au.config == cfg) & (au.variant == var)])
            cells += [f"{int(qq.flagged_intact.sum())}", f"{s['precision']:.4f}", f"{s['F1_mean']:.4f}", f"{s['FN']}"]
        op = {"robust only": "insert robust only", "fragile only": "insert fragile only", "dual": "insert dual"}[var]
        cells.append(f"{tvec[(cfg, op)]:.1f}")
        L.append(" & ".join(cells) + r" \\")
    L.append(r"\midrule" if cfg != MOM[-1] else r"\bottomrule")
L += [r"\end{tabular}", r"\end{table*}"]
open(os.path.join(RES, "exp9_layers_table.tex"), "w").write("\n".join(L) + "\n")

rev = ql[ql.variant == "reversed order"].groupby("config").flagged_intact.agg(["mean", "min", "max"])
out["reversed_order_flagged_blocks"] = rev.to_dict()
out["fragile_only"] = {k: (float(v) if isinstance(v, (np.floating, float)) else int(v)) for k, v in fr.items()}
out["dual_auth"] = {c: {k: (float(v) if isinstance(v, (np.floating, float)) else int(v))
                        for k, v in auth_stats(au[(au.config == c) & (au.variant == "dual")]).items()} for c in MOM}
out["quality_layers"] = ql.groupby(["config", "variant"])[["psnr", "ssim", "changed_pixels"]].mean().round(5).reset_index().to_dict("records")
out["times_ms"] = tim.groupby(["variant", "impl", "op"]).ms.median().round(3).reset_index().to_dict("records")
out["quality_transforms"] = {v: dict(psnr=list(map(float, ci_mean(q.loc[v].psnr))), ssim=list(map(float, ci_mean(q.loc[v].ssim))),
                                     ber0=float(cond(t[t.variant == v], "none", 0).ber.mean()))
                             for v in VAR}
json.dump(out, open(os.path.join(RES, "exp9_numbers.json"), "w"), indent=1, default=float)
print(json.dumps({k: out[k] for k in ["interaction", "reversed_order_flagged_blocks", "fragile_only", "quality_transforms"]}, indent=1, default=float))

# ================================================================== (B2) diferencias pareadas entre capas
pc = []
for cfg in MOM:
    p = rb[rb.config == cfg].pivot_table(index=["image", "attack", "level"], columns="variant", values="ber")
    d = (p["dual"] - p["robust only"]).rename("d").to_frame().join(p["robust only"].rename("ber_robust_only"))
    for (a_, l_), x in d.groupby(level=["attack", "level"]):
        m, lo, hi = ci_mean(x.d.values)
        pc.append(dict(config=cfg, attack=a_, level=l_, ber_robust_only=x.ber_robust_only.mean(),
                       diff_dual_minus_robust=m, ci_low=lo, ci_high=hi, p=wil(x.d.values)))
pc = pd.DataFrame(pc); pc.to_csv(os.path.join(RES, "exp9_layers_ber_differences.csv"), index=False)

L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Paired differences between layer configurations (" + f"{NB}" + r" images; mean [95\% CI]). "
     r"$\Delta$PSNR: robust only minus dual (distortion added by the fragile layer). $\Delta$BER: dual minus "
     r"robust only, largest absolute mean difference and largest CI bound over the attack conditions in which "
     r"the watermark is recoverable (BER $<0.2$), and over all conditions. $\Delta F_1$: dual minus fragile "
     r"only, block-level $F_1$ on the tampered images. Reversed: blocks flagged in untouched images when the "
     r"fragile tags are written before the robust embedding (mean of 4096).}",
     r"\label{tab:layer_differences}", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
     r"\begin{tabular}{lccccc}", r"\toprule",
     r" & $\Delta$PSNR (dB) & $\Delta$BER, BER $<0.2$ & $\Delta$BER, all & $\Delta F_1$ ($\times10^{-6}$) & Reversed \\",
     r"\midrule"]
for i, cfg in enumerate(MOM):
    it = inter[i]; x = pc[pc.config == cfg]; rec = x[x.ber_robust_only < 0.2]
    f = it["f1_diff"]; ps = it["psnr_cost_fragile_in_dual"]
    rmax = rec.diff_dual_minus_robust.abs().max(); rb_ = rec[["ci_low", "ci_high"]].abs().max().max()
    amax = x.diff_dual_minus_robust.abs().max(); ab_ = x[["ci_low", "ci_high"]].abs().max().max()
    g = lambda v: f"{v:+.1f}".replace("-", "$-$")
    L.append(" & ".join([TEX[cfg], f"{ps[0]:.3f} [{ps[1]:.3f}, {ps[2]:.3f}]",
                         f"${rmax * 1e5:.1f}\\times10^{{-5}}$ ($\\leq{rb_ * 1e5:.1f}\\times10^{{-5}}$)",
                         f"{amax:.4f} ($\\leq${ab_:.4f})",
                         f"{g(f[0] * 1e6)} [{g(f[1] * 1e6)}, {g(f[2] * 1e6)}]",
                         f"{out['reversed_order_flagged_blocks']['mean'][cfg]:.0f}"]) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(RES, "exp9_layers_differences_table.tex"), "w").write("\n".join(L) + "\n")
print(open(os.path.join(RES, "exp9_layers_differences_table.tex")).read())
