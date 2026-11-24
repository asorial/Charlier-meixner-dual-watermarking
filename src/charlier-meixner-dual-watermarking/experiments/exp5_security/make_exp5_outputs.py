"""
Tablas (LaTeX), figura (PDF) y texto en inglés del Experimento 5 a partir de results/.

Uso (desde la raíz del repositorio):
    python experiments/exp5_security/make_exp5_outputs.py
"""
import json, math, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
MUTED, ACC = "#52514e", "#e34948"
COL = {"none": "#2a78d6", "gaussian 0.10": "#eb6834", "salt_pepper 0.05": "#1baf7a"}
LAB = {"none": "No attack", "gaussian 0.10": r"Gaussian, $p=0.10$", "salt_pepper 0.05": r"S&P, $d=0.05$"}

R = json.load(open(os.path.join(RES, "exp5_results.json")))
t3 = pd.DataFrame(R["T3_robust_parameters"]).set_index("setting")
t4 = pd.read_csv(os.path.join(RES, "exp5_arnold_bruteforce.csv"))
t7 = pd.DataFrame(R["T7_removal"]).set_index("attack")
T2, T4, T5, T6 = R["T2_wrong_fragile_key"], R["T4_arnold_bruteforce"], R["T5_forgery"], R["T6_fragile_attacks"]
n_img, ETA = len(R["images"]), R["eta"]
pct = lambda x: f"{100 * x:.1f}"
pct3 = lambda x: f"{100 * x:.3f}"


def sci(v):
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(v))); return f"{v / 10 ** e:.2f}\\times10^{{{e}}}"


# ------------------------------------------------------------------ figura: búsqueda exhaustiva de Arnold
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharex=True)
for att in ["none", "gaussian 0.10", "salt_pepper 0.05"]:
    g = t4[t4.attack == att].groupby("eta")
    for ax, met in zip(axes, ["structure", "ber"]):
        m = g[met].median(); lo, hi = g[met].quantile(0.05), g[met].quantile(0.95)
        ax.plot(m.index, m.values, color=COL[att], lw=1.2, label=LAB[att])
        ax.fill_between(m.index, lo.values, hi.values, color=COL[att], alpha=0.15, lw=0)
for ax in axes:
    ax.axvline(ETA, color=ACC, lw=0.8, ls="--"); ax.grid(axis="y"); ax.set_xlim(1, 63)
    ax.set_xticks([1, 16, ETA, 48, 63]); ax.set_xlabel(r"Candidate number of Arnold iterations $\eta'$")
axes[0].set_ylabel("Structure score"); axes[0].set_title("(a) Blind criterion (no watermark needed)", loc="left")
axes[1].set_ylabel("BER"); axes[1].set_title("(b) BER of the decoded watermark", loc="left")
axes[0].text(ETA + 1, axes[0].get_ylim()[1] * 0.98, rf"true $\eta={ETA}$", color=ACC, fontsize=7.5, va="top")
axes[1].legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp5_arnold_bruteforce.pdf"), bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ tabla 1: espacio de claves y amenazas
L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{Secret elements of the scheme, their effective size and what the experiments show "
     r"(Kerckhoffs' principle: the algorithm is public).}", r"\label{tab:keyspace}", r"\footnotesize",
     r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{p{2.3cm}p{1.1cm}p{2.5cm}p{6.1cm}}", r"\toprule",
     r"Element & Layer & Size & Evidence and effective security \\", r"\midrule",
     r"Key $K$ $\to$ $\kappa=\mathrm{SHA\text{-}256}(K)$ & Fragile & 256-bit $\kappa$; entropy bounded "
     r"by $K$ ($\approx$52 bits for 8 printable characters) & A wrong key rejects "
     + pct3(T2["random_keys"]["mean_rejected"]) + r"\% of the blocks of a genuine image (one changed character: "
     + pct3(T2["one_character_keys"]["mean_rejected"]) + r"\%; one changed bit of $\kappa$: "
     + pct3(T2["one_bit_kappa"]["mean_rejected"]) + r"\%). Security depends on the entropy of $K$. \\",
     r"Tag length $L=16$ & Fragile & $2^{16}$ tags per block & Forged blocks accepted: "
     + f"{T5['accepted']} of {T5['trials']:,}".replace(",", r"\,") + f" (${sci(T5['rate'])}$; 95\\% CI "
     f"$[{sci(T5['ci95'][0])},\\,{sci(T5['ci95'][1])}]$), consistent with $2^{{-16}}$. With a verification "
     r"oracle a block is forged in $2^{15}$ queries on average. \\",
     r"Arnold iterations $\eta$ & Robust & 63 values ($\approx$6 bits) & A blind search over the 63 "
     r"candidates recovers $\eta$ in " + pct(T4["success_rate_by_attack"]["none"]) + r"\% of the images "
     r"(also after Gaussian and salt-and-pepper noise). Not a cryptographic key. \\",
     r"Coefficient, $\Delta$ & Robust & 64 positions; $\Delta$ in a small range & A wrong coefficient "
     r"gives BER $\geq" + f"{t3.loc[['coefficient 61', 'coefficient 63', 'coefficient 53', 'coefficient 0'], 'ber'].min():.2f}"
     + r"$, but $\Delta\in[140,152]$ decodes as $\Delta=150$; both are recoverable by exhaustive search. \\",
     r"Polynomial parameters, transform & Robust & Continuous & The watermark of C\,I is decoded with the "
     r"matrices of C\,II, M\,I or M\,II (BER $\leq" + f"{t3.loc[['C II matrix', 'M I matrix', 'M II matrix'], 'ber'].max():.4f}"
     + r"$, as with the true matrix): the parameters are not secret. \\",
     r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open(os.path.join(RES, "exp5_keyspace_table.tex"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ tabla 2: ataques
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Attacks on the fragile and robust layers (" + f"{n_img}" + r" images, C\,I, $\Delta=150$). "
     r"Detected: percentage of tampered blocks flagged; rejected: percentage of all blocks flagged.}",
     r"\label{tab:security_attacks}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
     r"\begin{tabular}{lcccc}", r"\toprule", r"Attack & PSNR (dB) & BER & NCC & Fragile layer \\", r"\midrule",
     r"\multicolumn{5}{l}{\emph{Fragile layer}} \\",
     r"Flip the LSBs of all carriers & -- & -- & -- & " + pct(T6["flip_all_carrier_lsbs"]) + r"\% rejected \\",
     r"Random LSB plane (all pixels) & -- & -- & -- & " + pct(T6["random_lsb_plane_all_pixels"]) + r"\% rejected \\",
     r"Copy-move of whole blocks (with tags) & -- & -- & -- & " + pct(T6["copy_move_same_image_detected"]) + r"\% detected \\",
     r"Collage from another image, other position & -- & -- & -- & " + pct(T6["collage_other_image_same_key_other_position_detected"]) + r"\% detected \\",
     r"Collage from another image, same position & -- & -- & -- & " + pct(T6["collage_other_image_same_key_same_position_detected"]) + r"\% detected \\",
     r"\midrule", r"\multicolumn{5}{l}{\emph{Robust layer (attacker who knows the algorithm)}} \\"]
for lab, name in [("move coefficient to decision boundary", "Coefficient moved to the decision boundary"),
                  ("flip each bit with p = 1/2", "Each bit flipped with probability 1/2"),
                  ("overwrite with attacker watermark", "Overwriting with another watermark$^{a}$"),
                  ("median filter 3x3 (blind)", "Median filter $3\\times3$ (blind attacker)")]:
    r = t7.loc[lab]
    L.append(f"{name} & {r.psnr_vs_marked:.1f} & {r.ber:.3f} & {r.ncc:.3f} & {pct(r.rejected_blocks)}\\% rejected \\\\")
L += [r"\bottomrule", r"\end{tabular}",
      r"\par\smallskip\raggedright $^{a}$ The attacker's watermark is then extracted with BER "
      + f"{t7.loc['overwrite with attacker watermark', 'attacker_wm_ber']:.3f}.", r"\end{table}"]
open(os.path.join(RES, "exp5_attacks_table.tex"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ texto en inglés
cm = t3.loc[["C II matrix", "M I matrix", "M II matrix"], "ber"]
txt = rf"""% Generated by make_exp5_outputs.py from results/exp5_results.json
\subsection{{Security analysis}}\label{{sec:security}}

\paragraph{{Threat model.}} Following Kerckhoffs' principle, the algorithm and its public parameters
are assumed to be known to the adversary, who may read, modify and redistribute watermarked images
but does not know the secret key $K$. The adversary may try to (i) forge or transplant content
without being detected by the fragile layer, and (ii) remove, invalidate or overwrite the robust
watermark. Table~\ref{{tab:keyspace}} lists the secret elements of the scheme and
Table~\ref{{tab:security_attacks}} the attacks evaluated on {n_img} images.

\paragraph{{Fragile layer.}} The only cryptographic secret is $K$, from which the HMAC key
$\kappa=\mathrm{{SHA\text{{-}}256}}(K)$ is derived. With the correct key no block of a genuine image is
rejected, while any wrong key rejects practically all of them: {pct3(T2['random_keys']['mean_rejected'])}\% of the
blocks for 100 random keys, {pct3(T2['one_character_keys']['mean_rejected'])}\% when a single character of $K$ is
changed and {pct3(T2['one_bit_kappa']['mean_rejected'])}\% when a single bit of $\kappa$ is flipped, in agreement
with the expected $1-2^{{-16}}$. Although $\kappa$ has 256 bits, the effective key space is bounded by the
entropy of $K$ (about 52 bits for eight printable characters), so $K$ should be a random key of at
least 128 bits or be derived with a password-hashing function. Flipping the carrier LSBs, replacing
the LSB plane, modifying content or copying whole blocks within the image are always detected. The
16-bit truncation of the tag is the main limitation: an adversary who modifies a block and writes a
random tag is accepted with probability $2^{{-16}}$; in {T5['trials']:,} simulated forgeries
{T5['accepted']} were accepted (${sci(T5['rate'])}$, 95\% confidence interval
$[{sci(T5['ci95'][0])},\,{sci(T5['ci95'][1])}]$), and an adversary with access to a verification oracle
needs only $2^{{15}}$ queries on average to forge one block. Longer tags reduce this probability
exponentially at the cost of more modified LSBs per block. Finally, because the tag depends only on
the key, the block index and the block content, blocks copied from \emph{{another}} image protected with
the same key at the \emph{{same}} position verify correctly ({pct(T6['collage_other_image_same_key_same_position_detected'])}\% detected),
whereas blocks taken from another position are always detected. Including an image identifier in the
authenticated message, or using a different key per image, prevents this collage attack.
""".replace(f"{T5['trials']:,}", f"{T5['trials']:,}".replace(",", r"\,")) + rf"""
\paragraph{{Robust layer.}} The robust watermark does not depend on $K$, and its parameters do not
provide cryptographic security. The Arnold scrambling has period 64 for a $64\times64$ watermark, so
$\eta$ takes only 63 values ($\approx$6 bits). A wrong $\eta$ yields BER $\approx0.45$, but an adversary
who does not know the watermark can identify the correct value by decoding the 63 candidates and
choosing the most structured image: this blind search recovered $\eta$ in
{pct(T4['success_rate_by_attack']['none'])}\% of the images, also after Gaussian and salt-and-pepper noise
(Fig.~\ref{{fig:arnold_bruteforce}}). Likewise, a wrong embedding coefficient prevents extraction
(BER $\geq{t3.loc[['coefficient 61', 'coefficient 63', 'coefficient 53', 'coefficient 0'], 'ber'].min():.2f}$),
but the 64 possible positions can also be searched exhaustively, $\Delta$ is tolerant to errors
($\Delta\in[140,152]$ decodes as $\Delta=150$), and the polynomial parameters are not secret: the
watermark embedded with C\,I is extracted with the matrices of C\,II, M\,I or M\,II with BER
{cm.min():.4f}--{cm.max():.4f}, as with the true matrix ({t3.loc['correct parameters', 'ber']:.4f}). Consequently,
an adversary who knows the algorithm can invalidate the robust watermark with a distortion similar
to that of the embedding: flipping each bit with probability 1/2 gives BER
{t7.loc['flip each bit with p = 1/2', 'ber']:.3f} at {t7.loc['flip each bit with p = 1/2', 'psnr_vs_marked']:.1f}\,dB,
and overwriting it with another watermark gives BER {t7.loc['overwrite with attacker watermark', 'ber']:.3f} for the
owner and {t7.loc['overwrite with attacker watermark', 'attacker_wm_ber']:.3f} for the adversary. These attacks
do not go unnoticed, however: the fragile layer rejects {pct(t7.loc['flip each bit with p = 1/2', 'rejected_blocks'])}\%
and {pct(t7.loc['overwrite with attacker watermark', 'rejected_blocks'])}\% of the blocks, respectively (every block whose
robust coefficient was changed), so the owner can show that the image was altered after
watermarking, although the original robust watermark cannot be recovered.

\paragraph{{Scope.}} The dual scheme therefore provides cryptographically keyed integrity verification
with a per-block forgery probability of $2^{{-16}}$, and a robust watermark that resists the signal
processing operations reported in the previous sections but not an informed adversary. Making the
robust layer secure would require keying it, for example with key-dependent dither values or a
key-dependent selection of coefficients and block order; these extensions are left for future work.
"""
open(os.path.join(RES, "exp5_security_section.tex"), "w").write(txt)
print("ok")
