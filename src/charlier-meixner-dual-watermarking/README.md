# Robust and fragile dual watermarking with Charlier and Meixner moments

Code and experiments for the paper *Robust and Fragile Dual Watermarking Based on Charlier and
Meixner Orthogonal Moment Approaches* (A. Soria-Lorente, J. Michel, E. J. Huertas; under review).

The scheme protects a colour image with two independent layers:

- **Robust layer (ownership).** A 64×64 binary watermark is scrambled with the Arnold transform
  and embedded, one bit per 8×8 block of the blue channel, by Dither Modulation of the moment
  coefficient ν₆₂ = 𝓜₇,₆, with 𝓜 = 𝒜ᵀ𝒞𝒜 and 𝒜 a Charlier or Meixner moment matrix. Extraction is blind.
- **Fragile layer (integrity).** After robust embedding, each 8×8 block receives a 16-bit tag
  `Trunc16(HMAC-SHA256_κ(k ‖ B⁰ₖ))`, where k is the block index, B⁰ₖ the RGB block with the 16 carrier
  LSBs set to zero and κ = SHA-256(K). Verification checks every block and returns a tamper map.

| Parameter | Value |
| --- | --- |
| Block size, channel | 8×8, blue |
| Marked coefficient | zigzag index 62, position (7,6) |
| Quantization | Δ = 150, d₀ = −Δ/4, d₁ = +Δ/4 |
| Watermark | 64×64 bits, Arnold p = q = 1, η = 27 iterations (inverse: 37) |
| Moment matrices | C I, C II (Charlier), M I, M II (Meixner): `orthmatrix(0..3)` in `src/helpers/orthonornal_matrix.py` |
| Fragile tag | HMAC-SHA256 truncated to 16 bits, in the LSBs of the first 16 blue pixels of each block |

## Repository layout

```
src/                 implementation of the scheme
  dual_scheme/       DualWatermarkingScheme: insert(), extraction(), authenticate()
  helpers/           moment matrices, zigzag scan, Arnold transform, metrics
  polynomials/, orthogonal_matrix/, ...   Charlier, Meixner and related polynomial families
  noises_attaks/     attacks used in Sections 3.1-3.4 of the paper
  main_*.py          scripts of the original experiments and of the compared methods
experiments/         experiments of the revised manuscript (E1-E9), with their results
examples/demo.py     minimal embed / extract / verify / localize example
tests/               unit tests (no dataset needed) and an integration test
scripts/             preparation of the test images
data/                image manifest with checksums
```

## Installation

Python 3.10 or later (tested with 3.13).

```bash
git clone https://github.com/<user>/charlier-meixner-dual-watermarking.git
cd charlier-meixner-dual-watermarking
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

```bash
python examples/demo.py                 # synthetic image if the test images are not installed
python examples/demo.py --cover my_image.bmp --config "M I" --key "<your secret key>"
python -m pytest tests/test_dual_scheme.py
```

`demo.py` embeds the watermark, reports the PSNR and the BER of the recovered watermark,
verifies the untouched image (no block flagged), pastes a 64×64 patch and prints the
flagged blocks. The key of the examples is only illustrative: the security of the fragile
layer depends on K, which should be a random key of at least 128 bits.

Use from Python:

```python
import sys; sys.path.insert(0, "src")
from dual_scheme.dual_watermarking_scheme import DualWatermarkingScheme
from helpers.orthonornal_matrix import orthmatrix

A = orthmatrix(0)                                   # C I
scheme = DualWatermarkingScheme(key, 150, 62, 0.58, 0.24, A, A)
marked = scheme.insert(cover, bits)                 # bits: 4096 '0'/'1' (Arnold-scrambled watermark)
bits_out = scheme.extraction(received, "name.png", save_tampered=False)
tamper_map = scheme.authenticate(received)          # 64×64 boolean map for a 512×512 image
```

## Test images

The images come from the RGB-BMP Steganalysis Dataset (Al-Jarrah, 2018,
[doi:10.17632/sp4g8h7v8k.1](https://doi.org/10.17632/sp4g8h7v8k.1)) and are not included.
Download and extract the dataset, then run

```bash
python scripts/prepare_images.py --source /path/to/extracted/dataset
```

This places the 24 test images in `src/static/cover_images/` and the 100 images of the
ablation study in `data/covers100/`, after checking their SHA-256 checksums
(see [data/README.md](data/README.md)).

## Reproducing the experiments

Run every command from the repository root. Each folder contains the results reported in the
paper (`results/`, CSV, JSON and LaTeX tables) and the figures (`figures/`, vector PDF).
Comments and console messages of the experiment scripts are in Spanish.

| Experiment | Paper (revised manuscript) | Commands |
| --- | --- | --- |
| E1 Tamper localization (24 images, 5,760 tampered images) | Sec. 3.7, Table 9, Figs. 21-23 | `python experiments/exp1_tamper_localization/run_exp1.py`<br>`python experiments/exp1_tamper_localization/plot_exp1.py` |
| E2 Parameter sensitivity (Δ, polynomial parameters, coefficient position) | Sec. 2.4 (coefficient position); response to the reviewers | `python experiments/exp2_parameter_sensitivity/run_exp2.py all`<br>`python experiments/exp2_parameter_sensitivity/plot_exp2.py` |
| E3 Ablation against the DCT (first version, superseded by E9) | — | `python experiments/exp3_ablation_dct/run_exp3.py`<br>`python experiments/exp3_ablation_dct/analyze_exp3.py` |
| E4 Complexity and execution time | Sec. 3.8, Table 10, Figs. 24-25 | `python experiments/exp4_complexity/run_exp4.py --label mymachine`<br>`python experiments/exp4_complexity/plot_exp4.py --label mymachine` |
| E5 Security and key sensitivity | Sec. 3.10, Tables 12-13, Fig. 27 | `python experiments/exp5_security/run_exp5.py`<br>`python experiments/exp5_security/make_exp5_outputs.py` |
| E6 Execution time versus noise intensity | Sec. 3.9, Table 11, Fig. 26 | `python experiments/exp6_runtime_noise/run_exp6.py --label mymachine`<br>`python experiments/exp6_runtime_noise/plot_exp6.py --label mymachine` |
| E8 Properties of the moment transform | Sec. 2.4, Table 3, Fig. 2 | `python experiments/exp8_moment_properties/run_exp8.py`<br>`python experiments/exp8_moment_properties/plot_exp8.py` |
| E9 Six-transform ablation and interaction of the two layers (100 images) | Secs. 3.5-3.6, Tables 5-8, Figs. 18-19 | `python experiments/exp9_ablation_layers/run_exp9.py transforms`<br>`python experiments/exp9_ablation_layers/run_exp9.py layers`<br>`python experiments/exp9_ablation_layers/run_exp9.py timing`<br>`python experiments/exp9_ablation_layers/analyze_exp9.py` |

Notes:

- Random attacks and tampered regions use fixed seeds, so the results are reproducible.
- The timings in `results_cloud/` (E4, E6) were measured on an Intel Xeon at 2.1 GHz with 2 logical
  cores. Times on another machine will differ; use `--label` to keep them in a separate folder.
- `experiments/exp2_parameter_sensitivity/fastwm.py` is a vectorized implementation of the scheme
  that produces bit-identical outputs to `DualWatermarkingScheme` and is used by most experiments.
- E7 does not exist as a separate folder; the numbering follows the revision plan.

## Status of the code

- The fragile layer was reimplemented in October 2026 (keyed HMAC over the block index and the
  canonical block). The previous version wrote the same key-derived bits in every block and did not
  authenticate the content. All authentication results of the revised manuscript use the new layer.
- The scripts in `src/` that reproduce Sections 3.1-3.4 of the original submission and the
  reimplementations of the compared methods (`src/main_*_et_al_*.py`) are kept as they were used
  and are being revised.

## Citation

If you use this code, please cite the paper (reference to be completed upon publication) and
this repository (see [CITATION.cff](CITATION.cff)).

## License

[MIT](LICENSE)
