# Test images

The experiments use 512×512 RGB images from the **RGB-BMP Steganalysis Dataset**
(M. Al-Jarrah, Mendeley Data, V1, 2018, [doi:10.17632/sp4g8h7v8k.1](https://doi.org/10.17632/sp4g8h7v8k.1)),
which contains images derived from the Caltech Birds and NRC collections.
The images are not redistributed here.

| Set | Images | Folder | Used by |
| --- | --- | --- | --- |
| `test24` | 12 Caltech Birds (C…) and 12 NRC (NB…) | `src/static/cover_images/` | Experiments 1, 2, 4, 5, 6, 8 and the scripts in `src/` |
| `ablation100` | C0001–C0050 and NB0001–NB0050 | `data/covers100/` | Experiments 3 and 9 |

`images_manifest.csv` lists every file with its SHA-256 checksum. After downloading and
extracting the dataset, run from the repository root:

```bash
python scripts/prepare_images.py --source /path/to/extracted/dataset
```

The script searches the folder recursively, copies the listed files to their folders and
reports any file that is missing or whose checksum differs.
