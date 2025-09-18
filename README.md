# Wrapper SD-DINO Feature Extractor

A wrapper for feature extractor from [**A Tale of Two Features**](https://github.com/Junyi42/sd-dino) that runs over an image dataset, extracts features and applies PCA dimensionality reduction.

Used as a part of [**DualPM**](https://github.com/DualPM/DualPM_Paper) project.

## 🚀 Installation
Setup environment following `install.sh`

## 📚 Usage
1. Modify `configs/extract_feats.yml` to configure `load_pca_path`, `test_root` and `features_out_root`
2. Run: `python scripts/extract.py configs=configs/extract_feats.yml`

## 📝 Attribution
Wrapper for: [github.com/Junyi42/sd-dino](https://github.com/Junyi42/sd-dino)

Please cite the original paper if you use this in your research.

## 🧑‍💻 Contributors
- **Tomas Jakab** (tomj@robots.ox.ac.uk)
- **Ben Kaye** (kaye@robots.ox.ac.uk)
