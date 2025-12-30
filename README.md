# Mask-Guided Colour-Mix Data Augmentation for Rice Sheath Blight Severity Estimation

## Overview
This repository provides the complete implementation accompanying the research work on **rice sheath blight (RSB) disease severity estimation** using deep learning. The primary contribution is a **Mask-Guided Colour-Mix data augmentation strategy** designed to preserve plant anatomical structure while improving dataset diversity under real-field conditions.

Baseline augmentation methods—traditional augmentation, severity-aware Mixup, and GAN-based image synthesis—are also included for fair comparison.

---

## Key Contributions
- Mask-Guided Colour-Mix augmentation preserving leaf anatomy
- HSV-space color manipulation decoupling symptoms from lighting
- Two-stage augmentation pipeline (Traditional → Advanced)
- Severity-aware Mixup strategy
- GAN-based augmentation as a comparative baseline
- Fully reproducible training and evaluation pipeline

---

## Repository Structure

Mask-Guided-ColourMix/
├── augmentation/
│   ├── traditional_augmentation.py
│   ├── severity_aware_mixup.py
│   ├── mask_guided_colormix.py
│   └── gan_based_augmentation.py
├── training/
│   └── model_training_and_evaluation.py
├── data/
│   └── original_dataset/   (not included)
├
├── LICENSE
└── README.md

---

## Dataset Description
- Crop: Rice (*Oryza sativa L.*)
- Disease: Rice Sheath Blight (*Rhizoctonia solani*)
- Severity Levels: 6
- Image Type: RGB field images

## Dataset Source
The original rice sheath blight dataset used in this study is available at:

**DOI:** Varghese K, Sonu  (2025), “Colour-Mix Augmentation Dataset for Rice Sheath Blight Severity Estimation”, Mendeley Data, V1, doi: 10.17632/489spnkhp5.1

Only the original dataset is published. All augmented samples are generated using the provided scripts and are not redistributed.

---

## How to Run

Install dependencies:
```
pip install -r requirements.txt
```

Run proposed augmentation:
```
python augmentation/mask_guided_colormix.py
```

Train and evaluate model:
```
python training/model_training_and_evaluation.py
```

---

## Citation
If you use this code, please cite:

@misc{colormix2025,
  author = {Sonu Varghese K},
  title = {Mask-Guided Colour-Mix Data Augmentation for Rice Sheath Blight Severity Estimation},
  year = {2025},
  url = {https://github.com/your-username/Mask-Guided-ColourMix}
}

---

## License
MIT License
