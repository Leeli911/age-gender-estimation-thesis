```markdown
# Age and Gender Estimation from Historical Portraits

This repository implements five deep learning models for facial age and gender estimation using historical portrait images. The models were developed as part of the master's thesis:

**"From Faces to Ages: Enhancing Historical Recognition with Transfer Learning"**

---

## Overview

The project investigates how task formulation, gender supervision, and dataset balancing affect prediction performance and fairness. It includes:

| Model | Description                                     |
|-------|-------------------------------------------------|
| M1    | Multi-task regression with gender classification |
| M2    | Single-task age regression only                  |
| M3    | Grouped age classification with gender           |
| M4    | Grouped age classification only                  |
| M5    | Cascaded regression conditioned on gender        |

---

## Code Structure

```

.
├── train\_age76\_gender.py         # M1
├── train\_age76\_nongender.py     # M2
├── train\_group5\_gender.py       # M3
├── train\_group5\_nongender.py    # M4
├── train\_cascade\_gender\_age.py  # M5
├── model.py                     # Backbone and architecture
├── loss.py                      # Custom loss functions
├── helperT.py                   # Dataset loading and splitting
├── run\_models.sh                # Full model training across datasets
├── analyze\_merged\_data.ipynb    # Dataset analysis notebook
└── README.md                    # Project description

````

---

## Setup

This project requires:

- Python 3.11
- PyTorch ≥ 2.0
- torchvision ≥ 0.15
- timm ≥ 0.9.2
- scikit-learn, matplotlib, pandas

Install all dependencies:

```bash
pip install -r requirements.txt
````

---

## Data Preparation

Each training script expects:

* Pre-cropped and aligned face images (via MTCNN or InsightFace)
* Metadata CSVs with columns: `image_name`, `age`, `gender`, and optionally `group` (for age group labels)

Ensure that data organization aligns with the logic in `helperT.py`.

---

## Model Training

To train a specific model:

```bash
python train_age76_gender.py  # Example: Model 1
```

To train all five models under both MTCNN and InsightFace data setups:

```bash
bash run_models.sh
```

Trained weights, logs, and visualizations will be saved under model-specific folders.

---

## Output and Evaluation

Each model outputs:

* Training/validation loss curves
* Group-wise and overall MAE
* Gender classification metrics (if applicable)
* Residual plots and scatter plots for calibration inspection

These are saved automatically during training.

---

## Thesis Reference

This repository supports the Master's thesis submitted to Uppsala University:

**From Faces to Ages: Enhancing Historical Recognition with Transfer Learning**
\[Insert final thesis PDF link here]

---

## License and Contact

This code is released for academic use only.

Author: Li Li
Program: MSc in Data Science, Uppsala University
Email: `li.li.5064@student.uu.se`

```
