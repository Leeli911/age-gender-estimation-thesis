# Age and Gender Estimation from Historical Portraits

This repository contains the full implementation of five deep learning models for age and gender prediction from historical facial images. The models were developed and evaluated as part of the master's thesis titled:

**"From Faces to Ages: Enhancing Historical Recognition with Transfer Learning"**

---

## Overview

The study explores the impact of task design, gender supervision, and data balancing strategies on facial age estimation. Five models are implemented and evaluated:

| Model   | Description                                      |
|---------|--------------------------------------------------|
| Model 1 | Multi-task age regression + gender classification |
| Model 2 | Single-task age regression only                   |
| Model 3 | Multi-task age group classification + gender      |
| Model 4 | Single-task age group classification only         |
| Model 5 | Cascaded gender-specific age regression           |

Each model is implemented in its own training script and shares common data processing and architecture utilities.

---

## Directory Structure

```bash
.
├── train_age76_gender.py            # Model 1 training script
├── train_age76_nongender.py        # Model 2 training script
├── train_group5_gender.py          # Model 3 training script
├── train_group5_nongender.py       # Model 4 training script
├── train_cascade_gender_age.py     # Model 5 training script
├── helperT.py                      # Data loading, preprocessing, stratified splitting
├── model.py                        # Backbone and output head definitions (ConvNeXtV2, SE-ResNeXt50)
├── loss.py                         # Custom loss functions (AgeGenderLoss, AgeOnlyLoss, CascadeLoss)
├── run_all.py                      # Batch training automation script
├── requirements.txt                # Python package requirements
└── README.md                       # This file
````

---

## Reproducing Results

### 1. Environment Setup

This project was developed and tested with:

* Python 3.11
* PyTorch 2.0+
* torchvision 0.15+
* timm >= 0.9.2
* scikit-learn
* matplotlib
* pandas

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 2. Dataset Preparation

The models require preprocessed datasets containing:

* Cropped and aligned face images (using MTCNN)
* Associated CSV files with `image_name`, `age`, `gender`, and optionally `group` annotations

Organize your data in a directory structure compatible with `helperT.py`.

---

### 3. Training a Model

Example: Train Model 1 (age + gender regression)

```bash
python train_age76_gender.py
```

Model checkpoints, logs, and plots will be saved automatically under:

```
logs_age76_gender/
├── best_model.pth
├── train.log
├── loss_curve.png
├── val_mae_curve.png
├── test_scatter.png
└── test_predictions.csv
```

To train all five models in sequence:

```bash
python run_all.py
```

---

## Outputs and Evaluation

Each model logs:

* Training and validation loss
* Validation MAE per epoch
* Final test MAE and group-wise MAE
* Scatter plots of predicted vs. ground-truth age

---

## Thesis Link

This repository supports the thesis submitted to Uppsala University, Department of Information Technology, for the degree of Master in Data Science.

**\[Insert PDF Thesis Link Here]**

---

## License

This code is intended for academic research purposes only. Contact the author for any commercial or redistribution inquiries.

---

## Contact

Li Li
Master's Student, Data Science
Uppsala University, Sweden
Email: `li.li.5064@student.uu.se`

