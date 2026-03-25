# two-stage-defect-classification-basedonViT

This repository contains a two-stage machine learning pipeline developed for the March 2026 Semiconductor Solutions Challenge. The project addresses semiconductor defect classification under limited training data and strong class imbalance.

## Overview

The proposed system combines pretrained visual embeddings, unsupervised routing, and specialized downstream classifiers:

1. DINOv2 (frozen Vision Transformer) extracts both global CLS embeddings and local patch-level features.
2. KMeans clustering routes samples into visually coherent groups.
3. Specialized per-cluster classifiers perform defect prediction within each routed subset.
4. A confidence-aware inference module flags ambiguous cases for manual review.
5. A Good-Gate threshold controls false positives on normal wafers.

## Highlights

- Designed a two-stage routing framework for small-sample, imbalanced defect classification.
- Built a hybrid representation pipeline using global and local DINOv2 embeddings.
- Implemented 12 specialized routed classifiers with fallback handling for uncertain assignments.
- Added confidence-aware manual-review logic for human-in-the-loop inspection.
- Achieved 84.5% classification accuracy, 0.986 AUROC, and 0.903 average precision, while maintaining <5% false positive rate on normal wafers.
