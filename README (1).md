# CIE 2 – Activity 2: Binary Classification, Logistic Regression (AD3304)

Course: Regression and Classification Algorithms, T.Y. B.Tech AI \& DS, Sem V, 2026–27
Author: Ayush Rajgure | Roll No.: AD3304 | AI \& DS, Div C
https://github.com/complexiota

## What this activity does

Predicts whether a breast-mass sample is malignant (positive class) or benign from 10 nucleus measurements, and studies how the decision threshold trades precision against recall.

## Dataset

Wisconsin Diagnostic Breast Cancer (569 samples, 10 "mean" features; 37 % malignant).
Source: Wolberg, Street \& Mangasarian (1995), UCI ML Repository, https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic (loaded with `sklearn.datasets.load\\\_breast\\\_cancer`). Coding flipped so malignant = 1.

## Work completed

Class balance, stratified 80:20 split (455 / 114), train-only scaling, sigmoid plot, coefficients and odds ratios, probabilities and labels, confusion matrix, full metric table with formulas, classification report, majority-class baseline, ROC and PR curves, threshold table (0.3 / 0.5 / 0.7), train-vs-test check, decision-behaviour plot, new prediction, decision log and limitations.

## Key results (test set, threshold 0.5)

Accuracy 0.930, precision 0.886, recall 0.929, specificity 0.931, F1 0.907, ROC-AUC 0.984 (baseline accuracy 0.632 with recall 0).
Confusion matrix: TN 67, FP 5, FN 3, TP 39. A missed cancer costs more than a false alarm, so a threshold around 0.3 (recall 0.95, precision 0.82) is recommended, to be confirmed on validation data. Decision aid only, not a diagnosis.

## Reproduce

`pip install numpy pandas scikit-learn matplotlib`, then run the notebook top to bottom (seed 42).

