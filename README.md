# Histopathologic Cancer Detection — Deep Learning Pipeline

A high-performance PyTorch pipeline for histopathologic cancer detection using convolutional neural networks, GPU optimization, and confidence-aware evaluation.

This project implements and compares **two CNN architectures** on a large Kaggle histopathology dataset, focusing on:

* Efficient GPU training
* Dynamic data augmentation with RAM caching
* Confidence-based reliability analysis
* Detailed performance evaluation and visualization

---

## Project Overview

This repository presents an complete end-to-end deep learning workflow for binary cancer classification from histopathology images.

Two models are implemented and evaluated:

* **Baseline Model** — Lightweight CNN for fast and stable training
* **Advanced Model** — Deeper CNN with regularization and stronger feature extraction

The pipeline emphasizes:

* GPU-optimized training
* mixed precision acceleration (AMP)
* asynchronous data prefetching
* structured evaluation and visualization
* confidence-tier performance analysis

---

## Dataset

**Dataset:** Histopathologic Cancer Detection (Kaggle)

* ~220k RGB images
* Resolution: 96 × 96
* Binary classification (Cancer / No Cancer)
* Nearly balanced class distribution

### Class Distribution

![Class Distribution](class_distribution.png)

The dataset is close to balanced, enabling fair comparison between models without heavy class weighting.

---

## System Architecture

### Data Pipeline

* RAM caching of decoded images
* Dynamic augmentation per epoch
* GPU prefetching for faster training
* Mixed precision training (AMP)

### Models

#### Baseline Model

* 2 convolution blocks
* Batch normalization
* Max pooling
* Fully connected classifier

Designed for speed and stability.

#### Advanced Model

* 4 convolution blocks
* Batch normalization
* Dropout regularization
* Global average pooling
* Deeper classifier

Designed for improved generalization and robustness.

---

## Training Configuration

| Parameter       | Value             |
| --------------- | ----------------- |
| Batch size      | 192               |
| Epochs          | 15                |
| Optimizer       | Adam              |
| Learning rate   | 1e-4              |
| Loss            | BCEWithLogitsLoss |
| Image size      | 96 × 96           |
| Mixed precision | Enabled           |

---

## Training Performance

### Baseline Model Training

![Baseline Training](baseline_model_training_history.png)

### Advanced Model Training

![Advanced Training](advanced_model_training_history.png)

Both models converge smoothly, with the advanced model showing stronger validation stability.

---

## Model Evaluation

### Confusion Matrices

**Baseline Model**

![Baseline Confusion Matrix](baseline_model_confusion_matrix.png)

**Advanced Model**

![Advanced Confusion Matrix](advanced_model_confusion_matrix.png)

The advanced model demonstrates improved error handling and reduced misclassification.

---

### ROC Curve Comparison

![ROC Curve](roc_curve_comparison.png)

Both models achieve strong separability. The advanced model achieves a higher AUC.

---

### Precision-Recall Curve

![Precision Recall](precision_recall_curve.png)

The advanced model maintains stronger precision at higher recall levels.

---

## Confidence-Based Reliability Analysis

A key contribution of this project is analyzing predictions by confidence tiers:

* Low confidence (< 50%)
* Medium confidence (50–90%)
* High confidence (> 90%)

### Confidence Tier Comparison

![Confidence Analysis](confidence_analysis.png)

### Error vs Confidence

**Baseline Model**

![Baseline Error Confidence](baseline_model_error_by_confidence.png)

**Advanced Model**

![Advanced Error Confidence](advanced_model_error_by_confidence.png)

### Key Insight

The advanced model significantly reduces **false negatives**, especially in **high-confidence predictions**, which is critical in cancer detection systems.

This indicates:

* better feature representation
* stronger decision boundaries
* improved reliability in confident predictions

---

## Misclassification Analysis

![Misclassification Comparison](misclassification_comparison.png)

The advanced model achieves:

* fewer false negatives
* improved balance between precision and recall
* more reliable high-confidence predictions

---

## Quantitative Comparison

A full metric comparison is available in:

* `model_comparison.csv`
* `model_results.json`

### Performance Summary

![Model Comparison](model_comparison.png)

Metrics include:

* Accuracy
* Precision
* Recall
* F1 score
* AUC-ROC
* Average precision

The advanced model consistently outperforms the baseline.

---

## Sample Predictions

**Baseline Model**

![Baseline Predictions](baseline_model_predictions.png)

**Advanced Model**

![Advanced Predictions](advanced_model_predictions.png)

---

## Generated Outputs

The pipeline automatically produces:

### Figures

* class distribution
* training history
* confusion matrices
* ROC & precision-recall curves
* confidence analysis
* misclassification comparisons
* prediction visualizations

### Models

* best checkpoints
* final trained weights

### Data

* metric summaries (CSV + JSON)
* confidence tier analysis

---

## How to Run

### Requirements

* Python 3.10+
* PyTorch with CUDA
* torchvision
* pandas
* matplotlib
* seaborn
* scikit-learn
* tqdm

### Training

```bash
python shot.py
```

The script automatically:

* loads the dataset
* trains both models
* generates metrics and visualizations
* saves checkpoints and results

---

## Project Goals

This project demonstrates:

* practical deep learning engineering
* GPU-optimized training pipelines
* comparative model analysis
* reliability-focused evaluation
* professional visualization and reporting

It serves as a foundation for more advanced architectures and scaling experiments.

---

## Future Work

Planned improvements include:

* transfer learning with modern architectures
* ensemble models
* advanced augmentation strategies
* cross-validation experiments
* model calibration studies

---

## License

MIT License

---

## Author

Developed as a deep learning portfolio project focused on cancer detection and high-performance model engineering.
