# Histopathologic Cancer Detection

A PyTorch research project comparing two convolutional neural networks for binary classification of histopathology image patches.

The repository contains one training and analysis script, example chart artifacts, and an MIT software license. It is a portfolio experiment: independent patient-level or clinical validation has not been established.

## Project status

The current script supports dynamic training augmentation, in-memory image caching, CUDA prefetching, optional mixed precision, validation checkpointing, and diagnostic plots.

Known batch-shape, checkpoint-resume, evaluation, and confidence-bin limitations remain in the current script and are described below. Address them before treating a run as reproducible evidence. This documentation update does not fix those implementation issues.

## Models and workflow

| Component | Current implementation |
| --- | --- |
| Baseline model | Two convolution blocks with batch normalization and pooling, followed by a dense classifier |
| Advanced model | Four convolution blocks, batch normalization, dropout, global average pooling, and a dense classifier |
| Output | One logit per patch; sigmoid probability and a strict `p > 0.5` classification threshold |
| Training data | Decoded RGB images cached in RAM; random flips, rotation, and color jitter applied during access |
| Validation data | Loaded from disk with resize, tensor conversion, and normalization; no random augmentation |
| Model selection | Best checkpoint by validation accuracy; learning-rate scheduling and early stopping use validation loss |
| Final analysis | Selected models evaluated again on the same validation split |

`main()` creates a label-stratified 80/20 training/validation split using `random_state=17`. The `test/` directory is counted for image statistics, but the script does not produce test predictions or evaluate a labeled test set.

## Set up a local experiment

### 1. Prepare Python and dependencies

The previous project guidance targets Python 3.10+, but the repository does not yet provide a tested dependency lockfile or a verified Python/PyTorch/torchvision/CUDA version matrix.

From a local checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn tqdm pillow
```

On Windows, activate with `.venv\Scripts\activate` instead. The install command is an unpinned starting point, not a tested reproducible environment. Select mutually compatible PyTorch and torchvision builds for your hardware and record the versions used for each run.

A CUDA-capable GPU is optional: the script selects CPU when CUDA is unavailable. For CPU experiments, set `USE_AMP = False` in the configuration section before running, because validation currently enters a CUDA autocast context whenever that flag is true.

### 2. Place the dataset

The original project describes the Kaggle Histopathologic Cancer Detection dataset. Obtain the data separately under its source terms; no images or labels are included here. Record the exact dataset source, version, label definition, and reuse terms with your experiment.

By default, launch from a working directory containing:

```text
small_CNN_models.py
train_labels.csv
train/
    <image_id>.tif
test/
    <test_image_id>.tif
```

`train_labels.csv` must contain `id` and binary `label` columns. Each training row is loaded from `train/<id>.tif`, converted to RGB, and resized to 96 × 96. IDs should omit the `.tif` suffix. Use the source dataset's label definition when interpreting class 0 and class 1; a patch label does not establish a patient diagnosis.

Both `train/` and `test/` must exist with the current defaults because `main()` lists both directories. Test images are not otherwise used. To use different paths, edit `LABELS_PATH`, `TRAIN_DIR`, and `TEST_DIR` near the top of [small_CNN_models.py](small_CNN_models.py).

### 3. Review configuration and run

| Setting | Default |
| --- | --- |
| `BATCH_SIZE` | 192 |
| `LEARNING_RATE` | 0.0001 |
| `EPOCHS` | 15 maximum per training call |
| `IMAGE_SIZE` | 96 |
| `NUM_WORKERS` | 4 for validation; training uses 0 |
| `PREFETCH_FACTOR` | 2 |
| `USE_AMP` | True |
| Optimizer / loss | Adam / BCEWithLogitsLoss |

Configuration is currently edited in the script; there is no command-line configuration interface. Training drops its last incomplete batch. Validation retains it, and a final validation batch of one item currently triggers the known shape bug.

Training caches the training split's decoded images in RAM. At roughly 176,000 RGB 96 × 96 patches, raw pixels alone would occupy about 4.9 GB in decimal units, before Python and image-object overhead. This is a capacity estimate, not a measured memory requirement. Reduce the experiment size or add a disk-backed loading mode before attempting a run on a memory-constrained machine.

For a new experiment, use a clean working directory or archive previous output files first. The script automatically loads existing `baseline_model_best.pth` and `advanced_model_best.pth` files and writes fixed output filenames. Its current loading behavior is partial-state extra training, not a faithful resume. Only load checkpoints you trust: the current code uses `torch.load(..., weights_only=False)`.

```bash
python small_CNN_models.py
```

The script displays plots with `plt.show()`, so an interactive run may pause at plot windows. A full run needs local data, sufficient resources, and a compatible environment. No end-to-end training result is guaranteed by the setup steps above.

## Outputs and interpretation

A successful run writes outputs to the **current working directory**, not automatically into `chart/`.

| Output | Contents |
| --- | --- |
| `baseline_model_best.pth`, `advanced_model_best.pth` | Validation-accuracy-selected weights with optimizer state, epoch, and score |
| `baseline_model_final.pth`, `advanced_model_final.pth` | State dictionaries of the selected models reloaded for final analysis |
| `model_results.json` | Per-model metrics, confidence-tier metrics, and some training parameters |
| `model_comparison.csv` | Accuracy, precision, recall, F1, ROC AUC, and average precision |
| `confidence_analysis.csv` | Per-model metrics for populated confidence tiers |
| PNG files | Training histories, confusion matrices, ROC/PR curves, predictions, and confidence/error comparisons |

The CSV/JSON files and trained weights above are generated outputs; they are **not committed** in the current repository. The example PNGs in [chart/](chart/) do not include enough split, dataset-version, environment, or run metadata to independently reconstruct their results.

All metrics currently emitted by the script are **selected-validation results**. Validation is reused for checkpoint selection and final reporting, so these are not untouched test estimates. Patient/slide independence is unknown from the committed material. Label stratification alone does not demonstrate group independence.

The confidence value is `max(p, 1 - p)`, which is always at least 0.5. The current “Low (<50%)” tier is therefore empty by definition, and the error-by-confidence chart excludes values exactly equal to 1.0. Empty categories plotted as zero should not be interpreted as measured accuracy or error. Raw sigmoid confidence also does not establish calibration.

### Example artifacts

These images are retained as historical project artifacts; no new numerical performance claim is made here.

| Example | Artifact |
| --- | --- |
| Training history | [Baseline](chart/baseline_model_training_history.png) · [Advanced](chart/advanced_model_training_history.png) |
| Confusion matrices | [Baseline](chart/baseline_model_confusion_matrix.png) · [Advanced](chart/advanced_model_confusion_matrix.png) |
| Ranking metrics | [ROC comparison](chart/roc_curve_comparison.png) · [Precision-recall comparison](chart/precision_recall_curve.png) |
| Overall comparison | [Model comparison](chart/model_comparison.png) |

## Development priorities

Start with singleton batches and checkpoint preservation, then establish an independent evaluation protocol and reproducible run artifacts before expanding the model architectures.

## License

The software is available under the [MIT License](LICENSE). Dataset access and reuse terms must be checked separately with the data provider.
