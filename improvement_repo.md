# Repository improvement plan

## Scope and evidence

This plan translates the supplied 20 October 2026 audit into implementation work. Current source and the complete repository tree were rechecked on 27 October 2026 at [`16df6d74a8cd927dc87632b61f9795f5c5a74d67`](https://github.com/b23bb1023/Cancer-Detection-Model/commit/16df6d74a8cd927dc87632b61f9795f5c5a74d67), which still matches the audited snapshot. No open pull requests were present at that check.

All four audit findings below remain visible in the current source. This PR changes documentation only; every technical item remains open. Priorities describe implementation order, not a new audit severity rating.

Verification was limited to source/tree inspection, Python syntax parsing, and documentation checks. No dataset, model weights, PyTorch execution, GPU training, or fresh performance measurement was available. Patient/slide leakage has neither been demonstrated nor excluded.

## Delivery order

| Priority | Work item | Completion signal |
| --- | --- | --- |
| P1 | Preserve the batch dimension everywhere (CDM-01) | One-item evaluation and prediction tests pass |
| P1 | Separate faithful resume from fine-tuning (CDM-03) | Worse resumed epochs cannot overwrite a better checkpoint |
| P1 | Separate selection from final evaluation (CDM-02) | Frozen models evaluated on an independent, documented split |
| P2 | Repair confidence summaries (CDM-04) | Every valid probability enters one bin; empty bins remain unavailable |
| P2 | Establish tested environments and CPU CI | Clean environment passes focused synthetic-data checks |
| P2 | Make data/run configuration and caching explicit | Isolated runs with validated inputs and bounded memory behavior |
| P3 | Package evidence and contribution guidance | A new contributor can reproduce a small run and inspect its provenance |

## P1 — Preserve singleton batches (CDM-01, Medium in audit)

**Current evidence:** [`train_epoch` and `validate_epoch`](https://github.com/b23bb1023/Cancer-Detection-Model/blob/16df6d74a8cd927dc87632b61f9795f5c5a74d67/small_CNN_models.py#L350-L419), [`get_predictions`](https://github.com/b23bb1023/Cancer-Detection-Model/blob/16df6d74a8cd927dc87632b61f9795f5c5a74d67/small_CNN_models.py#L463-L489), and [`show_predictions`](https://github.com/b23bb1023/Cancer-Detection-Model/blob/16df6d74a8cd927dc87632b61f9795f5c5a74d67/small_CNN_models.py#L752-L789) use unconstrained `.squeeze()`. A model output shaped `[1, 1]` becomes a scalar. Loss expects the labels' batch dimension, and prediction collection attempts to extend a list from that scalar. Validation retains its final incomplete batch.

**Implementation:**

- Replace each unconstrained output squeeze with `.squeeze(-1)` or an explicit batch-preserving reshape; standardize the contract as logits and labels shaped `[batch]`.
- Apply the same contract to training, validation, prediction collection, and sample visualization.
- Keep all validation samples. Dropping the final batch would hide the defect and silently change evaluation coverage.

**Acceptance checks:** Use both real model classes on CPU in evaluation mode with batch sizes 1 and 2; assert output/probability lengths match inputs. Exercise validation and prediction over a synthetic loader ending in one sample, compute finite loss, and verify no sample is lost. Add a one-item visualization check with a noninteractive plotting backend. Test training loss shape independently without conflating it with any batch-normalization constraints.

## P1 — Preserve best checkpoints and resume state (CDM-03, Medium in audit)

**Current evidence:** [checkpoint helpers](https://github.com/b23bb1023/Cancer-Detection-Model/blob/16df6d74a8cd927dc87632b61f9795f5c5a74d67/small_CNN_models.py#L318-L350) save model/optimizer/epoch/accuracy. [`train_model`](https://github.com/b23bb1023/Cancer-Detection-Model/blob/16df6d74a8cd927dc87632b61f9795f5c5a74d67/small_CNN_models.py#L422-L460) recreates scheduler, scaler, early stopping, history, and `best_val_acc=0`. [`main`](https://github.com/b23bb1023/Cancer-Detection-Model/blob/16df6d74a8cd927dc87632b61f9795f5c5a74d67/small_CNN_models.py#L1011-L1049) ignores the returned `start_epoch` when calling training. A resumed score can overwrite a better checkpoint.

**Implementation:**

- Introduce explicit new-run, resume, and fine-tune modes with separate run directories.
- For resume, restore the next global epoch, best validation score, scheduler/scaler state, early-stopping state, training history, and NumPy/PyTorch CPU/CUDA random states where applicable.
- Store a versioned checkpoint schema and run configuration; reject incompatible model or dataset settings clearly.
- Keep a latest-resume checkpoint separate from the best-selected model. Update the best model only on improvement under the documented selection rule.
- Use restricted state-dictionary loading where possible. Document trusted-only loading for any remaining legacy pickle checkpoints.

**Acceptance checks:** Begin with a saved best accuracy of 0.95, simulate a resumed epoch at 0.80, and assert the best checkpoint contents are unchanged. Verify epoch numbering and learning-rate progression across a short interrupted/resumed run against an uninterrupted run in a controlled deterministic CPU setting. Cover a missing checkpoint, a legacy weights-only file, and incompatible schema/configuration errors. Fine-tuning must create a distinct run rather than overwrite the source experiment.

## P1 — Establish independent final evaluation (CDM-02, Medium in audit)

**Current evidence:** [the split](https://github.com/b23bb1023/Cancer-Detection-Model/blob/16df6d74a8cd927dc87632b61f9795f5c5a74d67/small_CNN_models.py#L906-L911) creates only train/validation data. Validation drives selection and early stopping, and [final predictions](https://github.com/b23bb1023/Cancer-Detection-Model/blob/16df6d74a8cd927dc87632b61f9795f5c5a74d67/small_CNN_models.py#L1063-L1080) reuse `val_loader`. `TEST_DIR` is only counted. Committed charts lack the numeric result files and provenance needed to reconstruct the reported run.

**Implementation:**

- Document the dataset source, version, labeling rule, exclusions, and available original sample/slide/patient identifiers before choosing a split protocol.
- Persist immutable train, validation, and final-evaluation sample IDs. Use group-aware splitting when reliable grouping exists. If it does not, state that limitation and avoid patient-independent claims.
- Keep validation for model selection. Freeze the selected weights and threshold before evaluating once on a separate labeled holdout; use nested/grouped evaluation if required by the data and research question.
- Do not treat an unlabeled competition test directory as a scored holdout. Add a distinct labeled evaluation input and explicit schema.
- Export numeric metrics, per-sample predictions, split manifest/hash, dataset version/hash, Git commit, run configuration, environment versions, and model checksum. Preserve historical plots as selected-validation artifacts until rerun.

**Acceptance checks:** Assert disjoint original sample IDs and, when available, group IDs. Instrument the pipeline to prove final-evaluation data is never passed into fitting, scheduling, threshold selection, or checkpoint selection. A synthetic fixture must exercise the frozen evaluation path. A real experiment must publish its documented split and machine-readable results before new generalization claims are added.

## P2 — Make confidence summaries mathematically valid (CDM-04, Low in audit)

**Current evidence:** [prediction confidence](https://github.com/b23bb1023/Cancer-Detection-Model/blob/16df6d74a8cd927dc87632b61f9795f5c5a74d67/small_CNN_models.py#L480-L487) is `max(p, 1-p)`, so it lies in `[0.5, 1]`. [Tier definitions](https://github.com/b23bb1023/Cancer-Detection-Model/blob/16df6d74a8cd927dc87632b61f9795f5c5a74d67/small_CNN_models.py#L542-L560) include an impossible “Low (<50%)” category; [plots](https://github.com/b23bb1023/Cancer-Detection-Model/blob/16df6d74a8cd927dc87632b61f9795f5c5a74d67/small_CNN_models.py#L659-L749) replace missing values with zero and exclude exact-confidence-one from the final error bin.

**Implementation:**

- Define shared intervals entirely within `[0.5, 1]`, with an inclusive final endpoint and unambiguous labels.
- Represent empty groups as unavailable, display sample counts, and reuse the same binning logic in CSV export, plots, and printed interpretation.
- Remove conclusions that equate high raw confidence with reliability. Add a held-out calibration study with an appropriate score and reliability visualization after establishing the evaluation protocol.

**Acceptance checks:** For probabilities `[0, 0.1, 0.5, 0.9, 1]`, every sample belongs to exactly one applicable tier/bin, including both endpoints. Empty bins must not display measured zero error or accuracy. Tier counts must sum to the prediction count in plots and exports. Calibration claims require separately recorded held-out measurements.

## P2 — Add a tested environment and focused CPU CI

**Current evidence:** The current tree contains no dependency manifest, lockfile, tests, or Actions workflow. Imports include torch, torchvision, NumPy, pandas, Pillow, matplotlib, seaborn, scikit-learn, and tqdm. The code uses `torch.amp`; a broad Python version claim alone does not establish package compatibility.

**Implementation:**

- Choose and actually test a supported Python/PyTorch/torchvision pair; publish a dependency manifest and reproducible constraints/lock strategy. Document CPU and CUDA installation paths separately.
- Add focused tests for the four audited defects and a small generated-image/CSV smoke run that needs no external data, checkpoint, network access, or GPU.
- Gate CUDA AMP on both the flag and CUDA device consistently. Verify a CPU run with AMP disabled.
- Run the focused suite on pull requests with read-only repository permissions. Add optional hardware validation separately; avoid downloading the full dataset in CI.

**Acceptance checks:** A clean environment resolves the recorded dependencies and passes syntax/import checks plus the focused suite. The smoke run emits the expected numeric output schema in a temporary directory. Record exact package and hardware versions for the CUDA check rather than claiming GPU coverage from CPU CI.

## P2 — Make runs and resource use explicit

**Current evidence:** [configuration constants](https://github.com/b23bb1023/Cancer-Detection-Model/blob/16df6d74a8cd927dc87632b61f9795f5c5a74d67/small_CNN_models.py#L49-L61), [eager image caching](https://github.com/b23bb1023/Cancer-Detection-Model/blob/16df6d74a8cd927dc87632b61f9795f5c5a74d67/small_CNN_models.py#L91-L144), and fixed output paths currently couple a run to its working directory. Input validation and a cache budget are absent. Memory concerns are an estimate, not a reproduced out-of-memory failure.

**Implementation:**

- Add a command-line/config-file entry point for data paths, output directory, seed, epochs, batch size, workers, AMP, and cache mode/budget.
- Validate CSV columns, binary labels, unique/nonempty IDs, image existence, and sufficient samples before caching or plotting. Do not require an unused test directory.
- Offer disk-backed loading and estimate cache usage before allocation. Fail clearly or fall back when a configured budget is exceeded.
- Use isolated run directories with explicit overwrite/resume behavior. Add a headless plotting option and persist configuration plus split IDs before training.
- Separate data, model, training, evaluation, and reporting functions into importable modules while preserving a simple entry point.

**Acceptance checks:** Cover a missing directory/image, malformed labels, duplicate IDs, an undersized dataset, and a cache-budget boundary with small fixtures. Two runs must write to different directories without loading each other's checkpoints. A headless smoke run must complete without GUI interaction.

## P3 — Finish the reproducibility and contribution package

- Add a data-provenance note and a model card describing intended research use, label semantics, split protocol, limitations, and evidence available for the actual trained weights.
- Publish one small reproducible reference run with its numeric metrics, command/configuration, split hashes, dependency versions, and model checksum. Link plots to that run rather than inferring model superiority from unversioned images.
- Add concise contribution instructions describing environment setup, the focused checks, and how results are regenerated. Ignore local datasets, checkpoints, virtual environments, and run outputs by default; share large artifacts through an intentional release process.
- Once CI exists, configure project-appropriate merge checks. Repository settings should be changed separately and verified; this documentation PR does not change governance.

**Acceptance:** A contributor unfamiliar with the project can create the recorded environment, run the synthetic smoke experiment, locate its outputs, and explain which metrics are validation versus independent evaluation. Each published result is traceable to code, data, split, configuration, and weights.
