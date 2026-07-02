# Fruit RUL Prediction

This repository supports research on predicting the remaining useful life (RUL) of fruit from sequential image data and optional numeric factors such as temperature, humidity, and firmness.

The active implementation is currently focused on strawberry experiments, while the shared early pipeline is designed to also support avocado experiments. Model development differs by fruit context, so this README prioritizes the unified data, preprocessing, labeling, and tracking rules that every team member must follow.

## Research Context

The project frames fruit spoilage as an hourly regression problem:

```text
RUL hours = EOL timestamp for the fruit - sample timestamp
```

Each fruit is tracked over time through repeated visual observations. Environmental measurements are aligned by timestamp. Firmness is optional at the schema level: avocado experiments include daily firmness measurements, while the current strawberry workflow does not.

## Active Scope

- Shared early pipeline for avocado and strawberry.
- Current active dataset: newly recorded strawberry data with timestamped images/video and environmental readings.
- Legacy strawberry dataset: low-quality prototype data for model-development experiments only.
- Avocado methodology: placeholder for future attention-based multimodal models.
- App deployment: future proof-of-concept after the MVP research pipeline is stable.

## Pipeline

| Stage | Name | Current Focus | Main Outputs |
| --- | --- | --- | --- |
| 1 | Data acquisition | Raw captures, sensor logs, experiment metadata | Raw videos/images, numeric CSV/JSON files, acquisition report |
| 1.5 | Image inventory and readiness | Raw image integrity, master image inventory, numeric coverage checks | Inventory CSV, QC reports, debug report, property charts |
| 2 | Data preprocessing | Frame extraction, ROI/ID separation, masks, invalid-frame detection | Fruit-ID image folders, mask outputs, preprocessing reports |
| 2.5 | Labeling | EOL anchors and RUL labels | `labels.csv`, `eol_anchors.csv` |
| Parallel | EDA | Dataset inspection and paper evidence | EDA graphs/reports |
| 3.1 | Model training | Placeholder | Fruit-specific training pipelines |
| 3.2 | Model tuning | Placeholder | Tuned checkpoints and tuning reports |
| 4 | Evaluation and XAI | Placeholder | Metrics, XAI outputs, paper-ready evidence |

The code currently stores preprocessing scripts under `src/stage3_preprocessing/` from an older naming scheme. In the project plan, those scripts are treated as Stage 2 preprocessing until the folder naming is refactored.

## Team Roles

| Member | Primary Role | Review Responsibility |
| --- | --- | --- |
| Hung | Data acquisition and technical integration | Confirms capture setup, sensor mapping, and model-development feasibility |
| Cong | Preprocessing execution | Produces fruit-ID folders, masks, CSV/JSON reports, and invalid-frame logs |
| Hai | EDA and paper evidence | Builds dataset statistics, temporal/spatial analyses, and quality summaries |
| Gate checker | Project consistency and approval | Checks rules, leakage prevention, task consistency, and acceptance criteria |

The team reviews progress twice per week. Each review should answer:

- What was completed?
- What is blocked?
- Which files or reports prove the result?
- What must be reviewed before the next step?

## Core Rules

- Raw data is immutable. Do not edit or delete raw captures; exclude bad frames only from model-ready manifests.
- Fruit identity is based on fixed 3x2 ROI position. A physical fruit must return to its original position after daily measurement.
- Dataset splitting must happen by fruit ID, not by frame. Final evaluation should use leave-one-fruit-out cross-validation (LOOCV).
- Every model-ready image must map to a timestamp, fruit ID, experiment ID, and numeric fields where available.
- Environmental data is measured once per box/room and mapped to frames by timestamp.
- Avocado firmness is measured once per day per fruit from five points and averaged; that daily value is mapped to all frames for that fruit-day.
- Strawberry firmness is currently unavailable and must be represented as missing/optional, not invented.
- Invalid frames, hands/devices, black/blank frames, unreadable files, and failed masks must be logged before exclusion.
- EOL is fruit-specific. Avocado EOL uses visual decay plus firmness collapse; strawberry EOL uses the agreed visual anchor unless another numeric basis is introduced.

## Repository Structure

```text
data/
  01_raw/            Raw videos, frames, sensor logs, and acquisition metadata
  02_processed/      Cropped, segmented, assigned, and model-ready intermediate files
  03_split/          Fruit-ID-safe train/val/test split (F01-F04/F06/F05)
  model_A_outputs/   Training history, predictions, metrics for Model A
  model_B_outputs/   Training history, predictions, metrics for Model B
  model_C_outputs/   (same for Model C)
  model_D_outputs/   (same for Model D)

docs/
  TRAINING_GUIDE.md       Model training guide (architecture, training, comparison)
  DATA_PROTOCOL.md        Required metadata, naming rules, integrity checks
  PREPROCESSING_SPEC.md   Stage 2 workflow, outputs, QC checks
  PREPROCESSING_GUIDE.md  Operator guide for running preprocessing
  LABELING_PROTOCOL.md    EOL approval flow, RUL formula, label schema
  EDA_PLAN.md             EDA scope and required reports/graphs
  PROJECT_PLAN.md         Phase plan, owners, acceptance criteria
  PROGRESS_TRACKER.md     Milestone tracker for twice-weekly review
  model_A/ - model_D/     Per-model README and training details

output/
  graphs/
    evaluation/      Comparison charts from compare_models.py
    training/        Per-model training curve plots
  reports/
    evaluation/      Comparison report (auto-generated)
    training/        Per-model training reports
  results/           Final metrics and prediction CSVs

src/
<<<<<<< HEAD
  shared/
    cbam.py               CBAM attention module (used by all 4 models)
  stage3_preprocessing/   Preprocessing scripts (crop, segment, assign, label, split)
  stage4_training/
    model_A/              EfficientNet-B0 + CBAM + GRU
    model_B/              MobileNetV2 + CBAM + LSTM
    model_C/              EfficientNet-B0 + CBAM + LSTM
    model_D/              MobileNetV2 + CBAM + GRU
  stage5_evaluation/
    compare_models.py     Cross-model comparison charts and report
=======
  stage1_5_image_inventory/  Raw image inventory, integrity, and numeric coverage checks
  stage3_preprocessing/  Current preprocessing scripts from the older stage naming
  rul_android_app/       Prototype app surface for later deployment work
>>>>>>> 84323c4dab20ad24dbb39471cd39190483f7d0bd

models/
  model_A/best_model.pth  Trained checkpoints (one per model)
  model_B/best_model.pth
  model_C/best_model.pth
  model_D/best_model.pth
```

## Documentation Map

- [Training Guide](docs/TRAINING_GUIDE.md): **Start here for model training.** Architecture, how to train, evaluation, comparison.
- [Data protocol](docs/DATA_PROTOCOL.md): required metadata, naming rules, integrity checks, and modality handling.
- [Stage 1.5 image inventory plan](docs/STAGE_1_5_IMAGE_INVENTORY_PLAN.md): raw image inventory, readiness checks, reports, and numeric coverage cross-checks.
- [Preprocessing spec](docs/PREPROCESSING_SPEC.md): Stage 2 workflow, outputs, QC checks, and current script mapping.
- [Preprocessing guide](docs/PREPROCESSING_GUIDE.md): Step-by-step instructions for running preprocessing scripts.
- [Labeling protocol](docs/LABELING_PROTOCOL.md): EOL approval flow, RUL formula, label schema, and leakage rules.
- [EDA plan](docs/EDA_PLAN.md): Hai's analysis scope and expected reports/graphs.
- [Project plan](docs/PROJECT_PLAN.md): phase plan, owners, acceptance criteria, and review rhythm.
- [Progress tracker](docs/PROGRESS_TRACKER.md): GitHub-friendly tracker for twice-weekly review.

## Current Script Notes

The repository already contains useful preprocessing and prototype-model scripts, but several still use machine-specific absolute paths. Prefer command-line arguments when scripts support them. Any script without CLI arguments should be refactored before being used as a team-standard workflow.

Useful current components:

- `src/stage3_preprocessing/extracting_frames.py`: extracts frames from video at a fixed sampling interval.
- `src/stage3_preprocessing/crop_images.py`: center-crops raw frames.
- `src/stage3_preprocessing/segmentation.py`: segments strawberries and writes transparent PNG masks/crops.
- `src/stage3_preprocessing/frame_differencing.py`: detects motion/unstable frames and validates masks.
- `src/stage3_preprocessing/assign_id.py`: groups segmented fruit images by ID.
- `src/stage3_preprocessing/label_rul.py`: creates prototype RUL labels.
- `src/stage3_preprocessing/split_data.py`: performs prototype fruit-ID split.

Example for a script that already supports CLI paths:

```bash
python src/stage3_preprocessing/frame_differencing.py --input-dir data/01_raw/<experiment>/<date>/cropped --mask-dir data/02_processed/<experiment>/<date>/segmented --output-csv output/reports/processed/frame_differencing_report.csv
```

## Model Development

### Strawberry RUL Models (Active)

Four hybrid CNN-Attention-RNN architectures have been implemented for strawberry RUL prediction. All models share the same pipeline:

```
CNN Backbone -> CBAM Attention -> Temporal Model (GRU/LSTM) -> Regression Head -> RUL (hours)
```

| Model | CNN | Attention | Temporal | Params | Status |
|-------|-----|-----------|----------|--------|--------|
| **A** | EfficientNet-B0 | CBAM | GRU | 4.76M | Trained |
| **B** | MobileNetV2 | CBAM | LSTM | 3.16M | Trained |
| **C** | EfficientNet-B0 | CBAM | LSTM | 4.94M | Ready to train |
| **D** | MobileNetV2 | CBAM | GRU | 2.98M | Ready to train |

Training produces checkpoints (`models/model_X/`), metrics, predictions, and training history (`data/model_X_outputs/`). A comparison script at `src/stage5_evaluation/compare_models.py` generates charts and a report across all trained models.

**Full training documentation:** [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)

### Quick Start (Training)

```bash
cd src/stage4_training/model_A && python train.py   # train one model
python src/stage5_evaluation/compare_models.py       # compare all trained models
```

### Avocado (Future)

Avocado experiments may later use attention-based multimodal approaches such as ViT/Mamba backbones with multimodal bottleneck fusion, using visual sequences, environmental data, and firmness.

## Minimum Definition of Done

A dataset batch is ready for EDA/model preparation only when:

- Every raw capture has an experiment ID and timestamp.
- Every usable frame has a fruit ID, timestamp, and traceable raw source.
- Temperature and humidity are mapped or explicitly marked missing.
- Firmness is mapped for avocado or explicitly missing for strawberry.
- Excluded frames are logged with reason codes.
- EOL anchors are proposed, reviewed, and approved per fruit.
- RUL labels are calculated in hours.
- Splits are fruit-ID safe and documented.
