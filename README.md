# Cross-Architecture-Multi-Granularity-Knowledge-Distillation-for-Tomato-Leaf-Disease-Detection
## Dataset Download

The tomato leaf disease dataset is available on the
[dataset release page](https://github.com/huangyi1042/Cross-Architecture-Multi-Granularity-Knowledge-Distillation-for-Tomato-Leaf-Disease-Detection/releases/tag/Dataset).

Open the link and download the dataset archive under **Assets**.
[README.md](https://github.com/user-attachments/files/34455320/README.md)
# Cross-Architecture Multi-Granularity Knowledge Distillation for Tomato Leaf Disease Detection

This repository contains training and knowledge distillation scripts associated with **Cross-Architecture Multi-Granularity Knowledge Distillation for Tomato Leaf Disease Detection**.

The framework transfers knowledge from an **RT-DETR-L teacher** to a lightweight **YOLO11-MobileNetV2 student** through three complementary forms of supervision:

- **Localization decision supervision:** reliable teacher predictions are matched to annotated objects and used to refine training bounding boxes.
- **Semantic feature distillation:** feature adapters align intermediate student and teacher representations, with distillation restricted to foreground regions.
- **Spatial-awareness distillation:** the student learns the teacher's spatial attention distributions within expanded foreground regions.

Feature and spatial-awareness losses are introduced progressively during training.

## Repository Contents

| File | Description |
|---|---|
| `RT-DETR.py` | RT-DETR teacher training script; currently configured to resume an existing training checkpoint. |
| `yolov11_mobilenetv2.yaml` | Four-class student architecture configuration with a custom MobileNetV2 backbone and P3/P4/P5 detection outputs. |
| `yolov11_moblienetv2.py` | Training entry point for the student baseline using the original annotated dataset. The filename is retained exactly as uploaded. |
| `generate_matched_response_dataset.py` | Generates a separate dataset with teacher-refined training boxes while preserving the original validation annotations. |
| `rtdetr_yolo_mobile_kd.py` | Feature extraction hooks, feature adapters, foreground masks, feature distillation model, and feature distillation trainer. |
| `rtdetr_yolo_mobile_attention_kd.py` | Multi-granularity distillation model and trainer, including spatial attention losses and progressive weight schedules. |
| `train_response_feature_attention_kd.py` | Main entry point for training with response-label, feature, and spatial-awareness supervision. |

## Requirements and Required Assets

The scripts use **Python**, **PyTorch**, **Ultralytics**, and **PyYAML**. Their current configuration targets an NVIDIA GPU with CUDA support (`device=0`). Install a PyTorch build compatible with your CUDA environment, then install the remaining dependencies:

```bash
python -m pip install ultralytics pyyaml
```

Exact training-environment versions are not currently pinned in this repository. The distillation modules depend on internal Ultralytics classes, so compatibility with every Ultralytics version is not guaranteed. Record the versions used for each experiment.

The following assets must be supplied before training:

1. The original images, YOLO annotation files, and dataset `data.yaml`.
2. The custom `MobileNetV2Backbone` implementation and the Ultralytics registration/model-parser integration required by `yolov11_mobilenetv2.yaml`.
3. A trained RT-DETR-L checkpoint whose class IDs and class names match the dataset.

The student architecture YAML is included. Its backbone entry is `MobileNetV2Backbone` with arguments `[0.5, False]`. This custom class is not defined in the currently uploaded Python files, so the YAML alone is insufficient to construct the student in a standard Ultralytics installation. Supply the custom implementation and its model-parser integration before training. The teacher checkpoint is also not included.

## Dataset

**[Download fanqie2.zip](https://github.com/huangyi1042/Cross-Architecture-Multi-Granularity-Knowledge-Distillation-for-Tomato-Leaf-Disease-Detection/releases/download/Dataset/fanqie2.zip)**

Dataset release page:

**[Tomato Leaf Disease Dataset](https://github.com/huangyi1042/Cross-Architecture-Multi-Granularity-Knowledge-Distillation-for-Tomato-Leaf-Disease-Detection/releases/tag/Dataset)**

The dataset is attached to the release as `fanqie2.zip`. Use the direct download link above, or select this file under **Assets** on the release page. GitHub's automatically generated **Source code** archives contain the repository files, not the image dataset.

The local dataset version inspected for this documentation contains 4,166 images: 3,332 training images and 834 validation images. Its class order is:

| Class ID | Label |
|---|---|
| 0 | `healthy_tomato` |
| 1 | `early_blight_alternaria_solani` |
| 2 | `tomato_groundnut_bud_necrosis_virus` |
| 3 | `tomato_alternaria` |

Expected original dataset structure:

```text
fanqie2/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

Example dataset configuration; replace `path` with the actual dataset directory:

```yaml
path: /absolute/path/to/fanqie2
train: train/images
val: valid/images
nc: 4
names:
  0: healthy_tomato
  1: early_blight_alternaria_solani
  2: tomato_groundnut_bud_necrosis_virus
  3: tomato_alternaria
```

Each annotation row uses YOLO format:

```text
class_id x_center y_center width height
```

Coordinates are normalized to image dimensions. Preserve the class order throughout teacher training, label refinement, student training, and evaluation. Dataset attribution and redistribution terms must be checked against the original sources; hosting a copy does not establish ownership of the images.

## Training Workflow

Run commands from the repository directory. Paths and experiment settings are configured inside the scripts; these entry points do not expose a command-line argument parser.

### 1. Configure the local paths

Replace the original `/root/autodl-tmp/fanqie/...` paths with paths on your machine:

| Script | Settings to update |
|---|---|
| `RT-DETR.py` | Checkpoint passed to `RTDETR(...)`, dataset path in `model.train(...)`, `resume`, output directory, and device. |
| `yolov11_moblienetv2.py` | `MODEL_YAML`, `DATA_YAML`, `PROJECT`, `NAME`, and device/batch settings. |
| `generate_matched_response_dataset.py` | `SOURCE_DATA_YAML`, `TEACHER_WEIGHT`, `OUTPUT_ROOT`, and device. |
| `train_response_feature_attention_kd.py` | `DATA_YAML`, `TEACHER_WEIGHT`, `STUDENT_YAML`, `PROJECT`, `NAME`, and device/batch settings. |

The scripts are configured for locally available model files; provide all required checkpoints before execution rather than relying on automatic downloads.

### 2. Prepare the RT-DETR-L teacher

For an existing training run, set the checkpoint path in `RT-DETR.py` to the appropriate `last.pt` and retain `resume=True`.

For a fresh training run, supply an appropriate RT-DETR-L initialization checkpoint and change `resume=False`. Check all training settings explicitly, because resuming may restore settings from the saved run.

```bash
python RT-DETR.py
```

Use the resulting selected teacher checkpoint, normally `weights/best.pt`, in both the dataset-generation and distillation scripts.

### 3. Train the student baseline

Point `DATA_YAML` to the original manually annotated dataset and `MODEL_YAML` to the student architecture configuration:

```bash
python yolov11_moblienetv2.py
```

The baseline script constructs the student from its architecture YAML with `pretrained=False`.

### 4. Generate teacher-refined training annotations

```bash
python generate_matched_response_dataset.py
```

The current settings are:

| Setting | Value |
|---|---:|
| Teacher confidence threshold (`TEACHER_CONF`) | 0.75 |
| Teacher prediction NMS IoU (`TEACHER_NMS_IOU`) | 0.70 |
| Minimum teacher/annotation matching IoU (`MATCH_IOU`) | 0.70 |
| Maximum box blending coefficient (`BOX_BLEND_MAX`) | 0.10 |
| Add unmatched teacher predictions (`ADD_UNMATCHED_PSEUDO`) | `False` |

For matched objects, the teacher-box contribution is confidence dependent: `BOX_BLEND_MAX * teacher_confidence`. The validation annotations remain unchanged.

**Output-directory handling:** `REBUILD_OUTPUT=True` deletes an existing `OUTPUT_ROOT` before regeneration. Select a separate output directory that contains no source data or other files you need to preserve.

The generated dataset uses a different directory layout:

```text
<OUTPUT_ROOT>/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### 5. Train with multi-granularity distillation

Set `DATA_YAML` in `train_response_feature_attention_kd.py` to the **exact `data.yaml` produced in Step 4**. The current hard-coded generator output directory and training input directory have different names, so they must be synchronized manually.

```bash
python train_response_feature_attention_kd.py
```

The main launcher configures the following settings:

| Setting | Current value |
|---|---:|
| Image size | 640 |
| Epochs | 150 |
| Batch size | 64 |
| Optimizer | SGD |
| Initial learning rate | 0.01 |
| Random seed | 42 |
| Feature distillation scales | P4 and P5 |
| Maximum feature loss weight | 0.30 |
| Feature weight start / ramp duration | 50 / 40 epochs |
| Spatial-awareness distillation scales | P4 and P5 |
| Maximum spatial-awareness loss weight | 0.10 |
| Spatial-awareness weight start / ramp duration | 90 / 30 epochs |
| Spatial attention power | 2.0 |
| Foreground mask expansion radius | 1 feature-map cell |

These values are read from the training launcher, which overrides some defaults in the trainer classes. An expansion radius of 1 corresponds to a 3 × 3 max-pooling kernel with stride 1 and padding 1.

Response-level supervision is provided through the refined training labels in the detector's standard loss. Feature and spatial-awareness losses are added by the custom trainer.

## Outputs and Reproducibility

Training outputs are written under the configured `PROJECT` and `NAME`. Inspect the run directory for checkpoints, metrics, plots, and saved training arguments.

For reproducible comparisons:

- Keep the dataset split, class order, image size, teacher checkpoint, and evaluation settings fixed.
- Group original images and their augmented variants before creating splits; check for duplicate images across training and validation sets.
- Record package versions, hardware, seeds, and all training arguments.
- Verify the current launcher settings against the experiment you intend to reproduce. The uploaded baseline and distillation launchers currently use different initial learning rates (`0.008` and `0.01`) and warm-up durations (`5` and `3` epochs).

The commands above document the uploaded source code. End-to-end training has not been verified as part of this README update, and the current script defaults should not be assumed to reproduce every result in the manuscript without the corresponding architecture, checkpoints, data version, and experiment configuration.

## Citation

If this repository supports your research, please cite the accompanying manuscript, **Cross-Architecture Multi-Granularity Knowledge Distillation for Tomato Leaf Disease Detection**, using its final bibliographic details when available. Cite the original dataset sources separately.
