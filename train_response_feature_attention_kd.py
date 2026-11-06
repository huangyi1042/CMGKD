import math
import os
from pathlib import Path

os.environ["ULTRALYTICS_AUTO_DOWNLOAD"] = "0"
os.environ["YOLO_OFFLINE"] = "1"
os.environ["ULTRALYTICS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from ultralytics import YOLO

from rtdetr_yolo_mobile_attention_kd import MultiGranularityKDTrainer


DATA_YAML = (
    "/root/autodl-tmp/fanqie/dataset/"
    "fanqie2_rtdetr_matched_response/data.yaml"
)

TEACHER_WEIGHT = (
    "/root/autodl-tmp/fanqie/runs/detect/runs/"
    "rtdetr2/teacher_rtdetr_l2-5/weights/best.pt"
)

STUDENT_YAML = (
    "/root/autodl-tmp/fanqie/"
    "yolov11_mobilenetv2.yaml"
)

PROJECT = (
    "/root/autodl-tmp/fanqie/runs/"
    "multigranularity_kd"
)

NAME = (
    "rtdetr_l_to_yolo_mobile_"
    "response_feature_attention_b64_e150"
)

IMG_SIZE = 640
EPOCHS = 150
BATCH_SIZE = 64
DEVICE = 0
WORKERS = 0
SEED = 42

OPTIMIZER = "SGD"
LR0 = 0.01
LRF = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005

# Feature KD：延续当前较好的配置
FEATURE_KD_WEIGHT = 0.3
FEATURE_START_EPOCH = 50
FEATURE_RAMP_EPOCHS = 40
FEATURE_LEVEL_INDICES = (1, 2)
FEATURE_BACKGROUND_WEIGHT = 0.0

# Attention KD：在 Feature KD 基本稳定后再加入
ATTENTION_KD_WEIGHT = 0.10
ATTENTION_START_EPOCH = 90
ATTENTION_RAMP_EPOCHS = 30
ATTENTION_LEVEL_INDICES = (1, 2)
ATTENTION_BACKGROUND_WEIGHT = 0.0
ATTENTION_POWER = 2.0
ATTENTION_MASK_EXPAND = 1

IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff",
}


def check_file(path: str, description: str) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"{description}不存在：{file_path}"
        )
    print(f"{description}：{file_path}")
    return file_path


def count_training_images() -> int:
    train_dir = Path(DATA_YAML).parent / "images" / "train"
    if not train_dir.exists():
        raise FileNotFoundError(
            f"训练图像目录不存在：{train_dir}"
        )

    image_count = sum(
        1
        for path in train_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    )

    if image_count <= 0:
        raise RuntimeError(
            f"训练目录没有图像：{train_dir}"
        )
    return image_count


def configure_trainer(iterations_per_epoch: int) -> None:
    MultiGranularityKDTrainer.teacher_weight = TEACHER_WEIGHT
    MultiGranularityKDTrainer.probe_size = IMG_SIZE
    MultiGranularityKDTrainer.iters_per_epoch = iterations_per_epoch

    MultiGranularityKDTrainer.feature_kd_weight = FEATURE_KD_WEIGHT
    MultiGranularityKDTrainer.feature_start_epoch = FEATURE_START_EPOCH
    MultiGranularityKDTrainer.feature_ramp_epochs = FEATURE_RAMP_EPOCHS
    MultiGranularityKDTrainer.feature_level_indices = FEATURE_LEVEL_INDICES
    MultiGranularityKDTrainer.feature_background_weight = (
        FEATURE_BACKGROUND_WEIGHT
    )

    MultiGranularityKDTrainer.attention_kd_weight = ATTENTION_KD_WEIGHT
    MultiGranularityKDTrainer.attention_start_epoch = ATTENTION_START_EPOCH
    MultiGranularityKDTrainer.attention_ramp_epochs = ATTENTION_RAMP_EPOCHS
    MultiGranularityKDTrainer.attention_level_indices = (
        ATTENTION_LEVEL_INDICES
    )
    MultiGranularityKDTrainer.attention_background_weight = (
        ATTENTION_BACKGROUND_WEIGHT
    )
    MultiGranularityKDTrainer.attention_power = ATTENTION_POWER
    MultiGranularityKDTrainer.attention_mask_expand = ATTENTION_MASK_EXPAND


def print_config(image_count: int, iterations_per_epoch: int) -> None:
    print("\n========== Response + Feature + Attention KD ==========")
    print(f"数据集：{DATA_YAML}")
    print(f"教师权重：{TEACHER_WEIGHT}")
    print(f"学生结构：{STUDENT_YAML}")
    print(f"训练图像数：{image_count}")
    print(f"训练轮数：{EPOCHS}")
    print(f"批次大小：{BATCH_SIZE}")
    print(f"每轮迭代：{iterations_per_epoch}")
    print(f"Feature KD：{FEATURE_KD_WEIGHT}, "
          f"start={FEATURE_START_EPOCH}, ramp={FEATURE_RAMP_EPOCHS}")
    print(f"Attention KD：{ATTENTION_KD_WEIGHT}, "
          f"start={ATTENTION_START_EPOCH}, ramp={ATTENTION_RAMP_EPOCHS}")
    print(f"输出：{PROJECT}/{NAME}")
    print("=======================================================\n")


def main() -> None:
    check_file(DATA_YAML, "多粒度蒸馏数据集 YAML")
    check_file(TEACHER_WEIGHT, "正确四类 RT-DETR 教师权重")
    check_file(STUDENT_YAML, "YOLO-MobileNetV2 学生 YAML")

    image_count = count_training_images()
    iterations_per_epoch = math.ceil(image_count / BATCH_SIZE)

    configure_trainer(iterations_per_epoch)
    print_config(image_count, iterations_per_epoch)

    model = YOLO(
        STUDENT_YAML,
        task="detect",
    )

    model.train(
        data=DATA_YAML,
        pretrained=False,
        resume=False,

        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,

        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,

        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        cos_lr=True,
        patience=150,

        mosaic=1.0,
        close_mosaic=10,
        mixup=0.0,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,

        amp=True,
        deterministic=True,
        seed=SEED,
        compile=False,

        project=PROJECT,
        name=NAME,
        exist_ok=False,

        save=True,
        save_period=-1,
        plots=True,

        trainer=MultiGranularityKDTrainer,
    )


if __name__ == "__main__":
    main()