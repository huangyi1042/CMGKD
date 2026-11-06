import os
from pathlib import Path

os.environ["ULTRALYTICS_AUTO_DOWNLOAD"] = "0"
os.environ["YOLO_OFFLINE"] = "1"
os.environ["ULTRALYTICS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from ultralytics import YOLO


# =========================================================
# 路径配置
# =========================================================
MODEL_YAML = (
    "/root/autodl-tmp/fanqie/"
    "yolov11_mobilenetv2.yaml"
)

# 纯学生基线必须使用原始人工标注数据集
DATA_YAML = (
    "/root/autodl-tmp/fanqie/dataset/"
    "fanqie2/data.yaml"
)

PROJECT = (
    "/root/autodl-tmp/fanqie/runs/"
    "mobilenetv2"
)

# 新实验名称，避免覆盖以前的batch=32结果
NAME = (
    "yolo11_mobilenetv2_"
    "baseline_b64_lr008_e150_strict"
)


# =========================================================
# 与batch=64蒸馏实验统一的参数
# =========================================================
IMG_SIZE = 640
EPOCHS = 150
BATCH_SIZE = 64

DEVICE = 0
WORKERS = 0
SEED = 42

OPTIMIZER = "SGD"

LR0 = 0.008
LRF = 0.01

MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005

WARMUP_EPOCHS = 5
WARMUP_MOMENTUM = 0.8
WARMUP_BIAS_LR = 0.1


def check_file(path: str, description: str) -> None:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"{description}不存在：{file_path}"
        )

    print(f"{description}：{file_path}")


def train():
    check_file(
        MODEL_YAML,
        "YOLO-MobileNetV2模型配置",
    )

    check_file(
        DATA_YAML,
        "原始数据集配置",
    )

    print("\n========== 学生模型batch=64基线 ==========")
    print(f"模型结构：{MODEL_YAML}")
    print(f"数据集：{DATA_YAML}")
    print(f"训练轮数：{EPOCHS}")
    print(f"输入尺寸：{IMG_SIZE}")
    print(f"批次大小：{BATCH_SIZE}")
    print(f"初始学习率：{LR0}")
    print(f"最终学习率比例：{LRF}")
    print(f"预热轮数：{WARMUP_EPOCHS}")
    print(f"优化器：{OPTIMIZER}")
    print(f"随机种子：{SEED}")
    print(f"输出目录：{PROJECT}/{NAME}")
    print("=========================================\n")

    # 仅从YAML构建模型，不加载学生预训练权重
    model = YOLO(
        MODEL_YAML,
        task="detect",
    )

    results = model.train(
        # 原始人工标注数据
        data=DATA_YAML,

        # 随机初始化
        pretrained=False,
        resume=False,

        # 基础训练参数
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,

        # 优化器
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,

        # 学习率预热
        warmup_epochs=WARMUP_EPOCHS,
        warmup_momentum=WARMUP_MOMENTUM,
        warmup_bias_lr=WARMUP_BIAS_LR,

        # 余弦学习率衰减
        cos_lr=True,

        # 尽量完整训练150轮
        patience=150,

        # 数据增强
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.0,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        fliplr=0.5,
        flipud=0.0,

        # 训练设置
        amp=True,
        deterministic=True,
        seed=SEED,
        compile=False,

        # 输出设置
        project=PROJECT,
        name=NAME,
        exist_ok=False,

        save=True,
        save_period=-1,
        plots=True,
    )

    return results


if __name__ == "__main__":
    train()