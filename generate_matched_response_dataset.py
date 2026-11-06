import csv
import gc
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
from ultralytics import RTDETR


# =========================================================
# 离线模式
# =========================================================
os.environ["ULTRALYTICS_AUTO_DOWNLOAD"] = "0"
os.environ["YOLO_OFFLINE"] = "1"
os.environ["ULTRALYTICS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# =========================================================
# 路径配置
# =========================================================

# 原始人工标注数据集
SOURCE_DATA_YAML = (
    "/root/autodl-tmp/fanqie/dataset/"
    "fanqie2/data.yaml"
)

# 正确的四类RT-DETR教师权重
TEACHER_WEIGHT = (
    "/root/autodl-tmp/fanqie/runs/detect/runs/rtdetr2/"
    "teacher_rtdetr_l2-5/weights/best.pt"
)

# 新生成的GT匹配响应蒸馏数据集
OUTPUT_ROOT = Path(
    "/root/autodl-tmp/fanqie/dataset/"
    "fanqie2_rtdetr_matched_response_conf075_iou070_blend010"
)

OUTPUT_DATA_YAML = OUTPUT_ROOT / "data.yaml"


# =========================================================
# 教师推理参数
# =========================================================

DEVICE = 0
IMG_SIZE = 640

# 教师预测最低置信度
TEACHER_CONF = 0.75

# 教师预测NMS IoU
TEACHER_NMS_IOU = 0.70

# 教师框与人工框的最低匹配IoU
MATCH_IOU = 0.70

# 教师框对人工框的最大修正比例
#
# 最终比例为：
# blend = BOX_BLEND_MAX * teacher_confidence
#
# 当置信度为0.8时：
# blend = 0.25 * 0.8 = 0.20
#
# 即最终框：
# 80%人工框 + 20%教师框
BOX_BLEND_MAX = 0.10

# 默认不添加未匹配教师框
ADD_UNMATCHED_PSEUDO = False

# 如果以后希望添加未匹配教师框，
# 只有置信度大于该值才允许加入
UNMATCHED_CONF = 0.95

# 是否删除并重新生成输出目录
REBUILD_OUTPUT = True


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def normalize_names(names) -> List[str]:
    """
    将names统一转换为顺序列表。
    """
    if isinstance(names, dict):
        return [
            names[key]
            for key in sorted(
                names.keys(),
                key=lambda value: int(value),
            )
        ]

    if isinstance(names, (list, tuple)):
        return list(names)

    raise TypeError(
        f"不支持的names类型：{type(names)}"
    )


def load_yaml(path: str) -> Dict:
    """
    读取YAML文件。
    """
    yaml_path = Path(path)

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"YAML不存在：{yaml_path}"
        )

    with open(
        yaml_path,
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise RuntimeError(
            f"YAML内容格式错误：{yaml_path}"
        )

    return config


def get_dataset_root(
    config: Dict,
    yaml_path: str,
) -> Path:
    """
    获取数据集根目录。
    """
    yaml_parent = Path(yaml_path).parent

    root_value = config.get(
        "path",
        str(yaml_parent),
    )

    root_path = Path(root_value)

    if not root_path.is_absolute():
        root_path = yaml_parent / root_path

    return root_path.resolve()


def resolve_split_path(
    config: Dict,
    yaml_path: str,
    split: str,
) -> Path:
    """
    解析train或val图像路径。
    """
    if split not in config:
        raise KeyError(
            f"data.yaml中缺少字段：{split}"
        )

    entry = config[split]

    if isinstance(entry, (list, tuple)):
        if len(entry) != 1:
            raise ValueError(
                f"当前脚本要求{split}只有一个目录：{entry}"
            )
        entry = entry[0]

    entry_path = Path(entry)

    if entry_path.is_absolute():
        return entry_path.resolve()

    dataset_root = get_dataset_root(
        config,
        yaml_path,
    )

    return (
        dataset_root / entry_path
    ).resolve()


def find_images(directory: Path) -> List[Path]:
    """
    递归查找图像。
    """
    if not directory.exists():
        raise FileNotFoundError(
            f"图像目录不存在：{directory}"
        )

    images = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    )

    if not images:
        raise RuntimeError(
            f"目录中没有找到图像：{directory}"
        )

    return images


def image_to_label_path(
    image_path: Path,
) -> Path:
    """
    将：
        .../images/train/abc.jpg

    转换为：
        .../labels/train/abc.txt
    """
    parts = list(image_path.parts)

    image_indices = [
        index
        for index, value in enumerate(parts)
        if value == "images"
    ]

    if not image_indices:
        raise RuntimeError(
            f"图像路径中不包含images目录：{image_path}"
        )

    # 使用最后一个images目录
    image_index = image_indices[-1]
    parts[image_index] = "labels"

    return Path(*parts).with_suffix(".txt")


def create_link_or_copy(
    source: Path,
    destination: Path,
) -> None:
    """
    优先创建软链接。
    若软链接失败，则复制文件。
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists() or destination.is_symlink():
        destination.unlink()

    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(
            source,
            destination,
        )


def read_yolo_labels(
    label_path: Path,
) -> List[Dict]:
    """
    读取YOLO格式人工标签。
    """
    labels = []

    if not label_path.exists():
        return labels

    with open(
        label_path,
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) < 5:
                raise ValueError(
                    f"标签字段不足：{label_path}:"
                    f"{line_number} -> {line}"
                )

            class_value = float(values[0])
            class_id = int(class_value)

            if class_value != class_id:
                raise ValueError(
                    f"类别编号不是整数：{label_path}:"
                    f"{line_number} -> {line}"
                )

            x_center = float(values[1])
            y_center = float(values[2])
            width = float(values[3])
            height = float(values[4])

            labels.append(
                {
                    "class_id": class_id,
                    "xywhn": (
                        x_center,
                        y_center,
                        width,
                        height,
                    ),
                }
            )

    return labels


def xywhn_to_xyxy(
    xywhn: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """
    归一化xywh转换为像素xyxy。
    """
    x_center, y_center, width, height = xywhn

    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height

    x1 = x_center - width / 2.0
    y1 = y_center - height / 2.0
    x2 = x_center + width / 2.0
    y2 = y_center + height / 2.0

    return torch.tensor(
        [x1, y1, x2, y2],
        dtype=torch.float32,
    )


def clip_xyxy(
    box: torch.Tensor,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """
    将边界框限制在图像范围内。
    """
    box = box.clone()

    box[0] = box[0].clamp(
        min=0.0,
        max=float(image_width - 1),
    )
    box[1] = box[1].clamp(
        min=0.0,
        max=float(image_height - 1),
    )
    box[2] = box[2].clamp(
        min=0.0,
        max=float(image_width),
    )
    box[3] = box[3].clamp(
        min=0.0,
        max=float(image_height),
    )

    return box


def xyxy_to_xywhn(
    box: torch.Tensor,
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    """
    像素xyxy转换为归一化xywh。
    """
    box = clip_xyxy(
        box,
        image_width,
        image_height,
    )

    x1, y1, x2, y2 = box.tolist()

    width = max(
        x2 - x1,
        1e-6,
    )
    height = max(
        y2 - y1,
        1e-6,
    )

    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0

    return (
        x_center / image_width,
        y_center / image_height,
        width / image_width,
        height / image_height,
    )


def calculate_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
) -> float:
    """
    计算两个xyxy框的IoU。
    """
    inter_x1 = max(
        float(box1[0]),
        float(box2[0]),
    )
    inter_y1 = max(
        float(box1[1]),
        float(box2[1]),
    )
    inter_x2 = min(
        float(box1[2]),
        float(box2[2]),
    )
    inter_y2 = min(
        float(box1[3]),
        float(box2[3]),
    )

    inter_width = max(
        inter_x2 - inter_x1,
        0.0,
    )
    inter_height = max(
        inter_y2 - inter_y1,
        0.0,
    )

    intersection = (
        inter_width * inter_height
    )

    area1 = max(
        float(box1[2] - box1[0]),
        0.0,
    ) * max(
        float(box1[3] - box1[1]),
        0.0,
    )

    area2 = max(
        float(box2[2] - box2[0]),
        0.0,
    ) * max(
        float(box2[3] - box2[1]),
        0.0,
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def greedy_match(
    gt_boxes: List[Dict],
    teacher_boxes: List[Dict],
) -> List[Tuple[int, int, float]]:
    """
    对同类别GT框和教师框进行一对一匹配。

    返回：
        (gt索引, teacher索引, IoU)
    """
    candidates = []

    for gt_index, gt_item in enumerate(gt_boxes):
        for teacher_index, teacher_item in enumerate(
            teacher_boxes
        ):
            if (
                gt_item["class_id"]
                != teacher_item["class_id"]
            ):
                continue

            iou = calculate_iou(
                gt_item["xyxy"],
                teacher_item["xyxy"],
            )

            if iou >= MATCH_IOU:
                candidates.append(
                    (
                        iou,
                        float(teacher_item["confidence"]),
                        gt_index,
                        teacher_index,
                    )
                )

    # 优先IoU高、置信度高的匹配
    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )

    matched_gt = set()
    matched_teacher = set()
    matches = []

    for (
        iou,
        _confidence,
        gt_index,
        teacher_index,
    ) in candidates:
        if gt_index in matched_gt:
            continue

        if teacher_index in matched_teacher:
            continue

        matched_gt.add(gt_index)
        matched_teacher.add(teacher_index)

        matches.append(
            (
                gt_index,
                teacher_index,
                iou,
            )
        )

    return matches


def validate_class_names(
    teacher: RTDETR,
    dataset_names: List[str],
) -> None:
    """
    确保教师与数据集类别名称、顺序完全一致。
    """
    teacher_names = normalize_names(
        teacher.model.names
    )

    print(f"数据集类别：{dataset_names}")
    print(f"教师类别：{teacher_names}")

    if teacher_names != dataset_names:
        raise ValueError(
            "教师类别名称或顺序与数据集不一致。\n"
            f"数据集：{dataset_names}\n"
            f"教师：{teacher_names}"
        )


def predict_teacher(
    teacher: RTDETR,
    image_path: Path,
    class_count: int,
) -> List[Dict]:
    """
    对单张图片执行教师推理。
    """
    results = teacher.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        conf=TEACHER_CONF,
        iou=TEACHER_NMS_IOU,
        device=DEVICE,
        max_det=300,
        verbose=False,
    )

    if not results:
        return []

    result = results[0]

    if result.boxes is None:
        return []

    xyxy_tensor = result.boxes.xyxy.detach().cpu()
    class_tensor = result.boxes.cls.detach().cpu()
    confidence_tensor = result.boxes.conf.detach().cpu()

    teacher_boxes = []

    for box, class_value, confidence in zip(
        xyxy_tensor,
        class_tensor,
        confidence_tensor,
    ):
        class_id = int(class_value.item())

        if class_id < 0 or class_id >= class_count:
            raise ValueError(
                f"教师输出类别越界：{class_id}，"
                f"合法范围为0～{class_count - 1}，"
                f"图像：{image_path}"
            )

        teacher_boxes.append(
            {
                "class_id": class_id,
                "confidence": float(
                    confidence.item()
                ),
                "xyxy": box.float(),
            }
        )

    return teacher_boxes


def process_train_image(
    teacher: RTDETR,
    image_path: Path,
    destination_label: Path,
    class_count: int,
) -> Dict:
    """
    处理一张训练图像：
    1. 读取人工标签；
    2. 教师推理；
    3. 匹配同类框；
    4. 使用教师框轻微修正人工框；
    5. 默认不添加未匹配教师框。
    """
    source_label = image_to_label_path(
        image_path
    )

    gt_labels = read_yolo_labels(
        source_label
    )

    # 读取图像尺寸
    from PIL import Image

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    gt_boxes = []

    for label in gt_labels:
        class_id = label["class_id"]

        if class_id < 0 or class_id >= class_count:
            raise ValueError(
                f"人工标签类别越界：{class_id}，"
                f"文件：{source_label}"
            )

        gt_boxes.append(
            {
                "class_id": class_id,
                "xyxy": xywhn_to_xyxy(
                    label["xywhn"],
                    image_width,
                    image_height,
                ),
            }
        )

    teacher_boxes = predict_teacher(
        teacher=teacher,
        image_path=image_path,
        class_count=class_count,
    )

    matches = greedy_match(
        gt_boxes=gt_boxes,
        teacher_boxes=teacher_boxes,
    )

    match_by_gt = {
        gt_index: (
            teacher_index,
            iou,
        )
        for (
            gt_index,
            teacher_index,
            iou,
        ) in matches
    }

    matched_teacher_indices = {
        teacher_index
        for (
            _gt_index,
            teacher_index,
            _iou,
        ) in matches
    }

    output_labels = []
    matched_confidences = []
    matched_ious = []

    # 对每个人工标注框生成最终监督框
    for gt_index, gt_item in enumerate(gt_boxes):
        final_box = gt_item["xyxy"].clone()

        if gt_index in match_by_gt:
            teacher_index, match_iou = (
                match_by_gt[gt_index]
            )

            teacher_item = teacher_boxes[
                teacher_index
            ]

            confidence = teacher_item[
                "confidence"
            ]

            # 置信度控制融合比例
            blend = min(
                BOX_BLEND_MAX * confidence,
                BOX_BLEND_MAX,
            )

            # 人工标签占主导，教师框只做小幅修正
            final_box = (
                (1.0 - blend)
                * gt_item["xyxy"]
                + blend
                * teacher_item["xyxy"]
            )

            matched_confidences.append(
                confidence
            )
            matched_ious.append(
                match_iou
            )

        final_xywhn = xyxy_to_xywhn(
            final_box,
            image_width,
            image_height,
        )

        output_labels.append(
            (
                gt_item["class_id"],
                *final_xywhn,
            )
        )

    added_unmatched = 0

    # 默认关闭，避免教师误检成为硬标签
    if ADD_UNMATCHED_PSEUDO:
        for teacher_index, teacher_item in enumerate(
            teacher_boxes
        ):
            if teacher_index in matched_teacher_indices:
                continue

            if (
                teacher_item["confidence"]
                < UNMATCHED_CONF
            ):
                continue

            final_xywhn = xyxy_to_xywhn(
                teacher_item["xyxy"],
                image_width,
                image_height,
            )

            output_labels.append(
                (
                    teacher_item["class_id"],
                    *final_xywhn,
                )
            )

            added_unmatched += 1

    destination_label.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        destination_label,
        "w",
        encoding="utf-8",
    ) as file:
        for (
            class_id,
            x_center,
            y_center,
            width,
            height,
        ) in output_labels:
            file.write(
                f"{class_id} "
                f"{x_center:.8f} "
                f"{y_center:.8f} "
                f"{width:.8f} "
                f"{height:.8f}\n"
            )

    mean_iou = (
        sum(matched_ious) / len(matched_ious)
        if matched_ious
        else 0.0
    )

    mean_confidence = (
        sum(matched_confidences)
        / len(matched_confidences)
        if matched_confidences
        else 0.0
    )

    return {
        "image": str(image_path),
        "gt_boxes": len(gt_boxes),
        "teacher_boxes": len(teacher_boxes),
        "matched_boxes": len(matches),
        "unmatched_added": added_unmatched,
        "mean_match_iou": mean_iou,
        "mean_teacher_confidence": mean_confidence,
    }


def prepare_validation_split(
    validation_images: List[Path],
    validation_root: Path,
) -> None:
    """
    验证集保持原始人工标注，不使用教师伪标签。
    """
    output_image_root = (
        OUTPUT_ROOT / "images" / "val"
    )
    output_label_root = (
        OUTPUT_ROOT / "labels" / "val"
    )

    for index, image_path in enumerate(
        validation_images,
        start=1,
    ):
        relative_path = image_path.relative_to(
            validation_root
        )

        destination_image = (
            output_image_root / relative_path
        )

        destination_label = (
            output_label_root
            / relative_path.with_suffix(".txt")
        )

        source_label = image_to_label_path(
            image_path
        )

        create_link_or_copy(
            image_path,
            destination_image,
        )

        destination_label.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if source_label.exists():
            shutil.copy2(
                source_label,
                destination_label,
            )
        else:
            destination_label.write_text(
                "",
                encoding="utf-8",
            )

        if index % 200 == 0:
            print(
                f"验证集处理进度："
                f"{index}/{len(validation_images)}"
            )


def save_output_yaml(
    dataset_names: List[str],
) -> None:
    """
    保存新数据集data.yaml。
    """
    output_config = {
        "path": str(OUTPUT_ROOT),
        "train": "images/train",
        "val": "images/val",
        "nc": len(dataset_names),
        "names": dataset_names,
    }

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_DATA_YAML,
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            output_config,
            file,
            allow_unicode=True,
            sort_keys=False,
        )


def main() -> None:
    source_config = load_yaml(
        SOURCE_DATA_YAML
    )

    if "names" not in source_config:
        raise KeyError(
            "原始data.yaml缺少names字段。"
        )

    dataset_names = normalize_names(
        source_config["names"]
    )

    dataset_nc = int(
        source_config.get(
            "nc",
            len(dataset_names),
        )
    )

    if dataset_nc != len(dataset_names):
        raise ValueError(
            f"nc与names数量不一致："
            f"nc={dataset_nc}，"
            f"names={dataset_names}"
        )

    train_root = resolve_split_path(
        source_config,
        SOURCE_DATA_YAML,
        "train",
    )

    validation_root = resolve_split_path(
        source_config,
        SOURCE_DATA_YAML,
        "val",
    )

    train_images = find_images(
        train_root
    )

    validation_images = find_images(
        validation_root
    )

    print(f"训练图像数：{len(train_images)}")
    print(f"验证图像数：{len(validation_images)}")
    print(f"类别数：{dataset_nc}")
    print(f"类别名称：{dataset_names}")

    if REBUILD_OUTPUT and OUTPUT_ROOT.exists():
        print(f"删除旧输出目录：{OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)

    (
        OUTPUT_ROOT / "images" / "train"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        OUTPUT_ROOT / "labels" / "train"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    teacher = RTDETR(
        TEACHER_WEIGHT
    )

    validate_class_names(
        teacher=teacher,
        dataset_names=dataset_names,
    )

    manifest_records = []

    total_gt = 0
    total_teacher = 0
    total_matched = 0
    total_unmatched_added = 0

    output_train_images = (
        OUTPUT_ROOT / "images" / "train"
    )

    output_train_labels = (
        OUTPUT_ROOT / "labels" / "train"
    )

    for index, image_path in enumerate(
        train_images,
        start=1,
    ):
        relative_path = image_path.relative_to(
            train_root
        )

        destination_image = (
            output_train_images / relative_path
        )

        destination_label = (
            output_train_labels
            / relative_path.with_suffix(".txt")
        )

        create_link_or_copy(
            source=image_path,
            destination=destination_image,
        )

        record = process_train_image(
            teacher=teacher,
            image_path=image_path,
            destination_label=destination_label,
            class_count=dataset_nc,
        )

        manifest_records.append(record)

        total_gt += record["gt_boxes"]
        total_teacher += record["teacher_boxes"]
        total_matched += record["matched_boxes"]
        total_unmatched_added += record[
            "unmatched_added"
        ]

        if (
            index % 50 == 0
            or index == len(train_images)
        ):
            print(
                f"训练集处理进度："
                f"{index}/{len(train_images)}，"
                f"累计匹配框：{total_matched}"
            )

        # 定期清理缓存，降低长时间推理的显存碎片
        if index % 200 == 0:
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    prepare_validation_split(
        validation_images=validation_images,
        validation_root=validation_root,
    )

    save_output_yaml(
        dataset_names=dataset_names
    )

    manifest_path = (
        OUTPUT_ROOT
        / "response_match_manifest.csv"
    )

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image",
                "gt_boxes",
                "teacher_boxes",
                "matched_boxes",
                "unmatched_added",
                "mean_match_iou",
                "mean_teacher_confidence",
            ],
        )

        writer.writeheader()
        writer.writerows(
            manifest_records
        )

    total_mean_iou_values = [
        record["mean_match_iou"]
        for record in manifest_records
        if record["matched_boxes"] > 0
    ]

    overall_mean_iou = (
        sum(total_mean_iou_values)
        / len(total_mean_iou_values)
        if total_mean_iou_values
        else 0.0
    )

    summary = {
        "train_images": len(train_images),
        "validation_images": len(
            validation_images
        ),
        "dataset_nc": dataset_nc,
        "dataset_names": dataset_names,
        "teacher_conf": TEACHER_CONF,
        "match_iou": MATCH_IOU,
        "box_blend_max": BOX_BLEND_MAX,
        "add_unmatched_pseudo": (
            ADD_UNMATCHED_PSEUDO
        ),
        "total_gt_boxes": total_gt,
        "total_teacher_boxes": total_teacher,
        "total_matched_boxes": total_matched,
        "total_unmatched_added": (
            total_unmatched_added
        ),
        "overall_image_mean_match_iou": (
            overall_mean_iou
        ),
    }

    summary_path = (
        OUTPUT_ROOT
        / "response_match_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            summary,
            json_file,
            ensure_ascii=False,
            indent=2,
        )

    del teacher
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n========== 数据集生成完成 ==========")
    print(f"输出目录：{OUTPUT_ROOT}")
    print(f"数据配置：{OUTPUT_DATA_YAML}")
    print(f"人工框数量：{total_gt}")
    print(f"教师预测框数量：{total_teacher}")
    print(f"成功匹配框数量：{total_matched}")
    print(
        f"未匹配教师框加入数量："
        f"{total_unmatched_added}"
    )
    print(f"匹配记录：{manifest_path}")
    print(f"统计文件：{summary_path}")
    print("===================================\n")


if __name__ == "__main__":
    main()