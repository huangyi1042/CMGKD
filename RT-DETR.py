from ultralytics import RTDETR

def train_rtdetr_teacher():

    # 1. 加载预训练模型（推荐 rtdetr-l）
    model = RTDETR("/root/autodl-tmp/fanqie/runs/detect/runs/rtdetr2/teacher_rtdetr_l2-5/weights/last.pt")

    # 2. 开始训练
    results = model.train(
        data="/root/autodl-tmp/fanqie/dataset/fanqie2/data.yaml",   # 数据集配置
        epochs=150,                    # 训练轮数
        imgsz=640,                     # 输入尺寸
        batch=32,                      # RT-DETR较吃显存
        device=0,                      # GPU
        workers=0,
        rect=False,
        resume=True,
        # ====== 关键论文设置 ======
        lr0=2e-5,                     # DETR类模型建议小学习率
        optimizer="AdamW",
        weight_decay=0.0001,
        # ====== 提升稳定性 ======
        cos_lr=True,
        patience=30,
        warmup_epochs=10,

        # ====== 数据增强 ======
        mosaic=0,
        mixup=0,
        close_mosaic=0,
        cache=False,
        multi_scale=False,
        # ====== 输出路径 ======
        project="runs/rtdetr2",
        name="teacher_rtdetr_l333",

        pretrained=True,
        deterministic=False,
        amp=True
    )

    return results


if __name__ == "__main__":
    train_rtdetr_teacher()