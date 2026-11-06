from copy import copy

import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK

from rtdetr_yolo_mobile_kd import RTDETRToYOLOMobileFeatureKDModel


class RTDETRToYOLOMobileMultiGranularityKDModel(
    RTDETRToYOLOMobileFeatureKDModel
):
    """
    RT-DETR-L -> YOLO11-MobileNetV2 多粒度知识蒸馏模型。

    Response KD:
        由训练数据集中的教师响应标签实现，因此包含在 YOLO 检测损失中。

    Feature KD:
        对齐并蒸馏 P4、P5 中间特征。

    Attention KD:
        将多尺度特征沿通道聚合为空间注意力图，使学生模仿教师
        对病斑区域的空间关注分布。

    总损失:
        L = L_det(response labels)
            + lambda_f(t) * L_feature
            + lambda_a(t) * L_attention

    权重调度:
        1. ramp_epochs > 0:
           使用渐进式线性权重调度。
        2. ramp_epochs <= 0:
           从 start_epoch 开始直接使用最大蒸馏权重。
           该模式可用于“非渐进式同步蒸馏”对照实验。
    """

    def __init__(
        self,
        cfg,
        ch=3,
        nc=None,
        verbose=True,
        teacher_weight=None,
        feature_kd_weight=0.3,
        feature_start_epoch=50,
        feature_ramp_epochs=40,
        feature_level_indices=(1, 2),
        feature_background_weight=0.0,
        probe_size=640,
        iters_per_epoch=105,
        attention_kd_weight=0.1,
        attention_start_epoch=90,
        attention_ramp_epochs=20,
        attention_level_indices=(1, 2),
        attention_background_weight=0.0,
        attention_power=2.0,
        attention_mask_expand=1,
    ):
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
            teacher_weight=teacher_weight,
            kd_weight=feature_kd_weight,
            probe_size=probe_size,
            kd_start_epoch=feature_start_epoch,
            kd_ramp_epochs=feature_ramp_epochs,
            iters_per_epoch=iters_per_epoch,
            distill_level_indices=feature_level_indices,
            background_weight=feature_background_weight,
        )

        self.attention_kd_weight = float(attention_kd_weight)
        self.attention_start_epoch = int(attention_start_epoch)
        self.attention_ramp_epochs = int(attention_ramp_epochs)
        self.attention_level_indices = tuple(attention_level_indices)
        self.attention_background_weight = float(attention_background_weight)
        self.attention_power = float(attention_power)
        self.attention_mask_expand = max(int(attention_mask_expand), 0)

    def _get_ramp_weight(self, max_weight, start_epoch, ramp_epochs):
        """
        根据当前训练进度计算蒸馏权重。

        ramp_epochs > 0:
            从 start_epoch 开始线性增加，
            在 ramp_epochs 个 epoch 后达到 max_weight。

        ramp_epochs <= 0:
            从 start_epoch 开始直接使用 max_weight。
            用于非渐进式同步蒸馏对照实验。
        """
        if not self.training:
            return 0.0

        current_epoch = self.kd_step / max(self.iters_per_epoch, 1)

        if current_epoch < start_epoch:
            return 0.0

        # 非渐进式模式：启用后立即使用最大权重
        if ramp_epochs <= 0:
            return float(max_weight)

        # 渐进式模式：线性增长
        progress = (current_epoch - start_epoch) / ramp_epochs
        progress = max(0.0, min(1.0, progress))

        return float(max_weight) * progress

    def get_current_feature_weight(self):
        """当前 Feature KD 权重。"""
        return self._get_ramp_weight(
            max_weight=self.kd_weight,
            start_epoch=self.kd_start_epoch,
            ramp_epochs=self.kd_ramp_epochs,
        )

    def get_current_attention_weight(self):
        """当前 Attention KD 权重。"""
        return self._get_ramp_weight(
            max_weight=self.attention_kd_weight,
            start_epoch=self.attention_start_epoch,
            ramp_epochs=self.attention_ramp_epochs,
        )

    def _feature_to_spatial_attention(self, feature):
        """
        将 BxCxHxW 特征转换为 Bx1xHxW 空间注意力图。

        A(F) = mean_c(|F|^p)
        """
        if not isinstance(feature, torch.Tensor):
            raise TypeError(f"feature 必须是 Tensor，实际为 {type(feature)}")

        if feature.ndim != 4:
            raise ValueError(
                f"feature 应为四维 BxCxHxW，实际形状为 {feature.shape}"
            )

        return (
            feature.float()
            .abs()
            .pow(self.attention_power)
            .mean(dim=1, keepdim=True)
        )

    def _expand_foreground_mask(self, mask):
        """在特征图上扩张病斑框区域，保留边缘与邻域上下文。"""
        if self.attention_mask_expand <= 0:
            return mask

        kernel_size = 2 * self.attention_mask_expand + 1

        return F.max_pool2d(
            mask,
            kernel_size=kernel_size,
            stride=1,
            padding=self.attention_mask_expand,
        )

    def attention_kd_loss(self, batch):
        """病斑区域感知的多尺度空间注意力蒸馏。"""
        teacher_features = self._get_pyramid_features(
            self.teacher_cache,
            "教师模型",
        )

        student_features = self._get_pyramid_features(
            self.student_cache,
            "学生模型",
        )

        level_losses = []

        for level_index in self.attention_level_indices:
            if level_index < 0 or level_index >= len(student_features):
                raise IndexError(
                    f"非法注意力蒸馏层索引：{level_index}"
                )

            student_feature = student_features[level_index]
            teacher_feature = teacher_features[level_index].detach()

            student_attention = self._feature_to_spatial_attention(
                student_feature
            )

            teacher_attention = self._feature_to_spatial_attention(
                teacher_feature
            ).detach()

            if teacher_attention.shape[-2:] != student_attention.shape[-2:]:
                teacher_attention = F.interpolate(
                    teacher_attention,
                    size=student_attention.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            feature_h, feature_w = student_attention.shape[-2:]

            foreground_mask = self.build_foreground_mask(
                batch=batch,
                feature_h=feature_h,
                feature_w=feature_w,
                device=student_attention.device,
            ).float()

            foreground_mask = self._expand_foreground_mask(
                foreground_mask
            )

            spatial_weight = (
                foreground_mask
                + self.attention_background_weight
                * (1.0 - foreground_mask)
            )

            valid_samples = (
                spatial_weight.flatten(1).sum(dim=1) > 1e-6
            )

            if not bool(valid_samples.any()):
                continue

            student_vector = (
                student_attention * spatial_weight
            ).flatten(1)[valid_samples]

            teacher_vector = (
                teacher_attention * spatial_weight
            ).flatten(1)[valid_samples]

            student_vector = F.normalize(
                student_vector,
                p=2,
                dim=1,
                eps=1e-6,
            )

            teacher_vector = F.normalize(
                teacher_vector,
                p=2,
                dim=1,
                eps=1e-6,
            )

            level_loss = (
                student_vector - teacher_vector
            ).pow(2).sum(dim=1).mean()

            level_losses.append(level_loss)

        if not level_losses:
            return torch.zeros(
                (),
                device=batch["img"].device,
                dtype=torch.float32,
            )

        return torch.stack(level_losses).mean()

    def loss(self, batch, preds=None):
        """
        训练:
            检测损失 + Feature KD + Attention KD。

        验证:
            只计算检测损失，两个 KD 日志列补零。
        """
        if not self.training:
            det_loss, det_items = DetectionModel.loss(
                self,
                batch,
                preds,
            )

            kd_zeros = torch.zeros(
                2,
                device=det_loss.device,
                dtype=det_loss.dtype,
            )

            return det_loss, torch.cat(
                [
                    det_items.detach().flatten(),
                    kd_zeros,
                ]
            )

        self.teacher_cache.clear()
        self.student_cache.clear()

        self._move_teacher_to(batch["img"].device)

        with torch.no_grad():
            self.teacher(batch["img"])

        if preds is None:
            preds = self.predict(batch["img"])

        # 直接调用原始 DetectionModel.loss，
        # 避免父类 Feature KD loss 再次运行教师和学生前向传播。
        det_loss, det_items = DetectionModel.loss(
            self,
            batch,
            preds,
        )

        feature_weight = self.get_current_feature_weight()
        attention_weight = self.get_current_attention_weight()

        zero = torch.zeros(
            (),
            device=det_loss.device,
            dtype=torch.float32,
        )

        raw_feature_loss = (
            self.feature_kd_loss(batch)
            if feature_weight > 0.0
            else zero
        )

        raw_attention_loss = (
            self.attention_kd_loss(batch)
            if attention_weight > 0.0
            else zero
        )

        weighted_feature_loss = (
            feature_weight * raw_feature_loss
        )

        weighted_attention_loss = (
            attention_weight * raw_attention_loss
        )

        kd_total = (
            weighted_feature_loss
            + weighted_attention_loss
        ) * batch["img"].shape[0]

        total_loss = det_loss + kd_total

        self.kd_step += 1

        loss_items = torch.cat(
            [
                det_items.detach().flatten(),
                weighted_feature_loss.detach().reshape(1),
                weighted_attention_loss.detach().reshape(1),
            ]
        )

        self.teacher_cache.clear()
        self.student_cache.clear()

        return total_loss, loss_items


class MultiGranularityKDTrainer(DetectionTrainer):
    """
    Response + Feature + Attention 多粒度蒸馏 Trainer。

    通过启动脚本覆盖以下参数即可切换两种实验：

    渐进式：
        feature_start_epoch > 0
        feature_ramp_epochs > 0
        attention_start_epoch > 0
        attention_ramp_epochs > 0

    非渐进式同步蒸馏：
        feature_start_epoch = 0
        feature_ramp_epochs = 0
        attention_start_epoch = 0
        attention_ramp_epochs = 0
    """

    teacher_weight = None
    probe_size = 640
    iters_per_epoch = 105

    # Feature KD 默认使用渐进式配置
    feature_kd_weight = 0.3
    feature_start_epoch = 50
    feature_ramp_epochs = 40
    feature_level_indices = (1, 2)
    feature_background_weight = 0.0

    # Attention KD 默认使用渐进式配置
    attention_kd_weight = 0.1
    attention_start_epoch = 90
    attention_ramp_epochs = 20
    attention_level_indices = (1, 2)
    attention_background_weight = 0.0
    attention_power = 2.0
    attention_mask_expand = 1

    def get_model(self, cfg=None, weights=None, verbose=True):
        if self.teacher_weight is None:
            raise ValueError(
                "MultiGranularityKDTrainer.teacher_weight 未设置。"
            )

        model = RTDETRToYOLOMobileMultiGranularityKDModel(
            cfg=cfg,
            ch=self.data.get("channels", 3),
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
            teacher_weight=self.teacher_weight,
            feature_kd_weight=self.feature_kd_weight,
            feature_start_epoch=self.feature_start_epoch,
            feature_ramp_epochs=self.feature_ramp_epochs,
            feature_level_indices=self.feature_level_indices,
            feature_background_weight=self.feature_background_weight,
            probe_size=self.probe_size,
            iters_per_epoch=self.iters_per_epoch,
            attention_kd_weight=self.attention_kd_weight,
            attention_start_epoch=self.attention_start_epoch,
            attention_ramp_epochs=self.attention_ramp_epochs,
            attention_level_indices=self.attention_level_indices,
            attention_background_weight=self.attention_background_weight,
            attention_power=self.attention_power,
            attention_mask_expand=self.attention_mask_expand,
        )

        if weights is not None:
            model.load(weights)

        return model

    def get_validator(self):
        self.loss_names = (
            "box_loss",
            "cls_loss",
            "dfl_loss",
            "kd_feature_loss",
            "kd_attention_loss",
        )

        return DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )
