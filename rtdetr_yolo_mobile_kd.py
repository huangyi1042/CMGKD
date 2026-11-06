from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import RTDETR
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK


class PyramidInputHook:
    """
    捕获模块前向输入中的多尺度特征。

    RT-DETR Decoder 输入：
    P3、P4、P5

    YOLO Detect 输入：
    P3、P4、P5
    """

    def __init__(self, cache):
        self.cache = cache

    def __call__(self, module, inputs):
        if len(inputs) == 0:
            raise RuntimeError("Hook 未捕获到输入特征。")

        features = inputs[0]

        if isinstance(features, torch.Tensor):
            features = [features]

        if not isinstance(features, (list, tuple)):
            raise RuntimeError(
                f"期望获得 list 或 tuple 类型特征，实际为：{type(features)}"
            )

        valid_features = [
            feat
            for feat in features
            if isinstance(feat, torch.Tensor) and feat.ndim == 4
        ]

        if len(valid_features) == 0:
            raise RuntimeError("未捕获到四维特征图。")

        self.cache["features"] = valid_features


class FeatureAdapter(nn.Module):
    """
    将学生特征映射到教师通道空间。
    """

    def __init__(self, student_channels, teacher_channels):
        super().__init__()

        self.adapter = nn.Sequential(
            nn.Conv2d(
                in_channels=student_channels,
                out_channels=teacher_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(teacher_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.adapter(x)


class RTDETRToYOLOMobileFeatureKDModel(DetectionModel):
    """
    教师模型：RT-DETR-L
    学生模型：YOLO-MobileNetV2

    随机初始化学生模型时采用：
    1. 前 30 epoch 不蒸馏
    2. 30 到 70 epoch 逐渐增加 KD 权重
    3. 只蒸馏 P4、P5
    4. 使用标注框前景区域掩码
    """

    def __init__(
        self,
        cfg,
        ch=3,
        nc=None,
        verbose=True,
        teacher_weight=None,
        kd_weight=1.0,
        probe_size=640,
        kd_start_epoch=30,
        kd_ramp_epochs=40,
        iters_per_epoch=105,
        distill_level_indices=(1, 2),
        background_weight=0.05,
    ):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        if teacher_weight is None:
            raise ValueError("teacher_weight 不能为空。")

        self.kd_weight = float(kd_weight)
        self.kd_start_epoch = int(kd_start_epoch)
        self.kd_ramp_epochs = int(kd_ramp_epochs)
        self.iters_per_epoch = max(int(iters_per_epoch), 1)
        self.distill_level_indices = tuple(distill_level_indices)
        self.background_weight = float(background_weight)
        self.kd_step = 0

        # 教师模型不注册到 self._modules 中，避免被 Trainer 加入优化器或自动解冻
        teacher_model = RTDETR(teacher_weight).model
        object.__setattr__(self, "_teacher", teacher_model)

        self.teacher_cache = {}
        self.student_cache = {}

        self._teacher_hook_handle = None
        self._student_hook_handle = None

        self._freeze_teacher()
        self._register_hooks()
        self._build_adapters(probe_size)

    @property
    def teacher(self):
        return getattr(self, "_teacher", None)

    def _freeze_teacher(self):
        teacher = self.teacher

        if teacher is None:
            return

        teacher.eval()

        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    def _move_teacher_to(self, device):
        teacher = self.teacher

        if teacher is None:
            raise RuntimeError("教师模型为空，无法执行蒸馏训练。")

        teacher_device = next(teacher.parameters()).device

        if teacher_device != device:
            teacher.to(device)

        self._freeze_teacher()

    def get_current_kd_weight(self):
        """
        计算当前Feature KD权重。

        第1～50轮：
            KD权重为0。

        第51～90轮：
            从0逐渐增加到KD_WEIGHT。

        第91～150轮：
            保持KD_WEIGHT。
        """
        if not self.model.training:
            return 0.0

        current_epoch = (
            self.kd_step
            / max(self.iters_per_epoch, 1)
        )

        if current_epoch < self.kd_start_epoch:
            return 0.0

        ramp_progress = (
            current_epoch
            - self.kd_start_epoch
        ) / max(
            self.kd_ramp_epochs,
            1,
        )

        ramp_progress = max(
            0.0,
            min(
                1.0,
                ramp_progress,
            ),
        )

        return (
            self.kd_weight
            * ramp_progress
        )

    @staticmethod
    def build_foreground_mask(batch, feature_h, feature_w, device):
        """
        根据真实标注框生成特征图前景区域掩码。
        batch["bboxes"] 是归一化 xywh。
        """
        batch_size = batch["img"].shape[0]

        mask = torch.zeros(
            batch_size,
            1,
            feature_h,
            feature_w,
            device=device,
        )

        if batch["bboxes"].numel() == 0:
            return mask

        boxes = batch["bboxes"].to(device)
        batch_indices = batch["batch_idx"].view(-1).long().to(device)

        for image_index, box in zip(batch_indices, boxes):
            x_center, y_center, width, height = box

            x1 = int(
                torch.floor((x_center - width / 2) * feature_w)
                .clamp(0, feature_w - 1)
                .item()
            )

            y1 = int(
                torch.floor((y_center - height / 2) * feature_h)
                .clamp(0, feature_h - 1)
                .item()
            )

            x2 = int(
                torch.ceil((x_center + width / 2) * feature_w)
                .clamp(0, feature_w)
                .item()
            )

            y2 = int(
                torch.ceil((y_center + height / 2) * feature_h)
                .clamp(0, feature_h)
                .item()
            )

            if x2 > x1 and y2 > y1:
                mask[image_index, 0, y1:y2, x1:x2] = 1.0

        return mask

    @staticmethod
    def sort_features(features):
        """
        按特征图面积从大到小排序。
        输出顺序：P3、P4、P5。
        """
        return sorted(
            features,
            key=lambda x: x.shape[-2] * x.shape[-1],
            reverse=True,
        )

    def _remove_hooks(self):
        if self._teacher_hook_handle is not None:
            self._teacher_hook_handle.remove()
            self._teacher_hook_handle = None

        if self._student_hook_handle is not None:
            self._student_hook_handle.remove()
            self._student_hook_handle = None

    def _register_hooks(self):
        self._remove_hooks()

        teacher = self.teacher

        if teacher is None:
            raise RuntimeError("教师模型为空，无法注册 Hook。")

        teacher_decoder = teacher.model[-1]
        student_detect = self.model[-1]

        self._teacher_hook_handle = teacher_decoder.register_forward_pre_hook(
            PyramidInputHook(self.teacher_cache)
        )

        self._student_hook_handle = student_detect.register_forward_pre_hook(
            PyramidInputHook(self.student_cache)
        )

    def _get_pyramid_features(self, cache, name):
        if "features" not in cache:
            raise RuntimeError(f"{name} 特征为空。")

        features = self.sort_features(cache["features"])

        if len(features) != 3:
            raise RuntimeError(
                f"{name} 应输出 P3、P4、P5 三层特征，当前得到 {len(features)} 层。"
            )

        return features

    def _build_adapters(self, probe_size):
        """
        用虚拟输入自动探测教师与学生的实际通道数。
        """
        device = next(self.parameters()).device

        self.eval()
        self._move_teacher_to(device)

        self.teacher_cache.clear()
        self.student_cache.clear()

        dummy = torch.zeros(
            1,
            3,
            probe_size,
            probe_size,
            device=device,
        )

        with torch.no_grad():
            self.teacher(dummy)
            self.predict(dummy)

        teacher_features = self._get_pyramid_features(
            self.teacher_cache,
            "教师模型",
        )

        student_features = self._get_pyramid_features(
            self.student_cache,
            "学生模型",
        )

        teacher_channels = [feat.shape[1] for feat in teacher_features]
        student_channels = [feat.shape[1] for feat in student_features]

        LOGGER.info("\n教师 RT-DETR 特征：")
        for i, feat in enumerate(teacher_features):
            LOGGER.info(f"P{i + 3}: {tuple(feat.shape)}")

        LOGGER.info("\n学生 YOLO-MobileNetV2 特征：")
        for i, feat in enumerate(student_features):
            LOGGER.info(f"P{i + 3}: {tuple(feat.shape)}")

        LOGGER.info("\nAdapter 通道对齐：")
        for i, (student_c, teacher_c) in enumerate(
            zip(student_channels, teacher_channels)
        ):
            LOGGER.info(
                f"P{i + 3}: 学生 {student_c} 通道 -> 教师 {teacher_c} 通道"
            )

        self.adapters = nn.ModuleList(
            [
                FeatureAdapter(
                    student_channels=student_c,
                    teacher_channels=teacher_c,
                )
                for student_c, teacher_c in zip(
                    student_channels,
                    teacher_channels,
                )
            ]
        )

        self.teacher_cache.clear()
        self.student_cache.clear()

        self.train()

    def train(self, mode=True):
        super().train(mode)
        self._freeze_teacher()
        return self

    def __getstate__(self):
        """
        EMA deepcopy 和 checkpoint 保存时不保存教师模型。
        """
        state = self.__dict__.copy()

        state["_teacher"] = None
        state["teacher_cache"] = {}
        state["student_cache"] = {}
        state["_teacher_hook_handle"] = None
        state["_student_hook_handle"] = None

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        if "teacher_cache" not in self.__dict__:
            self.teacher_cache = {}

        if "student_cache" not in self.__dict__:
            self.student_cache = {}

        if "_teacher_hook_handle" not in self.__dict__:
            self._teacher_hook_handle = None

        if "_student_hook_handle" not in self.__dict__:
            self._student_hook_handle = None

    def feature_kd_loss(self, batch):
        """
        P4、P5 前景区域特征蒸馏。
        默认跳过 P3，降低低层纹理负迁移。
        """
        teacher_features = self._get_pyramid_features(
            self.teacher_cache,
            "教师模型",
        )

        student_features = self._get_pyramid_features(
            self.student_cache,
            "学生模型",
        )

        level_losses = []

        for level_index in self.distill_level_indices:
            adapter = self.adapters[level_index]
            student_feature = student_features[level_index]
            teacher_feature = teacher_features[level_index]

            aligned_student = adapter(student_feature)
            teacher_target = teacher_feature.detach()

            if aligned_student.shape[-2:] != teacher_target.shape[-2:]:
                teacher_target = F.interpolate(
                    teacher_target,
                    size=aligned_student.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            student_norm = F.normalize(
                aligned_student.float(),
                p=2,
                dim=1,
                eps=1e-6,
            )

            teacher_norm = F.normalize(
                teacher_target.float(),
                p=2,
                dim=1,
                eps=1e-6,
            )

            feature_h, feature_w = aligned_student.shape[-2:]

            foreground_mask = self.build_foreground_mask(
                batch=batch,
                feature_h=feature_h,
                feature_w=feature_w,
                device=aligned_student.device,
            )

            spatial_weight = (
                foreground_mask
                + self.background_weight * (1.0 - foreground_mask)
            )

            pixel_error = (
                student_norm - teacher_norm
            ).pow(2).sum(dim=1, keepdim=True)

            level_loss = (
                pixel_error * spatial_weight
            ).sum() / (spatial_weight.sum() + 1e-6)

            level_losses.append(level_loss)

        if len(level_losses) == 0:
            return torch.zeros(
                1,
                device=batch["img"].device,
            ).squeeze()

        return torch.stack(level_losses).mean()

    def loss(self, batch, preds=None):
        """
        训练阶段：
        YOLO 检测损失 + 渐进式前景区域特征蒸馏损失

        验证阶段：
        仅计算 YOLO 检测损失。
        """

        if not self.training:
            det_loss, det_items = super().loss(batch, preds)

            kd_zero = torch.zeros(
                1,
                device=det_loss.device,
                dtype=det_loss.dtype,
            )

            return det_loss, torch.cat(
                [
                    det_items.detach().flatten(),
                    kd_zero,
                ]
            )

        self.teacher_cache.clear()
        self.student_cache.clear()

        self._move_teacher_to(batch["img"].device)

        with torch.no_grad():
            self.teacher(batch["img"])

        if preds is None:
            preds = self.predict(batch["img"])

        det_loss, det_items = super().loss(batch, preds)

        current_kd_weight = self.get_current_kd_weight()

        kd_feature_loss = self.feature_kd_loss(batch)

        kd_total = (
            current_kd_weight
            * kd_feature_loss
            * batch["img"].shape[0]
        )

        self.kd_step += 1

        total_loss = det_loss + kd_total

        loss_items = torch.cat(
            [
                det_items.detach().flatten(),
                (current_kd_weight * kd_feature_loss).detach().reshape(1),
            ]
        )

        self.teacher_cache.clear()
        self.student_cache.clear()

        return total_loss, loss_items


class FeatureKDTrainer(DetectionTrainer):
    """
    RT-DETR 到 YOLO-MobileNetV2 渐进式前景区域特征蒸馏 Trainer。
    """

    teacher_weight = None
    kd_weight = 1.0
    probe_size = 640

    kd_start_epoch = 30
    kd_ramp_epochs = 40
    iters_per_epoch = 105
    distill_level_indices = (1, 2)
    background_weight = 0.05

    def get_model(self, cfg=None, weights=None, verbose=True):
        if self.teacher_weight is None:
            raise ValueError("FeatureKDTrainer.teacher_weight 未设置。")

        model = RTDETRToYOLOMobileFeatureKDModel(
            cfg=cfg,
            ch=self.data["channels"],
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
            teacher_weight=self.teacher_weight,
            kd_weight=self.kd_weight,
            probe_size=self.probe_size,
            kd_start_epoch=self.kd_start_epoch,
            kd_ramp_epochs=self.kd_ramp_epochs,
            iters_per_epoch=self.iters_per_epoch,
            distill_level_indices=self.distill_level_indices,
            background_weight=self.background_weight,
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
        )

        return DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )