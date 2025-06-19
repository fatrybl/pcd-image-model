# from mmdet3d.registry import MODELS
from mmdet.models.losses import FocalLoss as mmdetFocalLoss, L1Loss as mmdetL1Loss, CrossEntropyLoss as mmdetCrossEntropyLoss

from mmdet.models.task_modules.assigners import MaxIoUAssigner as mmdetMaxIoUAssigner
from mmdet3d.structures.ops import BboxOverlapsNearest3D as mmde3dBboxOverlapsNearest3D

from mmengine.registry import MODELS, TASK_UTILS

@MODELS.register_module()
class FocalLoss(mmdetFocalLoss):
    ...

@MODELS.register_module()
class L1Loss(mmdetL1Loss):
    ...

@MODELS.register_module()
class CrossEntropyLoss(mmdetCrossEntropyLoss):
    ...

@TASK_UTILS.register_module()
class MaxIoUAssigner(mmdetMaxIoUAssigner):
    ...

@TASK_UTILS.register_module()
class BboxOverlapsNearest3D(mmde3dBboxOverlapsNearest3D):
    ...