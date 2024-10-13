import numpy as np
import torch

def calculate_iou(pred, label, num_classes):
    """
    计算每个类别的 IoU（Intersection over Union）对于一批图像。

    参数:
    - pred: 预测的分割结果（logits），shape为(B, C, H, W)
    - label: 实际的标签结果，shape为(B, H, W)
    - num_classes: 类别的总数

    返回:
    - IoU: 每个类别的 IoU 列表
    """
    # 将 logits 转换为类别预测
    pred = torch.argmax(pred, axis=1)  # shape: (B, H, W)

    ious = []

    for cls in range(num_classes):
        intersection = 0
        union = 0

        for b in range(pred.shape[0]):  # 遍历每张图像
            pred_cls = (pred[b] == cls)
            label_cls = (label[b] == cls)

            intersection += np.logical_and(pred_cls, label_cls).sum()
            union += np.logical_or(pred_cls, label_cls).sum()

        if union == 0:
            iou = float('nan')  # 如果并集为 0，意味着该类别在标签和预测中都不存在
        else:
            iou = intersection / union

        ious.append(iou)

    return ious


def calculate_miou(pred, label, num_classes):
    """
    计算 mIoU（Mean Intersection over Union）对于一批图像。

    参数:
    - pred: 预测的分割结果（logits），shape为(B, C, H, W)
    - label: 实际的标签结果，shape为(B, H, W)
    - num_classes: 类别的总数

    返回:
    - mIoU: 所有类别的平均 IoU
    """
    ious = calculate_iou(pred, label, num_classes)

    # 计算所有类别的有效 IoU（跳过 nan 值）
    valid_ious = [iou for iou in ious if not np.isnan(iou)]

    if len(valid_ious) == 0:
        return 0.0  # 没有有效的 IoU
    else:
        miou = np.mean(valid_ious)
        return miou