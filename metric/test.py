import math
import numpy as np
from typing import List, Tuple
import numpy as np
import numpy as np

def stp3_l2_persecond(pred, gt, secs=(1, 2, 3), hz=2, reduction=None):
    """
    单条轨迹的 L2@k 指标（时间前缀平均）
    参数:
      pred, gt: (T, 2)
      secs:     统计到第 k 秒的时间前缀平均，如 (1,2,3)
      hz:       采样频率 (默认 2Hz -> 0.5s 一点)
    返回:
      [L2@1s, L2@2s, L2@3s]（或按 secs 决定）
    """
    pred = np.asarray(pred, dtype=np.float32)
    gt   = np.asarray(gt,   dtype=np.float32)

    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt 形状不一致: {pred.shape} vs {gt.shape}")
    if pred.ndim != 2 or pred.shape[1] != 2:
        raise ValueError(f"期望 (T,2)，实际 {pred.shape}")

    # (T,): 每步欧氏距离
    d = np.linalg.norm(pred - gt, axis=1)
    T = d.shape[0]

    out = []
    for s in secs:
        # 用到的前缀步数 p：至少 1，最多 T
        p = int(min(max(int(s * hz), 1), T))
        out.append(float(d[:p].mean()))
    return out

def compute_ADE(preds, gts):
    """
    ADE: 所有点（以及所有样本）的欧氏距离平均
    支持 (T,2) 或 (N,T,2)，返回标量
    """
    preds = np.asarray(preds, dtype=np.float32)
    gts   = np.asarray(gts,   dtype=np.float32)
    if preds.shape != gts.shape:
        raise ValueError(f"shape 不一致: {preds.shape} vs {gts.shape}")

    # 对最后一维(坐标维)求 L2；形状：(T,) 或 (N,T)
    dists = np.linalg.norm(preds - gts, axis=-1)
    # 全体平均 -> 标量
    return float(dists.mean())

def compute_L2_avg(preds, gts, secs=(1, 2, 3), hz=2):
    """
    L2@k（时间前缀平均）的均值，按 secs 的长度平均
    仅支持单条轨迹 (T,2)
    """
    l2_list = stp3_l2_persecond(preds, gts, secs=secs, hz=hz)
    return float(np.mean(l2_list))

def compute_ADE_L2pers_L2AVG(preds, gts, secs=(1, 2, 3), hz=2):
    """
    返回：
      ade:        标量 ADE（全体平均）
      l2_persec:  按 secs 返回的 L2@k 列表
      l2_avg:     l2_persec 的均值
    """
    ade = compute_ADE(preds, gts)
    l2_persec = stp3_l2_persecond(preds, gts, secs=secs, hz=hz)
    l2_avg = float(np.mean(l2_persec))
    return ade, l2_persec, l2_avg


