import numpy as np
import math

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