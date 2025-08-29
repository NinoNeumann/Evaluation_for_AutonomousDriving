import math
import numpy as np
from typing import List, Tuple
import numpy as np

def stp3_l2_persecond(preds, gts, secs=(1, 2, 3), hz=2, reduction='mean'):
    """
    ST-P3 风格的 L2@k秒：
      对前 (k*hz) 个点的欧氏距离先在时间维取平均，再按样本做 reduce。
    参数:
      preds, gts: (N, T, 2)  预测/GT 轨迹（单位: 米）
      secs: 统计的秒数
      hz: 采样频率 (默认 2Hz -> 0.5s 一点)
      reduction: 'mean' | 'sum' | 'none'
    返回:
      [L2@1s, L2@2s, L2@3s]（或按 secs 决定）
    """
    preds = np.asarray(preds, dtype=np.float32)
    gts   = np.asarray(gts,   dtype=np.float32)
    if preds.shape != gts.shape or preds.ndim != 3 or preds.shape[-1] != 2:
        raise ValueError(f"expect (N,T,2), got {preds.shape} and {gts.shape}")

    # (N, T): 每步欧氏距离
    d = np.linalg.norm(preds - gts, axis=2)
    N, T = d.shape
    out = []
    for s in secs:
        p = min(s * hz, T)
        per_traj = d[:, :p].mean(axis=1)  # 先对时间平均
        if reduction == 'mean':
            out.append(float(per_traj.mean()))   # 再对样本平均
        elif reduction == 'sum':
            out.append(float(per_traj.sum()))
        elif reduction == 'none':
            out.append(per_traj)                 # 返回 (N,)
        else:
            raise ValueError("reduction must be 'mean'|'sum'|'none'")
    return out



def compute_ADE(preds, gts):
    """
    Compute Average Displacement Error (ADE) over all points and trajectories.
    
    ADE 是每个点欧氏距离的平均值。
    
    Args:
        preds: list of predicted trajectories, shape (N, 8, 2)
        gts: list of ground truth trajectories, shape (N, 8, 2)
    
    Returns:
        float: ADE value
    """
    preds = np.array(preds, dtype=np.float32)
    gts = np.array(gts, dtype=np.float32)
    
    # 每个点的欧氏距离，shape (N, 8)
    dists = np.linalg.norm(preds - gts, axis=2)
    
    # 对所有轨迹、所有点求平均
    ade = np.mean(dists)
    return ade

def compute_L2_avg(preds, gts):
    """
    Compute the average of cumulative L2 distances per second.
    
    Args:
        preds: list of predicted trajectories
        gts: list of ground truth trajectories
    
    Returns:
        float: average L2 over 4 seconds
    """
    L2 = compute_L2_persecond(preds, gts)  # 得到每秒 L2 列表
    l2_avg = sum(L2) / 4                    # 平均值
    return l2_avg

def compute_ADE_L2pers_L2AVG(preds, gts):
    """
    Compute ADE, cumulative L2 per second, average L2, and average L2 for first 3 seconds.
    
    Args
        preds: list of predicted trajectories
        gts: list of ground truth trajectories
    
    Returns:
        ade: overall average displacement error
        l2_persecond: list of cumulative L2 distances per second [L2_1s, L2_2s, L2_3s, L2_4s]
        l2_avg: average L2 over 4 seconds
        l2_avg_3s: average L2 over first 3 seconds
    """
    ade = compute_ADE(preds, gts)
    l2_persecond = compute_L2_persecond(preds, gts)
    l2_avg = sum(l2_persecond) / 4
    l2_avg_3s = sum(l2_persecond[:3]) / 3
    return ade, l2_persecond, l2_avg, l2_avg_3s
