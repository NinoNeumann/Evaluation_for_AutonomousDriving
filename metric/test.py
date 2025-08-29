import math
import numpy as np
from typing import List, Tuple

def compute_L2_persecond_RSS(preds, gts, reduce='mean', secs=(1, 2, 3, 4), hz=2):
    """
    L2(RSS)@k秒 = sqrt( sum_{t<=k*hz} ||p_t - g_t||_2^2 )
    preds, gts: 形状 (N, T, 2)
    reduce: 'mean' | 'sum' | 'none'（返回每条轨迹）
    secs: 需要统计的秒数元组
    hz: 每秒采样点数（默认 2 -> 0.5s 间隔）
    """
    preds = np.asarray(preds, dtype=np.float32)
    gts   = np.asarray(gts,   dtype=np.float32)
    if preds.shape != gts.shape or preds.ndim != 3 or preds.shape[2] != 2:
        raise ValueError(f"输入形状应一致且为 (N, T, 2)，得到 {preds.shape} vs {gts.shape}")

    N, T, _ = preds.shape
    out = []
    for k in secs:
        p_len = min(k * hz, T)
        diff = preds[:, :p_len, :] - gts[:, :p_len, :]      # (N, p_len, 2)
        e2   = np.sum(diff**2, axis=2)                      # (N, p_len)
        rss_per_traj = np.sqrt(np.sum(e2, axis=1))          # (N,)

        if reduce == 'mean':
            out.append(float(rss_per_traj.mean()))
        elif reduce == 'sum':
            out.append(float(rss_per_traj.sum()))
        elif reduce == 'none':
            out.append(rss_per_traj)                        # 返回 (N,)
        else:
            raise ValueError("reduce 必须是 'mean' | 'sum' | 'none'")
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
