import math
import numpy as np
from typing import List, Tuple

def compute_L2_persecond(preds, gts):
    """
    Compute cumulative L2 distance per second for trajectories.
    
    每秒累积欧氏距离：
    - 1s: 前 2 个点
    - 2s: 前 4 个点
    - 3s: 前 6 个点
    - 4s: 前 8 个点
    
    Args:
        preds: list of predicted trajectories, shape (N, 8, 2)
        gts: list of ground truth trajectories, shape (N, 8, 2)
    
    Returns:
        L2_list: list of cumulative L2 distances for each second [L2_1s, L2_2s, L2_3s, L2_4s]
    """
    # 转为 NumPy 数组，方便向量化计算
    preds = np.array(preds, dtype=np.float32)  # shape (N, 8, 2)
    gts = np.array(gts, dtype=np.float32)      # shape (N, 8, 2)
    
    cum_points = [2, 4, 6, 8]  # 每秒对应的累计点数
    L2_list = []
    
    for p_len in cum_points:
        # 计算每条轨迹前 p_len 个点的欧氏距离
        # preds[:, :p_len, :] - gts[:, :p_len, :] 形状 (N, p_len, 2)
        # np.linalg.norm(..., axis=2) 得到每条轨迹每个点的距离，形状 (N, p_len)
        dists = np.linalg.norm(preds[:, :p_len, :] - gts[:, :p_len, :], axis=2)
        
        # 累加所有轨迹所有点的距离
        L2 = np.sum(dists)
        L2_list.append(L2)
    
    return L2_list

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
    
    Args:
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
