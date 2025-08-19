"""
ADE (Average Displacement Error) 和 L2 指标实现
自动驾驶轨迹预测评估脚本

主要功能:
1. ADE (Average Displacement Error): 计算预测轨迹与真实轨迹在所有时间步的平均欧几里得距离
2. L2: 计算预测轨迹与真实轨迹的L2范数距离

作者: shl
"""

import math
from typing import List, Tuple


def compute_ade(pred_traj: List[Tuple], gt_traj: List[Tuple]) -> float:
    """
    计算ADE (Average Displacement Error)
    
    ADE定义: 预测轨迹与真实轨迹在所有时间步上的平均欧几里得距离
    
    Args:
        pred_traj: 预测轨迹，格式为 [(x1, y1), (x2, y2), ..., (xT, yT)]
        gt_traj: 真实轨迹，格式为 [(x1, y1), (x2, y2), ..., (xT, yT)]
        
    Returns:
        ade: Average Displacement Error (单位: 米)
    """
    if len(pred_traj) != len(gt_traj):
        raise ValueError(f"预测轨迹和真实轨迹长度不匹配: {len(pred_traj)} vs {len(gt_traj)}")
    
    # 计算每个时间步的欧几里得距离
    distances = []
    for pred_point, gt_point in zip(pred_traj, gt_traj):
        pred_x, pred_y = pred_point[0], pred_point[1]
        gt_x, gt_y = gt_point[0], gt_point[1]
        
        distance = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        distances.append(distance)
    
    # 返回平均距离
    ade = sum(distances) / len(distances)
    return float(ade)


def compute_l2(pred_traj: List[Tuple], gt_traj: List[Tuple]) -> float:
    """
    计算L2范数距离
    
    L2定义: 预测轨迹与真实轨迹的L2范数（所有时间步距离的平方和再开方）
    
    Args:
        pred_traj: 预测轨迹，格式为 [(x1, y1), (x2, y2), ..., (xT, yT)]
        gt_traj: 真实轨迹，格式为 [(x1, y1), (x2, y2), ..., (xT, yT)]
        
    Returns:
        l2_distance: L2范数距离 (单位: 米)
    """
    if len(pred_traj) != len(gt_traj):
        raise ValueError(f"预测轨迹和真实轨迹长度不匹配: {len(pred_traj)} vs {len(gt_traj)}")
    
    # 计算所有时间步的距离平方和
    squared_distances_sum = 0.0
    for pred_point, gt_point in zip(pred_traj, gt_traj):
        pred_x, pred_y = pred_point[0], pred_point[1]
        gt_x, gt_y = gt_point[0], gt_point[1]
        
        squared_distance = (pred_x - gt_x)**2 + (pred_y - gt_y)**2
        squared_distances_sum += squared_distance
    
    # 计算L2范数
    l2_distance = math.sqrt(squared_distances_sum)
    return float(l2_distance)
