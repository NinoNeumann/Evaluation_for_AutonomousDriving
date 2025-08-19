"""
FDE (Final Displacement Error) 和 Failure Rate 评测指标实现
自动驾驶轨迹预测评估脚本

主要功能:
1. FDE (Final Displacement Error): 计算预测轨迹终点与真实轨迹终点的欧几里得距离
2. Failure Rate: 计算预测失败的比例（当FDE超过阈值时认为失败）

作者: shl
"""

import math
from typing import List, Tuple


def compute_fde(pred_traj: List[Tuple], gt_traj: List[Tuple]) -> float:
    """
    计算FDE (Final Displacement Error)
    
    FDE定义: 预测轨迹的最后一个时间步与真实轨迹最后一个时间步之间的欧几里得距离
    
    Args:
        pred_traj: 预测轨迹，格式为 [(x1, y1), (x2, y2), ..., (xT, yT)]
        gt_traj: 真实轨迹，格式为 [(x1, y1), (x2, y2), ..., (xT, yT)]
        
    Returns:
        fde: Final Displacement Error (单位: 米)
    """
    if len(pred_traj) != len(gt_traj):
        raise ValueError(f"预测轨迹和真实轨迹长度不匹配: {len(pred_traj)} vs {len(gt_traj)}")
    
    if len(pred_traj) == 0:
        raise ValueError("轨迹不能为空")
    
    # 获取最后一个时间步的坐标 (只考虑x, y坐标)
    pred_final = pred_traj[-1]
    gt_final = gt_traj[-1]
    
    pred_x, pred_y = pred_final[0], pred_final[1]
    gt_x, gt_y = gt_final[0], gt_final[1]
    
    # 计算欧几里得距离
    fde = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
    
    return float(fde)


def compute_failure_rate(pred_traj: List[Tuple], gt_traj: List[Tuple], 
                        threshold: float = 2.0) -> float:
    """
    计算Failure Rate (失败率)
    
    Failure Rate定义: 当预测轨迹的FDE超过给定阈值时，认为该预测失败。
    
    Args:
        pred_traj: 预测轨迹，格式为 [(x1, y1), (x2, y2), ..., (xT, yT)]
        gt_traj: 真实轨迹，格式为 [(x1, y1), (x2, y2), ..., (xT, yT)]
        threshold: 失败阈值 (单位: 米)，默认为2.0米
        
    Returns:
        failure_rate: 失败率，0表示成功，1表示失败
    """
    fde = compute_fde(pred_traj, gt_traj)
    return float(fde > threshold)