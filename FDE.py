"""
FDE (Final Displacement Error) 和 Failure Rate 评测指标实现
自动驾驶轨迹预测评估脚本

主要功能:
1. FDE (Final Displacement Error): 计算预测轨迹终点与真实轨迹终点的欧几里得距离
2. Failure Rate: 计算预测失败的比例（当FDE超过阈值时认为失败）

作者: shl
"""

import numpy as np
import torch
from typing import Union, List, Tuple, Optional
import warnings


def compute_fde(pred_traj: Union[np.ndarray, torch.Tensor], 
                gt_traj: Union[np.ndarray, torch.Tensor]) -> float:
    """
    计算FDE (Final Displacement Error)
    
    FDE定义: 预测轨迹的最后一个时间步与真实轨迹最后一个时间步之间的欧几里得距离
    
    Args:
        pred_traj: 预测轨迹，形状为 [..., timesteps, 2] 或 [..., timesteps, 3]
                   最后一维表示坐标 (x, y) 或 (x, y, z)
        gt_traj: 真实轨迹，形状与pred_traj相同
        
    Returns:
        fde: Final Displacement Error (单位: 米)
        
    Raises:
        ValueError: 当预测轨迹和真实轨迹形状不匹配时
        
    Examples:
        >>> pred = np.array([[0, 0], [1, 1], [2, 2]])  # 3个时间步的轨迹
        >>> gt = np.array([[0, 0], [1, 0], [2, 1]])
        >>> fde = compute_fde(pred, gt)
        >>> print(f"FDE: {fde:.4f} 米")
    """
    # 转换为numpy数组以便统一处理
    if torch.is_tensor(pred_traj):
        pred_traj = pred_traj.detach().cpu().numpy()
    if torch.is_tensor(gt_traj):
        gt_traj = gt_traj.detach().cpu().numpy()
    
    # 检查输入形状
    if pred_traj.shape != gt_traj.shape:
        raise ValueError(f"shape error: {pred_traj.shape} vs {gt_traj.shape}")
    
    if len(pred_traj.shape) < 2:
        raise ValueError(f"shape error: (timesteps, coordinates), shape now: {pred_traj.shape}")
    
    # 获取最后一个时间步的坐标 (只考虑x, y坐标)
    pred_final = pred_traj[..., -1, :2]  # [..., 2]
    gt_final = gt_traj[..., -1, :2]      # [..., 2]
    
    # 计算欧几里得距离
    displacement = pred_final - gt_final  # [..., 2]
    squared_distance = np.sum(displacement ** 2, axis=-1)  # [...]
    euclidean_distance = np.sqrt(squared_distance)  # [...]
    
    # 如果是批量数据，返回平均值；否则返回标量
    if euclidean_distance.ndim == 0:
        return float(euclidean_distance)
    else:
        return float(np.mean(euclidean_distance))


def compute_failure_rate(pred_traj: Union[np.ndarray, torch.Tensor], 
                        gt_traj: Union[np.ndarray, torch.Tensor],
                        threshold: float = 2.0) -> float:
    """
    计算Failure Rate (失败率)
    
    Failure Rate定义: 当预测轨迹的FDE超过给定阈值时，认为该预测失败。
    失败率是所有样本中失败样本的比例。
    
    Args:
        pred_traj: 预测轨迹，形状为 [..., timesteps, 2] 或 [..., timesteps, 3]
        gt_traj: 真实轨迹，形状与pred_traj相同
        threshold: 失败阈值 (单位: 米)，默认为2.0米
        
    Returns:
        failure_rate: 失败率，范围 [0, 1]
        
    Examples:
        >>> pred = np.array([[0, 0], [1, 1], [5, 5]])  # 终点误差较大
        >>> gt = np.array([[0, 0], [1, 0], [2, 1]])
        >>> failure_rate = compute_failure_rate(pred, gt, threshold=2.0)
        >>> print(f"失败率: {failure_rate:.4f} ({failure_rate*100:.1f}%)")
    """
    # 转换为numpy数组
    if torch.is_tensor(pred_traj):
        pred_traj = pred_traj.detach().cpu().numpy()
    if torch.is_tensor(gt_traj):
        gt_traj = gt_traj.detach().cpu().numpy()
    
    # 检查输入形状
    if pred_traj.shape != gt_traj.shape:
        raise ValueError(f"shape error: {pred_traj.shape} vs {gt_traj.shape}")
    
    # 获取最后一个时间步的坐标
    pred_final = pred_traj[..., -1, :2]
    gt_final = gt_traj[..., -1, :2]
    
    # 计算终点距离
    displacement = pred_final - gt_final
    distances = np.sqrt(np.sum(displacement ** 2, axis=-1))
    
    # 判断哪些预测失败 (距离超过阈值)
    failures = distances > threshold
    
    # 计算失败率
    if failures.ndim == 0:
        # 单个样本
        return float(failures)
    else:
        # 批量样本
        return float(np.mean(failures))

