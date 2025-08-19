"""
文件工具模块
提供JSON文件读取相关的实用函数

作者: shl
"""

import json
import os
from typing import Dict, List, Any, Optional, Union
from pathlib import Path


def read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    读取JSON文件
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        data: 解析后的JSON数据
        
    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON格式错误
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def read_json_safe(file_path: Union[str, Path], default: Any = None) -> Any:
    """
    安全读取JSON文件，出错时返回默认值
    
    Args:
        file_path: JSON文件路径
        default: 出错时返回的默认值
        
    Returns:
        data: 解析后的JSON数据或默认值
    """
    try:
        return read_json(file_path)
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"警告: 读取JSON文件失败 {file_path}: {e}")
        return default


def read_multiple_json(file_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
    """
    批量读取多个JSON文件
    
    Args:
        file_paths: JSON文件路径列表
        
    Returns:
        data_list: 解析后的JSON数据列表
    """
    data_list = []
    
    for file_path in file_paths:
        try:
            data = read_json(file_path)
            data_list.append(data)
        except Exception as e:
            print(f"警告: 读取文件 {file_path} 失败: {e}")
            continue
    
    return data_list


def read_json_from_directory(directory: Union[str, Path], 
                           pattern: str = "*.json") -> List[Dict[str, Any]]:
    """
    读取目录中所有匹配的JSON文件
    
    Args:
        directory: 目录路径
        pattern: 文件匹配模式，默认为"*.json"
        
    Returns:
        data_list: 所有JSON文件的数据列表
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    
    if not directory.is_dir():
        raise NotADirectoryError(f"路径不是目录: {directory}")
    
    json_files = list(directory.glob(pattern))
    
    if not json_files:
        print(f"警告: 在目录 {directory} 中没有找到匹配 {pattern} 的文件")
        return []
    
    print(f"找到 {len(json_files)} 个JSON文件")
    return read_multiple_json(json_files)




def extract_trajectories_from_json(json_data: Dict[str, Any], 
                                 traj_key: str = "trajectory") -> List[List[tuple]]:
    """
    从JSON数据中提取轨迹数据
    
    Args:
        json_data: JSON数据
        traj_key: 轨迹数据的键名
        
    Returns:
        trajectories: 轨迹列表，每个轨迹是 [(x, y), ...] 格式
    """
    if traj_key not in json_data:
        raise KeyError(f"JSON数据中没有找到轨迹键: {traj_key}")
    
    raw_trajectories = json_data[traj_key]
    trajectories = []
    
    for traj in raw_trajectories:
        # 假设轨迹数据格式为 [[x1, y1], [x2, y2], ...] 或 [(x1, y1), (x2, y2), ...]
        if isinstance(traj, list):
            trajectory = []
            for point in traj:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    trajectory.append((float(point[0]), float(point[1])))
                else:
                    print(f"警告: 轨迹点格式错误: {point}")
            
            if trajectory:
                trajectories.append(trajectory)
    
    return trajectories

def write_json(data: Dict[str, Any], file_path: Union[str, Path]):
    """
    写入JSON文件
    
    Args:
        data: JSON数据
        file_path: JSON文件路径
    """
    file_path = Path(file_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
