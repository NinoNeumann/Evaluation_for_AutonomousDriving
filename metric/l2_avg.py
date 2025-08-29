from metric.l2 import stp3_l2_persecond
import numpy as np

def compute_L2_avg(preds, gts, secs=(1, 2, 3), hz=2):
    """
    L2@k（时间前缀平均）的均值，按 secs 的长度平均
    仅支持单条轨迹 (T,2)
    """
    l2_list = stp3_l2_persecond(preds, gts, secs=secs, hz=hz)
    return float(np.mean(l2_list))
    