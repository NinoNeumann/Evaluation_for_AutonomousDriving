import math
import numpy as np
from typing import List, Tuple
from metric.l2_persecond import stp3_l2_persecond
from metric.ADE import compute_ADE


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


