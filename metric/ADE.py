import numpy as np
def compute_ADE(preds, gts):
    """
    ADE: 所有点（以及所有样本）的欧氏距离平均
    支持 (T,2) 或 (N,T,2)，返回标量
    """
    preds = np.asarray(preds, dtype=np.float32)
    gts   = np.asarray(gts,   dtype=np.float32)
    if preds.shape != gts.shape:
        raise ValueError(f"shape 不一致: {preds.shape} vs {gts.shape}")

    # 对最后一维(坐标维)求 L2；形状：(T,) 或 (N,T)
    dists = np.linalg.norm(preds - gts, axis=-1)
    # 全体平均 -> 标量
    return float(dists.mean())