import numpy as np

def compute_ADE(preds, gts):
    """
    计算平均位移误差 ADE
    :param preds: list[list[tuple]]，预测轨迹，每条轨迹有8个(x,y)
    :param gts: list[list[tuple]]，真实轨迹，每条轨迹有8个(x,y)
    :return: float，ADE值
    """
    preds = np.array(preds, dtype=np.float32)  # shape (N, 8, 2)
    gts = np.array(gts, dtype=np.float32)      # shape (N, 8, 2)

    # 逐点欧式距离
    dists = np.linalg.norm(preds - gts, axis=2)  # shape (N, 8)

    # 平均
    ade = np.mean(dists)
    return ade


# preds = [
#     [(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8)]   
# ]
# gts = [ 
#     [(34,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8)]
# ]

# print(compute_ADE(preds, gts))

git config --global user.email falsestunch@gmail.com
git config --global user.name ZZH