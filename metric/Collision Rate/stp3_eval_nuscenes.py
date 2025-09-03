import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

# ===== Default config (aligned with ST-P3 open-loop table) =====
STEP_HZ = 2.0            # 2 Hz -> 0.5s
FUTURE_STEPS = 6         # 3.0 s
EGO_L, EGO_W = 4.0, 1.8  # ego box length & width (meters)
CHECK_CATS = ("vehicle.", "human.pedestrian")
SMALL_EPS = 1e-9

# ----- math & geometry -----
def rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)

def obb_axes(yaw: float) -> Tuple[np.ndarray, np.ndarray]:
    R = rot2d(yaw)
    return R[:, 0], R[:, 1]  # ux, uy

def obb_overlap(c1, ux1, uy1, ex1, ey1, c2, ux2, uy2, ex2, ey2) -> bool:
    """OBB-OBB SAT on 4 axes (ux1,uy1,ux2,uy2)."""
    t = c2 - c1

    def proj(ux, uy, ex, ey, L):
        return ex * abs(np.dot(ux, L)) + ey * abs(np.dot(uy, L))

    for L in (ux1, uy1, ux2, uy2):
        if abs(np.dot(t, L)) > proj(ux1, uy1, ex1, ey1, L) + proj(ux2, uy2, ex2, ey2, L) + SMALL_EPS:
            return False
    return True

def ego_obb(center_xy: np.ndarray, yaw: float):
    ux, uy = obb_axes(yaw)
    ex, ey = EGO_L / 2.0, EGO_W / 2.0
    return center_xy, ux, uy, ex, ey

def box_to_obb2d(box: Box):
    yaw = Quaternion(box.orientation).yaw_pitch_roll[0]
    c = np.array([box.center[0], box.center[1]], dtype=np.float64)
    ux, uy = obb_axes(yaw)
    # 正确：ex 对应 length/2（wlh[1]），ey 对应 width/2（wlh[0]）
    ex, ey = box.wlh[1] / 2.0, box.wlh[0] / 2.0
    return c, ux, uy, ex, ey

# ----- nuScenes helpers -----
def get_sample_and_next_tokens(nusc: NuScenes, start_token: str, n_steps: int) -> List[str]:
    tokens = []
    tok = start_token
    for _ in range(n_steps):
        samp = nusc.get("sample", tok)
        if not samp["next"]:
            return []
        tok = samp["next"]
        tokens.append(tok)
    return tokens

def get_lidar_sd_token(nusc: NuScenes, sample_token: str) -> str:
    return nusc.get("sample", sample_token)["data"]["LIDAR_TOP"]

def get_ego_pose_yaw_xy(nusc: NuScenes, sample_token: str) -> Tuple[float, np.ndarray]:
    sd = nusc.get("sample_data", get_lidar_sd_token(nusc, sample_token))
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    yaw = Quaternion(pose["rotation"]).yaw_pitch_roll[0]
    xy = np.array(pose["translation"][:2], dtype=np.float64)
    return yaw, xy

def get_gt_boxes_2d(nusc: NuScenes, sample_token: str):
    boxes: List[Box] = nusc.get_boxes(get_lidar_sd_token(nusc, sample_token))
    out = []
    for b in boxes:
        name = getattr(b, "name", None)
        if name and name.startswith(CHECK_CATS):
            out.append(box_to_obb2d(b))
    return out

def ego_local_to_global(xy_local: np.ndarray, ego_translation: np.ndarray, ego_yaw: float) -> np.ndarray:
    R = rot2d(ego_yaw)
    return (R @ xy_local.T).T + ego_translation[None, :]

def get_gt_ego_future_xy_glb(nusc: NuScenes, start_token: str) -> Optional[np.ndarray]:
    toks = get_sample_and_next_tokens(nusc, start_token, FUTURE_STEPS)
    if len(toks) != FUTURE_STEPS:
        return None
    xy_list = []
    for tok in toks:
        _, xy = get_ego_pose_yaw_xy(nusc, tok)
        xy_list.append(xy)
    return np.asarray(xy_list, dtype=np.float64)

def gt_future_collides(nusc: NuScenes, start_token: str) -> bool:
    """Optional filter: whether GT ego path collides with other agents within horizon."""
    toks = get_sample_and_next_tokens(nusc, start_token, FUTURE_STEPS)
    if len(toks) != FUTURE_STEPS:
        return True  # treat as invalid
    for i, tok in enumerate(toks):
        yaw_i, xy_i = get_ego_pose_yaw_xy(nusc, tok)
        c1, ux1, uy1, ex1, ey1 = ego_obb(xy_i, yaw_i)
        for (c2, ux2, uy2, ex2, ey2) in get_gt_boxes_2d(nusc, tok):
            if obb_overlap(c1, ux1, uy1, ex1, ey1, c2, ux2, uy2, ex2, ey2):
                return True
    return False

# ----- evaluator -----
class STP3StyleEvaluator:
    def __init__(self,
                 nusc: NuScenes,
                 use_gt_yaw_each_step: bool = True,
                 filter_gt_colliding: bool = True,
                 zero_eps: float = 1e-2):
        self.nusc = nusc
        self.use_gt_yaw_each_step = use_gt_yaw_each_step
        self.filter_gt_colliding = filter_gt_colliding
        self.zero_eps = zero_eps

    def _zero_tiny_disp(self, traj_local: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(traj_local, axis=1)
        traj_local[d < self.zero_eps] = 0.0
        return traj_local

    def evaluate(self, preds: Dict[str, List[List[float]]]):
        num = 0
        coll_1 = coll_2 = coll_3 = 0

        for start_tok, traj in preds.items():
            traj = np.asarray(traj, dtype=np.float64)
            if traj.shape != (FUTURE_STEPS, 2):
                continue

            next_toks = get_sample_and_next_tokens(self.nusc, start_tok, FUTURE_STEPS)
            if len(next_toks) != FUTURE_STEPS:
                continue

            if self.filter_gt_colliding and gt_future_collides(self.nusc, start_tok):
                continue

            traj = self._zero_tiny_disp(traj.copy())

            # transform to global
            t0_yaw, t0_xy = get_ego_pose_yaw_xy(self.nusc, start_tok)
            traj_glb = ego_local_to_global(traj, t0_xy, t0_yaw)

            # GT ego future (for L2)
            gt_xy = get_gt_ego_future_xy_glb(self.nusc, start_tok)
            if gt_xy is None:
                continue


            # Collision per step
            step_ego_yaws = []
            for tok in next_toks:
                y_i, _ = get_ego_pose_yaw_xy(self.nusc, tok)
                step_ego_yaws.append(y_i)

            collided = [False] * FUTURE_STEPS
            for i, tok in enumerate(next_toks):
                ego_xy = traj_glb[i]
                ego_yaw = step_ego_yaws[i] if self.use_gt_yaw_each_step else t0_yaw
                c1, ux1, uy1, ex1, ey1 = ego_obb(ego_xy, ego_yaw)
                for (c2, ux2, uy2, ex2, ey2) in get_gt_boxes_2d(self.nusc, tok):
                    if obb_overlap(c1, ux1, uy1, ex1, ey1, c2, ux2, uy2, ex2, ey2):
                        collided[i] = True
                        break

            num += 1
            if any(collided[:2]): coll_1 += 1
            if any(collided[:4]): coll_2 += 1
            if any(collided[:6]): coll_3 += 1

        if num == 0:
            return {"num": 0, "CR@1s(%)": None, "CR@2s(%)": None, "CR@3s(%)": None}

        to_pct = lambda x: 100.0 * x / num
        return {
            "num": num,
            "CR@1s(%)": to_pct(coll_1),
            "CR@2s(%)": to_pct(coll_2),
            "CR@3s(%)": to_pct(coll_3),
        }

# ----- IO -----
def load_preds(path: Path) -> Dict[str, List[List[float]]]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    if path.suffix.lower() in (".pkl", ".pickle"):
        return pickle.loads(path.read_bytes())
    raise ValueError(f"Unsupported preds file: {path}")

# ----- CLI -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nusc_root", required=True, help="nuScenes dataroot (contains v1.0-*)")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--preds", required=True, help="JSON/PKL: {sample_token: [[x,y]*6] in ego(t0)}")
    ap.add_argument("--use_gt_yaw_each_step", action="store_true",
                    help="Use GT ego yaw at each step when placing ego box (recommended).")
    ap.add_argument("--filter_gt_colliding", action="store_true",
                    help="Skip samples whose GT ego path collides in horizon.")
    ap.add_argument("--zero_eps", type=float, default=1e-2,
                    help="Zero-out tiny displacements (<zero_eps meters).")
    args = ap.parse_args()

    nusc = NuScenes(dataroot=args.nusc_root, version=args.version, verbose=False)
    preds = load_preds(Path(args.preds))
    ev = STP3StyleEvaluator(
        nusc,
        use_gt_yaw_each_step=args.use_gt_yaw_each_step,
        filter_gt_colliding=args.filter_gt_colliding,
        zero_eps=args.zero_eps,
    )
    print(ev.evaluate(preds))

if __name__ == "__main__":
    main()
