from tools.FDE import *
from tools.ADE_L2 import *
from tools.file_utils import read_json_from_directory, write_json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir_root", type=str, default="output/jsons")
    parser.add_argument("--res_save_json_path", type=str, default="output/results.json")
    args = parser.parse_args()

    json_dir_root = args.json_dir_root
    res_save_json_path = args.res_save_json_path

    
    all_cases_metrics = []
    all_ades = []
    all_l2s = []
    all_fdes = []

    data = sorted(read_json_from_directory(json_dir_root))
    for idx, item in enumerate(data):
        gt_traj = item["gt_trajectory"]
        pred_traj = item["pred_trajectory"]
        ade = compute_ade(pred_traj, gt_traj)
        l2 = compute_l2(pred_traj, gt_traj)
        fde = compute_fde(pred_traj, gt_traj)
        # failure_rate = compute_failure_rate(pred_traj, gt_traj)
        all_cases_metrics.append({
            "case_idx": idx,
            "ade": ade,
            "l2": l2,
            "fde": fde,
        })
        all_ades.append(ade)
        all_l2s.append(l2)
        all_fdes.append(fde)
    print(f"average all_ades: {sum(all_ades) / len(all_ades)}")
    print(f"average all_l2s: {sum(all_l2s) / len(all_l2s)}")
    print(f"average all_fdes: {sum(all_fdes) / len(all_fdes)}")


    write_json(all_cases_metrics, res_save_json_path)