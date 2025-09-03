# 1) 预测文件：JSON 或 PKL
#    结构：{ "<start_sample_token>": [[x,y], ..., (6 points)], ... }  —— ego(t0)坐标系, 米, 0.5s/点

# 2) 运行
python metric\Collision Rate\stp3_eval_nuscenes.py \
  --nusc_root ...\nuscenes \ 
  --version v1.0-mini \
  --preds metric\Collision Rate\preds.json \
  --use_gt_yaw_each_step \
  --filter_gt_colliding
