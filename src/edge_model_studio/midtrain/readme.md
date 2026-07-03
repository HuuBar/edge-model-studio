# 自动生成多域数据 + mid-train

python midtrain_run.py --data_dir ./mid_data --out_dir ./out_mid --domain_prefix

# 从 pretrain ckpt 继续训练（mid-train）

python midtrain_run.py --data_dir ./mid_data --out_dir ./out_mid \
  --domain_prefix \
  --base_ckpt ./out_pretrain/ckpt_step_2000.pt

# 自定义域权重（更偏垂域）

python midtrain_run.py --data_dir ./mid_data --out_dir ./out_mid \
  --domain_prefix \
  --domain_weights "sports_health=4,code=1,general=1"

# DDP

torchrun --nproc_per_node=2 midtrain_run.py --ddp --data_dir ./mid_data --out_dir ./out_mid --domain_prefix