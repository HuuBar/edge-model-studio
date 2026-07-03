# 单卡/CPU

python sft_run.py --data_dir ./sft_data --out_dir ./out_sft

# 强制重做预处理

python sft_run.py --data_dir ./sft_data --out_dir ./out_sft --reprocess

# 仅跑预处理+benchmark

python sft_run.py --data_dir ./sft_data --out_dir ./out_sft --benchmark_only

# DDP

torchrun --nproc_per_node=2 sft_run.py --ddp --data_dir ./sft_data --out_dir ./out_sft