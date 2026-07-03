python pretrain_run.py --data_dir ./raw_data --out_dir ./out_pretrain
python pretrain_run.py --data_dir ./raw_data --out_dir ./out_pretrain --reprocess
python pretrain_run.py --data_dir ./raw_data --out_dir ./out_pretrain --benchmark_only
torchrun --nproc_per_node=2 pretrain_run.py --ddp --data_dir ./raw_data --out_dir ./out_pretrain