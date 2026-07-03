import os
import subprocess

# Configurations
BASE_MODEL = "/data2/jwllm/model_process/exercise_models/qwen3_0.6b_sft"
DATA_PATH = "/data2/jwllm/datasets/exercise_generate/exercise_dataset_train.jsonl"
OUTPUT_BASE_DIR = "/data2/jwllm/model_process/exercise_models"
SFT_SCRIPT = "./sft.py"
PRUNE_SCRIPT = "./pruning.py"
NUM_ROUNDS = 4

def main():
    base_model_name = os.path.basename(BASE_MODEL.rstrip("/"))
    current_model_path = BASE_MODEL

    for round_idx in range(1, NUM_ROUNDS + 1):
        print(f"\n--- Round {round_idx}/{NUM_ROUNDS} ---")
        
        pruned_dir = os.path.join(OUTPUT_BASE_DIR, f"{base_model_name}_pruned_round_{round_idx}_fp16")
        sft_dir = os.path.join(OUTPUT_BASE_DIR, f"{base_model_name}_round_{round_idx}_fp16")
        
        os.makedirs(pruned_dir, exist_ok=True)
        os.makedirs(sft_dir, exist_ok=True)

        # Step 1: Prune the current model
        print(f"[{round_idx}] Pruning -> {pruned_dir}")
        subprocess.run([
            "python", PRUNE_SCRIPT,
            "--model", current_model_path,
            "--pruning_ratio", "0.2",
            "--max_seq_len", "1024",
            "--save_model", pruned_dir
        ], check=True)

        # Step 2: Fine-tune the pruned model (SFT)
        print(f"[{round_idx}] Fine-tuning -> {sft_dir}")
        subprocess.run([
            "python", SFT_SCRIPT,
            "--model_path", pruned_dir,
            "--data_path", DATA_PATH,
            "--output_dir", sft_dir
        ], check=True)

        # Update input model path for the next iteration
        current_model_path = sft_dir

if __name__ == "__main__":
    main()