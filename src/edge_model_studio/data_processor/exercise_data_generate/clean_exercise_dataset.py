import argparse
import json
import random

OUTPUT_PATH = "/data2/jwllm/datasets/exercise_generate/summary"
SOURCE_DATASET = "/data2/jwllm/datasets/exercise_generate/summary/exercise_summary_dataset_total.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Clean and split jsonl data for SFT, RL, Test, and Prune.")
    parser.add_argument('--source_dataset', type=str, default=SOURCE_DATASET, help='Input jsonl file path')
    parser.add_argument('--sft_ratio', type=float, default=0.5, help='Ratio for SFT data (default: 0.5)')
    parser.add_argument('--test_ratio', type=float, default=0.05, help='Ratio for Test data (default: 0.2)')
    parser.add_argument('--rl_ratio', type=float, default=0.2, help='Ratio for RL data (default: 0.2)')
    parser.add_argument('--prune_ratio', type=float, default=0.25, help='Ratio for Prune data (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    return parser.parse_args()


# 将原始的prune数据抽取出来，然后将metrics+summary拼接起来。
def prune_data_test(prune_data):
    prune_result = []
    for unit in prune_data:
        unit_dict = json.loads(unit)
        string = unit_dict["metrics"] + unit_dict["summary"]
        json_unit = {"combined": string}
        prune_result.append(json.dumps(json_unit, ensure_ascii=False))
    return prune_result


def main():
    args = parse_args()
    random.seed(args.seed)

    # 读取所有数据
    with open(args.source_dataset, 'r', encoding='utf-8') as fin:
        data = [line.strip() for line in fin if line.strip()]
    total = len(data)

    # 归一化比例
    sum_ratio = args.sft_ratio + args.rl_ratio + args.test_ratio + args.prune_ratio
    sft_ratio = args.sft_ratio / sum_ratio
    test_ratio = args.test_ratio / sum_ratio
    rl_ratio = args.rl_ratio / sum_ratio
    prune_ratio = args.prune_ratio / sum_ratio

    # 打乱数据
    random.shuffle(data)
    sft_end = int(total * sft_ratio)
    test_end = sft_end + int(total * test_ratio)
    rl_end = test_end + int(total * rl_ratio)
    prune_end = rl_end + int(total * prune_ratio)

    sft_data = data[:sft_end]
    test_data = data[sft_end:test_end]
    rl_data = data[test_end:rl_end]
    prune_data = data[rl_end:]

    # prune_data需要单独进行一次处理
    prune_data = prune_data_test(prune_data)

    # 输出文件路径
    output_files = {
        'sft': f'{OUTPUT_PATH}/exercise_summary_dataset_sft.jsonl',
        'test': f'{OUTPUT_PATH}/exercise_summary_dataset_test.jsonl',
        'rl': f'{OUTPUT_PATH}/exercise_summary_dataset_rl.jsonl',
        'prune': f'{OUTPUT_PATH}/exercise_summary_dataset_prune.jsonl',
    }
    output_data = {
        'sft': sft_data,
        'test': test_data,
        'rl': rl_data,
        'prune': prune_data,
    }

    for k in output_files:
        with open(output_files[k], 'w', encoding='utf-8') as fout:
            for line in output_data[k]:
                fout.write(line + '\n')
        print(f"{k.upper()} 数据写入: {output_files[k]}, 条数: {len(output_data[k])}")


if __name__ == "__main__":
    main()
