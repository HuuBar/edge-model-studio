from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from collections import Counter
from tqdm import tqdm
import torch
import argparse
import os
from pathlib import Path
import sys
import pickle

LINE_PRINT = 10000

def load_data(data_path):
    """加载预处理数据集"""
    ds = load_from_disk(data_path)
    for e in ds:
        print(e)
        break
    return ds

def compute_token_counter(ds, tokenizer, num_lines, counter_file):
    """计算token频率，如果已存在则直接加载"""
    num_tokens = 0

    if not os.path.exists(counter_file):
        token_counter = Counter()
        for i, d in tqdm(enumerate(ds)):
            for con in d['conversations']:
                tokens = tokenizer.encode(con['content'])
                token_counter.update(tokens)
                num_tokens += len(tokens)

            if i % LINE_PRINT == 0:
                print(f"已处理 {i} 行...")
            if i == num_lines:
                break
        try:
            with open(counter_file, "wb") as f:
                pickle.dump(token_counter, f)
        except Exception as e:
            print(f"发生错误：{e}")
    else:
        try:
            with open(counter_file, 'rb') as f:
                token_counter = pickle.load(f)
        except Exception as e:
            print(f"发生错误：{e}")
        num_tokens = len(token_counter)

    return token_counter, num_tokens


def sort_token_ids(token_counter):
    """排序token并返回ID列表"""
    sort_by_freq = sorted(token_counter.items(), key=lambda x: x[1], reverse=True)
    ids, frequencies = zip(*sort_by_freq)
    return list(ids)


def generate_freq_files(ids, tokenizer, full_vocab_size, vocab_size_rates, prefix_path):
    """为每个词表大小生成频率文件"""
    eos_id = tokenizer.encode(tokenizer.special_tokens_map['eos_token'])

    for rate in vocab_size_rates:
        r = int(full_vocab_size * rate / 100)
        if r > len(ids):
            print(f'warning: r({r}) > ids len {len(ids)}, rate = {rate}%')
            continue

        if eos_id not in ids[:r]:
            not_in_ids = len(set(eos_id) - set(ids[:r]))
            print(f"not_in_ids = {not_in_ids}")
            freq_ids = ids[:r - not_in_ids] + eos_id
        else:
            freq_ids = ids[:r]

        print(f'r = {r}, len(freq_ids) = {len(freq_ids)}')
        draft_to_target = torch.tensor(freq_ids, dtype=torch.long) - torch.arange(
            r, dtype=torch.long
        )
        print(f'save d2t_{r}.pt, shape: {draft_to_target.shape}')
        try:
            with open(f'{prefix_path}/d2t_{r}.pt', 'wb') as f:
                torch.save(draft_to_target, f)

            fr_indices = torch.tensor(freq_ids, dtype=torch.int)
            print(f'save freq_{r}.pt, size:', len(freq_ids))
            with open(f'{prefix_path}/freq_{r}.pt', 'wb') as f:
                torch.save(fr_indices, f)
        except Exception as e:
            print(f"发生错误：{e}")

def main(args):
    data_path = args.data_path
    prefix_path = f"./index-{args.lan}-{args.num_lines}/{args.model_name}"
    counter_file = f"{prefix_path}/token_counters.pkl"

    os.makedirs(f'{prefix_path}', exist_ok=True)

    ds = load_data(data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    vs = tokenizer.vocab_size

    token_counter, num_tokens = compute_token_counter(
        ds, tokenizer, args.num_lines, counter_file
    )

    ids = sort_token_ids(token_counter)
    print(f"processed {args.num_lines} items, ids len={len(ids)}")
    print(f"processed {num_tokens} tokens")

    generate_freq_files(ids, tokenizer, vs, args.vocab_size, prefix_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--lan',
        type=str,
        default='full',
        help='Language suffix for data path.'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='/',
        help='The path to the preprocessed data.'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='Qwen3.5-9B',
        help='The name of the model.'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='',
        help='The path to the model.'
    )
    parser.add_argument(
        '--num_lines',
        type=int,
        default=100000,
        help='The number of SlimPajama lines to process.'
    )
    parser.add_argument(
        '--vocab_size',
        nargs='+',
        type=int,
        default=[10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90],
        help='The vocab sizes to process.(%)'
    )

    args = parser.parse_args()
    print(args)
    main(args)
