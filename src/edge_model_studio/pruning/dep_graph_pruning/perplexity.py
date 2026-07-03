import argparse
import json

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_jsonl_samples(jsonl_path):
    lines = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            if 'combined' in obj:
                lines.append(obj['combined'])
    print(f"读取到 {len(lines)} 条数据")
    return lines


def eval_ppl_per_sample(model, tokenizer, text_list, seqlen, device):
    model.eval()
    nlls = []
    total_tokens = 0
    num_eval = 0
    print(f"开始进行perplexity测试: ")

    for idx, text in enumerate(text_list):
        enc = tokenizer(text, return_tensors='pt').input_ids.to(device)
        if enc.shape[1] < seqlen:
            continue  # 跳过过短样本
        # 支持一个样本多块
        for start in range(0, enc.shape[1] - seqlen + 1, seqlen):
            inp = enc[:, start:start + seqlen]
            with torch.no_grad():
                lm_logits = model(inp).logits
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = inp[:, 1:]
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            nlls.append(loss.float() * (seqlen - 1))
            total_tokens += (seqlen - 1)
            num_eval += 1
            # 每处理50个block打印一次
            if num_eval % 50 == 0:
                print(f"[PPL] 已处理样本: {idx + 1}/{len(text_list)}, 当前评测块: {num_eval}, 当前loss: {loss.item():.4f}")
    print(f"所有样本已处理完成, 评测块数: {num_eval}, 总token: {total_tokens}")

    if total_tokens > 0:
        ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
        return ppl.item()
    else:
        print("警告：无有效评测样本，请调小seqlen或确认数据！")
        return float('nan')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='模型目录路径')
    parser.add_argument('--jsonl_path', type=str, required=True, help='jsonl数据路径')
    parser.add_argument('--seqlen', type=int, default=256, help='每块token长度')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("加载模型和tokenizer ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model.to(device)

    print(f"评测模型: {args.model}\n数据: {args.jsonl_path}")
    lines = load_jsonl_samples(args.jsonl_path)
    ppl = eval_ppl_per_sample(model, tokenizer, lines, args.seqlen, device)
    print(f"\n==== 困惑度 Perplexity: {ppl:.3f} ====")


if __name__ == '__main__':
    main()
