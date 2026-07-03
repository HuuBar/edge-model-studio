import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
import torch
import torch.nn as nn
import fnmatch
import numpy as np
from datasets import load_dataset
import json
import random


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def get_exercise_dataset_jsonl(jsonl_path, nsamples, seed, seqlen, tokenizer):
    lines = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            # print(f"line is :{line}")
            obj = json.loads(line)
            if 'combined' in obj:
                lines.append(obj['combined'])
    print(f"加载到 {len(lines)} 条样本")

    random.seed(seed)
    random.shuffle(lines)
    n_total = len(lines)
    n_train = int(n_total * 0.9)
    train_lines = lines[:n_train]
    valid_lines = lines[n_train:]

    train_text = "\n\n".join(train_lines)
    train_enc = tokenizer(train_text, return_tensors='pt')
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, train_enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = train_enc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valid_enc_list = []
    for text in valid_lines:
        enc = tokenizer(text, return_tensors='pt').input_ids
        valid_enc_list.append(enc)

    token_lengths = [enc.shape[1] for enc in valid_enc_list]
    print(f"验证集样本条数: {len(valid_enc_list)}")
    print(
        f"验证集每条token长度，最小: {min(token_lengths)}, 最大: {max(token_lengths)}, 均值: {np.mean(token_lengths):.2f}, 中位数: {np.median(token_lengths):.2f}")
    print(f"小于 seqlen={seqlen} 的验证样本数: {sum(l < seqlen for l in token_lengths)}")

    print(f"训练集文本长度: {train_enc.input_ids.shape[1]}，验证集样本条数: {len(valid_enc_list)}")
    print(f"训练样本数(用作模型剪枝校准的采样批数): {len(trainloader)}，每条验证样本平均长度: {np.mean([e.shape[1] for e in valid_enc_list]):.1f}")
    return trainloader, valid_enc_list


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
                             split='train')
    valdata = load_dataset('allenai/c4', 'allenai--c4',
                           data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None, jsonl_path=None):
    if 'exercise' in name:
        return get_exercise_dataset_jsonl(jsonl_path, nsamples, seed, seqlen, tokenizer)
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)


def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0")):
    dataset = "exercise"
    print(f"evaluating on {dataset}")
    trainloader, valid_enc_list = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer, jsonl_path=args.jsonl_path
    )
    with torch.no_grad():
        ppl_test = eval_ppl_per_sample(model, valid_enc_list, model.seqlen, device)
    return ppl_test


def eval_ppl_per_sample(model, valid_enc_list, seqlen, device):
    nlls = []
    total_tokens = 0
    num_eval = 0
    for enc in valid_enc_list:
        if enc.shape[1] < seqlen:  # 跳过过短的
            continue
        enc = enc.to(device)
        for start in range(0, enc.shape[1] - seqlen, seqlen):
            inp = enc[:, start:start + seqlen]
            lm_logits = model(inp).logits
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inp[:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            nlls.append(loss.float() * (seqlen - 1))
            total_tokens += (seqlen - 1)
            num_eval += 1
    print(f"有效PPL评测块数: {num_eval}，总token: {total_tokens}")
    if total_tokens > 0:
        ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
        return ppl.item()
    else:
        print("警告：无有效评测样本，请调小seqlen或确认数据！")
        return float('nan')


def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
    # Get input IDs
    # testenc = testenc.input_ids

    # Calculate number of samples
    # nsamples = testenc.numel() // model.seqlen
    nsamples = len(trainloader)

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i + bs, nsamples)

        # Prepare inputs and move to device
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j - i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j - i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()


def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i + bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j - i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j - i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()


def eval_zero_shot(model_name, model, tokenizer,
                   task_list=["boolq", "rte", "hellaswag", "winogrande", "arc_challenge", "arc_easy", "openbookqa"],
                   num_fewshot=0, use_accelerate=False, add_special_tokens=False):
    from lm_eval import tasks, evaluator
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)

    task_names = pattern_match(task_list, tasks.ALL_TASKS)
    model_args = f"pretrained={model_name}, cache_dir=./cache"
    limit = None
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained={model_name}, cache_dir=./cache, use_accelerate=True"
    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=None,
        no_cache=True,
        limit=limit,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        pretrained_model=model,
        tokenizer=tokenizer,
        add_special_tokens=add_special_tokens
    )

    return results


print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def get_llm(model_name, max_seq_len=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.seqlen = min(max_seq_len,
                       model.config.max_position_embeddings) if max_seq_len is not None else model.config.max_position_embeddings // 2
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--pruning_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument("--eval_zero_shot", action="store_true")
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument('--jsonl_path', type=str, default=None, help='自定义jsonl数据路径')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, max_seq_len=args.max_seq_len)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model:
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    import torch_pruning as tp
    tp.utils.print_tool.before_pruning(model)
    text = "Hello world."
    inputs = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(model.device)
    num_heads = {}
    out_channel_groups = {}
    seperate_qkv = False
    for name, m in model.named_modules():
        if name.endswith("self_attn"):
            if hasattr(m, "q_proj"):
                seperate_qkv = True
                num_heads[m.q_proj] = model.config.num_attention_heads
                num_heads[m.k_proj] = model.config.num_key_value_heads
                num_heads[m.v_proj] = model.config.num_key_value_heads
            elif hasattr(m, "qkv_proj"):
                seperate_qkv = False
                num_heads[m.qkv_proj] = model.config.num_attention_heads
        if name.endswith('mlp'):
            if hasattr(m, "gate_up_proj"):
                out_channel_groups[m.gate_up_proj] = 2

    _is_gqa = model.config.num_attention_heads != model.config.num_key_value_heads
    head_pruning_ratio = args.pruning_ratio
    hidden_size_pruning_ratio = args.pruning_ratio
    importance = tp.importance.GroupMagnitudeImportance(p=2,
                                                        group_reduction='mean')  # tp.importance.ActivationImportance(p=2, target_types=[torch.nn.Linear])
    pruner = tp.pruner.BasePruner(
        model,
        example_inputs=inputs,
        importance=importance,
        global_pruning=False,
        output_transform=lambda x: x.logits,
        pruning_ratio=hidden_size_pruning_ratio,
        ignored_layers=[model.lm_head],
        num_heads=num_heads,
        prune_num_heads=True,
        prune_head_dims=False,  # we do not prune head dims so that we don't need to prune the ROPE
        head_pruning_ratio=head_pruning_ratio,
        out_channel_groups=out_channel_groups,
        round_to=4,
    )

    for g in pruner.step(interactive=True):
        g.prune()

    model.config.hidden_size = model.lm_head.in_features
    for name, m in model.named_modules():
        if name.endswith("self_attn"):
            if seperate_qkv:
                m.hidden_size = m.q_proj.out_features
            else:
                m.hidden_size = m.qkv_proj.out_features // 3
            m.num_heads = m.hidden_size // m.head_dim
            model.config.num_attention_heads = m.num_heads
            if not _is_gqa:
                m.num_key_value_heads = m.num_heads
                model.config.num_key_value_heads = m.num_heads
            if hasattr(m, "num_key_value_groups"):
                m.num_key_value_groups = m.num_heads // model.config.num_key_value_heads

        elif name.endswith("mlp"):
            if hasattr(m, "gate_proj"):
                m.hidden_size = m.gate_proj.in_features
                model.config.intermediate_size = m.gate_proj.out_features
            elif hasattr(m, "gate_up_proj"):
                m.hidden_size = m.gate_up_proj.in_features
                model.config.intermediate_size = m.gate_up_proj.out_features // 2
            else:
                raise ValueError("Unknown mlp layer")

    if not _is_gqa:
        model.config.num_key_value_heads = model.config.num_attention_heads
    tp.utils.print_tool.after_pruning(model, do_print=True)
    print(model.config)

    del pruner
    torch.cuda.empty_cache()
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"num_params {num_params}")
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"perplexity is: {ppl_test}")

    if args.save_model:
        print("🔁 将模型转换为 BF16...")
        model = model.to(torch.bfloat16)

        print("保存模型为 BF16 格式，按照model.safetensors保存...")
        model.save_pretrained(args.save_model, safe_serialization=True)
        tokenizer.save_pretrained(args.save_model)

        print(f"模型已保存至 {args.save_model}，并已强制使用 BF16 格式")


if __name__ == '__main__':
    main()
