import os
import sys
import random
import argparse
from importlib.metadata import version

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch_pruning as tp

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# ---------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------

def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    print("Loading WikiText-2 dataset...")
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

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
    print("Loading C4 dataset...")
    traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

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

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    return trainloader, TokenizerWrapper(valenc)

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    raise ValueError(f"Unsupported dataset: {name}")

# ---------------------------------------------------------
# Evaluation & Model Utilities
# ---------------------------------------------------------

def eval_ppl(model, tokenizer, dataset="wikitext2", device=torch.device("cuda:0")):
    print(f"Evaluating perplexity on {dataset}...")
    _, testloader = get_loaders(dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer)

    with torch.no_grad():
        testenc = testloader.input_ids
        nsamples = testenc.numel() // model.seqlen
        nlls = []
        
        for i in range(nsamples):
            if i % 10 == 0 and i > 0:
                print(f"Processed {i}/{nsamples} samples")

            inputs = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(device)
            inputs = inputs.reshape(1, model.seqlen)

            lm_logits = model(inputs).logits
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            nlls.append(loss.float() * model.seqlen)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
        torch.cuda.empty_cache()
        
    return ppl.item()

def get_llm(model_name, max_seq_len=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.seqlen = min(max_seq_len, model.config.max_position_embeddings) if max_seq_len else model.config.max_position_embeddings // 2 
    return model

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples')
    parser.add_argument('--pruning_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model')
    parser.add_argument("--max_seq_len", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Environment: torch={version('torch')}, transformers={version('transformers')}, gpus={torch.cuda.device_count()}")
    print(f"Loading model: {args.model}")
    
    model = get_llm(args.model, max_seq_len=args.max_seq_len)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    
    device = model.hf_device_map.get("lm_head", torch.device("cuda:0"))
    print(f"Primary compute device: {device}")

    # Pruning setup
    print("Preparing structured pruning...")
    tp.utils.print_tool.before_pruning(model)
    
    dummy_text = "Hello world."
    dummy_inputs = torch.tensor(tokenizer.encode(dummy_text)).unsqueeze(0).to(device)
    
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
        elif name.endswith('mlp'):
            if hasattr(m, "gate_up_proj"):
                out_channel_groups[m.gate_up_proj] = 2

    _is_gqa = model.config.num_attention_heads != model.config.num_key_value_heads
    importance = tp.importance.GroupMagnitudeImportance(p=2, group_reduction='mean') 

    pruner = tp.pruner.BasePruner(
        model,
        example_inputs=dummy_inputs,
        importance=importance,
        global_pruning=False,
        output_transform=lambda x: x.logits,
        pruning_ratio=args.pruning_ratio,
        ignored_layers=[model.lm_head],
        num_heads=num_heads,
        prune_num_heads=True,
        prune_head_dims=False, 
        head_pruning_ratio=args.pruning_ratio,
        out_channel_groups=out_channel_groups,
        round_to=4,
    )

    print(f"Starting pruning iterations (Target ratio: {args.pruning_ratio})...")
    for g in pruner.step(interactive=True):
        g.prune()
    print("Pruning completed.")

    # Update model config
    print("Synchronizing model config...")
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

    if not _is_gqa:
        model.config.num_key_value_heads = model.config.num_attention_heads

    tp.utils.print_tool.after_pruning(model, do_print=True)
    del pruner
    torch.cuda.empty_cache()
    
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Post-pruning parameters: {num_params / 1e9:.2f} B")
    
    ppl_test = eval_ppl(model, tokenizer, dataset="wikitext2", device=device)
    print(f"WikiText Perplexity: {ppl_test:.4f}")

    if args.save_model:
        print("Converting model to FP16 for saving...")
        model = model.half() 
        print(f"Saving model to: {args.save_model}")
        model.save_pretrained(args.save_model, safe_serialization=False) 
        tokenizer.save_pretrained(args.save_model)
        print("Save completed.")

if __name__ == '__main__':
    main()