import json
import logging
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import jieba

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score

# 保持一贯的硬朗规范日志，移除无意义的符号
logging.basicConfig(
    level=logging.INFO, 
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# 禁用 jieba 的默认琐碎输出，保持日志纯净
jieba.logger.setLevel(logging.ERROR)


def load_and_index_json(path: Path, data_key: str, id_key: str = "id") -> dict:
    """
    流式读取并基于唯一 ID 构建哈希索引。
    彻底杜绝因样本缺失、乱序导致的对齐灾难。
    """
    indexed_data = {}
    if not path.exists():
        logging.error(f"Target path not found: {path}")
        return indexed_data

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for idx, entry in enumerate(data):
                if data_key not in entry:
                    continue
                # 如果明文里没有定义唯一 ID，降级使用物理行索引作为 Key 兜底
                uid = entry.get(id_key, f"line_{idx}")
                indexed_data[uid] = entry[data_key]
    except Exception as e:
        logging.error(f"Failed to read or parse JSON file {path}: {e}")
    return indexed_data


def compute_bleu_optimized(preds: list, refs_tokenized: list) -> float:
    """
    高性能 BLEU 测算。传入预先切好词的参考端，使 jieba 开销减半。
    """
    smoothie = SmoothingFunction().method4
    scores = []
    
    # preds 端无法逃避，必须切词，但统一转换为内存更紧凑的 tuple/list 结构
    for pred, ref_tokens in zip(preds, refs_tokenized):
        pred_tokens = list(jieba.cut(pred))
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
        scores.append(score)
        
    return float(np.mean(scores)) if scores else 0.0


def compute_rouge(preds: list, refs: list) -> dict:
    """
    利用预分配加速的多级指标累加器。
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    agg = defaultdict(lambda: defaultdict(float))
    
    for pred, ref in zip(preds, refs):
        scores = scorer.score(ref, pred)
        for k, metric in scores.items():
            agg[k]['f'] += metric.fmeasure
            agg[k]['p'] += metric.precision
            agg[k]['r'] += metric.recall
            
    n = len(preds)
    return {k: {m: val / n for m, val in metrics.items()} for k, metrics in agg.items()}


def compute_bertscore_batched(preds: list, refs: list, model_path: str, lang: str = "zh") -> dict:
    """
    并行批处理 BERTScore。指定 GPU 驱动与 Batch_size 提升矩阵吞吐效率。
    """
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    is_local = "/" in model_path or model_path.startswith(".")
    kwargs = {
        "lang": lang,
        "model_type": model_path,
        "verbose": False,
        "rescale_with_baseline": False,
        "device": device,
        "batch_size": 64  # 显式控制批处理大小，充分压榨 GPU VRAM
    }
    if is_local:
        kwargs["num_layers"] = 12
        
    p, r, f1 = bert_score(preds, refs, **kwargs)
    return {
        "f": float(f1.mean().item()),
        "p": float(p.mean().item()),
        "r": float(r.mean().item())
    }


def main():
    # 路径解耦与收拢
    pred_dir = Path("/home/shangyangyang/project/Dataset/compare_qwen0.6b_base_20250529")
    gt_path = Path("/home/shangyangyang/project/Dataset/compare_for_data_19819/Qwen3-0.6B-Base-full_v2_summary_19000_19819.json")
    output_csv = Path("./eval_results_summary.csv")
    bert_model_path = "/home/shangyangyang/bert_model/bert-base-chinese"

    # 1. 强安全对齐加载 Ground Truth
    logging.info("Indexing Ground Truth target entries...")
    gt_map = load_and_index_json(gt_path, data_key='ground_truth', id_key='id')
    
    if not gt_map:
        logging.critical("Ground Truth directory load empty or crashed. Process terminated.")
        return

    # 2. 核心性能提速点：在全局对 Ground Truth 进行【预分词】（一次性切词，后续百个模型直接共享复用）
    logging.info(f"Performance Optimization: Pre-tokenizing {len(gt_map)} Ground Truth elements...")
    gt_tokens_map = {uid: list(jieba.cut(text)) for uid, text in gt_map.items()}

    records = []

    # 3. 遍历模型推理目录
    for pred_path in sorted(pred_dir.glob("*.json")):
        model_name = pred_path.stem.split('_summary')[0]
        logging.info(f"Synchronizing target verification entries for model: {model_name}")
        
        # 加载当前待测模型的预测数据
        pred_map = load_and_index_json(pred_path, data_key='llm_answer', id_key='id')
        
        # 基于具有唯一约束的哈希 ID 集合求交集，确保空间几何对齐的绝对安全
        aligned_ids = [uid for uid in pred_map if uid in gt_map]
        
        if len(aligned_ids) != len(gt_map):
            logging.warning(
                f"[{pred_path.name}] Alignment Deviation! "
                f"Aligned: {len(aligned_ids)} / GT Total: {len(gt_map)}. "
                f"Proceeding only with strictly matched intersection."
            )

        if not aligned_ids:
            logging.error(f"No intersected IDs found for {pred_path.name}. Skipping evaluation.")
            continue

        # 按照统一对齐后的 ID 拓扑顺序重组平面列表
        preds_ordered = [pred_map[uid] for uid in aligned_ids]
        refs_ordered = [gt_map[uid] for uid in aligned_ids]
        refs_tokens_ordered = [gt_tokens_map[uid] for uid in aligned_ids]

        # 4. 触发高维离线数学统计
        logging.info(f"Computing BLEU metric (Optimized path)...")
        bleu = compute_bleu_optimized(preds_ordered, refs_tokens_ordered)
        
        logging.info(f"Computing ROUGE-1/2/L precision grid...")
        rouge = compute_rouge(preds_ordered, refs_ordered)
        
        logging.info(f"Computing BERTScore via neural embeddings (Device execution)...")
        bert = compute_bertscore_batched(preds_ordered, refs_ordered, bert_model_path)

        records.append({
            "model": model_name,
            "BLEU": round(bleu, 4),
            "ROUGE-1 F": round(rouge["rouge1"]["f"], 4),
            "ROUGE-1 P": round(rouge["rouge1"]["p"], 4),
            "ROUGE-1 R": round(rouge["rouge1"]["r"], 4),
            "ROUGE-2 F": round(rouge["rouge2"]["f"], 4),
            "ROUGE-2 P": round(rouge["rouge2"]["p"], 4),
            "ROUGE-2 R": round(rouge["rouge2"]["r"], 4),
            "ROUGE-L F": round(rouge["rougeL"]["f"], 4),
            "ROUGE-L P": round(rouge["rougeL"]["p"], 4),
            "ROUGE-L R": round(rouge["rougeL"]["r"], 4),
            "BERTScore F1": round(bert["f"], 4),
            "BERTScore P": round(bert["p"], 4),
            "BERTScore R": round(bert["r"], 4),
        })

    if not records:
        logging.warning("Pipeline termination: No static evaluation row committed.")
        return

    # 5. 高内聚追加写入结果
    df = pd.DataFrame(records)
    write_header = not output_csv.exists()
    
    try:
        df.to_csv(
            output_csv,
            mode='a',
            header=write_header,
            index=False,
            encoding='utf-8-sig'
        )
        logging.info(f"[SUCCESS] Consolidated report updated safely -> {output_csv}")
    except Exception as e:
        logging.error(f"Failed to append result records to storage matrix: {e}")


if __name__ == "__main__":
    main()