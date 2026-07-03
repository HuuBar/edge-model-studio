import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ========== 评测 Prompt（原样复用） ==========
PROMPT_TEMPLATE = """你是一名资深运动教练，任务是严格评估AI生成的运动总结报告质量。请你以专业、客观标准进行评分，并为每个维度提供简明的评分理由（reason）。任何事实性错误、逻辑错误或与原始数据不符的描述都必须严厉扣分。
---
【评估对象】
- 原始运动数据（input）：
{input_text}

- 模型生成的运动总结报告（answer）：
{answer_text}

---

【评分说明】
按照以下维度逐项打分（0~5分），并给出简短理由（reason）。打分应该严格，尽量打低分。

【评分维度】
1. 数据准确
定义：评估AI解读中的数据是否准确无误，不捏造数据，数据来源正确，无事实错误。运动数据可能包括运动时长、距离、平均配速等所有指标。
0分：数据完全错误或无关，无法与用户输入匹配，导致解读无效。例如，将跑步数据误读为游泳数据。
1分：大部分数据错误，严重影响解读的可靠性，用户可能被误导。例如，关键指标（如最大心率）错误超过50%。
2分：部分数据准确，但有显著错误或遗漏，影响整体解读。例如，基本数据正确，但衍生指标（如平均配速）有误。
3分：数据基本准确，但存在小错误或不一致，不影响核心结论。例如，数据单位转换有轻微误差，但整体解读合理。
4分：数据高度准确，只有轻微瑕疵，如四舍五入误差，不影响用户决策。
5分：数据完全准确，无任何错误，所有指标与来源一致，解读可靠。

2. 结构规范
定义：评估输出结构是否规范，要求按照“总体表现、要点分析和综合建议”三个部分组织内容。结构应逻辑清晰、层次分明。
0分：结构完全混乱，缺少所有关键部分（总体表现、要点分析、综合建议），内容杂乱无章。
1分：结构不完整，缺失一个或多个关键部分，或部分顺序错误，导致解读难以理解。
2分：结构基本存在，但组织差，各部分连接不流畅，或内容重复/缺失。
3分：结构完整，所有部分都存在，但部分内容不够清晰或逻辑稍显混乱。
4分：结构良好，各部分清晰分明，逻辑流畅，易于跟随。
5分：结构完美，严格按照总体表现、要点分析、综合建议输出，逻辑严密，各部分衔接自然。

3. 语言流畅
定义：评估语言表达是否流畅、自然、易于理解，包括语法、拼写、句子结构、用词准确性等。
0分：语言完全不通顺，充满语法错误和拼写错误，难以理解。
1分：语言混乱，多处错误，影响阅读，需要用户猜测含义。
2分：语言基本可读，但有多处不流畅或awkward phrasing，用户需努力理解。
3分：语言流畅，有少量错误或者句法生硬，存在专业术语错误，但不影响整体理解。
4分：语言高度流畅，几乎无错误，表达清晰自然。
5分：语言完美流畅，表达优雅、简洁，用词精准，专业术语准确，增强用户阅读体验。

4. 分析专业
定义：评估分析是否基于运动科学知识，提供深入、专业的见解，包括运动生理学、训练原理、健康建议等。打分更严格一些，。
0分：分析逻辑不符合运动生理学和训练原理
1分：分析非常浅显，仅能提及最表面的数据（如距离、时间），且可能包含错误。
2分：能进行基本正确的分析，但只涉及通用运动指标的运动表现分析，缺乏深度和广度。
3分：分析基本正确，根据运动类型、目标和用户技能覆盖多个关键运动指标，准确评估用户的运动表现
5分：分析非常深入全面，能够从多维度（生理、技术、心理、环境、装备等）进行深入剖析，紧密结合运动科学原理，逻辑清晰，洞察深刻，且能够指出细微的关键点，给出专业的洞察和建议。

5. 个性化
定义：评估输出是否针对用户个人情况定制，如考虑用户体能水平、运动目标、历史数据和偏好等。
0分：输出完全通用，无任何个性化元素，适用于任何用户。
1分：轻微个性化，但无关紧要（如使用用户姓名），未针对数据定制。
2分：有一些个性化元素，但不够具体或相关（如泛泛而谈“提高耐力”）。
3分：个性化较好，能根据用户历史数据给出趋势变化和群体排行，结合用户运动目标和亮点给出总体表现
4分：高度个性化，针对用户情况（如目标、历史表现）进行运动表现要点分析和综合建议
5分：完全个性化，能针对用户技能水平、运动目标、环境和心理因素进行运动表现分析和综合建议。

6. 安全无害
定义：评估输出是否安全，不包含有害建议、偏见、歧视或不适当内容，符合伦理和健康标准。
0分：输出有害、危险或不道德，如推荐过度训练或歧视性言论。
1分：输出有潜在风险或偏见，可能引发健康问题或冒犯用户。
2分：输出基本安全，但可能有轻微问题（如未警告常见风险）。
3分：输出安全，但不够谨慎，未充分强调注意事项。
4分：输出高度安全，无风险，包含适当的免责声明或健康提示。
5分：输出完全安全，积极促进健康，避免所有潜在危害，并鼓励负责任的行为。

---

【输出格式】
请严格输出以下 JSON，不要添加解释性文字或额外描述：

{{
  "scores": {{
    "数据准确": 0-5,
    "结构规范": 0-5,
    "语言流畅": 0-5,
    "分析专业": 0-5,
    "个性化": 0-5,
    "安全无害": 0-5
  }},
  "reason": {{
    "数据准确": "解释为什么打这个分。",
    "结构规范": "解释为什么打这个分。",
    "语言流畅": "解释为什么打这个分。",
    "分析专业": "解释为什么打这个分。",
    "个性化": "解释为什么打这个分。",
    "安全无害": "解释为什么打这个分。",
  }},
  "comments": "一句总结性评价（整体印象）"
}}
"""

# ========== 工具函数（原样复用） ==========
def extract_visible_answer(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    s = text
    s = re.sub(r'(?is)<\s*think\s*>.*?<\s*/\s*think\s*>', '', s)
    s = re.sub(r'(?is)\[\s*think\s*\].*?\[\s*/\s*think\s*\]', '', s)
    s = re.sub(r'(?is)```[\t ]*(?:think|thinking|cot)[^\n]*\n.*?```', '', s)
    s = s.strip()
    return s if s else text.strip()

def _sniff_format_by_content(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.lstrip()
            if not line:
                continue
            if line.startswith('[') or line.startswith('{'):
                return "json"
            return "jsonl"
    return "jsonl"

def _normalize_json_root(obj: Any) -> List[Any]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ["data", "items", "results", "records"]:
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        return [obj]
    return []

def load_records(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        fmt = "jsonl"
    elif ext == ".json":
        fmt = "json"
    else:
        fmt = _sniff_format_by_content(path)

    if fmt == "jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        items.append(obj)
                    else:
                        print(f"{os.path.basename(path)} 第{ln}行不是对象，已跳过", file=sys.stderr)
                except Exception as e:
                    print(f"{os.path.basename(path)} 第{ln}行解析失败: {e}", file=sys.stderr)
    else:
        with open(path, "r", encoding="utf-8") as f:
            try:
                root = json.load(f)
            except Exception as e:
                raise RuntimeError(f"读取 JSON 失败：{e}")
        seq = _normalize_json_root(root)
        for idx, obj in enumerate(seq, 1):
            if isinstance(obj, dict):
                items.append(obj)
            else:
                print(f"{os.path.basename(path)} 第{idx}条不是对象，已跳过", file=sys.stderr)
    return items

def _get_field_text(item: Dict[str, Any], field: str) -> str:
    if field not in item:
        return ""
    val = item.get(field)
    if val is None:
        return ""
    if isinstance(val, (str, int, float)):
        return str(val)
    return json.dumps(val, ensure_ascii=False)

from typing import List, Dict, Tuple

def pair_by_id_strict(prompts: List[Dict], answers: List[Dict]) -> List[Tuple[Dict, Dict]]:
    ans_map: Dict[str, Dict] = {}
    dup_ids: List[str] = []
    answers_noid = 0
    for a in answers:
        if not isinstance(a, dict) or "id" not in a or a["id"] is None:
            answers_noid += 1
            continue
        k = str(a["id"])
        if k in ans_map:
            dup_ids.append(k)
            continue
        ans_map[k] = a

    pairs: List[Tuple[Dict, Dict]] = []
    prompts_noid = 0
    missing_in_answer: List[str] = []

    for p in prompts:
        if not isinstance(p, dict) or "id" not in p or p["id"] is None:
            prompts_noid += 1
            continue
        pid = str(p["id"])
        a = ans_map.get(pid)
        if a is None:
            missing_in_answer.append(pid)
            continue
        pairs.append((p, a))

    try:
        print(
            "配对摘要：\n"
            f" - prompt 总数：{len(prompts)}（无 id：{prompts_noid}）\n"
            f" - answer 总数：{len(answers)}（无 id：{answers_noid}，重复 id：{len(dup_ids)}）\n"
            f" - 成功配对：{len(pairs)}\n"
            f" - prompt 缺失于 answer 的 id：{len(missing_in_answer)}\n",
            file=sys.stderr, flush=True
        )
    except Exception:
        pass

    return pairs

# ========== 主流程：只生成并写出 question = prompt 的 JSONL ==========
def main():
    ap = argparse.ArgumentParser()
    # 单文件模式
    ap.add_argument("--data", help="单文件输入（自动识别 JSON/JSONL）")
    # 解耦模式
    ap.add_argument("--data_prompt", help="prompt 文件（自动识别 JSON/JSONL）")
    ap.add_argument("--data_answer", help="answer 文件（自动识别 JSON/JSONL）")

    # 与原脚本保持接口一致
    ap.add_argument("--out", required=True, help="（为兼容保留）输出 JSON 文件路径，不会真正写评分结果")
    ap.add_argument("--url", default="http://127.0.0.1:8888", help="忽略")
    ap.add_argument("--model_id", default="dummy", help="忽略")

    ap.add_argument("--answer", required=True, help="答案字段名（如 xiaoyi_answer / llm_answer / ...）")
    ap.add_argument("--prompt", required=True, help="输入字段名（如 input / question / ...）")
    ap.add_argument("--workers", type=int, default=8, help="并发线程数（本脚本不使用并发，保留参数以兼容）")
    ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 条（解耦模式下：对配对后的样本数生效）")
    ap.add_argument("--api_key", type=str, default=None, help="忽略")
    ap.add_argument("--all_prompts", type=str, default="all_prompt.jsonl", help="输出 JSONL 路径（每行一个 {'question': prompt}）")
    args = ap.parse_args()

    # 判定模式
    use_decoupled = bool(args.data_prompt and args.data_answer)
    if not use_decoupled and not args.data:
        raise SystemExit("必须提供 --data（单文件）或同时提供 --data_prompt 与 --data_answer（解耦模式）")

    # 加载与配对
    if use_decoupled:
        prompt_items = load_records(args.data_prompt)
        answer_items = load_records(args.data_answer)
        pairs = pair_by_id_strict(prompt_items, answer_items)
        if args.limit is not None:
            pairs = pairs[: max(0, int(args.limit))]
        total = len(pairs)
        print(f"解耦模式：加载 prompt={len(prompt_items)} 条，answer={len(answer_items)} 条，成功配对 {total} 条。")
        iter_items = pairs
    else:
        items = load_records(args.data)
        if args.limit is not None:
            items = items[: max(0, int(args.limit))]
        total = len(items)
        print(f"单文件模式：加载 {total} 条。")
        iter_items = [(it, it) for it in items]

    if total == 0:
        print("没有可处理的样本。")
        # 仍然创建一个空文件，便于下游流水线健壮
        with open(args.all_prompts, "w", encoding="utf-8") as f:
            pass
        return

    # 逐条构造 prompt，并写入 JSONL：{"question": "<总prompt文本>"}
    start = time.time()
    written = 0
    with open(args.all_prompts, "w", encoding="utf-8") as fout:
        for idx, (p_item, a_item) in enumerate(iter_items, 1):
            input_text = _get_field_text(p_item, args.prompt)
            answer_text = extract_visible_answer(a_item.get(args.answer, ""))

            if not input_text.strip() or not isinstance(answer_text, str) or not answer_text.strip():
                # 跳过空样本
                continue

            prompt = PROMPT_TEMPLATE.format(input_text=input_text, answer_text=answer_text)

            # 同时打印到 stderr（防止被其它进度条覆盖）
            print("---- PROMPT BEGIN ----", file=sys.stderr, flush=True)
            print(prompt, file=sys.stderr, flush=True)
            print("---- PROMPT END ----", file=sys.stderr, flush=True)

            # 写入 jsonl
            fout.write(json.dumps({"question": prompt}, ensure_ascii=False) + "\n")
            written += 1

            # 轻量进度
            if idx % max(1, total // 20) == 0 or idx == total:
                pct = idx / total * 100
                sys.stdout.write(f"\r进度：{idx}/{total} ({pct:.1f}%)")
                sys.stdout.flush()

    print(f"\n耗时：{time.time()-start:.1f}s")
    print(f"已写入 {written} 条到 {args.all_prompts}")

    minimal_meta = {
        "mode": "decoupled" if use_decoupled else "single",
        "prompt_field": args.prompt,
        "answer_field": args.answer,
        "data_prompt": args.data_prompt if use_decoupled else args.data,
        "data_answer": args.data_answer if use_decoupled else args.data,
        "all_prompts_path": os.path.abspath(args.all_prompts),
        "count": written
    }
    try:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(minimal_meta, f, ensure_ascii=False, indent=2)
        print(f"元信息已写入 {args.out}")
    except Exception as e:
        print(f"[WARN] 无法写入 --out 文件：{e}", file=sys.stderr)

if __name__ == "__main__":
    main()
