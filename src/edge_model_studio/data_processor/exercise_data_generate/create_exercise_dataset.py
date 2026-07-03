import json
import random
import re
from datetime import datetime, timedelta

import requests
from transformers import AutoTokenizer

# 训练10000(50%)  测试1000(5%)  RL4000(20%)  prune5000(25%)
# TOTAL_CASES = 20000
TOTAL_CASES = 100000

model_id = "/data2/jwllm/models_origin/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
BATCH_SIZE = 128  # 批量推理数值
api_url = "http://0.0.0.0:8088/v1/completions"  # 根据你的实际情况修改
OUTPUT_DATASET = "/data2/jwllm/datasets/exercise_generate/pretrain/exercise_summary_dataset_total.jsonl"

# 使用的最大跑步相关的参数
METRICS_POOL = [
    "最大摄氧量", "心率变异性", "静息心率", "无氧阈值", "有氧适能",
    "步频", "步幅", "乳酸阈值", "呼吸率", "训练负荷", "恢复时间", "触地时间",
    "左右平衡", "垂直振幅", "跑力指数", "能量消耗", "血氧饱和度", "肌肉含量",
    "体脂率", "最大心率", "静息血压", "基础代谢率", "运动经济性", "步态稳定性", "跑步姿态得分",
    "乳酸积累速度", "训练压力分数", "最大力量输出", "踝关节活动度", "配速稳定性", "步幅变异系数",
    "能量转化效率", "跑步动力学指标", "恢复心率", "睡眠质量评分", "运动动机指数", "最大爆发速度"
]


# 按照年月日生成一个随机日期
def generate_random_date(start_year=2020, end_year=2024):
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    delta_days = (end_date - start_date).days
    random_days = random.randint(0, delta_days)
    return (start_date + timedelta(days=random_days)).strftime("%Y年%m月%d日")


# 构建一个
def build_analysis_prompt(date_str, metrics):
    metric_list = "\n".join([f"（{i + 1}）{m}" for i, m in enumerate(metrics)])
    messages = [
        {"role": "system", "content": f"""你是一个专业的跑步教练，请严格按照要求生成跑步指标解析。
        基于以下三个专业指标{metric_list}，在「{date_str}」当天的训练数据，仿照以上格式，填写这三个指标的内容，
        不得重复上面示例中的数值和结论，只能使用中文，不要添加其他文字。每个指标需要包括事实，数据解析和改善建议。"""},
        {"role": "user", "content": f"""
指标解析示例如下：
#指标解析
(1)最大摄氧量（VO₂max）：
1.1 事实：{date_str} VO₂max 为 [46.2 ml/kg/min]
1.2 数据解读：处于优秀水平，反映有良好的有氧耐力
1.3 改善建议：可继续进行长距离有氧训练维持优势

(2)步频：
1.1 事实：{date_str} 平均步频为 [172 步/分钟]
1.2 数据解读：略低于高效区间，可能影响跑步经济性
1.3 改善建议：尝试使用节奏训练提升步频到 180 左右

(3)HRV（心率变异性）：
1.1 事实：{date_str} HRV 值为 [68 ms]
1.2 数据解读：属于中等偏上，说明身体恢复情况较好
1.3 改善建议：保持充足睡眠与均衡饮食以培固恢复质量
"""}
    ]
    return apply_qwen_template_inference(messages, enable_thinking=False)


# 基于运动健康的prompt和metrics生成对应的prompt用于生成summary
def build_exercise_summary_prompt(dataset):
    messages = [
        {"role": "system", "content": f"{dataset['prompt']}"},
        {"role": "user", "content": f"{dataset['metrics']}"}
    ]
    return apply_qwen_template_inference(messages, enable_thinking=False)


# 根据特定的格式，生成符合Qwen的prompt模版
# tokenize: True返回分词后的数据列表; False只输出拼接后的人类可读的字符串;
# add_generation_prompt: 自动在最后拼接assistant开头，方便模型生成
# messages是[{"role": ..., "content": ...},{}]，可支持多轮对话
def apply_qwen_template_inference(messages, enable_thinking=False):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )


# 获取最终的答案
def extract_final_answer(text: str) -> str:
    text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
    patterns = [r"#指标解析.*", r"指标解析.*", r"#分析结果.*", r"分析结果.*"]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(0).strip()
    return text.strip()


# 处理Qwen3的特殊输出格式，取出真正需要的东西
def parse_qwen3_output(text):
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip(), text[think_match.end():].strip()
    thinking = ""
    answer = text.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
    answer = re.sub(r"^(assistant|user|system):?", "", answer, flags=re.IGNORECASE).strip()
    return thinking, answer


# 批量进行推理
def perform_batch_inference(prompts_temp, batch_size):
    results = []
    for i in range(0, len(prompts_temp), batch_size):
        print(f"正在进行推理，进度是 {i} / {len(prompts_temp)}, 当前时间为: {datetime.now()}")

        batch = prompts_temp[i:i + batch_size]
        payload = {
            "model": "qwen3-8B-jwllm",
            "prompt": batch,
            "max_tokens": 1024,
            "temperature": 0.9,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.15
        }

        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            response_data = response.json()
            # print(f"the response datasets is : {response_data}")
            for choice in response_data["choices"]:
                response_text = choice["text"].strip()
                _, content = parse_qwen3_output(response_text)
                content = extract_final_answer(content)
                results.append(content)
        else:
            print(f"请求失败，状态码: {response.status_code}，响应内容: {response.text}")
    return results


# 生成最终的prompt
def generate_prompt_list(prompts_list):
    prompts_temp = []
    for unit in prompts_list:
        str_temp = f"""
        你是一个跑步专家，你能理解跑步指标的解析，并会形成200字以内的跑步报告。
        #任务
        请生成一段回复用户提问{unit["date"]}跑步跑得怎么样？的答复。
        所有数据和定性判断不能有任何修改，需要固括三个指标:{unit["metric_1"]},{unit["metric_2"]},{unit["metric_3"]}
        #限制
        不要幻想，不要偷懒，回复字数限制在200字内，语言一定要精简简洁。
        """
        prompts_temp.append(str_temp)
    print(f"========== 已生成{len(prompts_temp)}个对应的prompt ==========")
    return prompts_temp


# 抽取生成数据的关键词，后续可以用作强化学习训练使用
def extract_keywords(metrics_list):
    keyword_list = []
    for metrics_text in metrics_list:
        keywords = set()
        # 匹配所有日期，格式如 2023年12月17日
        date_pattern = r'\d{4}年\d{1,2}月\d{1,2}日'
        for date in re.findall(date_pattern, metrics_text):
            keywords.add(date.strip())
        # 匹配所有中括号数据，支持各种单位/格式
        data_pattern = r'\[([^\[\]]+?)\]'
        for val in re.findall(data_pattern, metrics_text):
            keywords.add(val.strip())
        keyword_list.append(list(keywords))
    return keyword_list


if __name__ == "__main__":
    # =============== 生成基础的跑步指标和基于指标生成的提示词 ===============
    print(f"=====正在并发生成 {TOTAL_CASES} 条跑步指标解析数据=====")
    start_time = datetime.now()
    origin_prompt_list = []
    for _ in range(TOTAL_CASES):
        date_str = generate_random_date()
        metric_list_temp = random.sample(METRICS_POOL, 3)
        origin_prompt = build_analysis_prompt(date_str, metric_list_temp)
        unit = {
            "date": date_str,
            "metric_1": metric_list_temp[0],
            "metric_2": metric_list_temp[1],
            "metric_3": metric_list_temp[2],
            "origin_prompt": origin_prompt
        }
        origin_prompt_list.append(unit)
        print(f"当前进度: {len(origin_prompt_list)} / {TOTAL_CASES}, 当前时间: {datetime.now()}")
    print(f"=====已生成跑步原始prompt指标=====")

    # =============== 基于原始prompt生成metrics和用于SFT的prompts ===============
    print(f"=====开始并发推理，生成 {TOTAL_CASES} 条metrics,prompts=====")
    prompt_list = []
    for unit in origin_prompt_list:
        prompt_list.append(unit["origin_prompt"])
    metrics_list = perform_batch_inference(prompt_list, BATCH_SIZE)
    keywords_list = extract_keywords(metrics_list)
    prompts = generate_prompt_list(origin_prompt_list)
    if len(origin_prompt_list) != len(metrics_list) or len(prompts) != len(metrics_list):
        raise Exception(f"""
            对应文件的尺寸不一致，有问题origin_prompt_list:
            {len(origin_prompt_list)}, metrics_list:{len(metrics_list)}, prompts:{len(prompts)}""")
    result_list = []
    for num in range(len(origin_prompt_list)):
        unit = {
            "date": origin_prompt_list[num]["date"],
            "metric_1": origin_prompt_list[num]["metric_1"],
            "metric_2": origin_prompt_list[num]["metric_2"],
            "metric_3": origin_prompt_list[num]["metric_3"],
            "prompt": prompts[num].replace(" ", ""),
            "metrics": metrics_list[num].replace(" ", ""),
            "keywords": keywords_list[num]
        }
        result_list.append(unit)
    print(f"=====已生成跑步原始prompt和metrics指标=====")

    # =============== 基于prompt和metrics ===============
    print(f"=====基于prompt和metrics生成 {TOTAL_CASES} 条summary=====")
    inference_prompts = []
    for unit in result_list:
        inference_prompts.append(build_exercise_summary_prompt(unit))
    summary = perform_batch_inference(inference_prompts, BATCH_SIZE)
    if len(inference_prompts) != len(summary):
        raise Exception(f"对应文件的尺寸不一致，有问题inference_prompts:{len(inference_prompts)}, summary:{len(summary)}")
    for idx, unit in enumerate(result_list):
        unit["summary"] = summary[idx].replace(" ", "")
    print(f"===== 已生成所有的summary指标 =====")

    # =============== 数据保存 ===============
    print(f"===== 开始保存jsonl格式的文件 =====")
    with open(OUTPUT_DATASET, 'w', encoding='utf-8') as f:
        for unit in result_list:
            line = json.dumps(unit, ensure_ascii=False)
            f.write(line + '\n')
    print(f"===== 数据已保存到 {OUTPUT_DATASET} =====")



