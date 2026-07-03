# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import json
import random
import re
from datetime import datetime, timedelta

import requests
from transformers import AutoTokenizer

TOTAL_CASES = 5000
CYCLE_TIMES = 1
MODEL_ID = "Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
BATCH_SIZE = 96  # 批量推理数值
API_URL = "http://100.105.97.35:8888/v1/completions"  # 根据你的实际情况修改
OUTPUT_DATASET = "exercise_summary_dataset_total_demo1_summary2_new_think6_235_35_cs_koufen.jsonl"

def read_jsonl_file(file_path):
    """读取JSONL文件并返回解析后的数据列表"""
    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 去除行尾空白符并解析JSON
                line = line.strip()
                if line:  # 确保不是空行
                    data = json.loads(line)
                    data_list.append(data)
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
    except json.JSONDecodeError as e:
        print(f"JSON解析错误：{e}")
    return data_list

# 构建一个
def build_analysis_prompt(date):
    messages = [
        {"role": "system", "content": f"""
        你是一个跑步专家，你能理解跑步指标的解读，并且根据用户的运动目标，能力状态识别运动关键信息，输出280个到360个汉字以内的跑步报告。
        #任务
        请根据以下个人数据生成一段回复用户提问跑步跑得怎么样？的答复。
        一. **内容要求**\n语句流畅口语化，仅仅按照三个大的维度【总体表现】【要点分析】【总结建议】进行分析，不要有多余的其他话”的答复。
        请按照以下流程进行思考，思考过程简洁、关键，做出最终回答：
        step1 在思考过程中，需要思考提到的所有指标和数据
        step2 从提到的所有指标中，总结出两条最关键指标，思考过程中包含对不同指标的重要程度分析，并总结选择目标指标的原因，输出到【总体表现】一栏，最终形成两条总结，每条最多总结一条数据
        step3 围绕【总体表现】中最关键的两个指标，选择个人数据中最相关的其他三个指标进行分析，并详细说明为什么选择这三条，输出到【要点分析】一栏，最终形成三条分析，每条最多分析一条数据
        step4 围绕【总体表现】和【要点分析】提到的指标，选择两条最该提升的进行建议，并在思考过程中详细说明为什么提出这两条建议，输出到【综合建议】一栏，最终形成两条建议
        二. **输出格式参考**
        【总体表现】
        1.
        2.
        【要点分析】
        1.
        2.
        3.
        【综合建议】
        1.
        2.
        三. **其他**
        最后不要有总结也不要有注解！！
        """
        },
        {"role": "user", "content": f"""
        ## 个人数据
        {date}
        """
    }
    ]
    return apply_qwen_template_inference(messages, enable_thinking=True)

# True
def build_exercise_summary_prompt(dataset):
    messages = [
        {"role": "system", "content": f"{dataset['instruction']}"},
        {"role": "user", "content": f"{dataset['input']}"}
    ]
    return apply_qwen_template_inference(messages, enable_thinking=True)

# False
def build_modify_prompt(dataset):
    messages = [
        {"role": "system", "content": f"{dataset['instruction']}"},
        {"role": "user", "content": f"{dataset['input']}"}
    ]
    return apply_qwen_template_inference(messages, enable_thinking=False)

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
            "model": "Qwen3-235B-A22B-w8a8",
            "prompt": batch,
            "max_tokens": 2048,
            "temperature": 1,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.1
        }
        response = requests.post(API_URL, json=payload)      
        if response.status_code == 200:
            response_data = response.json()
            for choice in response_data["choices"]:
                response_text = choice["text"].strip()
                _, content = parse_qwen3_output(response_text)
                content2 = extract_final_answer("<think>\n"+_+"\n</think>\n"+content)
                results.append(content)
        else:
            print(f"请求失败，状态码: {response.status_code}，响应内容: {response.text}")
    return results

# 批量进行推理
def perform_batch_inference_suggestion(prompts_temp, batch_size):
    results = []
    for i in range(0, len(prompts_temp), batch_size):
        print(f"正在进行推理，进度是 {i} / {len(prompts_temp)}, 当前时间为: {datetime.now()}")

        batch = prompts_temp[i:i + batch_size]
        payload = {
            "model": "Qwen3-235B-A22B-w8a8",
            "prompt": batch,
            "max_tokens": 2048,
            "temperature": 0.0,
            "seed": 42,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.0
        }

        response = requests.post(API_URL, json=payload)      
        if response.status_code == 200:
            response_data = response.json()
            for choice in response_data["choices"]:
                response_text = choice["text"].strip()
                _, content = parse_qwen3_output(response_text)
                content2 = extract_final_answer("<think>\n"+_+"\n</think>\n"+content)
                results.append(content)
        else:
            print(f"请求失败，状态码: {response.status_code}，响应内容: {response.text}")
    return results

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

def get_scores(output):
    context_scores = output.split("【累计扣分】：")[-1].split("分")[0]
    try:
        context_scores = int(context_scores)
        return context_scores
    except ValueError:
        return -1

def get_suggestion(output):
    modify_suggestion = output.split("【不足及修改说明】：")[-1]
    return modify_suggestion

def suggestion_module(output, input):
    context_scores = get_scores(unit["output"])
    if context_scores == -1:
        return -1,"eorr"
    modify_suggestion = get_suggestion(unit["output"])
    modify_suggestion_prompt = {
            "instruction": f"""
                            你是一名资深语言学家，熟悉运动领域知识，任务是基于【评价意见】对待修改内容进行修改.
                            未评价的地方不做修改，保持整体结构不变，不要出现英文，确保语言流畅，可读性强，最后不要写注释！
                            """,
            "input": f"""待修改内容：\n{input}\n。【评价意见】\n不足之处及修改说明：{modify_suggestion}"""
        }
    return context_scores, modify_suggestion_prompt

def wrong_word_modify(metrics_list):
    for i in range(len(metrics_list)):
        if "心�t率" in metrics_list[i]:
            metrics_list[i] = metrics_list[i].replace("心�t率", "心率")
        if "心�t" in metrics_list[i]:
            metrics_list[i] = metrics_list[i].replace("心�t", "心率")
        if "肌�t" in metrics_list[i]:
            metrics_list[i] = metrics_list[i].replace("肌�t", "肌群")
        if "心_rate" in metrics_list[i]:
            metrics_list[i] = metrics_list[i].replace("心_rate", "心率")
        if "心rate" in metrics_list[i]:
            metrics_list[i] = metrics_list[i].replace("心rate", "心率")
if __name__ == "__main__":
    # =============== 生成基础的跑步指标和基于指标生成的提示词 ===============
    
    data = read_jsonl_file('test.jsonl')   
    print(f"=====正在并发生成 {TOTAL_CASES} 条跑步指标解析数据=====")
    start_time = datetime.now()
    origin_prompt_list = []
    for _ in range(len(data)):
        date_str = generate_random_date()
        origin_prompt = build_analysis_prompt(data[_])
        
        unit = {
            "instruction": f"""
            请你严格评估语言表达是否流畅、自然、易于理解，包括语法、拼写、句子结构、用词准确性等。
            严格根据扣分规则给出的分数进行扣分，同一个扣分规则可出现多次，指出每个扣分项的扣分内容、扣分原因和扣分分数，并给出详细的修改说明。
            【累计扣分计算方式】
            对【不足及修改说明】中的扣分每一项分数进行累加后求和，输出到【累计扣分】一栏
            【扣分规则】
            句式严重冗余扣2分！
            前后数据表述不一致扣5分（时间除外）！
            前后时间表述超过一分钟扣30分！
            口语化不足扣5分！
            表述不完整扣1分！
            错别字扣100分！
            搭配不当扣5分！
            【输出格式说明】
            必须严格按照以下两项输出（格式不要发生变化）
            1. 【不足及修改说明】：
            1)
            2)
            3)
            4)
            2. 【累计扣分】：xx分
            """,
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
    metrics_list = perform_batch_inference_suggestion(prompt_list, BATCH_SIZE)
    print(metrics_list)
    wrong_word_modify(metrics_list)
    keywords_list = extract_keywords(metrics_list)
    if len(origin_prompt_list) != len(metrics_list) :
        raise Exception(f"""
            对应文件的尺寸不一致，有问题origin_prompt_list:
            {len(origin_prompt_list)}, metrics_list:{len(metrics_list)}, prompts:{len(prompts)}""")
    result_list = []
    for num in range(len(origin_prompt_list)):
        unit = {
            "instruction": origin_prompt_list[num]["instruction"],
            "input": metrics_list[num].replace(" ", ""),
        }
        result_list.append(unit)
    print("..........................................................................")
    print(f"=====已生成跑步原始prompt和metrics指标=====")

    # =============== 基于prompt和metrics ===============
    print(f"=====基于prompt和metrics生成 {TOTAL_CASES} 条summary=====")
    
    final_result_list = []
    
    for i in range(CYCLE_TIMES):
        if len(result_list) == 0:
            break
        inference_prompts = []
        for unit in result_list:
            inference_prompts.append(build_exercise_summary_prompt(unit))
        summary = perform_batch_inference_suggestion(inference_prompts, BATCH_SIZE)
        print(".................................................................",i)
        if len(inference_prompts) != len(summary):
            raise Exception(f"对应文件的尺寸不一致，有问题inference_prompts:{len(inference_prompts)}, summary:{len(summary)}")
        for idx, unit in enumerate(result_list):
            unit["output"] = summary[idx].replace(" ", "")

        modify_list = []
        for unit in result_list:
            scores, suggestions = suggestion_module(unit["output"],unit["input"])
            print("scores:",scores)
            if scores == -1:
                continue
            elif scores > 20:
                modify_list.append(suggestions)
            else:
                final_result_list.append(unit["input"])
        print(result_list)
        result_list = []
        if len(modify_list) == 0:
            break
        modify_inference_prompts = []
        for unit in modify_list:
            modify_inference_prompts.append(build_modify_prompt(unit))
        context = perform_batch_inference_suggestion(modify_inference_prompts, BATCH_SIZE)        
        if len(modify_inference_prompts) != len(context):
            raise Exception(f"对应文件的尺寸不一致，有问题inference_prompts:{len(inference_prompts)}, summary:{len(summary)}")
        wrong_word_modify(context)
        
        for num in range(len(context)):
            unit = {
                "instruction":  f"""
                请你严格评估语言表达是否流畅、自然、易于理解，包括语法、拼写、句子结构、用词准确性等。
                严格根据扣分规则给出的分数进行扣分，同一个扣分规则可出现多次，指出每个扣分项的扣分内容、扣分原因和扣分分数，并给出详细的修改说明。
                【累计扣分计算方式】
                对【不足及修改说明】中的扣分每一项分数进行累加后求和，输出到【累计扣分】一栏
                【扣分规则】
                句式严重冗余扣2分！
                前后数据表述不一致扣5分（时间除外）！
                前后时间表述超过一分钟扣30分！
                口语化不足扣5分！
                表述不完整扣1分！
                错别字扣100分！
                搭配不当扣5分！
                【输出格式说明】
                必须严格按照以下两项输出（格式不要发生变化）
                1. 【不足及修改说明】：
                1)
                2)
                3)
                4)
                2. 【累计扣分】：xx分
                """,
                "input": context[num],
            }
            result_list.append(unit)

    print(f"===== 已生成所有的summary指标 =====")
    step1_final_result_list = []
    # =============== 数据保存 ===============
    print(f"===== 开始保存jsonl格式的文件 =====")
    for num in range(len(final_result_list)):
        unit = {
            "input": final_result_list[num],
        }
        step1_final_result_list.append(unit)
    with open(OUTPUT_DATASET, 'w', encoding='utf-8') as f:
        for unit in step1_final_result_list:
            line = json.dumps(unit, ensure_ascii=False)
            f.write(line + '\n')
    print(f"===== 数据已保存到 {OUTPUT_DATASET} =====")

    