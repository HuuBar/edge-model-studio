import json
import random
import re
from datetime import datetime, timedelta
import os

import requests
from transformers import AutoTokenizer

# 训练10000(50%)  测试1000(5%)  RL4000(20%)  prune5000(25%)
# TOTAL_CASES = 20000
# TOTAL_CASES = 100000
TOTAL_CASES = 500000  # 🔥 修改为50万

# 🔥 新增：分批保存配置
SAVE_BATCH_SIZE = 10000  # 每10000个数据保存一次
OUTPUT_DIR = "/data2/jwllm/datasets/exercise_generate/pretrain/"
BASE_FILENAME = "exercise_summary_dataset_batch"

model_id = "/data2/jwllm/models_origin/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
BATCH_SIZE = 256  # 批量推理数值
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


# 🔥 新增：清理文本内容
def clean_text_content(text):
    """清理文本内容，去除指定的符号和格式"""
    # 去除换行符
    text = text.replace('\n', ' ')
    
    # 去除括号及其内容 (包括中英文括号)
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'（[^）]*）', '', text)
    
    # 去除方括号及其内容
    text = re.sub(r'\[[^\]]*\]', '', text)
    
    # 去除编号格式 (1.1, 1.2, 1.3等)
    text = re.sub(r'\d+\.\d+', '', text)
    
    # 去除多余的空格
    text = re.sub(r'\s+', ' ', text)
    
    # 去除首尾空格
    text = text.strip()
    
    return text


# 🔥 新增：保存单个批次数据
def save_batch_data(data_list, batch_num):
    """保存单个批次的数据"""
    batch_filename = f"{BASE_FILENAME}_{batch_num:03d}.jsonl"
    batch_filepath = os.path.join(OUTPUT_DIR, batch_filename)
    
    # 确保目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(batch_filepath, 'w', encoding='utf-8') as f:
        for unit in data_list:
            line = json.dumps(unit, ensure_ascii=False)
            f.write(line + '\n')
    
    print(f"✅ 批次 {batch_num} 数据已保存到: {batch_filepath} (共 {len(data_list)} 条)")
    return batch_filepath


# 🔥 新增：合并所有批次文件
def merge_all_batches():
    """合并所有批次文件为最终文件"""
    print(f"🔄 开始合并所有批次文件...")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_DATASET), exist_ok=True)
    
    total_count = 0
    with open(OUTPUT_DATASET, 'w', encoding='utf-8') as f_out:
        batch_num = 1
        while True:
            batch_filename = f"{BASE_FILENAME}_{batch_num:03d}.jsonl"
            batch_filepath = os.path.join(OUTPUT_DIR, batch_filename)
            
            if not os.path.exists(batch_filepath):
                break
                
            print(f"📂 合并批次文件: {batch_filepath}")
            with open(batch_filepath, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    f_out.write(line)
                    total_count += 1
            batch_num += 1
    print(f"✅ 所有批次合并完成! 总计 {total_count} 条数据保存到: {OUTPUT_DATASET}")


# 🔥 新增：处理单个批次的完整流程
def process_single_batch(start_idx, batch_size, batch_num):
    """处理单个批次的完整流程"""
    print(f"\n🔥 开始处理批次 {batch_num} (索引 {start_idx} - {start_idx + batch_size - 1})")
    batch_start_time = datetime.now()
    
    # =============== 生成基础的跑步指标和基于指标生成的提示词 ===============
    print(f"📝 生成 {batch_size} 条跑步指标解析数据...")
    origin_prompt_list = []
    for i in range(batch_size):
        
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
        
        if (i + 1) % 1000 == 0:
            print(f"  生成进度: {i + 1} / {batch_size}")

    # =============== 基于原始prompt生成metrics和用于SFT的prompts ===============
    print(f"🚀 开始推理生成 {batch_size} 条 metrics...")
    prompt_list = [unit["origin_prompt"] for unit in origin_prompt_list]
    metrics_list = perform_batch_inference(prompt_list, BATCH_SIZE)
    prompts = generate_prompt_list(origin_prompt_list)
    
    if len(origin_prompt_list) != len(metrics_list) or len(prompts) != len(metrics_list):
        raise Exception(f"批次 {batch_num} 数据尺寸不一致: origin_prompt_list:{len(origin_prompt_list)}, metrics_list:{len(metrics_list)}, prompts:{len(prompts)}")
    
    result_list = []
    for num in range(len(origin_prompt_list)):
        unit = {
            "prompt": prompts[num].replace(" ", ""),
            "metrics": metrics_list[num].replace(" ", ""),
        }
        result_list.append(unit)

    # =============== 基于prompt和metrics生成summary ===============
    print(f"📋 基于 prompt 和 metrics 生成 {batch_size} 条 summary...")
    inference_prompts = [build_exercise_summary_prompt(unit) for unit in result_list]
    summary = perform_batch_inference(inference_prompts, BATCH_SIZE)
    
    if len(inference_prompts) != len(summary):
        raise Exception(f"批次 {batch_num} summary 尺寸不一致: inference_prompts:{len(inference_prompts)}, summary:{len(summary)}")
    
    # 🔥 合并 prompt + metrics + summary 到一个字段，并清理内容
    final_result_list = []
    for idx, unit in enumerate(result_list):
        # 原始内容
        clean_prompt = unit["prompt"].replace(" ", "")
        clean_metrics = unit["metrics"].replace(" ", "")
        clean_summary = summary[idx].replace(" ", "")
        
        # 🔥 清理所有内容，去除指定符号
        clean_prompt = clean_text_content(clean_prompt)
        clean_metrics = clean_text_content(clean_metrics)
        clean_summary = clean_text_content(clean_summary)
        
        # 合并内容：去掉prompt中的系统提示部分，只保留用户问题
        user_question = extract_user_question(clean_prompt)
        user_question = clean_text_content(user_question)  # 同样清理用户问题
        
        # 组合最终内容，用空格分隔而不是换行
        combined_content = f"{user_question} {clean_metrics} {clean_summary}"
        
        # 🔥 最终清理整个合并内容
        combined_content = clean_text_content(combined_content)
        
        final_unit = {
            "exercise_pretrain": combined_content
        }
        final_result_list.append(final_unit)

    # =============== 保存当前批次数据 ===============
    batch_filepath = save_batch_data(final_result_list, batch_num)
    
    batch_end_time = datetime.now()
    batch_duration = batch_end_time - batch_start_time
    print(f"⏱️ 批次 {batch_num} 完成，耗时: {batch_duration}")
    
    return batch_filepath, len(final_result_list)


# 🔥 新增：提取用户问题部分
def extract_user_question(prompt_text):
    """从完整prompt中提取用户问题部分"""
    # 匹配日期模式，提取用户问题
    date_pattern = r'(\d{4}年\d{1,2}月\d{1,2}日)跑步跑得怎么样？'
    match = re.search(date_pattern, prompt_text)
    if match:
        return f"用户问题：{match.group(1)}跑步跑得怎么样？"
    
    # 如果没有匹配到，尝试其他模式
    if "跑步跑得怎么样" in prompt_text:
        # 找到包含日期和问题的句子
        lines = prompt_text.split(' ')  # 改为按空格分割
        for line in lines:
            if "跑步跑得怎么样" in line:
                return f"用户问题：{line.strip()}"
    
    # 默认返回一个通用问题
    return "用户问题：请分析我的跑步数据"


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
        print(f"  推理进度: {i} / {len(prompts_temp)}, 当前时间: {datetime.now()}")

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
    print(f"  已生成 {len(prompts_temp)} 个对应的 prompt")
    return prompts_temp


if __name__ == "__main__":
    print(f"🚀 开始生成 {TOTAL_CASES:,} 条数据，每 {SAVE_BATCH_SIZE:,} 条保存一次")
    print(f"📂 输出目录: {OUTPUT_DIR}")
    print(f"📄 最终文件: {OUTPUT_DATASET}")
    print(f"⏰ 开始时间: {datetime.now()}")
    
    total_start_time = datetime.now()
    total_processed = 0
    batch_files = []
    
    # 🔄 分批处理数据
    num_batches = (TOTAL_CASES + SAVE_BATCH_SIZE - 1) // SAVE_BATCH_SIZE  # 向上取整
    
    for batch_num in range(1, num_batches + 1):
        start_idx = (batch_num - 1) * SAVE_BATCH_SIZE
        current_batch_size = min(SAVE_BATCH_SIZE, TOTAL_CASES - start_idx)
        
        try:
            batch_filepath, batch_count = process_single_batch(start_idx, current_batch_size, batch_num)
            batch_files.append(batch_filepath)
            total_processed += batch_count
            
            print(f"📊 总进度: {total_processed:,} / {TOTAL_CASES:,} ({total_processed/TOTAL_CASES*100:.1f}%)")
            
            # 🔥 估算剩余时间
            elapsed_time = datetime.now() - total_start_time
            if total_processed > 0:
                avg_time_per_item = elapsed_time.total_seconds() / total_processed
                remaining_items = TOTAL_CASES - total_processed
                estimated_remaining_time = remaining_items * avg_time_per_item
                eta = datetime.now() + timedelta(seconds=estimated_remaining_time)
                print(f"⏱️ 预计完成时间: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            print(f"❌ 批次 {batch_num} 处理失败: {e}")
            print(f"🔄 跳过该批次，继续处理下一批次...")
            continue
    
    # =============== 合并所有批次文件 ===============
    print(f"\n🔄 开始合并所有批次文件...")
    merge_all_batches()
    
    total_end_time = datetime.now()
    total_duration = total_end_time - total_start_time
    
    print(f"\n✅ 全部完成!")
    print(f"📊 总计处理: {total_processed:,} 条数据")
    print(f"⏱️ 总耗时: {total_duration}")
    print(f"🚀 平均速度: {total_processed / total_duration.total_seconds():.2f} 条/秒")
    print(f"📂 批次文件数: {len(batch_files)}")
    print(f"📄 最终文件: {OUTPUT_DATASET}")