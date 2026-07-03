# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import json
import random
import re
from datetime import datetime, timedelta
import requests
from transformers import AutoTokenizer

class DataDistillationSystem():
    def __init__(self, config):
        self.metrics_pool = config.metrics_pool
        self.true_data_q_user = config.true_data_q_user
        self.case_numbers = config.case_numbers
        self.batch_size = config.batch_size
        self.text = config.text
        self.scores_keyword = config.scores_keyword
        self.suggestion_keyword = config.suggestion_keyword
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
        self.thre = config.thre
        self.final_result_list = config.final_result_list
        self.api_url = config.api_url
        self.tag = config.tag
        self.think_tag = config.think_tag

    def _apply_qwen_template_inference(self, messages, enable_thinking=False):
        #对输入进行格式化，适配大模型输入要求
        tokenizer = self.tokenizer
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
    def _build_generate_prompt(self, enable_thinking=True):
        #构建生成类prompt
        messages = [
            {"role": "system", "content": f"""
            你是一个专业的跑步教练，请严格按照要求生成跑步指标解析；
            基于以下专业指标{self.metrics_pool}，仿照提供格式，填写指标的内容；
            输出字符1000字以内，提示中出现的数值和结论不得重复；
            不同专业指标之间关系要合理。
            """},
            {"role": "user", "content": f"{self.true_data_q_user}"}
        ]
        return self._apply_qwen_template_inference(messages, enable_thinking)

    def _build_prompt(self, dataset, enable_thinking=True):
        #构建通用类prompt
        messages = [
            {"role": "system", "content": f"{dataset['instruction']}"},
            {"role": "user", "content": f"{dataset['input']}"}
        ]
        return self._apply_qwen_template_inference(messages, enable_thinking)
    
    def _extract_final_answer(self, text):
        # 去除标签
        text = re.sub(self.tag, "", text)
        # 定义匹配模式
        patterns = [r"#指标解析.*", r"指标解析.*", r"#分析结果.*", r"分析结果.*"]
        # 搜索匹配内容
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(0).strip()
        return text.strip()
    
    def _parse_qwen3_output(self, response_text):
        #解析输出
        match = re.search(self.think_tag, response_text, re.DOTALL)
        if match:
            # 提取<think>标签内的文本并去除前后空白
            thinking = match.group(1).strip()
            # 去除<think>标签后的内容并去除前后空白
            answer = response_text[match.end():].strip()
            return thinking, answer
        
        # 初始化思考内容为空字符串
        thinking = ""
        # 去除标签
        answer = re.sub(r"<.*?>", "", text).strip()
        # 去除开头的标识符（如assistant:、user:）并忽略大小写
        answer = re.sub(r"^(assistant|user|system):?", "", answer, flags=re.IGNORECASE).strip()
        return thinking, answer
    
    def _get_scores(self, input_suggestion):
        #获取生成数据的质量分数
        context_scores = input_suggestion.split(self.scores_keyword)[-1]
        try:
            context_scores = int(context_scores)
            return context_scores
        except ValueError:
            return -1
    
    def _get_suggestion(self, input_suggestion):
        #获取生成数据的修改意见
        modify_suggestion = input_suggestion.split(self.suggestion_keyword)[-1]
        return modify_suggestion
    
    def _perform_batch_inference(self, prompts_list):
        #进行批量推理
        results = []
        batch_size = self.batch_size
        for i in range(0, len(prompts_list), batch_size):
            batch = prompts_list[i:i + batch_size]
            payload = {
                "model": "Qwen3-235B-A22B-w8a8",
                "prompt": batch,
                "max_tokens": 2048,
                "temperature": 1,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.1
            }
            response = requests.post(self.api_url, json=payload)      
            response_data = response.json()
            for choice in response_data["choices"]:
                response_text = choice["text"].strip()
                thinking, content = self._parse_qwen3_output(response_text)
                content = self._extract_final_answer("<think>\n"+thinking+"\n</think>\n"+content)
                results.append(content)
        return results
        
    def generate_module(self):
        #生成模块：用于生成原始数据
        generate_prompt_list = []
        for _ in range(self.case_numbers):
            generate_prompt = self._build_generate_prompt(False)
            generate_prompt_list.append(generate_prompt)
        generate_result_list = self._perform_batch_inference(generate_prompt_list)
        return generate_result_list
    
    def review_module(self, generate_result_list):
        #评价模块：对原始数据进行质量评估，并打分，提出修改意见
        review_prompt_list = []
        for num in range(len(generate_result_list)):
            review_unit = {
                "instruction": f"""
                请你严格评估语言表达是否流畅、自然、易于理解，包括语法、拼写、句子结构、用词准确性等。
                严格根据扣分规则给出的分数进行扣分，同一个扣分规则可出现多次，指出每个扣分项的扣分内容、扣分原因和扣分分数，并给出详细的修改说明。
                【累计扣分计算方式】
                对【不足及修改说明】中的扣分每一项分数进行累加后求和，输出到【累计扣分】一栏
                【扣分规则】
                1. 句式严重冗余扣5分！
                2. 前后数据表述不一致扣5分5！
                3. 口语化不足扣5分！
                4. 表述不完整扣5分！
                5. 错别字扣5分！
                6. 搭配不当扣5分！
                【输出格式说明】
                必须严格按照以下两项输出（格式不要发生变化）
                1. 【不足及修改说明】：
                1)
                2)
                3)
                4)
                2. 【累计扣分】：xx分
                """,
                "input": generate_result_list[num],
            }
            review_prompt_list.append(review_unit)
        
        for _ in range(self.case_numbers):
            if len(review_prompt_list) == 0:
                break
            review_inference_prompts = []
            for _ in result_list:
                review_inference_prompts.append(self._build_prompt(review_prompt_list))
         review_result_list = self._perform_batch_inference(review_inference_prompts)
        
        return review_result_list

    def suggestion_module(self, review_result_list, generate_result_list):
        #建议模块：获取分数及修改建议，生成新的prompt
        suggestion_prompt_list = []
        for idx, review_result in enumerate(review_result_list):
            scores = self._get_scores(review_result)
            suggestions = self._get_suggestion(review_result)
            if scores > self.thre:
                #用于修改的prompt
                suggestion_prompt = {
                    "instruction": f"""
                    你是一名资深语言学家，熟悉运动领域知识；
                    任务是基于【评价意见】对待修改内容进行修改，未评价的地方不做修改，保持整体结构不变；
                    不要出现英文，确保语言流畅，可读性强。
                    """,
                    "input": f"待修改内容：{generate_result_list[idx]}\n【评价意见】\n不足之处及修改说明：{suggestions}",
                }
                suggestion_prompt_list.append(suggestion_prompt)
            else:
                self.final_result_list.append(unit["input"]) 
        return suggestion_prompt_list

    def modify_module(self, suggestion_prompt_list):
        #修改模块：对生成内容进行修改
        modify_prompt_list = []
        for suggestion_prompt in suggestion_prompt_list:
            modify_prompt_list.append(self._build_prompt(suggestion_prompt))
        modify_result_list = self._perform_batch_inference(modify_prompt_list)   
        return modify_result_list
