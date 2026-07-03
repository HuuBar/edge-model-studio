# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import json
import re
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex, sympy_parser
from sympy import simplify, N, Matrix

def max_word_length(sentence):
    """输出单词最大长度"""
    # 将句子分割成单词列表
    words = sentence.split()
    # 使用max函数和len函数找出最长的单词长度
    max_length = max(len(word) for word in words)
    return max_length

def _fix_sqrt(s):
    # 处理平方根符号
    return re.sub(r'\\sqrt\{(\d+)\}', r'\1^{1/2}', s)

def _fix_fracs(s):
    # 处理分数符号
    return re.sub(r'\\frac\{(\d+)\}\{(\d+)\}', r'\1/\2', s)

def _fix_a_slash_b(s):
    # 处理斜杠分隔的分数
    return re.sub(r'(\d+)\s*/\s*(\d+)', r'\1/\2', s)

def clean_string(input_str):
    # 基础清理
    cleaned = str(input_str).strip().replace('\n', '').rstrip('.')
    
    # 替换LaTeX符号
    replacements = {
        '\\begin{array}{.*?}': '\\begin{pmatrix}',
        '\\end{array}': '\\end{pmatrix}',
        'bmatrix': 'pmatrix',
        'tfrac': 'frac', 'dfrac': 'frac',
        '\\neq': '\\ne', '\\leq': '\\le', '\\geq': '\\ge',
        '\\left': '', '\\right': '', '\\{': '{', '\\}': '}',
        '\\$': '', '$': '', '\\(': '', '\\)': '',
        '\\emptyset': '{}', '(-\\infty,\\infty)': '\\mathbb{R}',
        '\\%': '', '%': '', ' .': ' 0.', '{.': '{0.',
        'infinity': '\\infty', 'inf': '\\infty', '+\\inity': '\\infty',
        'and': '', '\\mathbf': '', '\\mbox{.*?}': ''
    }
    for pattern, repl in replacements.items():
        cleaned = re.sub(pattern, repl, cleaned)
    
    # 特殊字符处理
    cleaned = cleaned.replace('\\text{', '').replace('}', '').replace("'", '').replace('"', '')
    cleaned = cleaned.replace('^{\\circ}', '').replace('^\\circ', '').replace('j', 'i')
    
    # 数字格式化
    cleaned = re.sub(r'(\d+)\.0*([^\d])', r'\1\2', cleaned)
    cleaned = re.sub(r'(\d+)\.0*$', r'\1', cleaned)
    
    # 结尾修正
    if cleaned.startswith('{') and cleaned.endswith('}') and cleaned.isalnum():
        cleaned = cleaned[1:-1]
    
    # 特殊条件处理
    if cleaned.startswith('.'):
        cleaned = '0' + cleaned
    if len(cleaned.split('=')) == 2 and len(cleaned.split('=')[0]) <= 2:
        cleaned = cleaned.split('=')[1]
    
    # 调用辅助函数
    cleaned = _fix_sqrt(cleaned)
    cleaned = _fix_fracs(cleaned)
    cleaned = _fix_a_slash_b(cleaned)
    cleaned = re.sub(r'\\(?=\-?\d+(\\|\)|,|\]|$))', '', cleaned)
    cleaned = re.sub(r'thgrade$', '', cleaned)
    
    return cleaned.strip()
def max_length_between_punctuations(s):
    # 输出标点符号间中文和英文最大的长度
    parts = re.split(r'(\W+)', s)
    max_length = 0
    current_length = 0
    in_chinese = False
    in_english = False
    
    for part in parts:
        if re.match(r'[\u4e00-\u9fff]+', part):  # 中文字符
            current_length += len(part)
            in_chinese = True
            in_english = False  # 重置英文状态
        elif re.match(r'[a-zA-Z]+', part):  # 英文字符
            current_length += len(part)
            in_english = True
            in_chinese = False  # 重置中文字符状态
        else:  # 非中英文部分（标点符号）
            if in_chinese and current_length > max_length:
                max_length = current_length
            elif in_english and current_length > max_length:
                max_length = current_length
            current_length = 0  # 重置当前长度
            in_chinese = False
            in_english = False
    
    # 检查最后一个连续部分是否为最大长度
    if (in_chinese and current_length > max_length):
        max_length = current_length
    if (in_english and current_length > max_length):
        max_length = current_length
    return max_length

def contains_special_characters_list(text_list):
    """
    检查列表中的每个元素是否包含特殊符号。
    
    参数:
    text_list -- 待检查的列表
    
    返回:
    True 如果列表中至少有一个元素包含特殊符号，否则返回 False
    """
    # 正则表达式匹配非字母数字字符
    pattern_regex = re.compile(r'[^a-zA-Z0-9]')
    
    for text in text_list:
        if pattern_regex.search(str(text)):  # 将元素转换为字符串，以便进行搜索
            return True
    return False

def read_jsonl_file(file_path):
    """读取JSONL文件并返回解析后的数据列表"""
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file1:
        for line1 in file1:
            # 去除行尾空白符并解析JSON
            line1 = line1.strip()
            if line1:  # 确保不是空行
                data1 = json.loads(line1)
                data_list.append(data1)
    return data_list

def save_list_to_jsonl(data_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data_list:
            # 将字典转换为JSON字符串并写入文件
            json_line = json.dumps(entry, ensure_ascii=False)
            file.write(json_line + '\n')

def main():
    #经常出现的乱码
    delet_key = ['combustiblecharacters','combus','y','TRIMP','w','f','F','BCAA','ss']
    data = read_jsonl_file('all_new.jsonl')
    for keys in delet_key:
        for item in data:
            if keys in item["output"]:
                data.remove(item)
                continue
            if keys in item["input"]:
                data.remove(item)
                continue
            if contains_special_characters_str(item["output"]):
                data.remove(item)
                continue
            if contains_special_characters_str(item["input"]):
                data.remove(item)
                continue
            if max_length_between_punctuations(item["output"])> 30:
                data.remove(item)
                continue
            if max_length_between_punctuations(item["input"])> 30:
                data.remove(item)
                continue
            if max_word_length(item["output"])> 15:
                data.remove(item)
                continue
            if max_word_length(item["input"])> 15:
                data.remove(item)
                continue
    # 调用函数保存数据
    save_list_to_jsonl(data, 'all_new.jsonl')