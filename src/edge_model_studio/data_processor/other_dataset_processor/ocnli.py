import json
import sys
from typing import List, Dict, Any


def convert_text_to_json(input_file_path: str) -> List[Dict[str, Any]]:
    """
    将每行包含JSON对象的文本文件转换为标准JSON数组文件

    参数:
        input_file_path: 输入文本文件路径
        output_file_path: 输出JSON文件路径
    """
    data_list: List[Dict[str, Any]] = []
    label_frequency = {}
    try:
        # 读取输入文件
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    # 解析每行JSON
                    json_obj = json.loads(line)
                    sentence1 = json_obj["sentence1"]
                    sentence2 = json_obj["sentence2"]
                    label = json_obj["label"]

                    if label not in ['entailment', 'neutral', 'contradiction']:
                        continue

                    json_obj_new = {
                        "number": json_obj["id"],
                        "sentence1": sentence1,
                        "sentence2": sentence2,
                        "label": label
                    }

                    if label in label_frequency:
                        label_frequency[label] += 1
                    else:
                        label_frequency[label] = 1

                    data_list.append(json_obj_new)
                except json.JSONDecodeError as e:
                    print(f"第 {line_num} 行JSON解析错误: {e}")
                    print(f"错误行内容: {line[:100]}...")  # 只显示前100个字符
                    sys.exit(1)

        print(f"最终统计出来的每个标签的数量：")
        for char, count in label_frequency.items():
            print(f"'{char}': {count}")

    except FileNotFoundError:
        print(f"错误: 文件 {input_file_path} 未找到")
        sys.exit(1)

    except Exception as e:
        print(f"发生未知错误: {e}")
        sys.exit(1)

    return data_list


# 解析jsondata
def filter_json(jsondata: List[Dict[str, Any]]):
    label_frequency = {
        'entailment': 0,
        'neutral': 0,
        'contradiction': 0
    }
    max_test_num = 400

    data_list_train: List[Dict[str, Any]] = []
    data_list_test: List[Dict[str, Any]] = []

    global_train_num = 0
    global_test_num = 0

    for unit in jsondata:
        sentence1 = unit["sentence1"]
        sentence2 = unit["sentence2"]
        label = unit["label"]

        # 进入测试集
        if label_frequency[label] < max_test_num:
            label_frequency[label] += 1
            json_obj_new = {
                "number": global_test_num,
                "sentence1": sentence1,
                "sentence2": sentence2,
                "label": label
            }
            global_test_num += 1
            data_list_test.append(json_obj_new)
        else:  # 进入训练集
            json_obj_new = {
                "number": global_train_num,
                "sentence1": sentence1,
                "sentence2": sentence2,
                "label": label
            }
            global_train_num += 1
            data_list_train.append(json_obj_new)

    return data_list_train, data_list_test


# 文件保存
def save_to_json(json_data: List[Dict[str, str]], output_path: str) -> None:
    try:
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, ensure_ascii=False, indent=2)
            print(f"Json saved to {output_path}")
    except Exception as e:
        print(f"Error saving Json: {e}")


if __name__ == "__main__":
    # 默认文件路径
    input_file = 'D:\llmDataSet\OCNLI\OCNLI-origin\data\ocnli\\train.30k.json'
    train_file = 'D:\llmDataSet\OCNLI\OCNLI-after-process\ocnli_train.json'
    test_file = 'D:\llmDataSet\OCNLI\OCNLI-after-process\ocnli_test.json'

    # 将原始文件转成json模式
    json_data = convert_text_to_json(input_file)

    # 从json中抽取1200个测试集，和余下作为训练集
    json_data_train, json_data_test = filter_json(json_data)

    # 文件存储
    save_to_json(json_data_train, train_file)
    save_to_json(json_data_test, test_file)
