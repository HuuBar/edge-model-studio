import json
from typing import List, Dict

import pandas as pd


# 读取csv文件
def convert_csv_to_json(input_csv_path: str):
    # 尝试使用更全面的中文编码
    try:
        # 优先尝试 GB18030（覆盖所有中文字符）
        df = pd.read_csv(input_csv_path, encoding='gb18030')
    except UnicodeDecodeError:
        # 尝试 GBK（比 GB2312 更全面）
        df = pd.read_csv(input_csv_path, encoding='gbk')
    except UnicodeDecodeError:
        # 尝试 Windows 中文默认编码
        df = pd.read_csv(input_csv_path, encoding='cp936')

    # 检查列是否存在
    if "ask" not in df.columns:
        raise ValueError(f"CSV文件中不存在名为 'ask' 的列")
    if "answer" not in df.columns:
        raise ValueError(f"CSV文件中不存在名为 'answer' 的列")
    if "department" not in df.columns:
        raise ValueError(f"CSV文件中不存在名为 'department' 的列")

    # 构建JSON数据
    json_data_train = []
    json_data_test = []
    train_number = 0
    test_number = 0
    for _, row in df.iterrows():
        ask = row["ask"].replace('\n', '').replace('\r', '').replace(' ', '')
        answer = row["answer"].replace('\n', '').replace('\r', '').replace(' ', '')

        # 筛除过于长的数据
        if len(ask) > 500 or len(ask) < 100 or len(answer) > 300 or len(answer) < 100:
            continue

        if test_number < 1000:
            json_entry = {
                "number": test_number,
                "ask": ask,
                "answer": answer
            }
            json_data_test.append(json_entry)
            test_number += 1
        else:
            json_entry = {
                "number": train_number,
                "ask": ask,
                "answer": answer
            }
            json_data_train.append(json_entry)
            train_number += 1
    return json_data_train, json_data_test


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
    input_file = 'D:\llmDataSet\Chinese-medical-dialogue-data-master\Chinese-medical-dialogue-data-origin\Data_数据\Pediatric_儿科\儿科5-14000.csv'
    train_file = 'D:\llmDataSet\Chinese-medical-dialogue-data-master\chinese-medical-dialogue-data-after-process\medical_dialogue_train.json'
    test_file = 'D:\llmDataSet\Chinese-medical-dialogue-data-master\chinese-medical-dialogue-data-after-process\medical_dialogue_test.json'

    # 解析csv到json格式
    json_data_train, json_data_test = convert_csv_to_json(input_file)

    # 文件存储
    save_to_json(json_data_train, train_file)
    save_to_json(json_data_test, test_file)
