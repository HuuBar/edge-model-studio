import json
from typing import List, Dict, Any

import opencc
import pandas as pd


# 将繁体字转换为简体字进行输出
def traditional_to_simplified(text: str) -> str:
    converter = opencc.OpenCC('t2s.json')
    return converter.convert(text)


# 读取csv文件，并将其转换为简体中文
def convert_csv_to_json(input_csv_path: str) -> List[Dict[str, Any]]:
    try:
        df = pd.read_csv(input_csv_path)

        # 检查列是否存在
        if "text" not in df.columns:
            raise ValueError(f"CSV文件中不存在名为 'text' 的列")
        if "emotion" not in df.columns:
            raise ValueError(f"CSV文件中不存在名为 'emotion' 的列")

        # 转换繁体字为简体字
        df["text"] = df["text"].apply(traditional_to_simplified)
        df["emotion"] = df["emotion"].apply(traditional_to_simplified)

        # 输出每一种emotion的类型
        emotion_frequency = {}
        for emotion in df["emotion"]:
            if emotion in emotion_frequency:
                emotion_frequency[emotion] += 1
            else:
                emotion_frequency[emotion] = 1
        print("本项目中，每种感情的标签数量：")
        for emotion, count in emotion_frequency.items():
            print(f"'{emotion}': {count}")

        # 构建JSON数据
        json_data = []
        for _, row in df.iterrows():
            json_entry = {
                "text": row["text"],
                "emotion": row["emotion"][:2]
            }
            json_data.append(json_entry)

        return json_data

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return []


# 将所有数据，拆分成训练集和测试集
def filter_json(jsondata: List[Dict[str, Any]]):
    label_frequency = {
        '平淡': 0,
        '开心': 0,
        '悲伤': 0,
        '愤怒': 0,
        '惊奇': 0,
        '厌恶': 0,
        '疑问': 0,
        '关切': 0
    }
    max_test_num = 30

    data_list_train: List[Dict[str, Any]] = []
    data_list_test: List[Dict[str, Any]] = []

    global_train_num = 0
    global_test_num = 0

    for unit in jsondata:
        text = unit["text"]
        emotion = unit["emotion"]

        # 进入测试集
        if label_frequency[emotion] < max_test_num:
            label_frequency[emotion] += 1
            json_obj_new = {
                "number": global_test_num,
                "text": text,
                "emotion": emotion
            }
            global_test_num += 1
            data_list_test.append(json_obj_new)
        else:  # 进入训练集
            json_obj_new = {
                "number": global_train_num,
                "text": text,
                "emotion": emotion
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
    input_file = 'D:\llmDataSet\chinese_multi_emotion_dialogue_dataset\chinese_multi_emotion_dialogue_dataset_origin\data.csv'
    train_file = 'D:\llmDataSet\chinese_multi_emotion_dialogue_dataset\chinese_multi_emotion_dialogue_dataset_after_process\multi_emotion_train.json'
    test_file = 'D:\llmDataSet\chinese_multi_emotion_dialogue_dataset\chinese_multi_emotion_dialogue_dataset_after_process\multi_emotion_test.json'

    # 解析csv到json格式，并且转换为简体中文
    json_data = convert_csv_to_json(input_file)

    # 从json中抽取1200个测试集，和余下作为训练集
    json_data_train, json_data_test = filter_json(json_data)

    # 文件存储
    save_to_json(json_data_train, train_file)
    save_to_json(json_data_test, test_file)




