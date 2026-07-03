import json
import os
from typing import List, Tuple, Dict


# 原始数据是.src和.tgt用来表示原始数据和结果，需要成对匹配
def get_file_pairs(dataset_path: str) -> List[Tuple[str, str]]:
    file_pairs = []
    src_files = set()

    # 收集所有.src文件
    for filename in os.listdir(dataset_path):
        if filename.endswith('.src'):
            base_name = filename[:-4]
            src_files.add(base_name)

    # 分别匹配.tgt文件
    for base_name in src_files:
        tgt_file = f"{base_name}.tgt"
        if os.path.exists(os.path.join(dataset_path, tgt_file)):
            file_pairs.append((
                os.path.join(dataset_path, f"{base_name}.src"),
                os.path.join(dataset_path, tgt_file)
            ))

    return file_pairs


# 将数据转换成特定结构
def create_json_data(file_pairs: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    json_data = []
    global_index = 0
    for src_path, tgt_path in file_pairs:
        with open(src_path, 'r', encoding='utf-8') as file1, open(tgt_path, 'r', encoding='utf-8') as file2:
            lines1 = file1.read().splitlines()
            lines2 = file2.read().splitlines()

            # 检查两个文件行数是否相同
            # if len(lines2) != len(lines1):
            #     raise ValueError(f"两个文件的行数不一致:{src_path}有{len(lines1)}行，{tgt_path}有{len(lines2)}行")

            # 组合对应元素成为json
            # zip将两个line打包成一个元组
            # enumerate将一个可迭代组合转成 编号+元素 的格式
            for line1, line2 in zip(lines1, lines2):
                line1 = line1.replace('\n', '').replace('\r', '').replace(' ', '')
                line2 = line2.replace('\n', '').replace('\r', '').replace(' ', '')
                json_data.append({
                    "index": global_index,
                    "context": line1,
                    "ground_truth": line2
                })
                global_index += 1
    return json_data


# 文件保存
def save_to_json(json_data: List[Dict[str, str]], output_path: str) -> None:
    try:
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, ensure_ascii=False, indent=2)
            print(f"Json saved to {output_path}")
    except Exception as e:
        print(f"Error saving Json: {e}")


# 按照一些要求过滤数据集中的数据，只保留满足要求的部分数据
# 前1000个作为测试集，之后为训练集
def filter_json(json_data: List[Dict[str, str]]):
    json_data_train = []
    json_data_test = []
    train_index = 0
    test_index = 0
    for unit in json_data:
        context = unit["context"]
        ground_truth = unit["ground_truth"]

        # 过滤超出取值范围的实例
        if len(context) >= 1000 or len(ground_truth) <= 77:
            continue

        if test_index < 1000:
            json_data_test.append({
                "number": test_index,
                "context": context,
                "ground_truth": ground_truth
            })
            test_index += 1
        else:
            json_data_train.append({
                "number": train_index,
                "context": context,
                "ground_truth": ground_truth
            })
            train_index += 1
    return json_data_train, json_data_test


def main():
    dataset_path = "D:\llmDataSet\pre-CLTs\clts-origin"
    output_path = "D:\Code\school_service_chendong\datasets\clts_datasets\clts.json"
    filter_train = "D:\Code\school_service_chendong\datasets\clts_datasets\clts_filter_train.json"
    filter_test = "D:\Code\school_service_chendong\datasets\clts_datasets\clts_filter_test.json"

    if not os.path.isdir(dataset_path):
        print(f"错误，数据集路径不存在: {dataset_path}")

    file_pairs = get_file_pairs(dataset_path)
    print(f"找到 {len(file_pairs)} 对文件")

    if file_pairs:
        json_data = create_json_data(file_pairs)
        save_to_json(json_data, output_path)

        json_data_train, json_data_test = filter_json(json_data)
        save_to_json(json_data_train, filter_train)
        save_to_json(json_data_test, filter_test)
    print("dataset save successfully!")


if __name__ == "__main__":
    main()
