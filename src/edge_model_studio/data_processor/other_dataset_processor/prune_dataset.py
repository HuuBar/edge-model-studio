import json

import pandas as pd
from datasets import load_dataset

train_json_path = "/datasets/total_dataset/total_train.json"
train_parquet_path = "/datasets/prune_parquet/prune_train.parquet"
train_csv_path = "/datasets/prune_parquet/train_csv.csv"
test_json_path = "/datasets/total_dataset/total_test.json"
test_parquet_path = "/datasets/prune_parquet/prune_test.parquet"
test_csv_path = "/datasets/prune_parquet/test_csv.csv"


def test_load_dataset() -> None:
    train_data = load_dataset(
        'parquet',
        data_files=train_parquet_path)
    # split='train')
    test_data = load_dataset(
        'parquet',
        data_files=test_parquet_path)
    # split='test')
    print(train_data)
    print(test_data)

    # 打印validation中的前10个
    split = test_data[list(test_data.keys())[0]]
    # 打印数据集基本信息
    print(f"数据集结构: {split}")
    # 显示前10个样本
    for i in range(min(10, len(split))):
        print(f"\n样本 {i + 1}:")
        for key, value in split[i].items():
            print(f"  {key}: {value}")


def test_parquet2csv() -> None:
    df = pd.read_parquet(train_parquet_path)
    df.to_csv(train_csv_path, index=False)
    print(f"成功将 {train_parquet_path} 转换为 {train_csv_path}")

    df = pd.read_parquet(test_parquet_path)
    df.to_csv(test_csv_path, index=False)
    print(f"成功将 {test_parquet_path} 转换为 {test_csv_path}")


# 将训练/测试集转成.qu
def list_to_parquet():
    # 训练集使用等距存储约800个
    with open(train_json_path, 'r', encoding='utf-8') as file:
        # 读取JSON内容
        content = file.read()
        data = json.loads(content)
        total = len(data)
        sample_size = 800
        step = (total - 1) / (sample_size - 1)  # 计算抽样间隔
        indices = [int(round(i * step)) for i in range(sample_size)]  # 生成均匀分布的索引
        train_list = [data[i]['context'] for i in indices]  # 提取context值

        # 存储
        df = pd.DataFrame({'text': train_list})
        df.to_parquet(train_parquet_path, engine='pyarrow')

        print(f"训练集存储成功：{train_parquet_path}")

        train_data = load_dataset('parquet', data_files=train_parquet_path)
        print(f"训练集存储结果: {train_data}")

    # 测试集等距存储约400个
    test_json_path = "/datasets/total_dataset/total_test.json"
    test_parquet_path = "/datasets/prune_parquet/prune_test.parquet"

    with open(test_json_path, 'r', encoding='utf-8') as file:
        # 读取JSON内容
        content = file.read()
        data = json.loads(content)
        total = len(data)
        sample_size = 400
        step = (total - 1) / (sample_size - 1)  # 计算抽样间隔
        indices = [int(round(i * step)) for i in range(sample_size)]  # 生成均匀分布的索引
        test_list = [data[i]['context'] for i in indices]  # 提取context值

        # 存储
        df = pd.DataFrame({'text': test_list})
        df.to_parquet(test_parquet_path, engine='pyarrow')

        print(f"测试集存储成功：{test_parquet_path}")

        test_data = load_dataset('parquet', data_files=test_parquet_path)
        print(f"训练集存储结果: {test_data}")


if __name__ == "__main__":
    list_to_parquet()
    test_load_dataset()
    test_parquet2csv()
