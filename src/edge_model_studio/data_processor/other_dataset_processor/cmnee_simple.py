import json
from typing import List, Dict


# 读取json文件
def read_json_data(json_file: str):
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 处理数据
    result = []
    for item in data:
        number = item['number']
        context = item['context']
        keywords = [keyword['text'] for keyword in item['keywords']]
        keywords = ",".join(keywords)
        result_item = {
            "number": number,
            "context": context,
            "keywords": keywords
        }
        result.append(result_item)
    return result


# 文件保存
def save_to_json(json_data: List[Dict[str, str]], output_path: str) -> None:
    try:
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, ensure_ascii=False, indent=2)
            print(f"Json saved to {output_path}")
    except Exception as e:
        print(f"Error saving Json: {e}")


if __name__ == "__main__":
    # 文件读取
    train_file = "/datasets/cmnee/cmnee_train.json"
    test_file = "/datasets/cmnee/cmnee_test.json"

    # 文件保存
    train_file_storage = "/datasets/cmnee/cmnee_train_simple.json"
    test_file_storage = "/datasets/cmnee/cmnee_test_simple.json"

    # 将原始文件转成json模式
    json_data_train = read_json_data(train_file)
    json_data_test = read_json_data(test_file)

    # 文件存储
    save_to_json(json_data_train, train_file_storage)
    save_to_json(json_data_test, test_file_storage)




