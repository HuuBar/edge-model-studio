import json

# 读取JSON数据
with open('D:\llmDataSet\cmnee\cmnee-origin\\train.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 用于存储最终结果的列表
results = []

# 遍历每个文档
for doc in data:
    context = doc['text']
    keywords = []
    seen_texts = set()

    # 遍历事件列表
    for event in doc['event_list']:
        trigger = event['trigger']
        trigger_text = trigger['text']
        if trigger_text not in seen_texts:
            keywords.append({
                "text": trigger_text,
                "offset": trigger['offset']
            })
            seen_texts.add(trigger_text)

        # 遍历事件参数
        for arg in event['arguments']:
            arg_text = arg['text']
            if arg_text not in seen_texts:
                keywords.append({
                    "text": arg_text,
                    "offset": arg['offset']
                })
                seen_texts.add(arg_text)

    # 遍历共指参数
    for coref_group in doc['coref_arguments']:
        for coref_arg in coref_group:
            coref_text = coref_arg['text']
            if coref_text not in seen_texts:
                keywords.append({
                    "text": coref_text,
                    "offset": coref_arg['offset']
                })
                seen_texts.add(coref_text)

    result = {
        "context": context,
        "keywords": keywords
    }
    results.append(result)

# 训练集保存
train_file = "D:\llmDataSet\cmnee\cmnee-after-process\cmnee_train.json"
test_file = "D:\llmDataSet\cmnee\cmnee-after-process\cmnee_test.json"
train_data = []
test_data = []
train_num = 0
test_num = 0
for unit in results:
    context = unit["context"]

    # 暂时去除较长的值
    if len(context) > 1000:
        continue

    keywords = unit["keywords"]
    if test_num < 1000:
        json_unit = {
            "number": test_num,
            "context": context,
            "keywords": keywords
        }
        test_data.append(json_unit)
        test_num += 1
    else:
        json_unit = {
            "number": train_num,
            "context": context,
            "keywords": keywords
        }
        train_data.append(json_unit)
        train_num += 1

try:
    with open(train_file, 'w', encoding='utf-8') as json_file:
        json.dump(train_data, json_file, ensure_ascii=False, indent=2)
        print(f"Json saved to {train_file}")
except Exception as e:
    print(f"Error saving Json: {e}")

try:
    with open(test_file, 'w', encoding='utf-8') as json_file:
        json.dump(test_data, json_file, ensure_ascii=False, indent=2)
        print(f"Json saved to {test_file}")
except Exception as e:
    print(f"Error saving Json: {e}")
