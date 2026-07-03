import json

# 将5种数据集合并到一起
clts_prompt = "你是一位专业新闻编辑，精通文章写作与摘要技巧，能够准确提炼关键信息，不得虚构、猜测或省略原文中的重要数据，确保摘要真实、客观、精炼。请基于上述内容，生成一段不超过70字的摘要。"
cmnee_prompt = "你是一位专业文本分析师，擅长从文章中提取关键词。关键词应该是文章中的核心概念、重要事件或关键人物，不需要包含普通词汇或无关信息。请确保关键词准确且不超过 10 个。"
medicial_dialogue_prompt = "你是一位经验丰富的儿科医生，擅长根据患儿病情描述给出专业、清晰、有同理心的医学建议。不超过200字."
multi_emotion_prompt = "你是一位资深情感分析专家，擅长判断用户语句中所表达的主要情绪。情绪类别：「平淡」「开心」「悲伤」「愤怒」「惊奇」「厌恶」「疑问」「关切」。输出情绪类别，不要包含任何其他文字或符号"
ocnli_prompt = "你是一位逻辑推理专家，擅长判断两个句子之间的逻辑关系。从「entailment」「contradiction」「neutral」选择一个输出，不需要额外输出任何其他信息。"


def merge_datasets(clts_test, cmnee_test, medicial_dialogue_test, multi_emotion_test, ocnli_test):
    global_num = 0
    json_data = []

    # 打开并读取 JSON 文件
    with open(clts_test, 'r', encoding='utf-8') as file:
        clts_json = json.load(file)
        for clts_unit in clts_json:
            json_data.append({
                "number": global_num,
                "dataset_base": "clts_datasets",
                "dataset_base_number": clts_unit["number"],
                "prompt": clts_prompt,
                "context": clts_unit["context"],
                "output": clts_unit["ground_truth"]
            })
            global_num += 1

    with open(cmnee_test, 'r', encoding='utf-8') as file:
        cmnee_json = json.load(file)
        for cmnee_unit in cmnee_json:
            json_data.append({
                "number": global_num,
                "dataset_base": "cmnee_datasets",
                "dataset_base_number": cmnee_unit["number"],
                "prompt": cmnee_prompt,
                "context": cmnee_unit["context"],
                "output": cmnee_unit["keywords"]
            })
            global_num += 1

    with open(medicial_dialogue_test, 'r', encoding='utf-8') as file:
        medicial_dialogue_json = json.load(file)
        for medicial_dialogue_unit in medicial_dialogue_json:
            json_data.append({
                "number": global_num,
                "dataset_base": "medicial_dialogue_datasets",
                "dataset_base_number": medicial_dialogue_unit["number"],
                "prompt": medicial_dialogue_prompt,
                "context": medicial_dialogue_unit["ask"],
                "output": medicial_dialogue_unit["answer"]
            })
            global_num += 1

    with open(multi_emotion_test, 'r', encoding='utf-8') as file:
        multi_emotion_json = json.load(file)
        for multi_emotion_unit in multi_emotion_json:
            json_data.append({
                "number": global_num,
                "dataset_base": "multi_emotion_datasets",
                "dataset_base_number": multi_emotion_unit["number"],
                "prompt": multi_emotion_prompt,
                "context": multi_emotion_unit["text"],
                "output": multi_emotion_unit["emotion"]
            })
            global_num += 1

    with open(ocnli_test, 'r', encoding='utf-8') as file:
        ocnli_json = json.load(file)
        for ocnli_unit in ocnli_json:
            json_data.append({
                "number": global_num,
                "dataset_base": "ocnli_datasets",
                "dataset_base_number": ocnli_unit["number"],
                "prompt": ocnli_prompt,
                "context": f"sentence1:{ocnli_unit['sentence1']},sentence2:{ocnli_unit['sentence2']}",
                "output": ocnli_unit["label"]
            })
            global_num += 1

    return json_data


def save_to_json(json_data, output_path: str) -> None:
    try:
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, ensure_ascii=False, indent=2)
            print(f"Json saved to {output_path}")
    except Exception as e:
        print(f"Error saving Json: {e}")


if __name__ == "__main__":
    # all test datasets
    clts_test = "D:\Code\school_service_chendong\datasets\clts_datasets\clts_filter_test.json"
    cmnee_test = "D:\Code\school_service_chendong\datasets\cmnee\cmnee_test_simple.json"
    medicial_dialogue_test = "D:\Code\school_service_chendong\datasets\medicial_dialogue\medical_dialogue_test.json"
    multi_emotion_test = "D:\Code\school_service_chendong\datasets\multi_emotion\multi_emotion_test.json"
    ocnli_test = "D:\Code\school_service_chendong\datasets\ocnli\ocnli_test.json"

    total_test_file = "/datasets/total_dataset/total_test.json"
    json_data_test = merge_datasets(clts_test, cmnee_test, medicial_dialogue_test, multi_emotion_test, ocnli_test)
    save_to_json(json_data_test, total_test_file)

    # all train datasets
    clts_train = "D:\Code\school_service_chendong\datasets\clts_datasets\clts_filter_train.json"
    cmnee_train = "D:\Code\school_service_chendong\datasets\cmnee\cmnee_train_simple.json"
    medicial_dialogue_train = "D:\Code\school_service_chendong\datasets\medicial_dialogue\medical_dialogue_train.json"
    multi_emotion_train = "D:\Code\school_service_chendong\datasets\multi_emotion\multi_emotion_train.json"
    ocnli_train = "D:\Code\school_service_chendong\datasets\ocnli\ocnli_train.json"

    total_train_file = "/datasets/total_dataset/total_train.json"
    json_data_train = merge_datasets(clts_train, cmnee_train, medicial_dialogue_train, multi_emotion_train, ocnli_train)
    save_to_json(json_data_train, total_train_file)




