from typing import Annotated, TypedDict, List
#from langchain_core.messages import HumanMessage, SystemMessage
#from langchain_openai import ChatOpenAI
from mem0 import Memory
#from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from openai import OpenAI
from openai.types.chat import ChatCompletion
import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_models import ChatOpenAI
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = {
    "llm": {
        "provider": "openai",  # 使用 OpenAI 提供的模型
        "config": {
            "model": "gpt-4o-mini",  # 模型名称
        }
    },
    "vector_store": {
        "provider": "chroma",       # 使用 Chroma 作为向量存储
        "config": {
            "collection_name": "test",  # 数据集合名
            "path": "db",               # Chroma 数据库存储路径
        }
    }
}


llm = ChatOpenAI(
    base_url="http://10.44.161.72:8090/v1",  # 本地 vLLM 服务地址
    api_key="EMPTY",                         # vLLM 默认不需要 API 鉴权
    model="Qwen3-8B",                        # 模型名称
)

mem0 = Memory().from_config(config)

class State(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    mem0_user_id: str

def chatbot(state: State):
    messages = state["messages"]
    user_id = state["mem0_user_id"]
    memoriesHuge = mem0.search(messages[-1].content, user_id=user_id)
    memories = memoriesHuge['results']
    context = "Relevant information from previous conversations:\n"
    for memory in memories:
        context += f"- {memory['memory']}\n"
    system_message = SystemMessage(content=f"""
    You are a helpful customer support assistant. 
    Use the provided context to personalize your responses and remember user preferences and past interactions.
    Answer the following question in one concise sentence only, no elaboration or explanation.{context}./no_think""")
    full_messages = [system_message] + messages
    response = llm.invoke(full_messages)
    # Store the interaction in Mem0
    add_messages = [{"role": "user", "content": f"{messages[-1].content}\nAssistant: {response.content}"}]
    a = mem0.add(add_messages, user_id=user_id)
    return {"messages": [response]}

graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", "chatbot")

compiled_graph = graph.compile()

def run_conversation(user_input: str, mem0_user_id: str):
    config = {"configurable": {"thread_id": mem0_user_id}}
    state = {"messages": [HumanMessage(content=user_input)], "mem0_user_id": mem0_user_id}
    for event in compiled_graph.stream(state, config):
        for value in event.values():
            if value.get("messages"):
                return value["messages"][-1].content

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_dialogues(json_data):
    """
    从JSON数据中提取所有会话中的说话者和对话内容
    参数:
        json_data: 包含对话数据的JSON对象
    返回:
        dict: {说话者: [对话内容列表]}的字典
    """
    dialogues = {}
    
    for item in json_data:
        if "conversation" in item:
            conv = item["conversation"]
            # 遍历所有以session_开头的键
            for key in conv:
                if key.startswith("session_"):
                    if len(key)>12:
                        data_time = conv[key]
                        dialogues[data_time] = []
                        dialogues[data_time] .append("The current time")
                    else:
                        for utterance in conv[key]:
                            speaker = utterance.get("speaker")
                            text = utterance.get("text")
                            if speaker and text:
                                if text not in dialogues:
                                    dialogues[text] = []
                                    dialogues[text].append(speaker)
    return dialogues

def extract_qa_pairs(json_data):
    """
    从嵌套JSON结构中提取问题-答案键值对
    参数:
        json_data (list/dict): 包含QA数据的JSON对象
    返回:
        dict: 问题为键、答案为值的字典
    """
    qa_dict = {}
    # 处理列表形式的顶层结构
    if isinstance(json_data, list):
        for item in json_data:
            if "qa" in item and isinstance(item["qa"], list):
                for qa_item in item["qa"]:
                    if "question" in qa_item and "answer" in qa_item:
                        qa_dict[qa_item["question"]] = qa_item["answer"]
                    elif "question" in qa_item and "adversarial_answer" in qa_item:
                        qa_dict[qa_item["question"]] = qa_item["adversarial_answer"]
                    elif "question" in qa_item and "answer" not in qa_item:
                        qa_dict[qa_item["question"]] = "None"

    return qa_dict
if __name__ == "__main__":
    print("Welcome to Customer Support! How can I assist you today?")    
    jsonData = read_json_file("/home/l00495039/data/cut_speak_1.json")
    mem0_user_id = "c6"  # You can generate or retrieve this based on your user management system
    
    talk_dict = extract_dialogues(jsonData)
    qa_dict = extract_qa_pairs(jsonData)
    result_dict_all=[]
    for key in qa_dict:
        result_dict = {}
        print(key)
        llm_answer = run_conversation(key, mem0_user_id)
        print(llm_answer)
        result_dict["question"] = key
        result_dict["origin_answer"] = qa_dict[key]
        result_dict["llm_answer"] = llm_answer
        result_dict_all.append(result_dict)
    
    with open('/home/l00495039/code/result_test.json', 'w') as f:
        json.dump(result_dict_all, f, indent=2)
    
