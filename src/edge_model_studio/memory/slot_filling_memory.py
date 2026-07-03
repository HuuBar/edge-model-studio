import logging
import json
from typing import List, Dict, Any, Optional

from mem0.utils.factory import LlmFactory

logger = logging.getLogger(__name__)

class EpisodicKeyValueMemory:
    """
    基于特定事实键值的槽位情节记忆管理器 (Episodic Key-Value Memory)
    """
    def __init__(self, config):
        self.config = config
        self._setup_kv_store()
        self._configure_llm()

    def _setup_kv_store(self):
        """配置常驻持久化 KV 存储（如连接 Redis 或者 MongoDB）"""
        from mem0.kv_stores.redis import RedisClient
        self.kv_client = RedisClient(
            host=self.config.kv_store.config.get("host", "localhost"),
            port=self.config.kv_store.config.get("port", 6379),
            password=self.config.kv_store.config.get("password", None)
        )

    def _configure_llm(self):
        self.llm_provider = self.config.llm.provider or "openai_structured"
        self.llm = LlmFactory.create(self.llm_provider, self.config.llm.config)

    def add(self, data: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        通过大模型提取有价值的 K-V 对，对已存在的 Slot 进行增量 Merge 与覆写
        """
        user_id = filters["user_id"]
        # 获取当前已存储的 Profile 全貌，方便进行上下文内冲突排查
        current_profile = self.get_all(filters)

        # 1. 大模型介入做多级 Slot-Filling 的增量合并决策
        system_prompt = (
            f"You are a structured profile sync engine. Current User Profile is: {json.dumps(current_profile)}.\n"
            f"Analyze the new dialog text, identify key-value pairs that need to be updated or added. "
            f"Output the complete merged new profile in strict JSON format."
        )
        
        response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data}
            ]
        )
        
        try:
            updated_profile = json.loads(response) if isinstance(response, str) else response
            # 2. 持久化落盘覆盖原冷数据
            db_key = f"profile:{user_id}"
            self.kv_client.set(db_key, json.dumps(updated_profile))
            return updated_profile
        except Exception as e:
            logger.error(f"Failed to merge key-value profile slots: {e}")
            return current_profile

    def get_all(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """完全精确的主键 K-V 定向召回"""
        user_id = filters["user_id"]
        db_key = f"profile:{user_id}"
        raw_data = self.kv_client.get(db_key)
        if raw_data:
            return json.loads(raw_data)
        return {}

    def search(self, query: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        K-V 的检索。先将全量结构化 Profile 读取出来，
        由大模型过滤筛选其中与给定 query 相关的关键键值槽返回。
        """
        current_profile = self.get_all(filters)
        if not current_profile:
            return []

        system_prompt = "Filter the provided profile dictionary. Return ONLY the key-value items relevant to answering the user's query as a JSON subset."
        response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\nProfile: {json.dumps(current_profile)}"}
            ]
        )
        try:
            filtered_slots = json.loads(response) if isinstance(response, str) else response
            return [{"key": k, "value": v} for k, v in filtered_slots.items()]
        except Exception:
            return []

    def delete_all(self, filters: Dict[str, Any]):
        """根据过滤条件物理抹除该 K-V 槽"""
        user_id = filters["user_id"]
        db_key = f"profile:{user_id}"
        self.kv_client.delete(db_key)