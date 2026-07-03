import logging
import time
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from mem0.utils.factory import EmbedderFactory, LlmFactory

logger = logging.getLogger(__name__)

class VectorSemanticMemory:
    """
    基于传统向量数据库的语义短记/事实记忆管理器 (Vector Semantic Memory)
    """
    def __init__(self, config):
        self.config = config
        self.threshold = 0.7
        
        # 建立向量数据库连接客户端 (以假定的通用抽象 Vector Client 为例)
        self._setup_vector_store()
        # 初始化与图记忆一致的 Embedding 模型与 LLM
        self._initialize_embedding_model()
        self._configure_llm()

    def _setup_vector_store(self):
        """初始化向量存储后端连接"""
        # 实际生产中可自适应对接 Chroma, Milvus, Qdrant 等
        from mem0.vector_stores.chroma import ChromaStore
        self.vector_store = ChromaStore(
            collection_name="semantic_memory",
            path=self.config.vector_store.config.get("path", "./chroma_db")
        )

    def _initialize_embedding_model(self):
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            {"enable_embeddings": True},
        )

    def _configure_llm(self):
        self.llm_provider = self.config.llm.provider or "openai_structured"
        self.llm = LlmFactory.create(self.llm_provider, self.config.llm.config)

    def add(self, data: str, filters: Dict[str, Any]) -> List[str]:
        """
        提取文本中的 Facts 并注入向量空间。
        """
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id")

        # 1. 驱动 LLM 提取扁平的事实陈述句 (Extract Facts)
        extracted_facts = self._extract_facts_from_text(data, user_id)
        if not extracted_facts:
            return []

        inserted_facts = []
        for fact in extracted_facts:
            # 2. 生成嵌入向量
            embedding = self.embedding_model.embed(fact)
            
            payload = {
                "text": fact,
                "user_id": user_id,
                "agent_id": agent_id,
                "created_at": int(time.time())
            }
            # 3. 写入向量库
            self.vector_store.insert(
                vector=embedding,
                payload=payload
            )
            inserted_facts.append(fact)
            
        return inserted_facts

    def search(self, query: str, filters: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """
        向量检索 + BM25 文本重排 (保持与你的 Mem0Graph 核心算法链路绝对一致)
        """
        query_embedding = self.embedding_model.embed(query)
        
        # 1. 向量一阶段召回
        raw_results = self.vector_store.search(
            vector=query_embedding,
            filters=filters,
            limit=limit
        )
        
        if not raw_results:
            return []

        # 2. 准备二阶段重排的数据形态
        sequences = [item["payload"]["text"] for item in raw_results]
        bm25 = BM25Okapi([seq.split(" ") for seq in sequences])

        tokenized_query = query.split(" ")
        reranked_texts = bm25.get_top_n(tokenized_query, sequences, n=min(5, len(sequences)))

        search_results = []
        for text in reranked_texts:
            search_results.append({
                "fact": text,
                "type": "semantic_fact"
            })

        logger.info(f"Vector search returned {len(search_results)} reranked results")
        return search_results

    def _extract_facts_from_text(self, text: str, user_id: str) -> List[str]:
        """通过大模型精炼文本事实"""
        prompt = f"You are an advanced memory engine. Extract core, evergreen facts about user {user_id} from the text. Return them as a flat JSON list of strings."
        response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]
        )
        try:
            # 兼容 structured 或普通文本解析逻辑
            if isinstance(response, dict) and "facts" in response:
                return response["facts"]
            return json.loads(response.strip())
        except Exception:
            return [text]

    def delete_all(self, filters: Dict[str, Any]):
        """根据过滤条件清除向量空间相关事实"""
        self.vector_store.delete(filters=filters)