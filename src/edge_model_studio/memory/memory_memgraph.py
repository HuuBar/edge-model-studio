# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import logging

from mem0.memory.utils import format_entities
from langchain_memgraph import Memgraph
from rank_bm25 import BM25Okapi

from mem0.graphs.tools import (
    DELETE_MEMORY_STRUCT_TOOL_GRAPH,
    DELETE_MEMORY_TOOL_GRAPH,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    EXTRACT_ENTITIES_TOOL,
    RELATIONS_STRUCT_TOOL,
    RELATIONS_TOOL,
)
from mem0.graphs.utils import EXTRACT_RELATIONS_PROMPT, get_delete_messages
from mem0.utils.factory import EmbedderFactory, LlmFactory

logger = logging.getLogger(__name__)

class Mem0Graph:

    def __init__(self, config):
        # 初始化配置对象
        self.config = config
        self.user_id = None
        self.threshold = 0.7
        
        # 建立图数据库连接
        self._setup_graph_connection()
        
        # 初始化嵌入模型
        self._initialize_embedding_model()
        
        # 配置语言模型
        self._configure_llm()
        
        # 创建数据库索引
        self._create_database_indexes()
    
    def _setup_graph_connection(self):
        """建立Memgraph数据库连接"""
        self.graph = Memgraph(
            self.config.graph_store.config.url,
            self.config.graph_store.config.username,
            self.config.graph_store.config.password,
        )
    
    def _initialize_embedding_model(self):
        """初始化嵌入模型"""
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            {"enable_embeddings": True},
        )
    
    def _configure_llm(self):
        """配置语言模型"""
        self.llm_provider = "openai_structured"
        
        if self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
            
        if self.config.graph_store.llm:
            self.llm_provider = self.config.graph_store.llm.provider
            
        self.llm = LlmFactory.create(self.llm_provider, self.config.llm.config)
    
    def _create_database_indexes(self):
        """创建数据库索引以优化查询性能"""
        embedding_dims = self.config.embedder.config["embedding_dims"]
        
        # 创建向量索引用于相似性搜索
        vector_index_query = f'''CREATE VECTOR INDEX memzero ON :
        Entity(embedding) WITH CONFIG {{'dimension': {embedding_dims}, 'capacity': 1000, 'metric': 'cos'}}
        '''
        self.graph.query(vector_index_query, params={})
        
        # 创建标签属性索引优化用户查询
        label_prop_index_query = "CREATE INDEX ON :Entity(user_id);"
        self.graph.query(label_prop_index_query, params={})
        
        # 创建标签索引优化节点查询
        label_index_query = "CREATE INDEX ON :Entity;"
        self.graph.query(label_index_query, params={})

    def delete_all(self, filters):
        # 删除用户或特定代理的所有节点和关系
        if filters.get("agent_id"):
            cypher = """
            MATCH (n:Entity {user_id: $user_id, agent_id: $agent_id})
            DETACH DELETE n
            """
            params = {"user_id": filters["user_id"], "agent_id": filters["agent_id"]}
        else:
            cypher = """
            MATCH (n:Entity {user_id: $user_id})
            DETACH DELETE n
            """
            params = {"user_id": filters["user_id"]}
        self.graph.query(cypher, params)


    def get_all(self, filters, limit=100):
        # 从图数据库中检索所有节点和关系，支持可选的过滤条件。  
        # filters（字典）：包含在检索过程中要应用的过滤条件的字典。  
        # limit（整数）：要检索的节点和关系的最大数量。 
        
        # Build query based on whether agent_id is provided
        if filters.get("agent_id"):
            query = """
            MATCH (n:Entity {user_id: $user_id, agent_id: $agent_id})-[r]->(m:Entity {user_id: $user_id, agent_id: $agent_id})
            RETURN n.name AS source, type(r) AS relationship, m.name AS target
            LIMIT $limit
            """
            params = {"user_id": filters["user_id"], "agent_id": filters["agent_id"], "limit": limit}
        else:
            query = """
            MATCH (n:Entity {user_id: $user_id})-[r]->(m:Entity {user_id: $user_id})
            RETURN n.name AS source, type(r) AS relationship, m.name AS target
            LIMIT $limit
            """
            params = {"user_id": filters["user_id"], "limit": limit}
        
        results = self.graph.query(query, params=params)

        final_results = []
        for result in results:
            final_results.append(
                {
                    "source": result["source"],
                    "relationship": result["relationship"],
                    "target": result["target"],
                }
            )

        return final_results

    def search(self, query, filters, limit=100):
        #相关记忆搜索
        
        entity_type_map = self._retrieve_nodes_from_data(query, filters)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)

        if not search_output:
            return []

        search_outputs_sequence = [
            [item["source"], item["relationship"], item["destination"]] for item in search_output
        ]
        bm25 = BM25Okapi(search_outputs_sequence)

        tokenized_query = query.split(" ")
        reranked_results = bm25.get_top_n(tokenized_query, search_outputs_sequence, n=5)

        search_results = []
        for item in reranked_results:
            search_results.append({"source": item[0], "relationship": item[1], "destination": item[2]})

        logger.info(f"Returned {len(search_results)} search results")

        return search_results

    def _retrieve_nodes_from_data(self, data, filters):
        #提取查询中提到的所有信息
        _tools = [EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]
        search_results = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a smart assistant who understands entities and their types in a given text. 
                    If user message contains self reference such as 'I', 'me', 'my' etc.
                     then use {filters['user_id']} as the source entity. Extract all the entities from the text. 
                     ***DO NOT*** answer the question itself if the given text is a question.",
                },
                {"role": "user", "content": data},
            ],
            tools=_tools,
        )

        entity_type_map = {}

        try:
            for tool_call in search_results["tool_calls"]:
                if tool_call["name"] != "extract_entities":
                    continue
                for item in tool_call["arguments"]["entities"]:
                    entity_type_map[item["entity"]] = item["entity_type"]
        except Exception as e:
            logger.exception(
                f"Error in search tool: {e}, llm_provider={self.llm_provider}, search_results={search_results}"
            )

        entity_type_map = {k.lower().replace(" ", "_"): v.lower().replace(" ", "_") for k, v in entity_type_map.items()}
        logger.debug(f"Entity type map: {entity_type_map}\n search_results={search_results}")
        return entity_type_map

    def _search_graph_db(self, node_list, filters, limit=100):
        #在节点之间及其各自的传入和传出关系中搜索相似的节点
        result_relations = []

        for node in node_list:
            n_embedding = self.embedding_model.embed(node)

            # Build query based on whether agent_id is provided
            if filters.get("agent_id"):
                cypher_query = """
                MATCH (n:Entity {user_id: $user_id, agent_id: $agent_id})-[r]->(m:Entity)
                WHERE n.embedding IS NOT NULL;
                """
                params = {
                    "n_embedding": n_embedding,
                    "threshold": self.threshold,
                    "user_id": filters["user_id"],
                    "agent_id": filters["agent_id"],
                    "limit": limit,
                }
            else:
                cypher_query = """
                MATCH (n:Entity {user_id: $user_id})-[r]->(m:Entity)
                WHERE n.embedding IS NOT NULL;
                """
                params = {
                    "n_embedding": n_embedding,
                    "threshold": self.threshold,
                    "user_id": filters["user_id"],
                    "limit": limit,
                }
            
            ans = self.graph.query(cypher_query, params=params)
            result_relations.extend(ans)

        return result_relations
        
    def _establish_nodes_relations_from_data(self, data, filters, entity_type_map):
        #在提取的节点之间建立关系
        if self.config.graph_store.custom_prompt:
            messages = [
                {
                    "role": "system",
                    "content": EXTRACT_RELATIONS_PROMPT.replace("USER_ID", filters["user_id"]).replace(
                        "CUSTOM_PROMPT", f"4. {self.config.graph_store.custom_prompt}"
                    ),
                },
                {"role": "user", "content": data},
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": EXTRACT_RELATIONS_PROMPT.replace("USER_ID", filters["user_id"]),
                },
                {
                    "role": "user",
                    "content": f"List of entities: {list(entity_type_map.keys())}. \n\nText: {data}",
                },
            ]

        _tools = [RELATIONS_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [RELATIONS_STRUCT_TOOL]

        extracted_entities = self.llm.generate_response(
            messages=messages,
            tools=_tools,
        )

        entities = []
        if extracted_entities["tool_calls"]:
            entities = extracted_entities["tool_calls"][0]["arguments"]["entities"]

        entities = self._remove_spaces_from_entities(entities)
        return entities

    def _get_delete_entities_from_search_output(self, search_output, data, filters):
        #从搜索结果中获取要删除的实体
        search_output_string = format_entities(search_output)
        system_prompt, user_prompt = get_delete_messages(search_output_string, data, filters["user_id"])

        _tools = [DELETE_MEMORY_TOOL_GRAPH]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [
                DELETE_MEMORY_STRUCT_TOOL_GRAPH,
            ]

        memory_updates = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=_tools,
        )
        to_be_deleted = []
        for item in memory_updates["tool_calls"]:
            if item["name"] == "delete_graph_memory":
                to_be_deleted.append(item["arguments"])
        # in case if it is not in the correct format
        to_be_deleted = self._remove_spaces_from_entities(to_be_deleted)
        logger.debug(f"Deleted relationships: {to_be_deleted}")
        return to_be_deleted

    def _delete_entities(self, to_be_deleted, filters):
        #从图中删除实体
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        results = []
        
        for item in to_be_deleted:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # Build the agent filter for the query
            agent_filter = ""
            params = {
                "source_name": source,
                "dest_name": destination,
                "user_id": user_id,
            }
            
            if agent_id:
                agent_filter = "AND n.agent_id = $agent_id AND m.agent_id = $agent_id"
                params["agent_id"] = agent_id

            # Delete the specific relationship between nodes
            cypher = f"""
            MATCH (n:Entity {{name: $source_name, user_id: $user_id}})
            -[r:{relationship}]->
            (m:Entity {{name: $dest_name, user_id: $user_id}})
            WHERE 1=1 {agent_filter}
            DELETE r
            RETURN 
                n.name AS source,
                m.name AS target,
                type(r) AS relationship
            """
            
            result = self.graph.query(cypher, params=params)
            results.append(result)
        
        return results

    def _remove_spaces_from_entities(self, entity_list):
        for item in entity_list:
            item["source"] = item["source"].lower().replace(" ", "_")
            item["relationship"] = item["relationship"].lower().replace(" ", "_")
            item["destination"] = item["destination"].lower().replace(" ", "_")
        return entity_list

    def _search_source_node(self, source_embedding, filters, threshold=0.9):
        """搜索具有相似嵌入的源节点"""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        
        if agent_id:
            cypher = """
                CALL vector_search.search("memzero", 1, $source_embedding) 
                YIELD distance, node, similarity
                WITH node AS source_candidate, similarity
                WHERE source_candidate.user_id = $user_id 
                AND source_candidate.agent_id = $agent_id 
                AND similarity >= $threshold
                RETURN id(source_candidate);
                """
            params = {
                "source_embedding": source_embedding,
                "user_id": user_id,
                "agent_id": agent_id,
                "threshold": threshold,
            }
        else:
            cypher = """
                CALL vector_search.search("memzero", 1, $source_embedding) 
                YIELD distance, node, similarity
                WITH node AS source_candidate, similarity
                WHERE source_candidate.user_id = $user_id 
                AND similarity >= $threshold
                RETURN id(source_candidate);
                """
            params = {
                "source_embedding": source_embedding,
                "user_id": user_id,
                "threshold": threshold,
            }

        result = self.graph.query(cypher, params=params)
        return result

    def _search_destination_node(self, destination_embedding, filters, threshold=0.9):
        """搜索具有相似嵌入的目标节点"""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        
        if agent_id:
            cypher = """
                CALL vector_search.search("memzero", 1, $destination_embedding) 
                YIELD distance, node, similarity
                WITH node AS destination_candidate, similarity
                WHERE node.user_id = $user_id 
                AND node.agent_id = $agent_id 
                AND similarity >= $threshold
                RETURN id(destination_candidate);
                """
            params = {
                "destination_embedding": destination_embedding,
                "user_id": user_id,
                "agent_id": agent_id,
                "threshold": threshold,
            }
        else:
            cypher = """
                CALL vector_search.search("memzero", 1, $destination_embedding) 
                YIELD distance, node, similarity
                WITH node AS destination_candidate, similarity
                WHERE node.user_id = $user_id 
                AND similarity >= $threshold
                RETURN id(destination_candidate);
                """
            params = {
                "destination_embedding": destination_embedding,
                "user_id": user_id,
                "threshold": threshold,
            }

        result = self.graph.query(cypher, params=params)
        return result

            def _add_entities(self, to_be_added, filters, entity_type_map):
        #将新实体添加到图中。如果节点已存在，则进行合并
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        results = []
        
        for item in to_be_added:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            source_type = entity_type_map.get(source, "__User__")
            destination_type = entity_type_map.get(destination, "__User__")

            source_embedding = self.embedding_model.embed(source)
            dest_embedding = self.embedding_model.embed(destination)

            source_node_search_result = self._search_source_node(source_embedding, filters, threshold=0.9)
            destination_node_search_result = self._search_destination_node(dest_embedding, filters, threshold=0.9)

            agent_id_clause = ""
            if agent_id:
                agent_id_clause = ", agent_id: $agent_id"
            
            if not destination_node_search_result and source_node_search_result:
                cypher = f"""
                    MATCH (source:Entity)
                    WHERE id(source) = $source_id
                    MERGE (destination:{destination_type}:Entity {{name: $destination_name, user_id: $user_id{agent_id_clause}}})
                    ON CREATE SET
                        destination.created = timestamp(),
                        destination.embedding = $destination_embedding,
                        destination:Entity
                    MERGE (source)-[r:{relationship}]->(destination)
                    ON CREATE SET 
                        r.created = timestamp()
                    RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                    """

                params = {
                    "source_id": source_node_search_result[0]["id(source_candidate)"],
                    "destination_name": destination,
                    "destination_embedding": dest_embedding,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                
            elif destination_node_search_result and not source_node_search_result:
                cypher = f"""
                    MATCH (destination:Entity)
                    WHERE id(destination) = $destination_id
                    MERGE (source:{source_type}:Entity {{name: $source_name, user_id: $user_id{agent_id_clause}}})
                    ON CREATE SET
                        source.created = timestamp(),
                        source.embedding = $source_embedding,
                        source:Entity
                    MERGE (source)-[r:{relationship}]->(destination)
                    ON CREATE SET 
                        r.created = timestamp()
                    RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                    """

                params = {
                    "destination_id": destination_node_search_result[0]["id(destination_candidate)"],
                    "source_name": source,
                    "source_embedding": source_embedding,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                
            elif source_node_search_result and destination_node_search_result:
                cypher = f"""
                    MATCH (source:Entity)
                    WHERE id(source) = $source_id
                    MATCH (destination:Entity)
                    WHERE id(destination) = $destination_id
                    MERGE (source)-[r:{relationship}]->(destination)
                    ON CREATE SET 
                        r.created_at = timestamp(),
                        r.updated_at = timestamp()
                    RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                    """
                params = {
                    "source_id": source_node_search_result[0]["id(source_candidate)"],
                    "destination_id": destination_node_search_result[0]["id(destination_candidate)"],
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                
            else:
                cypher = f"""
                    MERGE (n:{source_type}:Entity {{name: $source_name, user_id: $user_id{agent_id_clause}}})
                    ON CREATE SET n.created = timestamp(), n.embedding = $source_embedding, n:Entity
                    ON MATCH SET n.embedding = $source_embedding
                    MERGE (m:{destination_type}:Entity {{name: $dest_name, user_id: $user_id{agent_id_clause}}})
                    ON CREATE SET m.created = timestamp(), m.embedding = $dest_embedding, m:Entity
                    ON MATCH SET m.embedding = $dest_embedding
                    MERGE (n)-[rel:{relationship}]->(m)
                    ON CREATE SET rel.created = timestamp()
                    RETURN n.name AS source, type(rel) AS relationship, m.name AS target
                    """
                params = {
                    "source_name": source,
                    "dest_name": destination,
                    "source_embedding": source_embedding,
                    "dest_embedding": dest_embedding,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                
            result = self.graph.query(cypher, params=params)
            results.append(result)
        return results