from __future__ import annotations
from typing import Any, Dict
from langchain_community.vectorstores import Neo4jVector
from langchain_core.vectorstores import VectorStoreRetriever

def build_retrievers(
    neo4j_env: Dict[str, str],
    idx_cfg: Dict[str, Any],
    top_k_thread: int,
    top_k_disease: int,
    embeddings,
) -> tuple[VectorStoreRetriever, VectorStoreRetriever]:
    """
    Create two Neo4jVector retrievers (threads, diseases) from existing indexes.
    """
    threads_vs = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=neo4j_env["NEO4J_URI"],
        username=neo4j_env["NEO4J_USERNAME"],
        password=neo4j_env["NEO4J_PASSWORD"],
        database=neo4j_env["NEO4J_DATABASE"],
        index_name=idx_cfg["threads"]["index_name"],
        text_node_property=idx_cfg["threads"]["text_node_property"],
        retrieval_query=idx_cfg["threads"]["retrieval_query"],
    )
    diseases_vs = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=neo4j_env["NEO4J_URI"],
        username=neo4j_env["NEO4J_USERNAME"],
        password=neo4j_env["NEO4J_PASSWORD"],
        database=neo4j_env["NEO4J_DATABASE"],
        index_name=idx_cfg["diseases"]["index_name"],
        text_node_property=idx_cfg["diseases"]["text_node_property"],
        retrieval_query=idx_cfg["diseases"]["retrieval_query"],
    )
    return (
        threads_vs.as_retriever(search_kwargs={"k": int(top_k_thread)}),
        diseases_vs.as_retriever(search_kwargs={"k": int(top_k_disease)}),
    )
