from __future__ import annotations
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def get_bge_m3(normalize: bool = True) -> HuggingFaceBgeEmbeddings:
    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={"normalize_embeddings": normalize},
        query_instruction="query: ",
        embed_instruction="passage: ",
    )
