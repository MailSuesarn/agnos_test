from __future__ import annotations
import os
from typing import Any
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

def make_chat_model() -> Any:
    provider = (os.getenv("MODEL_PROVIDER") or "openai").strip().lower()
    temperature = float(os.getenv("TEMPERATURE", "0.2"))

    if provider == "ollama":
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        return ChatOllama(base_url=host, model=model, temperature=temperature)

    # Default: OpenAI
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Fill your .env.")
    os.environ["OPENAI_API_KEY"] = api_key  # ensure underlying client sees it
    return ChatOpenAI(model=model, temperature=temperature)
