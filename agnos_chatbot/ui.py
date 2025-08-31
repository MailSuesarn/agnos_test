from __future__ import annotations
import os, base64, mimetypes
from typing import Any, Dict
from dotenv import load_dotenv
import gradio as gr

from .utils.config import load_yaml
from .utils.embeddings import get_bge_m3
from .utils.neo4j_vec import build_retrievers
from .model_factory import make_chat_model
from .retrieval import init_tools, chat_once

def _rgb_css(rgb_str: str) -> str:
    try:
        r, g, b = [int(x.strip()) for x in (rgb_str or "").split(",")]
        r = max(0, min(255, r)); g = max(0, min(255, g)); b = max(0, min(255, b))
        return f"rgb({r}, {g}, {b})"
    except Exception:
        return "rgb(245, 248, 250)"

def _data_uri(path: str | None) -> str | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        raw = f.read()
    mt = mimetypes.guess_type(path)[0] or "image/png"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mt};base64,{b64}"

def _read_text(path: str, default: str = "") -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return default

def launch_app(config_path: str = "configs/retrieval_chat.yaml") -> None:
    # Load env and config
    load_dotenv()
    cfg = load_yaml(config_path)

    # Neo4j env (required)
    neo4j_env = {
        "NEO4J_URI": os.getenv("NEO4J_URI", ""),
        "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME", ""),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", ""),
        "NEO4J_DATABASE": os.getenv("NEO4J_DATABASE", "neo4j"),
    }
    if not all(neo4j_env.values()):
        missing = [k for k,v in neo4j_env.items() if not v]
        raise RuntimeError(f"Missing Neo4j env(s): {', '.join(missing)}")

    # Embeddings + retrievers
    embed = get_bge_m3(normalize=True)
    threads_ret, diseases_ret = build_retrievers(
        neo4j_env=neo4j_env,
        idx_cfg=cfg["neo4j_index"],
        top_k_thread=cfg["retrieval"]["top_k_thread"],
        top_k_disease=cfg["retrieval"]["top_k_disease"],
        embeddings=embed,
    )
    init_tools(threads_ret, diseases_ret)

    # Model + prompt
    chat_model = make_chat_model()
    system_prompt = _read_text(cfg["prompts"]["system_th_path"], default="คุณคือผู้ช่วยแพทย์ของ Agnos")

    # UI bits
    provider = (os.getenv("MODEL_PROVIDER") or "openai").strip().lower()
    model_name = os.getenv("OLLAMA_MODEL") if provider == "ollama" else os.getenv("OPENAI_MODEL")
    intro_md = f"""# {cfg['ui']['title']}
- ใช้ **กระทู้ที่แพทย์ตอบ** (T#) และ **ความรู้เกี่ยวกับโรค** (D#) ในการตอบ
- *ข้อจำกัด*: ให้ข้อมูลทั่วไป ไม่ใช่การวินิจฉัยเฉพาะราย  
- **Provider:** `{provider}` | **Model:** `{model_name}`
"""

    cover_path = cfg["ui"]["cover_image"]
    cover_src = _data_uri(cover_path) or ""
    cover_h = int(os.getenv("COVER_HEIGHT_PX", "240"))
    app_bg = _rgb_css(os.getenv("BG_RGB", "143,182,232"))

    app_css = f"""
    :root {{ --app-bg: {app_bg}; }}
    body, .gradio-container {{ background: var(--app-bg) !important; }}
    #hero {{
      position: relative; width: 100%; height: {cover_h}px; margin: 0 auto 14px auto;
      border-radius: 16px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    }}
    #hero-img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
    #hero .hero-overlay {{ position: absolute; inset: 0; background: linear-gradient(0deg, rgba(0,0,0,0.18), rgba(0,0,0,0.18)); }}
    #hero .hero-text {{
      position: absolute; bottom: 14px; left: 18px; right: 18px;
      color: #fff; font-weight: 700; font-size: 26px; line-height: 1.15;
      text-shadow: 0 1px 6px rgba(0,0,0,0.35);
    }}
    #panel {{
      background: rgba(255,255,255,0.85); backdrop-filter: blur(6px);
      border-radius: 16px; padding: 12px 16px; box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }}
    """

    def _chat_fn(message, history):
        return chat_once(system_prompt=system_prompt, user_text=message, history_messages=history, chat_model=chat_model)

    with gr.Blocks(css=app_css, title=cfg["ui"]["title"]) as demo:
        gr.HTML(f"""
        <div id="hero">
          <img id="hero-img" src="{cover_src}" alt="cover">
          <div class="hero-overlay"></div>
          <div class="hero-text">{cfg['ui']['title']}</div>
        </div>
        """)
        with gr.Group(elem_id="panel"):
            gr.Markdown(intro_md)
            gr.ChatInterface(fn=_chat_fn, type="messages", title="", description="")

    try:
        demo.queue().launch(share=False)
    except Exception:
        demo.launch(share=False)
