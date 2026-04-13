from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from rag_answer import LLM_MODEL, TOP_K_SEARCH, TOP_K_SELECT, rag_answer


APP_TITLE = "VinUni RAG Chat"
APP_TAGLINE = "Internal assistant for SLA, refund, access control, and HR policy."
DEFAULT_SESSION_TITLE = "New chat"
SAMPLE_PROMPTS = [
    "SLA xử lý ticket P1 là bao lâu?",
    "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?",
    "Ai phải phê duyệt để cấp quyền Level 3?",
    "Approval Matrix để cấp quyền là tài liệu nào?",
]


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');

:root {
  --bg: #f3f7f1;
  --bg-soft: #edf3eb;
  --panel: rgba(255, 255, 255, 0.82);
  --ink: #101713;
  --muted: #5d6a62;
  --line: rgba(16, 23, 19, 0.09);
  --accent: #1e7a4d;
  --accent-soft: #dff2e6;
  --user-ink: #f4fff7;
  --assistant-ink: #101713;
  --user-top: #17372a;
  --user-bottom: #0f241b;
  --sidebar-top: #121a16;
  --sidebar-bottom: #0c130f;
}

html, body, [class*="css"] {
  font-family: "IBM Plex Sans", sans-serif;
}

.stApp {
  color: var(--ink);
  background:
    radial-gradient(circle at top left, rgba(30, 122, 77, 0.14), transparent 28%),
    radial-gradient(circle at top right, rgba(16, 23, 19, 0.04), transparent 32%),
    linear-gradient(180deg, #f8fbf7 0%, var(--bg-soft) 46%, #f6f8f5 100%);
}

[data-testid="stHeader"] {
  background: transparent;
}

#MainMenu, footer {
  visibility: hidden;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--sidebar-top) 0%, var(--sidebar-bottom) 100%);
  border-right: 1px solid rgba(255, 255, 255, 0.08);
}

[data-testid="stSidebar"] * {
  color: #edf6f0;
}

[data-testid="stSidebar"] .stButton > button {
  width: 100%;
  border-radius: 18px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.04);
  color: #edf6f0;
  padding: 0.65rem 0.9rem;
}

[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(255, 255, 255, 0.08);
  border-color: rgba(255, 255, 255, 0.16);
}

[data-testid="stSidebar"] .stRadio label {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 16px;
  padding: 0.45rem 0.55rem;
  margin-bottom: 0.45rem;
}

[data-testid="stSidebar"] .stRadio label:hover {
  background: rgba(255, 255, 255, 0.06);
}

.block-container {
  max-width: 1120px;
  padding-top: 1.2rem;
  padding-bottom: 7rem;
}

.workspace {
  max-width: 920px;
  margin: 0 auto;
}

.topbar {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
  margin-bottom: 1.7rem;
}

.brand-kicker {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent);
}

.brand-title {
  font-family: "Space Grotesk", sans-serif;
  font-size: clamp(1.75rem, 3vw, 2.7rem);
  line-height: 0.96;
  margin: 0.25rem 0 0.45rem;
}

.brand-copy {
  max-width: 38rem;
  color: var(--muted);
  margin: 0;
  line-height: 1.65;
}

.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem;
}

.chip {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.72);
  padding: 0.42rem 0.75rem;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.75rem;
  color: var(--muted);
}

.empty-stage {
  margin-top: 10vh;
  text-align: center;
  padding: 0 0 2rem;
}

.empty-title {
  font-family: "Space Grotesk", sans-serif;
  font-size: clamp(2.4rem, 5vw, 4.8rem);
  line-height: 0.94;
  margin: 0;
}

.empty-copy {
  margin: 1rem auto 0;
  max-width: 40rem;
  color: var(--muted);
  line-height: 1.7;
}

.hint-copy {
  margin: 1.2rem auto 0;
  max-width: 36rem;
  font-size: 0.94rem;
  color: var(--muted);
}

.chat-row {
  display: flex;
  margin: 0 0 1rem;
}

.chat-row.user {
  justify-content: flex-end;
}

.chat-row.assistant {
  justify-content: flex-start;
}

.chat-card {
  width: min(82%, 760px);
  border-radius: 24px;
  padding: 1rem 1.1rem 1rem 1.15rem;
  box-shadow: 0 22px 60px rgba(13, 20, 17, 0.06);
}

.chat-card.user {
  color: var(--user-ink);
  background: linear-gradient(180deg, var(--user-top) 0%, var(--user-bottom) 100%);
  border: 1px solid rgba(23, 55, 42, 0.42);
}

.chat-card.assistant {
  color: var(--assistant-ink);
  background: var(--panel);
  border: 1px solid var(--line);
  backdrop-filter: blur(14px);
}

.chat-card.user,
.chat-card.user div,
.chat-card.user span,
.chat-card.user p,
.chat-card.user a {
  color: var(--user-ink) !important;
  -webkit-text-fill-color: var(--user-ink);
}

.chat-card.assistant,
.chat-card.assistant div,
.chat-card.assistant span,
.chat-card.assistant p,
.chat-card.assistant a {
  color: var(--assistant-ink) !important;
  -webkit-text-fill-color: var(--assistant-ink);
}

.chat-label {
  font-family: "Space Grotesk", sans-serif;
  font-size: 0.78rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 0.55rem;
  opacity: 0.78;
}

.chat-card.user .chat-label {
  opacity: 0.92;
}

.chat-text {
  font-size: 1rem;
  line-height: 1.72;
  white-space: pre-wrap;
  word-break: break-word;
  opacity: 1;
}

.source-stack {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  margin: 0.6rem 0 1.35rem;
}

.source-pill {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.78);
  padding: 0.45rem 0.75rem;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.74rem;
  color: #365244;
}

.status-note {
  border: 1px dashed var(--line);
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.42);
  padding: 0.9rem 1rem;
  color: var(--muted);
  margin-bottom: 1rem;
}

[data-testid="stExpander"] {
  border: 1px solid var(--line);
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.55);
}

[data-testid="stChatInput"] {
  background: transparent;
}

[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input {
  border-radius: 22px !important;
  border: 1px solid var(--line) !important;
  background: rgba(255, 255, 255, 0.92) !important;
  box-shadow: 0 16px 40px rgba(13, 20, 17, 0.06) !important;
}

@media (max-width: 900px) {
  .topbar {
    flex-direction: column;
  }

  .chat-card {
    width: 100%;
  }
}
</style>
"""


def _new_session() -> Dict[str, Any]:
    chat_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return {
        "id": chat_id,
        "title": DEFAULT_SESSION_TITLE,
        "created_at": datetime.now().isoformat(),
        "messages": [],
    }


def _init_state() -> None:
    if "chat_sessions" not in st.session_state:
        first_session = _new_session()
        st.session_state.chat_sessions = [first_session]
        st.session_state.active_chat_id = first_session["id"]

    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = st.session_state.chat_sessions[0]["id"]

    defaults = {
        "retrieval_mode": "dense",
        "use_rerank": False,
        "transform_strategy": "none",
        "show_debug": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_sessions() -> List[Dict[str, Any]]:
    return st.session_state.chat_sessions


def _get_active_session() -> Dict[str, Any]:
    sessions = _get_sessions()
    lookup = {session["id"]: session for session in sessions}
    active_id = st.session_state.active_chat_id
    if active_id not in lookup:
        st.session_state.active_chat_id = sessions[0]["id"]
        return sessions[0]
    return lookup[active_id]


def _start_new_chat() -> None:
    session = _new_session()
    st.session_state.chat_sessions.insert(0, session)
    st.session_state.active_chat_id = session["id"]


def _truncate_title(text: str, width: int = 34) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= width:
        return cleaned
    return f"{cleaned[: width - 1].rstrip()}…"


def _ensure_title(session: Dict[str, Any], prompt: str) -> None:
    if session["title"] == DEFAULT_SESSION_TITLE:
        session["title"] = _truncate_title(prompt)


def _escape_text(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def _render_bubble(role: str, content: str) -> None:
    role_class = "assistant" if role == "assistant" else "user"
    role_label = "VinUni RAG" if role == "assistant" else "You"
    st.markdown(
        f"""
        <div class="chat-row {role_class}">
          <div class="chat-card {role_class}">
            <div class="chat-label">{role_label}</div>
            <div class="chat-text">{_escape_text(content)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sources(sources: List[str]) -> None:
    if not sources:
        return

    pills = "".join(
        f'<span class="source-pill">{html.escape(source)}</span>' for source in sources
    )
    st.markdown(f'<div class="source-stack">{pills}</div>', unsafe_allow_html=True)


def _chunk_preview(chunk: Dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {}) or {}
    source = metadata.get("source", "unknown")
    section = metadata.get("section", "")
    effective_date = metadata.get("effective_date", "")
    score = chunk.get("rerank_score", chunk.get("rrf_score", chunk.get("score", 0)))
    text = str(chunk.get("text", "")).strip()
    text = " ".join(text.split())
    preview = text[:280]
    if len(text) > 280:
        preview += "..."

    header = source
    if section:
        header += f" | {section}"
    if effective_date:
        header += f" | {effective_date}"
    if score is not None:
        try:
            header += f" | score={float(score):.2f}"
        except Exception:
            pass

    return f"{header}\n{preview}"


def _render_debug(message: Dict[str, Any]) -> None:
    chunks = message.get("chunks_used") or []
    config = message.get("config") or {}
    if not chunks and not config:
        return

    with st.expander("Debug retrieval", expanded=False):
        if config:
            st.json(config)
        if chunks:
            for idx, chunk in enumerate(chunks, 1):
                st.markdown(f"**Chunk {idx}**")
                st.code(_chunk_preview(chunk), language="text")


def _render_header() -> None:
    index_state = "ready" if Path("chroma_db").exists() else "missing"
    chips = [
        f"mode={st.session_state.retrieval_mode}",
        f"rerank={'on' if st.session_state.use_rerank else 'off'}",
        f"transform={st.session_state.transform_strategy}",
        f"top_k={TOP_K_SEARCH}/{TOP_K_SELECT}",
        f"model={LLM_MODEL}",
        f"index={index_state}",
    ]
    chip_markup = "".join(f'<span class="chip">{html.escape(chip)}</span>' for chip in chips)

    st.markdown(
        f"""
        <div class="workspace">
          <div class="topbar">
            <div>
              <div class="brand-kicker">Day 08 • Internal RAG Workspace</div>
              <h1 class="brand-title">{APP_TITLE}</h1>
              <p class="brand-copy">{APP_TAGLINE}</p>
            </div>
            <div class="chip-row">{chip_markup}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_empty_state() -> str | None:
    st.markdown(
        """
        <div class="workspace empty-stage">
          <h2 class="empty-title">Ask the lab directly.</h2>
          <p class="empty-copy">
            Hỏi thẳng về policy, SLA, access control, HR, hoặc tên tài liệu cũ.
            App sẽ gọi pipeline RAG hiện tại và giữ source phía dưới mỗi câu trả lời.
          </p>
          <p class="hint-copy">Bắt đầu với một trong các prompt mẫu hoặc nhập câu hỏi của bạn ở cuối màn hình.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(2, gap="small")
    for index, prompt in enumerate(SAMPLE_PROMPTS):
        with columns[index % 2]:
            if st.button(prompt, key=f"hero_prompt_{index}", use_container_width=True):
                return prompt
    return None


def _render_sidebar() -> str | None:
    sessions = _get_sessions()
    options = [session["id"] for session in sessions]

    with st.sidebar:
        st.markdown("### VinUni RAG")
        st.caption("Chat workspace for the Day 08 lab.")
        if st.button("+ New chat", key="new_chat", use_container_width=True):
            _start_new_chat()
            st.rerun()

        selected_chat = st.radio(
            "History",
            options=options,
            index=options.index(st.session_state.active_chat_id),
            format_func=lambda chat_id: next(
                session["title"] for session in sessions if session["id"] == chat_id
            ),
            label_visibility="collapsed",
        )
        if selected_chat != st.session_state.active_chat_id:
            st.session_state.active_chat_id = selected_chat
            st.rerun()

        st.markdown("---")
        st.markdown("#### Retrieval")
        st.selectbox(
            "Mode",
            options=["dense", "hybrid", "sparse"],
            key="retrieval_mode",
        )
        st.checkbox("Use rerank", key="use_rerank")
        st.selectbox(
            "Query transform",
            options=["none", "expansion", "decomposition", "hyde"],
            key="transform_strategy",
        )
        st.checkbox("Show debug chunks", key="show_debug")

        st.markdown("---")
        st.markdown("#### Prompt ideas")
        for index, prompt in enumerate(SAMPLE_PROMPTS):
            if st.button(prompt, key=f"sample_prompt_{index}", use_container_width=True):
                return prompt

    return None


def _submit_prompt(prompt: str) -> None:
    user_prompt = prompt.strip()
    if not user_prompt:
        return

    session = _get_active_session()
    session["messages"].append(
        {
            "role": "user",
            "content": user_prompt,
            "created_at": datetime.now().isoformat(),
        }
    )
    _ensure_title(session, user_prompt)

    try:
        with st.spinner("VinUni RAG đang tìm bằng chứng và soạn câu trả lời..."):
            result = rag_answer(
                user_prompt,
                retrieval_mode=st.session_state.retrieval_mode,
                use_rerank=st.session_state.use_rerank,
                transform_strategy=(
                    None
                    if st.session_state.transform_strategy == "none"
                    else st.session_state.transform_strategy
                ),
                verbose=False,
            )
        assistant_message = {
            "role": "assistant",
            "content": result.get("answer", ""),
            "sources": result.get("sources", []),
            "chunks_used": result.get("chunks_used", []),
            "config": result.get("config", {}),
            "created_at": datetime.now().isoformat(),
        }
    except Exception as exc:
        assistant_message = {
            "role": "assistant",
            "content": (
                "Không thể trả lời lúc này.\n\n"
                "Kiểm tra API key, ChromaDB index, và dependencies rồi thử lại.\n\n"
                f"Chi tiết: {exc}"
            ),
            "sources": [],
            "chunks_used": [],
            "config": {
                "retrieval_mode": st.session_state.retrieval_mode,
                "use_rerank": st.session_state.use_rerank,
                "transform_strategy": st.session_state.transform_strategy,
            },
            "error": True,
            "created_at": datetime.now().isoformat(),
        }

    session["messages"].append(assistant_message)


def _render_messages() -> None:
    session = _get_active_session()
    messages = session.get("messages", [])
    if not messages:
        st.markdown(
            """
            <div class="status-note">
              Conversation history is stored in the current browser session only.
              Start a new chat from the sidebar whenever you want a clean context.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for message in messages:
        _render_bubble(message["role"], message["content"])
        if message["role"] == "assistant":
            _render_sources(message.get("sources", []))
            if st.session_state.show_debug:
                _render_debug(message)


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    sidebar_prompt = _render_sidebar()
    _render_header()

    main_prompt = None
    if not _get_active_session()["messages"]:
        main_prompt = _render_empty_state()
    _render_messages()

    prompt = st.chat_input("Ask about SLA, refund, access control, HR policy...")
    pending_prompt = prompt or sidebar_prompt or main_prompt
    if pending_prompt:
        _submit_prompt(pending_prompt)
        st.rerun()


if __name__ == "__main__":
    main()
