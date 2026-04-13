"""
rag_answer.py — Sprint 2 + Sprint 3: Retrieval & Grounded Answer (Bản cập nhật Metadata Filter)
================================================================
"""

import json
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv

# Ensure stdout/stderr use UTF-8 so Vietnamese text prints correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

TOP_K_SEARCH = 10
TOP_K_SELECT = 3
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
RAG_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_lab")
RRF_K = 60
NO_DATA_MESSAGE = "Không đủ dữ liệu."

_CROSS_ENCODER = None
_BM25_CACHE = None


def _import_index_config() -> Tuple[Path, Optional[Any]]:
    try:
        from index import CHROMA_DB_DIR, get_embedding  # type: ignore
        return CHROMA_DB_DIR, get_embedding
    except Exception:
        return Path(__file__).parent / "chroma_db", None


CHROMA_DB_DIR, INDEX_GET_EMBEDDING = _import_index_config()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[\w\-\/\.]+", (text or "").lower())


def _get_embedding_fallback(text: str) -> List[float]:
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower().strip()
    if provider == "local":
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            "paraphrase-multilingual-MiniLM-L12-v2",
        )
        model = SentenceTransformer(model_name)
        return model.encode(text).tolist()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Không tìm thấy get_embedding() từ index.py và cũng chưa có OPENAI_API_KEY/EMBEDDING_PROVIDER phù hợp."
        )
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding


def _embed_text(text: str) -> List[float]:
    if INDEX_GET_EMBEDDING is not None:
        return INDEX_GET_EMBEDDING(text)
    return _get_embedding_fallback(text)


def _get_collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    return client.get_collection(RAG_COLLECTION)


def _make_chunk_key(item: Dict[str, Any]) -> str:
    meta = item.get("metadata", {}) or {}
    text = item.get("text", "") or ""
    return "||".join([
        str(meta.get("source", "")),
        str(meta.get("section", "")),
        _normalize_text(text)[:300],
    ])


def _dedupe_keep_best(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best = {}
    for item in items:
        key = _make_chunk_key(item)
        if key not in best or item.get("score", 0) > best[key].get("score", 0):
            best[key] = item
    return list(best.values())


# =============================================================================
# RETRIEVAL — DENSE (Có hỗ trợ filter)
# =============================================================================

def retrieve_dense(
    query: str, 
    top_k: int = TOP_K_SEARCH,
    meta_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    collection = _get_collection()
    query_embedding = _embed_text(query)
    
    # Chuẩn bị kwargs cho truy vấn ChromaDB
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"]
    }
    # Thêm bộ lọc metadata nếu có
    if meta_filter:
        query_kwargs["where"] = meta_filter

    results = collection.query(**query_kwargs)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    formatted = []
    for doc, meta, dist in zip(docs, metas, dists):
        score = 1 - float(dist) if dist is not None else 0.0
        formatted.append({
            "text": doc,
            "metadata": meta or {},
            "score": score,
            "retrieval_method": "dense",
        })
    return formatted


# =============================================================================
# RETRIEVAL — SPARSE / BM25 (Có hỗ trợ filter thủ công)
# =============================================================================

def _load_all_chunks() -> List[Dict[str, Any]]:
    collection = _get_collection()
    results = collection.get(include=["documents", "metadatas"])
    docs = results.get("documents", []) or []
    metas = results.get("metadatas", []) or []
    chunks = []
    for doc, meta in zip(docs, metas):
        chunks.append({
            "text": doc,
            "metadata": meta or {},
        })
    return chunks


def _get_bm25_index():
    global _BM25_CACHE
    if _BM25_CACHE is not None:
        return _BM25_CACHE

    from rank_bm25 import BM25Okapi
    chunks = _load_all_chunks()
    tokenized_corpus = [_tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    _BM25_CACHE = (bm25, chunks)
    return _BM25_CACHE


def retrieve_sparse(
    query: str, 
    top_k: int = TOP_K_SEARCH,
    meta_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    bm25, chunks = _get_bm25_index()
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    
    scored_items = []
    for idx, score in enumerate(scores):
        if score <= 0:
            continue
        chunk = chunks[idx]
        chunk_meta = chunk.get("metadata", {})
        
        # Nếu có filter, kiểm tra thủ công xem chunk có thỏa mãn không
        if meta_filter:
            match = True
            for k, v in meta_filter.items():
                if chunk_meta.get(k) != v:
                    match = False
                    break
            if not match:
                continue # Bỏ qua chunk này nếu không khớp filter

        scored_items.append({
            "text": chunk["text"],
            "metadata": chunk_meta,
            "score": float(score),
            "retrieval_method": "sparse",
        })

    # Sắp xếp lại theo điểm từ cao xuống thấp và lấy top_k
    scored_items.sort(key=lambda x: x["score"], reverse=True)
    return scored_items[:top_k]


# =============================================================================
# RETRIEVAL — HYBRID (RRF)
# =============================================================================

def retrieve_hybrid(
    query: str,
    top_k: int = TOP_K_SEARCH,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    meta_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    # Truyền meta_filter xuống cả 2 hàm con
    dense_results = retrieve_dense(query, top_k=top_k, meta_filter=meta_filter)
    sparse_results = retrieve_sparse(query, top_k=top_k, meta_filter=meta_filter)

    fused: Dict[str, Dict[str, Any]] = {}

    for rank, item in enumerate(dense_results, start=1):
        key = _make_chunk_key(item)
        base = fused.setdefault(key, {**item, "score": 0.0, "rrf_score": 0.0})
        base["rrf_score"] += dense_weight * (1.0 / (RRF_K + rank))
        base["score"] = max(base.get("score", 0.0), item.get("score", 0.0))
        base["retrieval_method"] = "hybrid"

    for rank, item in enumerate(sparse_results, start=1):
        key = _make_chunk_key(item)
        base = fused.setdefault(key, {**item, "score": 0.0, "rrf_score": 0.0})
        base["rrf_score"] += sparse_weight * (1.0 / (RRF_K + rank))
        base["score"] = max(base.get("score", 0.0), item.get("score", 0.0))
        base["retrieval_method"] = "hybrid"

    ranked = sorted(fused.values(), key=lambda x: x.get("rrf_score", 0.0), reverse=True)
    return ranked[:top_k]


# =============================================================================
# RERANK & QUERY TRANSFORMATION (Giữ nguyên)
# =============================================================================

def _load_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        from sentence_transformers import CrossEncoder
        model_name = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        _CROSS_ENCODER = CrossEncoder(model_name)
    return _CROSS_ENCODER


def _lexical_overlap_score(query: str, text: str) -> float:
    q = set(_tokenize(query))
    t = set(_tokenize(text))
    if not q or not t:
        return 0.0
    return len(q & t) / max(1, len(q))


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = TOP_K_SELECT,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    try:
        model = _load_cross_encoder()
        pairs = [[query, chunk["text"]] for chunk in candidates]
        scores = model.predict(pairs)
        ranked = []
        for chunk, score in zip(candidates, scores):
            item = dict(chunk)
            item["rerank_score"] = float(score)
            ranked.append(item)
        ranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return ranked[:top_k]
    except Exception:
        ranked = []
        for chunk in candidates:
            item = dict(chunk)
            item["rerank_score"] = _lexical_overlap_score(query, chunk.get("text", ""))
            ranked.append(item)
        ranked.sort(
            key=lambda x: (x.get("rerank_score", 0.0), x.get("score", 0.0)),
            reverse=True,
        )
        return ranked[:top_k]

def _parse_json_list(text: str) -> List[str]:
    text = (text or "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass

    matches = re.findall(r'"(.*?)"', text, flags=re.S)
    if matches:
        return [m.strip() for m in matches if m.strip()]

    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def transform_query(query: str, strategy: str = "expansion") -> List[str]:
    strategy = (strategy or "expansion").lower().strip()

    if strategy == "expansion":
        instruction = (
            f"Given the query: {query}\n"
            "Generate 2-3 alternative phrasings or aliases that would help retrieval. "
            "Output only a JSON array of strings."
        )
    elif strategy == "decomposition":
        instruction = (
            f"Break down this complex query into 2-3 simpler sub-queries: {query}\n"
            "Output only a JSON array of strings."
        )
    elif strategy == "hyde":
        instruction = (
            f"Write one short hypothetical answer passage that could help retrieve evidence for this question: {query}\n"
            "Output only a JSON array with one string."
        )
    else:
        return [query]

    try:
        raw = call_llm(instruction)
        transformed = _parse_json_list(raw)
        transformed = [query] + transformed
        deduped = []
        seen = set()
        for q in transformed:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(q.strip())
        return deduped[:3]
    except Exception:
        return [query]


# =============================================================================
# GENERATION (Đã cập nhật prompt Inline Citation)
# =============================================================================

def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {}) or {}
        source = meta.get("source", "unknown")
        section = meta.get("section", "")
        effective_date = meta.get("effective_date", "")
        score = chunk.get("rerank_score", chunk.get("rrf_score", chunk.get("score", 0)))
        text = chunk.get("text", "")

        header = f"[{i}] {source}"
        if section:
            header += f" | {section}"
        if effective_date:
            header += f" | effective_date={effective_date}"
        if score is not None:
            try:
                header += f" | score={float(score):.2f}"
            except Exception:
                pass

        context_parts.append(f"{header}\n{text}")

    return "\n\n".join(context_parts)


def _build_source_labels(chunks: List[Dict[str, Any]]) -> List[str]:
    labels = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {}) or {}
        source = meta.get("source", "unknown")
        section = meta.get("section", "")
        effective_date = meta.get("effective_date", "")

        label = f"[{i}] {source}"
        if section:
            label += f" | {section}"
        if effective_date:
            label += f" | effective_date={effective_date}"
        labels.append(label)

    return labels


def _print_sources(sources: List[str]) -> None:
    print("Sources:")
    for source in sources:
        print(f"  {source}")


def _filter_sources_by_answer(answer: str, sources: List[str]) -> List[str]:
    cited_indices = sorted({int(match) for match in re.findall(r"\[(\d+)\]", answer or "")})
    if not cited_indices:
        return sources

    filtered = []
    for idx in cited_indices:
        if 1 <= idx <= len(sources):
            filtered.append(sources[idx - 1])
    return filtered or sources


def build_grounded_prompt(query: str, context_block: str) -> str:
    # Đã sửa đổi prompt để mượt mà hơn như đã thống nhất
    return f"""You are a helpful internal knowledge assistant.
Answer only from the retrieved context below.
If the context is insufficient, contradictory, or does not directly answer the question, reply exactly: "{NO_DATA_MESSAGE}"
Do not use outside knowledge.

Formatting rules:
- Start with "Theo [source file]," to ground the answer (e.g., "Theo support/sla-p1-2026.pdf,")
- Write in natural, concise Vietnamese — 1 to 2 sentences max
- Include key numbers, deadlines, names directly in the sentence
- If multiple snippets contribute, naturally connect them in one paragraph
- Do NOT use bullet points or lists in the answer
- Do NOT cite bracket numbers like [1] — the source name is the citation
- Output ONLY the final answer

Question: {query}

Context:
{context_block}

Answer:"""


def call_llm(prompt: str) -> str:
    provider = os.getenv("LLM_PROVIDER", "openai").lower().strip()

    if provider == "gemini":
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Thiếu GOOGLE_API_KEY trong .env")
        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return (response.text or "").strip()

    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Thiếu OPENAI_API_KEY trong .env")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512,
    )
    return (response.choices[0].message.content or "").strip()


def _retrieve_by_mode(
    query: str, 
    retrieval_mode: str, 
    top_k: int,
    meta_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    if retrieval_mode == "dense":
        return retrieve_dense(query, top_k=top_k, meta_filter=meta_filter)
    if retrieval_mode == "sparse":
        return retrieve_sparse(query, top_k=top_k, meta_filter=meta_filter)
    if retrieval_mode == "hybrid":
        return retrieve_hybrid(query, top_k=top_k, meta_filter=meta_filter)
    raise ValueError(f"retrieval_mode không hợp lệ: {retrieval_mode}")


# =============================================================================
# PIPELINE
# =============================================================================

def rag_answer(
    query: str,
    retrieval_mode: str = "dense",
    top_k_search: int = TOP_K_SEARCH,
    top_k_select: int = TOP_K_SELECT,
    use_rerank: bool = False,
    verbose: bool = False,
    transform_strategy: Optional[str] = None,
    meta_filter: Optional[Dict[str, Any]] = None, # Đã thêm tham số này
) -> Dict[str, Any]:
    config = {
        "retrieval_mode": retrieval_mode,
        "top_k_search": top_k_search,
        "top_k_select": top_k_select,
        "use_rerank": use_rerank,
        "transform_strategy": transform_strategy,
        "meta_filter": meta_filter,
    }

    queries = [query]
    if transform_strategy:
        queries = transform_query(query, strategy=transform_strategy)

    candidates: List[Dict[str, Any]] = []
    for q in queries:
        # Truyền meta_filter vào hàm tìm kiếm
        retrieved = _retrieve_by_mode(q, retrieval_mode, top_k=top_k_search, meta_filter=meta_filter)
        for item in retrieved:
            new_item = dict(item)
            new_item["retrieved_for_query"] = q
            candidates.append(new_item)

    candidates = _dedupe_keep_best(candidates)
    candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    candidates = candidates[:top_k_search]

    if verbose:
        print(f"\n[RAG] Query: {query}")
        print(f"[RAG] Retrieval mode: {retrieval_mode}")
        if meta_filter:
            print(f"[RAG] Meta Filter: {meta_filter}")
        print(f"[RAG] Retrieved {len(candidates)} candidates")
        for i, c in enumerate(candidates[:5], 1):
            meta = c.get("metadata", {}) or {}
            print(
                f"  [{i}] score={c.get('score', 0):.3f} | "
                f"source={meta.get('source', '?')} | section={meta.get('section', '?')}"
            )

    if use_rerank:
        selected = rerank(query, candidates, top_k=top_k_select)
    else:
        selected = candidates[:top_k_select]

    if not selected:
        return {
            "query": query,
            "answer": NO_DATA_MESSAGE,
            "sources": [],
            "chunks_used": [],
            "config": config,
        }

    context_block = build_context_block(selected)
    prompt = build_grounded_prompt(query, context_block)

    if verbose:
        print(f"[RAG] Selected {len(selected)} chunks")

    answer = call_llm(prompt).strip()
    if not answer:
        answer = NO_DATA_MESSAGE

    sources = _filter_sources_by_answer(answer, _build_source_labels(selected))

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "chunks_used": selected,
        "config": config,
    }


# =============================================================================
# COMPARISON
# =============================================================================

def compare_retrieval_strategies(query: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print('=' * 60)

    strategies = ["dense", "hybrid", "sparse"]  # Thêm "sparse" sau khi implement

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        try:
            result = rag_answer(query, retrieval_mode=strategy, verbose=False)
            print(f"Answer: {result['answer']}")
            _print_sources(result["sources"])
        except Exception as e:
            print(f"Lỗi: {e}")

    print("\n--- Dense + Rerank ---")
    try:
        result = rag_answer(query, retrieval_mode="dense", use_rerank=True, verbose=False)
        print(f"Answer: {result['answer']}")
        _print_sources(result["sources"])
    except Exception as e:
        print(f"Lỗi: {e}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Test: RAG Pipeline with Metadata Filter")
    print("=" * 60)

    test_queries = [
        "SLA xử lý ticket P1 là bao lâu?",
        "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?",
    ]

    print("\n--- Test 1: Không dùng Filter ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = rag_answer(query, retrieval_mode="hybrid", verbose=True)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        # except NotImplementedError:
        #     print("Chưa implement — hoàn thành TODO trong retrieve_dense() và call_llm() trước.")
        except Exception as e:
            print(f"Lỗi: {e}")

    print("\n--- Test 2: CÓ DÙNG FILTER (Ví dụ ép lấy file policy) ---")
    query_filter = "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?"
    print(f"\nQuery: {query_filter}")
    # Giả sử bạn chỉ muốn lục tìm trong file có tên là policy/refund-v4.pdf
    my_filter = {"source": "policy/refund-v4.pdf"} 
    
    try:
        result_filtered = rag_answer(
            query=query_filter, 
            retrieval_mode="hybrid", 
            meta_filter=my_filter, # <<< Gọi bộ lọc tại đây
            verbose=True
        )
        print(f"Answer: {result_filtered['answer']}")
        _print_sources(result_filtered["sources"])
    except Exception as e:
        print(f"Lỗi filter: {e}")
