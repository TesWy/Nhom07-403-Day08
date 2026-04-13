# Tuning Log — RAG Pipeline (Day 08 Lab)

---

## Baseline (Sprint 2)

**Ngày:** 13-04-2026  
**Config:**
```
retrieval_mode = "dense"
chunk_size = 400 tokens
overlap = 80 tokens
top_k_search = 10
top_k_select = 3
use_rerank = False
llm_model = gpt-4o-mini
embedding_model = text-embedding-3-small
```

**Scorecard Baseline:**
| Metric | Average Score |
|--------|--------------|
| Faithfulness | 4.56/5 |
| Answer Relevance | 4.67/5 |
| Context Recall | 4.44/5 |
| Completeness | 4.22/5 |
| **AVERAGE** | **4.47/5** |

**Câu hỏi yếu nhất:**
1. Q07 (Approval Matrix) - Recall: 2/5
   - **Nguyên nhân:** Dense embedding không match "Approval Matrix" → "Access Control SOP"
   - Embedding similarity score: 0.72 (dưới ngưỡng 0.75)
   - **Ảnh hưởng:** Lỗi không tìm thấy tài liệu, Recall = 1/2 expected sources

2. Q03 (Level 3 Access) - Recall: 3/5
   - **Nguyên nhân:** Dense matches "Level 3" nhưng miss "IT Security"
   
3. Q10 (VIP Refund) - Completeness: 3/5
   - **Nguyên nhân:** Không có tài liệu liên quan đến chính sách cho VIP
   - Model nhận ra nhưng câu trả lời chưa hướng đến cách giải quyết với chính sách coi VIP như khách hàng thường.

**Giả thuyết nguyên nhân (Error Tree):**
- [x] **Retrieval:** Dense bỏ lỡ từ đồng nghĩa/alias ("Matrix" ≠ "SOP")
- [x] **Retrieval:** Dense gặp khó khăn khi xử lý các cụm nhiều từ (“Access Control”) vì chúng bị tách thành các token riêng lẻ.
- [ ] Indexing: Chunking tốt, metadata đầy đủ
- [ ] Indexing: Metadata không thiếu effective_date
- [ ] Generation: Prompt đầy đủ, LLM thực hiện đúng grounding rules
- [ ] Generation: Độ dài context vừa phải (avg 3 chunks × 400 tokens = 1200 chars)

---

## Variant 1 (Sprint 3): Hybrid + Rerank

**Ngày:** 13-04-2026  
**Biến thay đổi:** `retrieval_mode: "dense" → "hybrid"` + `use_rerank: False → True`  

**Lý do chọn biến này:**
- Q07 failure chỉ do keyword mismatch, không phải semantic issue
- Dense (0.72) + BM25 (exact term match) = combined signal sẽ mạnh hơn
- RRF (Reciprocal Rank Fusion) sẽ kết hợp cả dense + sparse scores
- Reranking (cross-encoder) sẽ confirmed relevance từ 2nd pass
- **Expected improvement:** Q07 recall 2 → 4-5 và các thay đổi nhỏ khác với các câu hỏi còn lại

**Config thay đổi:**
```diff
- retrieval_mode = "dense"
+ retrieval_mode = "hybrid"          # Dense + BM25

- use_rerank = False
+ use_rerank = True                  # Cross-encoder rerank
+ rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Giữ nguyên:
top_k_search = 10
top_k_select = 3
chunk_size = 400 tokens
```

**Scorecard Variant 1:**
| Metric | Baseline | Variant 1 | Delta | % Gain |
|--------|----------|-----------|-------|--------|
| Faithfulness | 4.56/5 | 4.78/5 | +0.22 | +4.8% |
| Answer Relevance | 4.67/5 | 4.89/5 | +0.22 | +4.7% |
| Context Recall | 4.44/5 | 4.89/5 | +0.45 | **+10.1%** |
| Completeness | 4.22/5 | 4.56/5 | +0.34 | +8.1% |
| **AVERAGE** | 4.47/5 | 4.78/5 | **+0.31** | **+6.9%** |

**Nhận xét per-question:**

| Q | Category | Baseline | Variant | Delta | Notes |
|----|----------|----------|---------|-------|-------|
| q01 | SLA | 5/5/5/5 | 5/5/5/5 | - | Không cần cải thiện |
| q02 | Refund | 5/5/5/5 | 5/5/5/5 | - | Không cần cải thiện |
| **q03** | Access | 4/5/3/4 | **5/5/5/5** | **+1.75** | BM25 tốt hơn cho việc tìm kiếm "Level 3" + "IT Security" |
| **q04** | Refund | 5/5/5/4 | **5/5/5/5** | **+0.25** | Rerank giúp liệt kê danh sách loại trừ đầy đủ hơn |
| q05 | IT | 5/5/5/5 | 5/5/5/5 | - | Không cần cải thiện |
| **q06** | SLA | 4/4/4/4 | **5/5/5/4** | **+0.75** | Bắt được "escalation" |
| **q07** | Access | **2/2/2/2** | **5/5/5/4** | **+3** | Hybrid giúp truy xuất được alias |
| **q08** | HR | 4/5/4/4 | **4/5/4/5** | **+0.25** | Rerank tìm được chính sách làm remote một cách đầy đủ |
| q09 | None | 5/5/5/5 | 5/5/5/5 | - | Không cần cải thiện |
| q10 | Refund | 4/4/4/3 | 4/4/4/4 | - | Vẫn không tìm được chính sách VIP (nhưng đây là do hạn chế về document) |

**Kết luận:**
Variant 1 rõ ràng tốt hơn. Sự cải thiện đến từ:
1. **Bước đột phá ở Q07 (+3):**  Hybrid retrieval tận dụng BM25 để khớp các từ khóa như “access” và “SOP”.
2. **Cải thiện ở Q03, Q04, Q06, Q08:** Reranking giúp đưa lên các chunk đầy đủ thông tin hơn.
3. Không có trường hợp nào bị giảm chất lượng; thấp nhất là Q10 do bị giới hạn bởi tài liệu.


### Tóm tắt học được

1. Lỗi phổ biến nhất trong pipeline này là gì?
> Lỗi phổ biến nhất là retrieval miss do mismatch từ khóa/alias.
> Multi-term phrase bị tách token → mất ngữ nghĩa (“Access Control”)
> Query và document dùng cách diễn đạt khác nhau
2. Biến nào có tác động lớn nhất tới chất lượng?
> Biến có tác động lớn nhất: retrieval strategy (dense → hybrid)
3. Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?
> Query Transformation
> Tune hybrid weights (dense_weight / sparse_weight)
> Chunk theo semantic boundary tốt hơn (section nhỏ hơn)
> Add fallback logic cho “no data”
