# Báo Cáo Cá Nhân, Lab Day 08: RAG Pipeline

**Họ và tên**: Huỳnh Khải Huy  
**MSHV**: 2A202600082  
**Vai trò trong nhóm**: Retrieval Owner  
**Ngày nộp**: 2026-04-13

---

## 1. Tôi đã làm gì trong lab này?

Tôi phụ trách phần Retrieval trong pipeline, tập trung vào Sprint 2 và Sprint 3.

Ở Sprint 2, tôi implement retrieve_dense() trong rag_answer.py: embed query bằng cùng model đã dùng khi index, query ChromaDB với cosine similarity, trả về list chunks kèm score (score = 1 - distance). Bên cạnh đó, thêm caching global cho ChromaDB client và collection để tránh load lại nhiều lần.

Ở Sprint 3, tôi implement thêm hai hàm: retrieve_sparse() dùng BM25Okapi (caching toàn bộ corpus khi lần đầu gọi) và retrieve_hybrid() dùng Reciprocal Rank Fusion (RRF) kết hợp dense rank và sparse rank với trọng số dense=0.75 / sparse=0.25. Lý do chọn 0.75 cho dense vì corpus tiếng Việt, embedding model nắm ngữ nghĩa tốt hơn, còn BM25 bổ sung cho các term kỹ thuật như ERR-403, P1, Level 4.

Commit 12b32e5 thêm implementation ban đầu; commit 877860791 refactor lại toàn bộ module cho sạch hơn: tách _import_index_config(), thêm _normalize_text(), chuẩn hóa constants. Ngoài ra, tôi còn fix chat UI trong commit 76ee835, sửa CSS dark mode cho hero buttons và chat input bar vì contrast bị mất trên dark background.

Hơn nữa, phần tôi làm kết nối trực tiếp với phần của Huỳnh Nhựt Huy (Tech Lead) ở khâu rerank: retrieve_hybrid() của tôi trả về top-10 candidates, sau đó cross-encoder của bạn, rerank xuống top-3 trước khi đưa vào prompt.

---

## 2. Điều tôi hiểu rõ hơn sau lab này

Trước lab, tôi hiểu hybrid retrieval ở mức lý thuyết "kết hợp dense và sparse". Sau khi implement thực tế, tôi mới thấy rõ vấn đề score normalization: dense score là cosine similarity (0–1), còn BM25 score là giá trị tuyệt đối không có upper bound, không thể cộng trực tiếp được. RRF giải quyết điều này bằng cách dùng rank thay vì score, nên hai nguồn luôn có thể merge được bất kể thang đo khác nhau. Công thức 1 / (k + rank) với k=60 cũng có tác dụng "làm mềm" sự chênh lệch giữa rank 1 và rank 2, tránh winner-takes-all.

Điều thứ hai là tầm quan trọng của caching: lần đầu build BM25 index từ toàn bộ ChromaDB mất khoảng 1–2 giây, nhưng nếu không cache thì mỗi query đều phải rebuild, với 10 grading questions sẽ tốn 10–20 giây thêm. Global cache _BM25_CACHE giải quyết điều này.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn

Khó khăn lớn nhất là Vietnamese text encoding trên Windows. Khi chạy rag_answer.py từ terminal, output tiếng Việt bị vỡ ký tự, "không đủ" hiển thị thành "kh├┤ng ─æß╗º". Ban đầu tôi nghĩ là lỗi ở ChromaDB hoặc embedding, mất gần 20 phút debug và prompt với AI tôi mới phát hiện ra stdout của Python trên Windows mặc định dùng cp1252 thay vì UTF-8. Sau đó, tôi thử fix bằng sys.stdout.reconfigure(encoding="utf-8") ngay đầu file và nó chạy được bình thường.

Điều ngạc nhiên thứ hai: BM25 hoạt động tốt hơn mong đợi cho các từ khóa kỹ thuật. Query "SLA P1 6 giờ", dense retrieval trả về chunk chung về SLA, còn BM25 rank chunk có "6 giờ" lên top-1 ngay vì exact term match. RRF kết hợp hai nguồn cho kết quả tốt hơn cả hai riêng lẻ.

---

## 4. Phân tích một câu hỏi trong grading

Câu hỏi chọn: gq02, "Khi làm việc remote, tôi phải dùng VPN và được kết nối trên tối đa bao nhiêu thiết bị?"

Pipeline của nhóm trả về: "Mỗi tài khoản được kết nối VPN trên tối đa 2 thiết bị cùng lúc", Partial (5/10).

Nhìn vào sources của gq02, retrieve_hybrid() đã lấy được chunk từ cả helpdesk-faq.md (giới hạn 2 thiết bị) lẫn hr/leave-policy-2026.pdf (VPN bắt buộc khi remote). Vậy retrieval không phải lỗi, cả hai doc đều có trong top-3.

Lỗi nằm ở generation step: model tổng hợp từ chunk helpdesk-faq (score cao hơn sau rerank) và bỏ qua thông tin từ chunk hr_leave_policy. Đây là failure mode "cross-document retrieval thành công nhưng synthesis thất bại", model ưu tiên chunk có relevance score cao nhất thay vì tổng hợp từ tất cả sources.

Root cause kỹ thuật: system prompt chưa có chỉ dẫn explicit "phải dùng thông tin từ TẤT CẢ sources được cung cấp". Nếu thêm câu đó vào prompt, model sẽ không bỏ sót hr_leave_policy chunk. 

Đây là một bài học: retrieval tốt chưa đủ, generation prompt phải hướng dẫn model synthesize đúng cách.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì?

Tôi sẽ thử query expansion cho retrieve_sparse(): thay vì tokenize query thô, mở rộng thêm bộ từ vựng tiếng Việt (ví dụ: "nghỉ phép" → ["nghỉ phép", "annual leave", "xin nghỉ"]). Scorecard cho thấy gq08 disambiguation gặp khó khăn với BM25 vì "3 ngày" là stop-word quá phổ biến, expansion theo context sẽ giúp BM25 differentiate tốt hơn.
