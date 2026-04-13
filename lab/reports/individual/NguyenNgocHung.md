# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Nguyễn Ngọc Hưng  
**Vai trò trong nhóm:** Evaluation Owner  
**Ngày nộp:** 13/04/2026  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Tôi phụ trách chính Sprint 4 — phần evaluation và A/B comparison của pipeline RAG. Cụ thể, tôi xây dựng `eval.py` để chạy pipeline trên 10 test questions và tính toán các metrics chính: Faithfulness, Relevance, Context Recall, và Completeness theo thang đo 1-5. Tôi chạy evaluation song song cho hai variant: baseline (dense retrieval) và variant (hybrid retrieval + rerank). Sau đó tôi tạo hai scorecard chi tiết để so sánh kết quả từng câu hỏi. Vai trò của tôi là cầu nối giữa phần rag_answer.py của các bạn với phần group report: tôi phải lấy kết quả thô từ pipeline, chuyển đổi thành metrics có ý nghĩa, xác định failure patterns, và viết lại root cause analysis để wyoming guide quyết định chọn variant nào. Phần này không chỉ là chạy code mà còn phải hiểu được tại sao một câu hỏi được điểm cao/thấp để có thể đưa ra khuyến nghị hợp lý cho nhóm.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Sau lab này, tôi hiểu rõ hơn về sự phức tạp của việc đánh giá RAG system. Ban đầu tôi nghĩ metrics chỉ là con số, nhưng thực tế mỗi metric phản ánh một khía cạnh khác nhau của pipeline. Faithfulness (đáp án có dựa trên context không) khác với Relevance (context có liên quan đến câu hỏi không). Khi baseline có Faithfulness cao (4.56) nhưng Completeness thấp (4.22), nó không phải pipeline sai mà là các chunk lấy về chưa đủ chi tiết để trả lời đầy đủ.

Điều quan trọng hơn là tôi nhận ra rằng evaluation không phải là chỉ để chấm điểm, mà để **tìm ra failure patterns cụ thể**. Trong trường hợp của chúng tôi, Q07 (Approval Matrix) là chỉ báo rõ nhất rằng dense embedding không xử lý được synonym/alias tốt. Nhờ scorecard chi tiết từng câu, tôi có thể chỉ ra chính xác cai gì bị sai (recall 2/5) và variant nào fix được nó (recall 5/5). Dữ liệu cụ thể như vậy giúp nhóm quyết định deploy variant một cách tự tin hơn là chỉ dựa trên số trung bình.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Điều làm tôi bất ngờ nhất là Q07 — "Approval Matrix để cấp quyền hệ thống là tài liệu nào?" Đơn giản là một câu tìm tên tài liệu, nhưng baseline chỉ cho được điểm 2/5 ở tất cả metrics. Khi tôi xem lại log retrieval, vấn đề không nằm ở generation mà ở retrieval: dense embedding của "Approval Matrix" không match được với "Access Control SOP" vì chúng quá khác nhau về từ vựng, dù semantically là cùng một tài liệu. Điều này cho tôi thấy rằng embedding space không phải lúc nào cũng semantic như mong đợi.

Phần tốn thời gian nhất là debug metric calculation. Ban đầu tôi không rõ cách tính Context Recall — liệu nó là phần trăm chunk được retrieve hay tỉ lệ source được tìm thấy? Khi tôi xem lại 10 câu test, tôi thấy một số câu multi-hop (retrieve từ 2-3 tài liệu khác nhau) nên "đúng" recall không phải dễ. Tôi phải viết script để verify lại điểm các câu và confirm là logic scoring của tôi align với cách tính của các bạn ở phần generation.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** `Approval Matrix để cấp quyền hệ thống là tài liệu nào?` (Q07)

**Phân tích:**

Đây là câu căn bản nhất trong 10 câu — chỉ cần tìm ra tên đúng của tài liệu là đã trả lời được. Tuy nhiên ở baseline, scorecard cho thấy câu này chỉ đạt **2/5 ở tất cả metrics** (Faithfulness, Relevance, Recall, Completeness). Nguyên nhân chính không phải ở phần generation mà chiếm 80% lỗi ở retrieval.

Khi tôi kiểm tra log retrieval baseline, dense embedding trả về các chunk từ "access_control_sop.txt" với similarity score khoảng 0.72 — thấp hơn threshold bình thường (0.75). Vấn đề là "Approval Matrix" (old term) không match tốt với "Access Control SOP" (document name) trong embedding space vì embedding chỉ hiểu về semantic closeness, không biết được mối quan hệ alias. Model generation thấy context có liên quan đến access control nhưng chưa rõ ràng, nên chỉ trả lời chung chung thay vì nêu tên cụ thể.

Ở variant (hybrid + rerank), câu này được cải thiện **từ 2 lên 5 ở tất cả metrics**. Hybrid retrieval kết hợp dense (semantic) + BM25 (exact keyword). BM25 bắt được từ khóa "SOP" từ document name và từ khóa in nội dung, push điểm similarity lên 0.85. Reranker sau đó confirm rằng chunk này phù hợp nhất với query. Nhờ đó, model generation nhận được context rõ ràng hơn, trả lời trực tiếp "đây là Access Control SOP" và đạt full score.

Kết luận: Câu này chứng minh rằng dense embedding là bottleneck cho alias/synonym retrieval. Hybrid + rerank là giải pháp hiệu quả vì nó kết hợp cả semantic và lexical matching.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, tôi sẽ làm hai điều. Thứ nhất, thêm error analysis chi tiết hơn: phân loại 10 câu theo failure mode (alias mismatch, multi-hop, fact lookup, abstention) để nhóm biết cần optimize cái gì tiếp theo. Thứ hai, tôi sẽ setup confusion matrix để tracking false positive (khi pipeline retrieve chunk không liên quan nhưng gen model vẫn dùng) — đây là dạng lỗi khó phát hiện nhưng rất nguy hiểm với RAG system. Scorecard hiện tại chỉ show được tổng quát theo metric, nhưng failure mode breakdown sẽ giúp nhóm prioritize cải tiến.

---

*Lưu file này với tên: `reports/individual/[ten_ban].md`*  
*Ví dụ: `reports/individual/nguyen_ngoc_hung.md`*
