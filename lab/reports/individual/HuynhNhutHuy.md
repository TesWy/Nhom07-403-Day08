# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Huỳnh Nhật Huy  
**Vai trò trong nhóm:** Tech Lead - Retrieval Owner  
**Ngày nộp:** 13/04/2026  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong lab này tôi phụ trách chính ở Sprint 2 và một phần Sprint 3, tập trung vào phần retrieval và generation của pipeline. Cụ thể, tôi tham gia hoàn thiện `rag_answer.py`, thử nhiều grounded prompt khác nhau để ép mô hình chỉ trả lời từ context, đồng thời bổ sung phần post-processing để làm sạch output và trình bày citation rõ hơn theo source/chunk. Tôi cũng tham gia thử nghiệm các chiến lược retrieval như dense, hybrid và rerank để so sánh chất lượng trả lời trên tập câu hỏi test. Vai trò của tôi kết nối trực tiếp với phần index và evaluation của các bạn khác: sau khi có chunk và metadata từ `index.py`, tôi dùng chúng để xây pipeline trả lời, rồi phối hợp với phần `eval.py` để đọc scorecard, tìm edge cases và điều chỉnh prompt/retrieval sao cho grounded hơn.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Sau lab này, tôi hiểu rõ hơn hai khái niệm là chunking và grounded prompt. Với chunking, trước đây tôi nghĩ chỉ cần chia tài liệu thành các đoạn nhỏ là đủ, nhưng khi làm thực tế tôi thấy cách chia ảnh hưởng rất mạnh đến retrieval. Nếu cắt không theo ranh giới tự nhiên như section hoặc paragraph thì chunk có thể mất ngữ nghĩa, kéo theo việc retrieve đúng file nhưng sai đoạn. Overlap cũng quan trọng vì nó giúp giữ lại ngữ cảnh ở ranh giới giữa hai chunk liên tiếp, đặc biệt với policy hoặc SOP có nhiều điều kiện nối tiếp nhau.

Với grounded prompt, tôi hiểu rằng prompt không chỉ để “hỏi model”, mà còn là cơ chế kiểm soát hành vi. Chỉ cần prompt lỏng, model sẽ thêm thông tin suy diễn hoặc trả lời theo kiến thức nền. Khi prompt chặt hơn, output ổn định hơn, bám sát context hơn và cũng dễ post-process, evaluate hơn.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Điều làm tôi bất ngờ nhất là query alias khó hơn tôi nghĩ. Trường hợp `Approval Matrix để cấp quyền hệ thống là tài liệu nào?` là ví dụ rõ nhất: con người dễ hiểu đây là đang hỏi tên cũ của tài liệu `Access Control SOP`, nhưng model và retriever không phải lúc nào cũng nối được hai cụm này với nhau. Dense baseline bị hụt ở chỗ đó, còn hybrid + rerank cải thiện rõ vì bắt được exact term tốt hơn.

Phần tốn nhiều thời gian debug nhất không hẳn là lỗi code mà là việc chỉnh grounded prompt và format output. Tôi phải thử nhiều cách để model vừa trả lời tự nhiên, vừa giữ đúng source, không bị echo prompt và không thêm chi tiết ngoài context. Giả thuyết ban đầu của tôi là hybrid sẽ luôn tốt hơn dense, nhưng thực tế không phải câu nào cũng cải thiện mạnh. Với tập dữ liệu nhỏ, hybrid và rerank giúp ở các câu alias hoặc edge case, còn các câu trực tiếp thì dense đã đủ tốt.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** `Approval Matrix để cấp quyền hệ thống là tài liệu nào?`

**Phân tích:**

Đây là câu tôi thấy thú vị nhất vì nó không phải câu hỏi fact đơn giản, mà là một dạng alias mapping. Ở baseline dense, hệ thống trả lời chưa tốt. Theo scorecard baseline, câu này chỉ đạt khoảng `Faithfulness = 2`, `Relevance = 2`, `Context Recall = 2`, `Completeness = 2`. Lý do không nằm hoàn toàn ở generation mà chủ yếu ở retrieval: dense retrieval không nối tốt cụm “Approval Matrix” với tên tài liệu thật là `Access Control SOP`, nên chunk lấy về chưa đủ đúng trọng tâm. Khi context đã lệch, model dù vẫn grounded cũng chỉ trả lời chung chung về tài liệu access control chứ chưa nêu rõ quan hệ “tên cũ → tên mới”.

Ở variant `hybrid + rerank`, câu này cải thiện rõ. Theo scorecard variant, kết quả đạt `Faithfulness = 5`, `Relevance = 5`, `Context Recall = 5`, `Completeness = 4`. Hybrid giúp bắt được từ khóa và alias tốt hơn, còn rerank đẩy các chunk liên quan trực tiếp lên trên. Tuy vậy completeness vẫn chưa tuyệt đối vì câu trả lời vẫn thiên về tên mới của tài liệu và chưa giải thích thật rõ rằng `Approval Matrix for System Access` là tên cũ. Trường hợp này cho tôi thấy lỗi chính nằm ở retrieval trước, còn generation chỉ là tầng khuếch đại chất lượng của context đã lấy về.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, tôi sẽ thử hai hướng. Thứ nhất, tôi sẽ thêm query expansion hoặc alias dictionary cho các cụm như `Approval Matrix -> Access Control SOP`, vì eval cho thấy hybrid đã cải thiện nhưng vẫn chưa hoàn hảo ở câu alias. Thứ hai, tôi sẽ tinh chỉnh prompt để hạn chế việc trả lời thừa chi tiết ở các câu trực tiếp như SLA hay approval, vì scorecard cho thấy nhiều câu đúng nhưng completeness chưa tối đa do model thêm thông tin ngoài phần expected answer.

---

*Lưu file này với tên: `reports/individual/[ten_ban].md`*  
*Ví dụ: `reports/individual/nguyen_van_a.md`*
