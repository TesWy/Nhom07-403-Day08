# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Huỳnh Lê Xuân Ánh 
**Vai trò trong nhóm:** Documentation Owner  
**Ngày nộp:** 13/04/2026  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

> Trong lab này, tôi chủ yếu tham gia ở Sprint 2 và Sprint 3, với vai trò Documentation Owner. Tôi chịu trách nhiệm ghi lại toàn bộ quá trình tuning pipeline, bao gồm việc cập nhật các cấu hình như chunk size, overlap, top-k retrieval và việc có/không sử dụng rerank. Tôi cũng xây dựng file tuning log để theo dõi sự thay đổi giữa baseline và các variant.

> Ngoài ra, tôi tổng hợp kết quả evaluation từ các thành viên khác cũng như ghi nhận toàn bộ kiến trúc của RAG pipeline này. Công việc của tôi kết nối trực tiếp với phần implement của team (retrieval và generation), vì tôi phải hiểu họ thay đổi gì để ghi nhận chính xác và giúp cả nhóm nhìn ra xu hướng cải thiện hoặc suy giảm hiệu suất.

_________________

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

> Sau lab này, tôi hiểu rõ hơn về chunking và retrieval quality. Trước đây tôi nghĩ chunk càng lớn thì càng giữ được nhiều context và sẽ tốt hơn, nhưng thực tế không đơn giản như vậy. Chunk quá lớn làm giảm độ chính xác khi retrieval vì embedding bị loãng, chứa nhiều thông tin không liên quan đến query.

> Ngược lại, chunk nhỏ hơn giúp retrieval chính xác hơn, nhưng nếu quá nhỏ thì lại mất ngữ cảnh, khiến LLM khó tổng hợp câu trả lời hoàn chỉnh. Vì vậy, chunking là một bài toán đánh đổi giữa precision và context completeness.

> Ngoài ra, tôi cũng hiểu rõ hơn vai trò của top-k selection. Không phải cứ tăng top-k là tốt, vì nếu đưa quá nhiều tài liệu vào prompt, LLM có thể bị nhiễu và trả lời kém chính xác hơn.

_________________

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

> Điều khiến tôi ngạc nhiên là việc baseline hoạt động không tệ như kỳ vọng, trong khi một số variant sau khi tuning lại không cải thiện đáng kể, thậm chí giảm điểm ở một số câu hỏi. Ban đầu tôi giả định rằng thêm rerank hoặc tăng top-k sẽ luôn giúp kết quả tốt hơn, nhưng thực tế không phải vậy.

> Khó khăn lớn nhất là việc debug khi câu trả lời sai. Không dễ để xác định lỗi nằm ở retrieval hay generation. Có trường hợp retrieval đã lấy đúng tài liệu, nhưng LLM vẫn trả lời sai do prompt chưa đủ rõ hoặc bị nhiễu bởi các chunk khác.

> Việc thiếu công cụ quan sát rõ ràng từng bước (indexing → retrieval → generation) cũng làm quá trình phân tích mất nhiều thời gian hơn dự kiến.

_________________

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)
**Câu hỏi:** Mật khẩu tài khoản công ty cần đổi định kỳ không? Nếu có, hệ thống sẽ nhắc nhở trước bao nhiêu ngày và đổi qua đâu?

**Phân tích:**
> Câu trả lời hiện tại chưa đạt yêu cầu đầy đủ, dù đúng một phần. Hệ thống đã xác nhận mật khẩu cần đổi định kỳ, nêu đúng chu kỳ 90 ngày và thời gian nhắc nhở 7 ngày trước, nhưng thiếu thông tin về kênh đổi mật khẩu (SSO portal hoặc Helpdesk). Với tiêu chí chấm điểm, câu này chỉ đạt khoảng 6 điểm.

> Failure mode chính: missing detail trong multi-part question. Đây không phải lỗi sai thông tin, mà là lỗi không extract đủ các ý cần thiết từ context.

> Root cause:
- Retrieval: Các chunk được retrieve có thể đã chứa thông tin về kênh đổi mật khẩu, nhưng không được ưu tiên (ranking chưa tối ưu cho multi-intent query).
- Generation: LLM không được hướng dẫn rõ để đảm bảo trả lời đầy đủ tất cả các phần của câu hỏi (what + when + how), dẫn đến việc bỏ sót ý cuối.
- Indexing: Có khả năng thông tin về “kênh đổi mật khẩu” nằm ở chunk khác hoặc không được chunk cùng với phần FAQ chính, gây phân mảnh nội dung.

> Fix đề xuất:
- Cải thiện prompt để yêu cầu liệt kê đầy đủ từng phần của câu hỏi multi-part.
- Điều chỉnh retrieval (tăng diversity hoặc coverage) để đảm bảo các ý liên quan đều xuất hiện trong context.
- Xem lại chiến lược chunking để gom các thông tin liên quan trong cùng một chunk.

_________________

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

> Nếu có thêm thời gian, tôi sẽ tập trung vào việc cải thiện query transformation để làm rõ intent của người dùng, đặc biệt với các câu hỏi dài hoặc nhiều ý. Đồng thời, tôi muốn thử tuning hybrid weights (dense_weight / sparse_weight) nhằm cân bằng tốt hơn giữa semantic matching và keyword matching.

> Ngoài ra, tôi sẽ điều chỉnh lại chunking theo semantic boundary (chia nhỏ theo section hợp lý hơn) để tăng độ chính xác khi retrieval. Cuối cùng, tôi sẽ bổ sung fallback logic cho các trường hợp “no data”, giúp hệ thống tránh trả lời sai khi không tìm được thông tin phù hợp.
_________________

---

*Lưu file này với tên: `reports/individual/[ten_ban].md`*
*Ví dụ: `reports/individual/nguyen_van_a.md`*
