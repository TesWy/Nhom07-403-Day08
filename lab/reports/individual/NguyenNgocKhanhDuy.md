# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Nguyễn Ngọc Khánh Duy  
**MSSV:** 2A202600189  
**Vai trò trong nhóm:** Retrieval Owner  
**Ngày nộp:** 2026-04-13  

---

## 1. Tôi đã làm gì trong lab này?

Trong lab này tôi làm chính phần **Sprint 1 — Build RAG Index**, tức là xây pipeline để đọc tài liệu, cắt thành từng đoạn nhỏ rồi lưu vào ChromaDB.

Cụ thể:

- **Đọc và làm sạch tài liệu:** Viết hàm đọc từng file `.txt`, tách phần header (Source, Department, Effective Date, Access) ra khỏi nội dung chính, rồi bỏ các dòng thừa.
- **Cắt chunk:** Cắt tài liệu theo từng section (`=== ... ===`) trước, nếu section nào quá dài thì cắt tiếp theo đoạn văn. Tôi chọn chunk size 400 tokens và overlap 80 tokens.
- **Embed và lưu:** Dùng OpenAI `text-embedding-3-small` để embed từng chunk rồi lưu vào ChromaDB. Cuối cùng index được 29 chunks từ 5 tài liệu.
- **Kiểm tra index:** Viết hàm `list_chunks()` và `inspect_metadata_coverage()` để xem thử chunk có đúng và đủ metadata không.

Phần này là nền tảng để nhóm làm tiếp Sprint 2 và 3.

---

## 2. Điều tôi hiểu rõ hơn sau lab này

Trước đây tôi chỉ biết khái niệm "chunking" trong lý thuyết, nhưng sau khi tự code mới thấy nó ảnh hưởng nhiều đến kết quả như thế nào.

Chunk nhỏ thì tìm kiếm chính xác hơn nhưng dễ mất ngữ cảnh. Chunk lớn thì giữ được ngữ cảnh nhưng khi embed lại bị "loãng" vì có quá nhiều thông tin. Vì vậy tôi cắt theo section trước — để nội dung của phần này không lẫn sang phần khác — rồi mới cắt nhỏ hơn nếu cần.

Ngoài ra tôi cũng hiểu hơn về metadata: hồi đầu tôi nghĩ nó chỉ là nhãn cho biết file nào. Nhưng thực ra nếu không parse đúng `department` ngay từ Sprint 1, thì Sprint 3 sẽ không filter được query theo từng phòng ban.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn

Khó nhất là parse header của tài liệu — các file không đồng đều nhau. Có file thì có dòng trống trước header, có file thì tên tài liệu viết hoa nằm trên cùng. Ban đầu code của tôi hay bỏ sót metadata vì không xử lý được những trường hợp đó.

Một điều khác làm tôi bối rối lúc đầu là phần overlap: khi chạy thử `list_chunks()`, tôi thấy chunk thứ hai bắt đầu bằng nội dung đã có ở chunk trước, tưởng là bug. Sau đó mới hiểu đây là cố ý — overlap để khi cắt đoạn không bị mất ngữ cảnh ở chỗ nối.

Một điểm nữa: `inspect_metadata_coverage()` báo một vài chunk có `effective_date = "unknown"` vì tài liệu đó không có trường này trong header. Không gây lỗi, nhưng về sau filter theo ngày sẽ không được với những chunk đó.

---

## 4. Phân tích câu hỏi Q07 trong scorecard

**Câu hỏi:** Q07 — Hỏi về "Approval Matrix" trong quy trình cấp quyền.

Đây là câu có điểm thấp nhất ở baseline: chỉ **2/5**. Lỗi không phải ở LLM mà ở bước retrieval.

Nguyên nhân: người dùng hỏi "Approval Matrix" nhưng trong tài liệu gốc (`access_control_sop.txt`) section đó có tiêu đề là "Access Control SOP" — hai cụm từ có nghĩa gần nhau nhưng khi embed thành vector thì lại khá xa nhau (similarity khoảng 0.72, chưa đủ để retrieve). Vì không lấy được chunk đúng, LLM không có thông tin để trả lời.

Điều này cho thấy: dù prompt có hay cỡ nào, nếu retrieval không lấy được chunk đúng thì câu trả lời vẫn sai.

Variant hybrid (dense + BM25) giải quyết được vì BM25 khớp được từ "SOP" trong tiêu đề tài liệu, đẩy chunk đó lên cao hơn. Combined score tăng lên 0.85, reranker xác nhận thêm → Q07 tăng từ **2/5 lên 5/5**.

Nếu ngay từ Sprint 1 tôi thêm trường `aliases` vào metadata của chunk (ví dụ: `"aliases": "Approval Matrix"`), BM25 có thể đã bắt được ngay mà không cần đợi đến Sprint 3.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì?

Tôi sẽ thêm trường **`aliases`** vào metadata khi index, ghi các tên gọi khác của mỗi section. Ví dụ section "Access Control SOP" có thể thêm alias "Approval Matrix, Access Request Form". Làm vậy thì BM25 ở Sprint 3 sẽ có thêm signal mà không cần sửa gì ở Sprint 2 hay 3.

Lý do chọn cải tiến này: scorecard cho thấy Q07 là câu duy nhất bị mất điểm nặng ở baseline, và nguyên nhân rõ ràng có thể xử lý ngay ở tầng indexing.

---
