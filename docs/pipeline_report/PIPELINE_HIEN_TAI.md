# Pipeline hệ thống hiện tại

Sơ đồ vector: [pipeline_he_thong_hien_tai.svg](pipeline_he_thong_hien_tai.svg)

## Cách đọc sơ đồ

- **Whisper small.en** nhận audio và tạo transcript tiếng Anh. Transcript cùng lịch sử hội thoại đi vào **Qwen3-0.6B base** để tạo câu trả lời; TTS đọc câu trả lời và giao diện highlight chữ theo voice.
- Giao diện giữ tối đa **25 lượt trao đổi gần nhất** (50 tin nhắn user/assistant) và gửi chúng cùng lượt mới. Prompt yêu cầu Qwen trả lời thẳng câu hỏi hiện tại trước, rồi có thể hỏi tối đa một câu mở rộng dựa trên chi tiết người học vừa kể. Nếu lịch sử vượt cửa sổ token của model, worker loại các lượt cũ nhất theo cặp để giữ lượt hiện tại.
- **wav2vec2-B** chạy trên audio để tạo bằng chứng CTC/phoneme. Bộ đối chiếu so bằng chứng này với câu tham chiếu, rồi cập nhật mục **Sound details**. Câu tham chiếu có thể đến từ prompt hoặc do người dùng sửa/xác nhận; transcript ASR tự sinh chưa được xác nhận.
- **Qwen pronunciation LoRA** chỉ được gọi khi có khác biệt âm đã được gắn cờ và có reference/alignment phù hợp. Nó viết phần giải thích; không thay ASR, không thay điểm âm học, và không nhận audio trực tiếp.
- **Grammar LoRA** là nhánh tùy chọn trong phần sửa câu. **Fluency** dùng timestamp từ Whisper nếu có.
- **Projector** không được dùng trong luồng chat runtime hiện tại.

## Rủi ro chính: wav2vec2-B báo nhầm

Đây là điểm cần ưu tiên trước khi tin vào phần chấm phát âm. Trong kiểm chứng SpeechOcean762 đã lưu, wav2vec2-B gắn cờ ít nhất một lỗi ở **173/179 câu** mà cả năm chuyên gia đều đánh giá phát âm đúng (96,65%). Ở toàn bộ 2.500 câu test của bộ này, phone-level precision là 11,94% và recall là 88,46%. Đây là tập người học tiếng Anh có tiếng mẹ đẻ Trung, có cả trẻ em; kết quả này không phải phép đo riêng trên audio IELTS, nhưng cho thấy rõ rủi ro false positive khi chuyển miền.

Calibration full-validation trên L2-Arctic-Plus cũng **không tìm được ngưỡng đáp ứng policy** (`no_threshold_meets_fit_policy`); calibration artifact ở trạng thái `ineligible`. Vì vậy, một từ bị gắn cờ hiện nên được hiểu là gợi ý cần kiểm tra, không phải kết luận người học phát âm sai. Qwen LoRA chỉ diễn giải những từ đã bị gắn cờ, nên không thể tự khử false positive của wav2vec2-B.

Để chứng minh độ tin cậy cần đánh giá lại trên bộ audio có cả phát âm đúng và sai, được chấm độc lập theo người nói và accent; báo cáo precision, false-positive rate, recall và tỷ lệ abstain riêng. Không chọn ngưỡng trên test set. Riêng audio IELTS cần có nhãn chuyên gia tương ứng trước khi tuyên bố hiệu năng trên IELTS.

Nguồn số liệu local:

- `data/generated/external/speechocean762/20261021T024600Z/B-full/metrics.json`
- `data/generated/pronunciation-calibration-v1/full-validation/calibration.json`
- `docs/training_report/report_content.json`, chương 10 và chương calibration.

## Sơ đồ nguồn

`scripts/local_speech_engine.py` điều phối ASR, phân tích âm và phản hồi; `scripts/local_chat_worker.py` nạp Qwen base/LoRA; `src/mdd/demo_chat.py` quản lý lượt chat, job và TTS.

Lưu ý: tăng số tin nhắn được gửi không làm tăng cửa sổ token 4.096 của Qwen. Vì vậy, hội thoại ngắn giữ được tối đa 25 lượt gần nhất; các lượt dài có thể khiến worker cắt bớt lịch sử cũ trước khi sinh câu trả lời.
