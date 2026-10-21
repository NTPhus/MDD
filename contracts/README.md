# Định dạng trao đổi giữa MDD LLM và TTS

Model âm học trả logits và chuỗi phone. Adapter của backend sẽ căn chỉnh, ánh xạ về từ và tạo JSON theo [mdd_assessment.schema.json](mdd_assessment.schema.json). LLM nhận JSON đó để tạo lời hướng dẫn theo [llm_feedback.schema.json](llm_feedback.schema.json), với [prompt](../prompts/pronunciation_feedback.txt).

Hai file trong [examples](examples/) được viết để minh họa định dạng. `data_kind=synthetic_example` đánh dấu ví dụ giả lập; chúng không phải output của model đã train hoặc kết quả gọi LLM.

Pipeline MDD mặc định chỉ dùng BASE với head phone CTC. `reference_text` là câu người học chọn/nhập; `asr_text=null` vì không chạy ASR nhận dạng chữ. Schema vẫn giữ trường nullable này để hỗ trợ nhánh ASR tùy chọn về sau; không suy transcript chữ từ chuỗi phoneme rồi trình bày như kết quả STT.

Kiểm tra định dạng và liên kết giữa hai JSON:

```bash
python3 scripts/validate_contracts.py contracts/examples/assessment.json contracts/examples/feedback.json
```

Script cần package `jsonschema`. Nó kiểm tra kiểu dữ liệu, ID của từ và lỗi, tính hợp lệ của timestamp, confidence chưa hiệu chỉnh, ngôn ngữ TTS và việc giữ nguyên câu mẫu. Schema hợp lệ chưa chứng minh LLM hướng dẫn đúng ngữ âm; phần đó cần đánh giá riêng.

Khi tích hợp, backend cần reject output sai schema hoặc sai liên kết, thử tạo lại tối đa một lần rồi dùng phản hồi mẫu dựa trên lỗi gốc. UI đánh dấu từ từ JSON MDD trực tiếp. Không để câu trả lời LLM ghi đè dự đoán MDD hoặc nhãn chuyên gia trong dữ liệu đánh giá.

Adapter LLM chính đã có ở [llm_client.py](../scripts/llm_client.py), cấu hình model `claude-sonnet-5` qua gateway `llm.mrdnd.dev`. `LLMClient.generate_feedback(assessment)` thực hiện kiểm tra, thử sửa và fallback như trên. Kết quả có `feedback` theo schema và `source`/`fallback_reason` riêng để phân biệt phản hồi LLM với lời nhắc mẫu; metadata không đưa API key vào contract.

TTS chỉ nhận từng `tts_segments[].text` với đúng ngôn ngữ. Nếu backend TTS chưa hỗ trợ một ngôn ngữ, vẫn hiển thị feedback bằng chữ thay vì đọc bằng giọng sai ngôn ngữ.

TTS đã chọn là **ai33 API**. Adapter [ai33_tts.py](../scripts/ai33_tts.py) nhận text và locale qua `AI33TTS.synthesize`; cách cấu hình API key local và nối segment xem [README.md](../README.md). Key không thuộc JSON MDD/LLM. Adapter TTS đã kiểm tra với HTTP giả lập. Demo local hiện dùng JSON suy luận riêng (features/phone/comparison), chưa phát assessment theo contract này và chưa nối LLM/TTS.
