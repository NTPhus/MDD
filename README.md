# MDD tiếng Anh cho người Việt

Quy trình chuẩn bị dữ liệu, fine-tune và đánh giá nằm trong [TRAINING.md](TRAINING.md). Protocol dữ liệu đã khóa ở [data/protocol/l2arctic-v1](data/protocol/l2arctic-v1/); giấy tờ tham khảo ở [papers](papers/README.md).

[TTS từ văn bản đến âm thanh](papers/tts/README.md) giải thích kiến trúc, input/output và các tác vụ, kèm 7 PDF nền tảng. `.gitignore` bỏ qua dataset gốc, parsed annotations, weights tải về, môi trường và kết quả train local; PDF bài báo, code, config, protocol và metadata vẫn được giữ.

## Lý thuyết STT TTS và kiến trúc model

[Lý thuyết và giả mã đầy đủ](docs/SPEECH_THEORY.md) giải thích waveform, mel-spectrogram, CNN, Transformer, **Quantization Module với Gumbel product Vector Quantization**, self-supervised pre-training, CTC và cách fine-tune đang được code thực thi. Tài liệu có sơ đồ, kích thước tensor, ví dụ và liên kết paper gốc.

| Tác vụ | Input | Output |
| --- | --- | --- |
| STT/ASR | Audio lời nói | Transcript chữ |
| Nhận dạng phoneme của project | Audio lời nói | Chuỗi âm người học phát ra |
| MDD của project | Audio và câu tham chiếu | Lỗi thay, thiếu, thêm âm và vị trí |
| TTS | Text và điều kiện giọng tùy model/API | Âm thanh lời nói |

Với wav2vec 2.0, **CNN** giảm chuỗi waveform thành đặc trưng theo frame; **Transformer** bổ sung ngữ cảnh giữa các frame. Trong **self-supervised pre-training**, nhánh quantizer tạo target rời rạc từ audio chưa bị che để model học dự đoán tại các vị trí bị che. Các codeword không được gán sẵn nhãn phoneme.

Khi fine-tune hiện tại, `Wav2Vec2ForCTC` giữ backbone, tạo head `768 → 83` token và học bằng **CTC loss** từ `perceived_phones`. Nhánh quantizer và loss contrastive/diversity không tham gia giai đoạn này. CNN được đóng băng; feature projection, Transformer và head được cập nhật. Lỗi MDD được suy ra bằng căn chỉnh phone dự đoán với phone chuẩn, không phải head phân loại trực tiếp bốn loại lỗi. [Code thực thi](src/mdd/runtime.py)

A học trên dữ liệu nguồn; B thích ứng giọng Việt có source replay; C học tiếp dữ liệu nguồn cùng ngân sách với B. Batch 1, accumulation 4, AdamW; preset local dùng CPU/FP32, preset Colab dùng CUDA/FP16. Chọn checkpoint bằng F1 validation; hiện chưa có scheduler, warm-up riêng hoặc LoRA. [Giả mã và phân tích fine-tuning](docs/SPEECH_THEORY.md), [hướng dẫn thực hành](TRAINING.md)

Phần TTS trong tài liệu trình bày frontend text, duration/pitch/energy, acoustic model và vocoder, kèm hai giả mã train/inference kiểu FastSpeech 2. Đây là ví dụ kiến trúc; project hiện gọi ai33 API để đọc câu mẫu và feedback.

## Setup và train trên máy local

Môi trường `.venv` đã có PyTorch CPU, BASE và dữ liệu 3.599 mẫu được kiểm tra checksum. Các preset `local_*.json` dùng 4 threads, batch 1, FP32 và gradient checkpointing. CNN được freeze, Transformer và head vẫn được học. Không cần CUDA, Google Drive hoặc key API để train MDD trên máy hiện tại.

```bash
bash scripts/setup_local_training.sh
bash scripts/run_local_training.sh smoke --run-id local-smoke-01
bash scripts/run_local_training.sh A --run-id local-A-seed42
```

Đã chạy smoke local thành công: **1 update kỹ thuật** với Transformer/head trainable, lưu và nạp được checkpoint. Cả tiến trình khoảng 12,51 giây, peak RSS khoảng 2,24 GiB trên một mẫu train 1,03 giây và hai mẫu dev ngắn. Đây chưa phải phép đo thời gian/RAM toàn bộ dữ liệu hoặc benchmark MDD; chưa chạy A/B/C đầy đủ.

Checkpoint, optimizer, log và dự đoán nằm ở `runs/<run_id>/`. [TRAINING.md](TRAINING.md) có lệnh B/C, resume, test và so sánh. CPU sẽ chậm hơn GPU; launcher không âm thầm giảm dữ liệu hoặc ngân sách nghiên cứu. [Notebook Colab](notebooks/train_mdd_colab.ipynb) vẫn là lựa chọn bổ sung khi cần GPU.

## Cơ sở chia dữ liệu

[Phân tích cách chia và đề xuất phù hợp](docs/DATA_SPLIT.md) giải thích vì sao giữ riêng người nói, vì sao mỗi fold Việt có 300/150/150 câu và cách các paper MDD chia dữ liệu. Audit toàn bộ 3.599 mẫu xác nhận không trùng người/audio giữa các vai trò, nhưng mọi mã câu Việt test đã có ở người train: phạm vi là người mới đọc câu trong miền đã học.

Đề xuất v2 giữ chia 2 người train / 1 validation / 1 test và cân đối một nam/một nữ trong target train của mỗi fold, cùng số lượt train như v1. Đề xuất được lưu riêng; code và notebook vẫn dùng v1. Cách chia giúp đánh giá đáng tin và chọn model, chưa có bằng chứng tự làm F1 cao hơn. [Audit dữ liệu](data/generated/setup/split_audit.md), [split v2 đề xuất](data/proposals/l2arctic-v2-balanced/proposal.json)

## Model local

Pipeline mặc định chỉ dùng **`models/pretrained/wav2vec2-base`** để fine-tune MDD. Bản `wav2vec2-base-960h` đã tải chỉ dành cho ASR tùy chọn, không bắt buộc cho bài test. Script tải/kiểm tra model mặc định cũng chỉ chọn BASE. Xem [hướng dẫn và manifest model](models/pretrained/README.md) để biết revision, weights và cách load sau khi cài thư viện inference.

```bash
python3 scripts/download_models.py --verify-only
```

Script kiểm tra checksum và cấu hình hoàn toàn local, không cần PyTorch. Vocabulary phoneme, parser và scorer đã có; checkpoint MDD A/B/C sẽ được tạo khi chạy các lượt train khoa học.

Demo MDD một model hiển thị **câu tham chiếu do người dùng chọn/nhập**, phone dự đoán và từ bị đánh dấu. Phone CTC không tự xuất transcript chữ; `asr_text` được đặt `null`. Yêu cầu nhận dạng chữ cho lời nói tự do cần nhánh ASR hoặc một model đa nhiệm được huấn luyện và đánh giá riêng.

## LLM chính

Sử dụng gateway `https://llm.mrdnd.dev/v1`, endpoint `/chat/completions`, model ID `claude-sonnet-5` theo cấu hình đã chọn. Adapter [scripts/llm_client.py](scripts/llm_client.py) hỗ trợ chat và phản hồi phát âm theo contract hiện có. Gateway nhận key bằng header `Authorization: Bearer ...`.

Key đã được cấu hình trong `.env` local; `.env.example` chỉ chứa biến mẫu, không chứa key thật. Có thể đổi `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_TIMEOUT_SEC`, `LLM_MAX_TOKENS`. Biến môi trường được ưu tiên hơn file. Adapter cần `jsonschema`, cùng dependency với script kiểm tra contract; không cần SDK riêng của nhà cung cấp.

Thử chat:

```bash
python3 scripts/llm_client.py --message "Hello, hãy trả lời tôi bằng tiếng Việt"
```

Tạo feedback từ một assessment, lưu JSON và metadata riêng:

```bash
python3 scripts/llm_client.py --assessment contracts/examples/assessment.json --output data/generated/llm/feedback.json
python3 scripts/validate_contracts.py contracts/examples/assessment.json data/generated/llm/feedback.json
```

Lệnh thứ hai dùng **assessment minh họa**, không phải kết quả model MDD đã train. File `.meta.json` đi kèm ghi `data_kind`, nguồn phản hồi `llm`/`template`, lý do fallback, model yêu cầu/model gateway báo lại, token usage và thời gian từng completion. Không coi metadata model gateway trả về là kiểm chứng nhà cung cấp model thực tế.

Metadata cũng ghi lỗi validation để theo dõi trường hợp cần sửa JSON. Khi chẩn đoán, thêm `--trace` để lưu response nguyên văn vào file `.trace.json` local cùng thư mục output. Trace trong `data/generated/` đã được Git ignore; không dùng nó làm bằng chứng model MDD đã nhận diện đúng lỗi.

Backend kiểm tra assessment trước khi gọi LLM, đưa prompt và schema vào request, sau đó validate JSON và liên kết lỗi. Nếu output sai, thử sửa một lần; nếu vẫn sai hoặc API lỗi, trả lời nhắc mẫu có `source=template`, không bịa lỗi hoặc điểm phát âm. Với audio `uncertain`/`insufficient_audio`, adapter trả hướng dẫn thu lại và không gọi LLM. Schema hợp lệ vẫn cần người có chuyên môn đánh giá nội dung hướng dẫn.

Kiểm tra kết nối thật với key mới đã nhận được câu trả lời tiếng Việt. Kết quả local ở `data/generated/llm/connection_check.json`. Một lần kiểm tra feedback từ assessment minh họa đã qua schema sau một lượt sửa lỗi locale TTS; kết quả ở `data/generated/llm/feedback_checked.json`, metadata ở `feedback_checked.meta.json`. Đây là kiểm tra tích hợp trên một ví dụ, chưa phải đánh giá độ tin cậy của LLM hoặc độ chính xác MDD. Phần MDD inference và UI vẫn cần triển khai.

## TTS qua ai33

Project dùng **ai33 API** cho câu mẫu tiếng Anh và lời giải thích tiếng Việt. Adapter [scripts/ai33_tts.py](scripts/ai33_tts.py) dùng thư viện chuẩn Python, không cần cài model TTS trên máy.

1. Nếu chưa có `.env`, sao chép cấu hình mẫu; lệnh dưới giữ nguyên cấu hình đã có:

   ```bash
   cp -n .env.example .env
   ```

2. Mở `.env` trên máy, nhập API key của tài khoản vào `AI33_API_KEY`. File này đã được thêm vào `.gitignore`. Script tự đọc `.env`, biến môi trường có sẵn được ưu tiên; không cần `source` file. Không đưa key vào frontend, JSON gửi LLM hoặc nội dung commit.

3. Tìm giọng tiếng Anh trong tài khoản:

   ```bash
   python3 scripts/ai33_tts.py --list-voices --provider edge --search en-US
   ```

   Lấy voice ID từ kết quả, điền vào `AI33_VOICE_EN`. Có thể đổi `--provider` sang `elevenlabs` hoặc `minimax` để tìm giọng khác; `--page 2` lấy trang tiếp theo. Chọn giọng đúng accent mục tiêu, nghe thử trước khi dùng để dạy. `AI33_VOICE_VI` đang lấy ID tiếng Việt từ ví dụ đã cung cấp, vẫn cần kiểm tra giọng còn khả dụng với tài khoản.

4. Tạo audio:

   ```bash
   python3 scripts/ai33_tts.py --text "I think this is useful." --locale en-US
   python3 scripts/ai33_tts.py --text "Bạn hãy luyện lại từ think." --locale vi-VN
   ```

   Lệnh in đường dẫn file audio để mở nghe hoặc đưa vào UI. Có thể dùng `--voice-id ID` để chọn giọng cho một lần gọi. `--locale` chọn cấu hình giọng và cache; accent thực tế phụ thuộc voice ID, không có bước tự chuyển accent trong API request.

Luồng API là `POST /v3/text-to-speech` → nhận `task_id` → polling `GET /v1/task/{task_id}` → tải audio. Xác thực bằng header `xi-api-key`. Đây là luồng một giọng cho mỗi đoạn, phù hợp `tts_segments` hiện có; chưa cần endpoint nhiều người nói `/dialogue`. [Tài liệu API](https://openspeaker.ai/developers)

`AI33_BASE_URL` mặc định là `https://api.ai33.pro` theo endpoint tài khoản đã cung cấp. Tài liệu công khai hiện dùng `https://api.openspeaker.ai`; chỉ đổi biến cấu hình nếu tài liệu trong tài khoản yêu cầu host đó. Adapter không tự chuyển host khi lỗi. [Hướng dẫn công khai](https://openspeaker.ai/api-guide.txt)

Audio và receipt được lưu trong `data/generated/tts/ai33/`. Receipt có task ID, host, voice, locale, speed và SHA-256 của text/audio; không lưu key hoặc URL audio có chữ ký. Lặp lại cùng text, host, voice, locale và speed sẽ dùng cache; nếu lần trước đã có task ID nhưng polling/tải bị lỗi, lệnh sẽ tiếp tục task đó. Giữ cache để dùng lại audio, giảm các lần tạo task tốn credit. Một tiến trình sử dụng một cache tại một thời điểm; khi làm backend nhiều worker, cần thêm khóa hoặc hàng đợi chống tạo trùng task.

POST không tự retry. Nếu mất kết nối ngay khi gửi, response JSON không đọc được hoặc không có `task_id`, kiểm tra lịch sử task trên tài khoản trước khi gửi lại vì nhà cung cấp có thể đã nhận yêu cầu. `AI33_TIMEOUT_SEC` giới hạn thời gian polling, `AI33_POLL_INTERVAL_SEC` là khoảng chờ; demo giới hạn mỗi đoạn 5.000 ký tự và audio 25 MiB.

Public OpenAPI chỉ cung cấp schema response tổng quát. Adapter hỗ trợ `task_id`, `status`/`state` và URL ở `audio_url`/`output_uri`/`output_url` trong các envelope thông dụng `data`, `result`, `metadata`, `output`. Khi đã có key cần chạy thử một câu để xác minh response của tài khoản và chỉnh parser nếu cần. Hiện **chưa gọi API thật hoặc kiểm chứng chất lượng giọng**. [Public OpenAPI](https://openspeaker.ai/openapi-generation.json)

## Nối với phản hồi LLM

Backend gọi LLM từ assessment, kiểm tra phản hồi, rồi gọi adapter ai33 cho từng segment hợp lệ:

```python
from pathlib import Path
from scripts.ai33_tts import AI33TTS, load_env as load_tts_env
from scripts.llm_client import LLMClient, load_env as load_llm_env
from scripts.validate_contracts import validate_pair

load_llm_env(Path(".env"))
load_tts_env(Path(".env"))
result = LLMClient().generate_feedback(assessment)  # assessment từ model MDD
feedback = result.feedback
validate_pair(assessment, feedback)
tts = AI33TTS()
audio_files = [
    tts.synthesize(segment["text"], locale=segment["language"])
    for segment in feedback["tts_segments"]
]
```

Đoạn trên là hướng dẫn tích hợp hai adapter; model MDD inference và UI chưa được triển khai, ai33 chưa được kiểm tra bằng key thật. Với feedback tiếng Anh Anh, chọn voice Anh Anh cho `AI33_VOICE_EN` hoặc truyền `voice_id` phù hợp. Giữ nguyên câu mẫu khi gửi TTS; ký hiệu IPA riêng lẻ cần audio mẫu đã kiểm tra.

## Kiểm tra local

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/validate_contracts.py contracts/examples/assessment.json contracts/examples/feedback.json
```

Test LLM/TTS dùng HTTP giả lập; parser/scorer/runtime và launcher/config CPU kiểm tra các trường hợp quan trọng. Unit tests không gọi API và không cập nhật weights pretrained. Smoke CPU được chạy riêng trên model copy; các lượt A/B/C và điểm số trước/sau thích ứng Việt chưa được chạy. GPU Colab vẫn cần xác nhận bằng lượt chạy thật nếu sử dụng.
