# MDD tiếng Anh cho người Việt

Quy trình chuẩn bị dữ liệu, fine-tune và đánh giá nằm trong [TRAINING.md](TRAINING.md). Protocol dữ liệu đã khóa ở [data/protocol/l2arctic-v1](data/protocol/l2arctic-v1/); giấy tờ tham khảo ở [papers](papers/README.md).

## Notebook local: demo, visualization và kiểm thử so sánh

[mdd_local_demo_comparison.ipynb](notebooks/mdd_local_demo_comparison.ipynb) chia từng bước thành cell riêng, có hướng dẫn tiếng Việt: kiểm tra môi trường/model, audit split, phân bố dữ liệu, nghe audio, waveform/spectrogram, annotation, suy luận BASE/phoneme, unit tests, contract, smoke, train/resume A/B/C, learning curves, đánh giá test và biểu đồ so sánh có kiểm chứng.

Trên Windows, chạy từ thư mục project:

```powershell
pwsh -File scripts/setup_notebook.ps1
pwsh -File scripts/run_notebook.ps1 -OpenBrowser
```

Setup bổ sung JupyterLab, pandas, matplotlib vào `.venv` hiện có và đăng ký kernel **MDD (.venv)** trong project. Máy mới cần chạy `pwsh -File scripts/setup_local_training.ps1` trước để cài môi trường CUDA/model và chuẩn bị dữ liệu. Dùng `-Offline` với script setup notebook để kiểm tra dependency đã cài mà không tải mạng. Đổi cổng bằng `pwsh -File scripts/run_notebook.ps1 -Port 8890 -OpenBrowser`; `Ctrl+C` dừng Jupyter.

Trong notebook, chọn **MDD (.venv)** rồi chạy **Shift+Enter** từng cell hoặc **Run All**. A/B/C đã train đủ ngân sách. Notebook bật `RUN_FINAL_TEST=True` để đánh giá best checkpoint trên toàn bộ test fold 1 và tạo bảng/biểu đồ so sánh. Các cell train giữ nguyên run đã hoàn tất; các cell test tái sử dụng kết quả đầy đủ đã có. Mỗi cell in lệnh PowerShell có thể sao chép để chạy ở terminal. B/C dùng cùng best checkpoint A, cùng fold và ngân sách tương ứng; test chỉ chạy sau khi cố định cấu hình bằng validation.

Chuỗi train thật chạy tuần tự A (2.000 updates) → B (400) → C (400), dùng cùng best checkpoint A. Lệnh khởi chạy trong terminal là `pwsh -File scripts/run_training_sequence.ps1 -Experiment windows-notebook-seed42 -Seed 42 -Fold 1`; nếu chuỗi đã chạy thì theo dõi log, không khởi chạy thêm. State và log nằm tại `data/generated/training/windows-notebook-seed42-fold1/`, checkpoint tại `runs/windows-notebook-seed42-*/`. Khi chuỗi đã dừng giữa chừng và có checkpoint, dùng cùng lệnh với `-ResumeExisting` để tiếp tục.

Hình PNG, bảng CSV, JSON và log được lưu theo phiên trong `data/generated/notebook/`. **Ctrl+S** lưu biểu đồ và output trực tiếp trong `.ipynb`. Notebook dùng audio demo từ source train. Kết quả smoke được ghi rõ là kiểm tra kỹ thuật; bảng/biểu đồ A/B/C chỉ tạo khi có đủ test khoa học và qua verifier, không tạo điểm minh họa thay cho kết quả còn thiếu. Notebook Colab hiện có vẫn nằm ở [train_mdd_colab.ipynb](notebooks/train_mdd_colab.ipynb).

## UI demo local

Chạy trên Windows từ thư mục project, rồi mở **http://127.0.0.1:8765**:

```powershell
pwsh -File scripts/run_demo.ps1
```

Trên Linux dùng `.venv/bin/python scripts/serve_demo.py`. Nhánh MDD local không cần Node, API key hoặc Internet; dùng dependency trong môi trường train. Phần hướng dẫn LLM và giọng đọc ai33 dùng các key server trong `.env`. Chọn câu, thu âm bằng mic, tải file hoặc dùng một bản thu L2-ARCTIC có sẵn. Audio được xử lý trong bộ nhớ, không lưu bản thu lên đĩa. Có thể tải kết quả JSON về máy.

Suy luận dùng **CPU, mặc định 2 threads**, để GPU tiếp tục train ở tiến trình riêng. Tab Huấn luyện đọc log thật trong `runs/`, cập nhật mỗi vài giây. Chọn checkpoint `best` hoặc `latest`; lần phân tích kế tiếp sẽ nạp lại khi weights thay đổi. Nếu checkpoint đang được ghi, chờ lưu xong rồi thử lại. `Ctrl+C` dừng server demo.

BASE chỉ trích xuất đặc trưng âm thanh, chưa có head nhận dạng phoneme. Checkpoint smoke đã có head nhưng chỉ train một bước: demo hiển thị chuỗi âm thử nghiệm và không chấm lỗi phát âm. Checkpoint nghiên cứu cho phép đối chiếu phone với câu mẫu có chuỗi âm chuẩn; câu tự nhập vẫn cần nguồn phoneme chuẩn trước khi đối chiếu. Phone CTC không tạo transcript chữ (`asr_text=null`); các mốc thời gian là xấp xỉ. Sau khi có đối chiếu phoneme theo từ hợp lệ, demo tự yêu cầu hướng dẫn từ LLM và giọng đọc ai33 nếu dịch vụ đã được cấu hình.

Các bản thu mẫu chỉ lấy từ **source train**, không dùng tập test. Demo không thay đổi weights, log hoặc trạng thái tiến trình train. Có thể đổi cổng với `-Port 8766` và threads với `-Threads 1`.

Để máy cùng Wi-Fi/LAN truy cập, chạy `pwsh -File scripts/run_demo.ps1 -Lan`. Launcher tìm IPv4 nội bộ có gateway; có thể chọn rõ bằng `-LanAddress 10.10.63.214`. Mặc định không có `-Lan` vẫn chỉ mở localhost. Chế độ LAN chỉ chấp nhận các địa chỉ đã chọn trong Host và Origin, giữ nguyên giới hạn truy cập file.

Trên máy này, Wi-Fi hiện có IP **10.10.63.214**, nên máy được phép mở **http://10.10.63.214:8765/**. Rule firewall giới hạn TCP 8765, đúng Python chạy server, địa chỉ Wi-Fi và subnet `10.10.63.0/24`; `-AllowedClient` cho phép thêm IP nội bộ cụ thể ở subnet khác. Máy Linux `10.10.60.25/24` dùng tham số này vì khác subnet với Wi-Fi máy chủ. Lệnh thiết lập dưới chạy trong PowerShell Administrator; `-DryRun` chỉ xem cấu hình trước khi áp dụng. Profile `Public` là profile hiện tại của kết nối, script không đổi phân loại toàn mạng.

```powershell
pwsh -File scripts/setup_demo_lan.ps1 -LanAddress 10.10.63.214 -InterfaceAlias Wi-Fi -Profile Public -AllowedClient 10.10.60.25
pwsh -File scripts/run_demo.ps1 -Lan -LanAddress 10.10.63.214
```

Máy ở subnet khác còn cần được router/chính sách LAN–Wi-Fi cho phép kết nối. Rule Windows ở trên không thay đổi ACL của mạng. Nếu client `10.10.60.25` vẫn timeout, nhờ quản trị mạng cho phép TCP từ `10.10.60.25` tới `10.10.63.214:8765` và lưu lượng phản hồi; có thể thử từ một máy thực sự thuộc `10.10.63.0/24` để đối chiếu. Cùng SSID chưa bảo đảm cùng subnet hoặc không bị cách ly. Trên Linux, xem đường đi bằng `ip -4 route get 10.10.63.214` và kiểm tra bằng `curl --noproxy '*' -v --connect-timeout 5 --max-time 10 http://10.10.63.214:8765/ -o /dev/null`.

Nếu IP/interface/profile thay đổi, chạy lại thiết lập với thông tin mạng mới; script cập nhật rule riêng của dự án. HTTP trên IP LAN dùng được tải file, audio mẫu, MDD, LLM và ai33. Microphone từ trình duyệt cần HTTPS hoặc localhost; UI nêu rõ điều này và vẫn cho dùng file/mẫu. [Giới hạn getUserMedia của trình duyệt](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)

Demo cũng đã được triển khai bằng **Nginx trên Windows** tại **http://10.10.63.214/**, cổng `80`, chuyển tiếp tới Python chỉ lắng nghe `127.0.0.1:8765`. Sau khi khởi động lại Windows, chạy lại:

```powershell
pwsh -File scripts/run_demo_nginx.ps1 -LanAddress 10.10.63.214
```

Xem trạng thái bằng `pwsh -File scripts/run_demo_nginx.ps1 -Action Status`; dừng bằng `-Action Stop`. Start lặp lại sẽ giữ deployment đang hoạt động. Launcher quản lý tiến trình nền trong phiên Windows hiện tại; chưa đăng ký Windows service hoặc tác vụ tự khởi động. Nếu thay IP/cổng, dừng deployment rồi chạy lại với tham số mới và cập nhật firewall.

Lần thiết lập đầu trên máy khác, dùng `-Action Setup -LanAddress <IPv4>` để tải bản Nginx đã cố định phiên bản/SHA256, tạo config và chạy `nginx -t`. Sau đó chạy lệnh firewall dưới trong PowerShell Administrator với IP/interface/profile/client thực tế, rồi chạy Start:

```powershell
pwsh -File scripts/setup_demo_nginx_firewall.ps1 -LanAddress 10.10.63.214 -InterfaceAlias Wi-Fi -Profile Public -ProgramPath .tools/nginx/nginx-1.31.7/nginx.exe -AllowedClient 10.10.60.25
```

Nginx kiểm tra Host/Origin, giữ giới hạn JSON 16 MiB, chờ phản hồi tối đa 1500 giây và không tự gửi lại yêu cầu LLM/ai33. Config và log nằm tại `data/generated/demo/nginx/`; không có thư mục static trỏ vào repository. Chuyển backend/restart sẽ tạo phiên phân tích mới, cần bấm phân tích lại nếu trang còn giữ analysis ID cũ. HTTP qua Nginx vẫn có giới hạn microphone của trình duyệt và vẫn phụ thuộc chính sách định tuyến LAN–Wi-Fi. Xem [cấu hình và kiểm thử Nginx](deploy/nginx/README.md).

Với Nginx, ACL mạng cần cho phép `10.10.60.25 → 10.10.63.214 TCP 80` cùng lưu lượng phản hồi. Nếu cả cổng `80` và `8765` đều timeout từ client, cần kiểm tra đường LAN–Wi-Fi hoặc thử từ client thuộc `10.10.63.0/24`; đổi reverse proxy chưa bảo đảm vượt qua chính sách mạng. Chưa có cấu hình HTTPS/domain công khai trong deployment này.

Để chia sẻ demo qua **ngrok HTTPS**, thêm `NGROK_AUTHTOKEN` vào `.env` từ [ngrok Dashboard](https://dashboard.ngrok.com/get-started/your-authtoken), lưu rồi chạy:

```powershell
pwsh -File scripts/run_demo_nginx.ps1 -LanAddress 10.10.63.214
pwsh -File scripts/run_demo_ngrok.ps1
```

Lệnh thứ hai in URL HTTPS để mở trên máy Linux hoặc thiết bị khác. Dùng `pwsh -File scripts/run_demo_ngrok.ps1 -Action Status` để xem URL/trạng thái; `-Action Stop` chỉ dừng tunnel và Nginx ingress riêng, giữ backend/demo LAN. `-Action Setup` chuẩn bị công cụ/config trước khi mở tunnel. Sau khi khởi động lại Windows, chạy lại hai lệnh Start ở trên và dùng URL vừa được in. Trình duyệt có thể hiện màn hình xác nhận của ngrok ở lần truy cập đầu; chọn **Visit Site** nếu có.

Tunnel đi qua Nginx ingress chỉ nghe localhost, chấp nhận đúng tên miền HTTPS được ngrok cấp rồi chuyển tới backend. HTTPS cho phép trình duyệt yêu cầu quyền microphone. Không dùng wildcard Host/Origin hoặc thay toàn bộ Origin trước khi kiểm tra; ngrok inspection bị tắt và API điều khiển chỉ nghe localhost. Token được nạp từ môi trường/`.env`, không đưa vào lệnh hoặc file config tunnel. State/config nằm tại `data/generated/demo/ngrok/`. Xem [cấu hình ingress](deploy/nginx/README.md).

[TTS từ văn bản đến âm thanh](papers/tts/README.md) giải thích kiến trúc, input/output và các tác vụ, kèm 7 PDF nền tảng. `.gitignore` bỏ qua dataset gốc, parsed annotations, weights tải về, môi trường và kết quả train local; PDF bài báo, code, config, protocol và metadata vẫn được giữ.

## Hướng dẫn phát âm và giọng đọc trong demo

Chọn mốc model trong danh sách: BASE trước fine-tune, A trước thích ứng giọng Việt, B sau thích ứng hoặc C đối chứng. So sánh khả năng nhận diện lỗi trước/sau thích ứng dùng A và B trên cùng bản thu; BASE chỉ trả đặc trưng âm thanh vì chưa có đầu nhận diện âm vị. Checkpoint B · best được chọn mặc định khi có sẵn.

Sau **Dùng bản thu mẫu → Phân tích bản thu**, demo hiện ngay kết quả MDD sau khi CPU suy luận xong: từ nào có sai khác, loại thêm/thiếu/thay âm và âm mục tiêu/âm nhận diện. Ba trạng thái cho biết tiến độ MDD, LLM và ai33. Trong khi LLM xử lý, bảng lỗi vẫn xem được; khi LLM trả hướng dẫn, nội dung xuất hiện ngay và ai33 tiếp tục tạo giọng đọc. Sau cùng các nút nghe tiếng Việt, câu mẫu tiếng Anh và từ cần luyện được cập nhật. Bạn cũng có thể thu âm đọc câu mẫu đang chọn.

Rê chuột hoặc đưa focus bàn phím vào từ được đánh dấu để xem các âm model nhận diện, âm mục tiêu và cách sửa; mỗi từ hiển thị đủ các lỗi được gắn với nó. Khi LLM trả kết quả, tooltip cập nhật thêm hướng dẫn luyện. Trên điện thoại, chạm vào từ để mở/đóng; nhấn Escape hoặc bấm ra ngoài để đóng. Tooltip dùng lại kết quả phân tích hiện có.

Backend giữ kết quả MDD theo một mã lượt phân tích; frontend chỉ gửi mã đó để xin hướng dẫn. LLM nhận assessment đã kiểm tra, không nhận bản thu gốc. ai33 chỉ nhận văn bản cần đọc. Confidence chưa hiệu chỉnh và mốc lỗi chưa căn chỉnh được giữ `null`; BASE, smoke hoặc câu chưa có phone chuẩn không được dùng để tự bịa chẩn đoán. Phản hồi LLM chỉ liên kết với các lỗi đã có từ MDD.

Điền `LLM_API_KEY`, `AI33_API_KEY`, `AI33_VOICE_EN`, `AI33_VOICE_VI` trong `.env`. Tìm voice ID đúng tài khoản bằng:

```powershell
& .\.venv\Scripts\python.exe -X utf8 scripts/ai33_tts.py --list-voices --provider edge --search en-US
& .\.venv\Scripts\python.exe -X utf8 scripts/ai33_tts.py --list-voices --provider edge --search vi-VN
```

Giao diện vẫn giữ kết quả MDD nếu LLM hoặc giọng đọc lỗi. Mỗi lượt phân tích dùng lại phản hồi đã tạo để tránh gửi lặp, audio ai33 lưu cache trong `data/generated/tts/ai33/`. Nếu API báo lỗi sau khi đã tạo tác vụ, giữ cache và kiểm tra lịch sử tài khoản trước khi gửi lại.

Phần 32 của notebook lưu riêng từng cell cho trạng thái demo, kết quả MDD, hướng dẫn LLM, bảng lỗi và ba nút nghe audio. Mặc định `RUN_ONLINE_GUIDANCE=False` đọc kết quả thật đã lưu trong `data/generated/demo/`; audio được kiểm SHA-256 rồi nhúng vào notebook. Chỉ bật cờ này khi muốn gọi một lượt dịch vụ mới có thể tính phí. Cấu hình mẫu dành cho hướng dẫn nhiều lỗi dùng `LLM_MAX_TOKENS=8192` và `LLM_TIMEOUT_SEC=240` để tránh cắt JSON hoặc hết thời gian chờ.

Metadata chẩn đoán LLM nằm trong `data/generated/demo/feedback/`, không gửi ra frontend và không chứa nội dung completion hoặc key. Khi một lượt đã gọi dịch vụ thất bại, nút tải lại đọc lại kết quả của lượt đó; sau khi sửa cấu hình, nhấn Phân tích để tạo lượt mới.

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

Smoke CPU trước đây đã chạy **1 update kỹ thuật** để kiểm tra lưu/nạp checkpoint. Thí nghiệm Windows CUDA hiện đã hoàn tất A 2.000 updates, B/C mỗi lượt 400 updates và đánh giá toàn bộ 150 câu test fold 1. F1 lần lượt A 66,37%, B 70,85%, C 65,20%; notebook lưu bảng, biểu đồ và kết quả verifier. Đây là kết quả một fold, seed 42, một người test; chưa phải đánh giá tổng quát trên mọi giọng/người nói.

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

Điền `LLM_API_KEY` vào `.env` local; `.env.example` chỉ chứa biến mẫu, không chứa key thật. Có thể đổi `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_TIMEOUT_SEC`, `LLM_MAX_TOKENS`. Biến môi trường được ưu tiên hơn file. Adapter cần `jsonschema`, cùng dependency với script kiểm tra contract; không cần SDK riêng của nhà cung cấp.

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

Đã kiểm tra LLM thật từ kết quả suy luận checkpoint B trên bản thu `SVBI/arctic_a0117`: 5 hướng dẫn liên kết với 5 sai khác model đánh dấu ở `genuine`, `and`, `voice`, qua schema và kiểm tra liên kết lỗi. Kết quả ở `data/generated/demo/live-feedback-text.json`; metadata riêng trong `data/generated/demo/feedback/`. Đây là kiểm tra tích hợp một ví dụ, chưa phải đánh giá độ tin cậy của LLM hoặc độ chính xác MDD. Giao diện ghi rõ nguồn LLM hay hướng dẫn mẫu.

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

   Lấy voice ID từ kết quả, điền vào `AI33_VOICE_EN`. Có thể đổi `--provider` sang `elevenlabs` hoặc `minimax` để tìm giọng khác; `--page 2` lấy trang tiếp theo. Cấu hình local đã tra tài khoản và chọn `edge_en-US-AvaNeural` cùng `edge_vi-VN-HoaiMyNeural`. Máy/tài khoản khác cần tra lại giọng; accent thực tế phụ thuộc voice ID.

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

Adapter chờ task hoàn tất trước khi dùng URL audio. Tải CDN dùng tối đa ba lần GET với khoảng chờ 1/2 giây cho lỗi tạm thời; không gửi key trong request tải file và không lặp lại POST tạo tác vụ.

Public OpenAPI chỉ cung cấp schema response tổng quát. Adapter hỗ trợ `task_id`, `status`/`state` và URL ở `audio_url`/`output_uri`/`output_url` trong các envelope thông dụng `data`, `result`, `metadata`, `output`. Đã gọi API thật, tạo và tải được MP3 cho câu mẫu tiếng Anh, hướng dẫn tiếng Việt và nhóm từ cần luyện. Đây là kiểm tra tạo/tải audio; chất lượng hướng dẫn và giọng đọc vẫn cần người nghe đánh giá. [Public OpenAPI](https://openspeaker.ai/openapi-generation.json)

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

Demo local đã tích hợp hai adapter theo luồng trên. Cần điền key và voice ID thật vào `.env` để kiểm tra với tài khoản đang dùng; khi chưa có key, bộ kiểm thử dùng provider giả lập. Với feedback tiếng Anh Anh, chọn voice Anh Anh cho `AI33_VOICE_EN` hoặc truyền `voice_id` phù hợp. Giữ nguyên câu mẫu khi gửi TTS; ký hiệu IPA riêng lẻ cần audio mẫu đã kiểm tra.

## Kiểm tra local

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/validate_contracts.py contracts/examples/assessment.json contracts/examples/feedback.json
```

Kiểm thử thứ tự cập nhật giao diện bằng Node (chỉ dùng cho tests, không cần cho server demo):

```powershell
node --test tests/demo_frontend.test.cjs
```

Bộ kiểm thử giữ phản hồi LLM/ai33 ở trạng thái chờ để xác nhận bảng lỗi và hướng dẫn xuất hiện đúng thứ tự, kiểm tra đổi model trong lúc chờ và xử lý dịch vụ lỗi. Provider được giả lập trong unit tests.

Test LLM/TTS dùng HTTP giả lập; parser/scorer/runtime và các launcher kiểm tra những trường hợp quan trọng. Unit tests không gọi API và không cập nhật weights pretrained. Kết quả train/test A/B/C thật trên Windows CUDA được lưu riêng trong notebook; GPU Colab vẫn cần xác nhận bằng lượt chạy thật nếu sử dụng.
