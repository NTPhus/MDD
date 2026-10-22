# MDD tiếng Anh cho người Việt

Quy trình chuẩn bị dữ liệu, fine-tune và đánh giá nằm trong [TRAINING.md](TRAINING.md). Protocol dữ liệu đã khóa ở [data/protocol/l2arctic-v1](data/protocol/l2arctic-v1/); giấy tờ tham khảo ở [papers](papers/README.md).

Bạn chủ động bắt đầu train bằng lệnh terminal hoặc bật cờ trong cell tương ứng. Hai notebook cascade/Projector mặc định đặt `RUN_TRAIN=False` / `RUN_PROJECTOR_TRAIN=False`, nên Run All không khởi chạy train. Qwen cascade và Projector alignment đã hoàn tất; chưa có checkpoint Projector feedback. Pipeline kiểm thử riêng được thực hiện theo yêu cầu, giữ nguyên cache và checkpoint đã train.

Workflow mới **wav2vec2 + Qwen3-0.6B trên L2-Arctic-plus** nằm trong [notebook train và so sánh cascade](notebooks/mdd_cascade_training_comparison.ipynb). Cả hai model dùng protocol mới [l2arctic-plus-cascade-v1](data/protocol/l2arctic-plus-cascade-v1/): 2.249 train / 450 validation / 900 test, không trùng speaker/utterance giữa các vai trò. wav2vec2 A dùng 1.949 source train, B thích ứng bằng 300 mẫu Việt Nam với source replay 50%; Qwen học lời hướng dẫn tiếng Anh gốc từ nhãn công bố. Checkpoint A/B cũ không được dùng cho thí nghiệm này vì source train cũ chứa 750 mẫu trong test Plus.

Chạy từ project root (Windows dùng `.venv/Scripts/python.exe`, Linux dùng `.venv/bin/python`):

```powershell
.venv/Scripts/python.exe -X utf8 scripts/run_cascade_training.py --phase prepare
.venv/Scripts/python.exe -X utf8 scripts/run_cascade_training.py --phase all
.venv/Scripts/python.exe -X utf8 scripts/run_cascade_training.py --phase status
.venv/Scripts/python.exe -X utf8 scripts/report_cascade.py
```

Launcher chạy tuần tự A → B → Qwen → audio test → Qwen oracle → bốn cấu hình A/Q0, A/Q1, B/Q0, B/Q1 → report, có log/checkpoint và resume. `--phase train` chỉ train, `--phase evaluate` chỉ đo các model đã train. Cấu hình ở [cascade_l2arctic_plus.json](configs/cascade_l2arctic_plus.json); giữ A 2.000 updates, B 400 updates và Qwen 3 epochs. Config C source-only cùng budget B được cung cấp riêng để làm đối chứng thích ứng khi cần. Các file cấu hình/code quan trọng được hash khi bắt đầu; thay đổi recipe cần run/report root mới.

Qwen chỉ nhận câu tham chiếu và các nhãn âm vị, không nhận audio. Oracle dùng nhãn công bố; cascade dùng âm wav2vec2 dự đoán và đối chiếu canonical, không đọc perceived/gold để tạo đầu vào. Trường hợp không gán được âm vào từ sẽ abstain và vẫn tính trong mẫu số. JSON sai không được thay template để nâng tỷ lệ thành công. Qwen train bằng NF4 QLoRA, đánh giá cả Q0/Q1 cùng BF16/FP16 và greedy; loss projection theo chunk giảm bộ nhớ mà giữ objective. Đã kiểm tra loss/gradient và một update thật trên các mẫu train dài nhất, LoRA đổi trọng số, peak allocated khoảng 1,82 GiB; đây là smoke kỹ thuật, không phải kết quả chất lượng sau train.

Notebook có audio mẫu và biểu đồ dữ liệu; report tại `data/generated/cascade-l2arctic-plus-seed42/report/` có learning curves, CSV, JSON, PNG/SVG và `report.html` tự chứa. Audio báo PER/F1/FRR; feedback báo word-key coverage, BLEU-2, ROUGE-L, NLL/perplexity và latency. Delta ROUGE-L bootstrap ghép cặp theo speaker; không đánh đồng lexical similarity với tính đúng ngữ âm hay tiến bộ của người học. Khi train/test chưa xong, report ghi đang chờ, không có điểm sau train giả định. Nhánh duyệt template trước đây vẫn được giữ cho dữ liệu tự tạo; workflow công bố mới không yêu cầu duyệt lại các mẫu đó.

## Nhánh Projector multimodal và đối chứng cascade

[Notebook Projector](notebooks/mdd_projector_training_comparison.ipynb) thêm nhánh **audio → wav2vec2 B đóng băng → Linear/GELU/Linear → Qwen3-0.6B**. Vector audio 768 được chiếu sang embedding 1024 rồi ghép với text; không đưa nhãn lỗi gold vào đầu vào. Giữ cascade hiện tại và cùng split 2.249/450/900. Cấu hình riêng ở [projector_l2arctic_plus.json](configs/projector_l2arctic_plus.json).

```powershell
.venv/Scripts/python.exe -X utf8 scripts/run_projector_training.py --phase prepare
.venv/Scripts/python.exe -X utf8 scripts/run_projector_training.py --phase all
.venv/Scripts/python.exe -X utf8 scripts/run_projector_training.py --phase status
.venv/Scripts/python.exe -X utf8 scripts/report_projector.py
```

Runner chờ B mới hoàn tất, cache đặc trưng trên CPU rồi chờ GPU lock của cascade. Giai đoạn 1 train Projector bằng audio/transcript train trong 1 epoch (prompt không chứa transcript); giai đoạn 2 train Projector + LoRA q/v của Qwen trong 3 epochs bằng feedback tiếng Anh. Mỗi giai đoạn có stress test một update thật trước khi chạy đầy đủ, checkpoint/resume và chọn best bằng validation. Khả năng chứa mô hình đầy đủ trên 4 GB chỉ được xác nhận khi stress test GPU hoàn tất; kiểm thử CPU không thay thế phép đo đó.

P0 là Projector đã alignment + Qwen gốc, P1 là sau task tuning. Report so sánh P0/P1 và cascade B_Q1 trên cùng 900 test, giữ JSON lỗi trong mẫu số, lưu loss curves, CSV/JSON, PNG/SVG, ví dụ cố định và bootstrap theo speaker tại `data/generated/projector-l2arctic-plus-seed42/report/`. Đây là thử nghiệm nhỏ: alignment chỉ dùng 2.249 audio/transcript, khác 200k CommonVoice và backbone lớn trong [bài báo gốc](https://aclanthology.org/2026.findings-eacl.190/); không gán điểm công bố cho Qwen3-0.6B. So sánh hai nhánh không cùng compute vì có thêm giai đoạn alignment. Chưa benchmark xong thì report hiển thị đang chờ.

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

## Đối thoại Qwen local

Mở **http://127.0.0.1:8765/#chat** hoặc chọn **Đối thoại AI**. Chọn một trong 8 chủ đề (đời sống, du lịch, ăn uống, công việc, sở thích, công nghệ, học tập, thành phố), hoặc chọn **Free conversation** / **Read a sentence**. Bot mở lời trước theo chủ đề. Giao diện dùng tím–trắng: sidebar và thẻ chủ đề nền trắng, nút chính màu tím; câu của bạn bên phải trên nền tím, câu của Qwen bên trái trên nền lavender nhạt. Câu trả lời hiện đầy đủ trước khi đọc, rồi highlight từng từ đang phát. Bảng **Thử dùng các từ sau** có từ khóa bấm nghe và mẫu mở đầu câu, nằm phía trên nút mic tròn ở giữa. Lịch sử được giữ riêng theo chủ đề và coach; mũi tên quay lại mở danh sách chủ đề. **Kết thúc** dừng mic/giọng đọc, gửi yêu cầu hủy lượt đang xử lý và quay lại danh sách; lịch sử vẫn được giữ.

Nhấn mic **Nhấn để nói**, nhấn lần nữa để gửi hoặc chọn **Tải audio**, không có ô nhập tin nhắn. Bản thu tối đa 30 giây; file WAV/MP3/M4A/OGG/FLAC/WebM được trình duyệt giải mã và đổi thành WAV mono 16 kHz. Dưới mỗi lượt nói có thanh **Xem lỗi phát âm**, luôn đóng khi có kết quả mới. Bấm mở mới thấy các từ model đánh dấu cần luyện, âm kỳ vọng/nhận được và hướng dẫn; mở panel không tự đọc chúng. Tùy chọn mode, coach, voice, cuộc trò chuyện mới và xuất JSON nằm trong nút **Giọng đọc** trên header (ngoài phòng là **Settings**). Có thể dừng AI/voice độc lập và nghe lại bản thu bằng **Nghe bản thu**.

Trong panel phát âm, **Câu dùng để đối chiếu** cho biết chính xác câu đang dùng làm mục tiêu. Nếu ASR trả từ ngoài CMUdict như `fravorit`, hệ thống hiện một trạng thái chờ xác nhận thay vì nhiều thông báo thiếu reference. Có thể có bản nháp chính tả (`favorite` hoặc `flavorist` đều là ứng viên), nhưng chưa dùng để chấm. Người dùng sửa/xác nhận câu mình đã nói rồi bấm **Đối chiếu audio này**; `/api/chat/reference` dùng lại phoneme đã dự đoán của bản thu, không cần ghi lại, chạy lại ASR/wav2vec2/Qwen hoặc tạo lượt chat mới. Transcript và lịch sử gốc giữ nguyên; lời khuyên cũ được xóa khi câu đích thay đổi. Đây là tham chiếu do người dùng xác nhận, không phải câu gold độc lập của bài test. Từ chưa có âm chuẩn sẽ không bị tự gán âm hoặc âm của từ đó sang từ bên cạnh.

Trong panel, mỗi từ được đánh dấu có một dòng **từ · phiên âm IPA · Nghe**; bấm vào từ để mở chi tiết sai khác âm. **Nghe cả câu** nằm cuối phần **Sound details**, đọc đúng câu đang dùng làm mục tiêu, không đọc câu trả lời bot hoặc câu hoàn thiện thay cho nó. Các nút này dùng giọng tiếng Anh của trình duyệt và chỉ phát khi bấm. Phiên âm rộng theo CMUdict Mỹ, giữ trọng âm trong metadata `pronunciation_phones` riêng; chuỗi `phones` dùng so sánh âm giữ nguyên. Bản ghi cũ thiếu metadata trọng âm chỉ hiện các âm có sẵn, không tự đoán trọng âm.

Bot được yêu cầu phản hồi trực tiếp ý người học, không nhắc lại hoặc đọc lại câu của người học. Phần hoàn thiện ý vẫn giữ riêng trong mục con **Hoàn thiện câu (tùy chọn)**, đóng mặc định bên trong panel phát âm và chỉ hiện khi có bổ sung. Không lặp lại khối **You said**. **Listen** chỉ đọc mẫu khi người dùng bấm. Ví dụ, sau câu hỏi đi đâu, `Japan` có thể thành `I would love to travel to Japan.`; `Yesterday I go to the park with my friend` giữ nguyên từ `go`. Parser chặn thay từ, đổi thì, xóa/đảo từ và chèn từ vào giữa câu đã nói. Chỉ thêm khung đầu câu đã nhận diện từ câu hỏi trước hoặc kết thúc câu bằng nội dung chính người học đã nói. Thiếu ngữ cảnh thì giữ nguyên; output vi phạm bị withheld/unavailable và hội thoại vẫn tiếp tục. Kết quả mang `mode=completion_only`; thẻ sửa ngữ pháp của phiên cũ được ẩn để tránh nhầm.

**Practise this sentence** chọn câu đã hoàn thiện làm mục tiêu cho bản thu tiếp theo. Browser gửi `correction_job_id`; server lấy câu từ một lượt completion-only đã hoàn tất cùng topic/coach, không nhận câu mục tiêu do client tự gán hoặc gợi ý sửa ngữ pháp cũ. Khi đọc lại, giữ tên trường tương thích `reference_source=corrected_sentence`, so âm trên audio mới và bỏ qua bước hoàn thiện câu. **Back to conversation**, đổi topic/model hoặc rời màn hình xóa lựa chọn luyện lại. Câu đã lưu sau reload vẫn xem/nghe được; luyện lại cần lượt có ID còn hiệu lực trong phiên. Audio gốc luôn đối chiếu transcript ASR gốc, không dùng câu đã hoàn thiện để chấm lại audio cũ.

Chủ đề, lời mở đầu và ngân hàng từ/mẫu câu là nội dung biên soạn sẵn tại [speech_topics.py](src/mdd/speech_topics.py), không phải kết quả dự đoán. Qwen nhận lời mở đầu, lịch sử lời nói gốc và câu trả lời bot; câu hoàn thiện không ghi đè lịch sử. Bot được yêu cầu trả lời tự nhiên, không bắt buộc hỏi ở mỗi lượt, cho phép câu hỏi yes/no, hỏi rõ khi nghe không hiểu và đi theo khi người học đổi chủ đề. Code không nối câu hỏi mẫu sau câu trả lời hợp lệ của Qwen. Context dài được rút theo từng cặp lượt, giữ nguyên câu hiện tại. Từ gợi ý chỉ hiện khi khớp câu hỏi thực tế của bot; câu trả lời thông thường hoặc đổi chủ đề sẽ ẩn gợi ý cũ. Chức năng này không train thêm model; Qwen3-0.6B vẫn có thể hiểu nhầm hoặc phản hồi chưa tự nhiên.

Chữ trả lời hiện dần theo từ từ output thật của Qwen. Mặc định **Browser · Live voice** chỉ đọc câu trả lời của bot theo từng câu/đoạn ngay khi đủ chữ. Trình duyệt hỗ trợ word boundaries sẽ tô sáng từ đang đọc; trình duyệt khác vẫn hiện chữ dần và đọc theo đoạn. Lỗi/hướng dẫn phát âm đến sau không được đọc thêm; browser live, **Phát lại** và ai33 đều chỉ đọc câu trả lời. **Stop voice**, đổi tab hoặc tắt **Read aloud** chặn audio đang đọc và phần đến muộn. Nếu trình duyệt chặn tự phát, nhấn **Phát lại** khi có kết quả. **ai33 · Full reply** chờ lượt xử lý hoàn tất rồi tạo audio, có thể dùng API credits. Nghe từ mẫu hoặc câu đã hoàn thiện cần bấm nút riêng trong panel.

Bộ lọc stream chặn câu lặp hoàn chỉnh trước khi hiện chữ/phát voice, kể cả câu ngắn, câu lặp sau lời dẫn và dạng chuyển ngôi `I like pizza` → `You like pizza`. Câu có thông tin mới như `Japan sounds exciting` và câu hỏi làm rõ ngắn vẫn được giữ. Câu chào/câu hỏi dài của bot trong lịch sử cũng được kiểm tra để chặn phát lại lời mở đầu. Raw output giữ tại `metadata.raw_reply_text`, phần bị bỏ tại `omitted_echoes` (cùng alias tương thích `omitted_leading_echoes`). Nếu output chỉ có câu lặp, Qwen thử sinh lại đúng một lần trên cùng lịch sử/câu nói, tối đa 96 token/35 giây, trước khi dùng lời xin làm rõ cuối cùng. `reply_attempts`, `reply_generation_attempts` và `reply_source=qwen_retry` ghi rõ lượt thử lại; không tạo lượt chat hay đổi câu gốc. Đây là lọc mẫu câu xác định, không đảm bảo nhận biết mọi cách diễn đạt lại cùng ý.

Riêng hội thoại chọn `generation_profile=conversation`: sampling `temperature=0.7`, `top_p=0.8`, `top_k=20`, `min_p=0` theo [hướng dẫn Qwen3-0.6B cho non-thinking](https://huggingface.co/Qwen/Qwen3-0.6B#best-practices). Hoàn thiện ý và coaching có cấu trúc giữ cấu hình sinh trước đó; thiết lập này không đổi trọng số, quá trình train hay benchmark nghiên cứu.

Với chủ đề café, một lựa chọn rõ như `I would like to order one coffee and one milk tea please` mở frame gọi món có bằng chứng từ chính lời người học. Đã chọn đồ uống thì Qwen được yêu cầu hỏi một chi tiết chưa có: nóng/lạnh, kích cỡ, rồi dùng tại chỗ/mang đi. Trạng thái và lời đã trả lời được dựng từ lịch sử thật; câu hỏi mở đầu biên soạn sẵn không được chèn lại vào prompt của frame này. Lời chào café ban đầu cũng chỉ hỏi gọi món, không hỏi thêm lý do món yêu thích. Câu trả lời ngắn được nối với câu hỏi trước, còn câu không nhận diện chắc hoặc chuyển sang chuyện khác sẽ rời frame để tiếp tục hội thoại tự do. Đây là bộ quy tắc café có giới hạn, không phải bộ hiểu ý tổng quát cho mọi chủ đề.

Trong frame café, mỗi câu Qwen sinh được kiểm tra trước khi chuyển tới UI/voice: chặn Welcome lại, hỏi lựa chọn đã biết, hỏi lại nhiệt độ/kích cỡ đã có hoặc nói nhầm vai khách hàng. Khi bước tiếp theo cần một câu hỏi, lời xác nhận chỉ được công bố cùng câu hỏi hợp lệ; chỉ có lời xác nhận thì thử Qwen lại một lần. `conversation_state`, `blocked_questions` và `reply_attempts` ghi rõ bằng chứng, nguyên nhân chặn và output gốc. Không thay bằng câu hỏi soạn sẵn dưới tên Qwen. Nếu cả hai lượt không tạo được câu hợp lệ, trạng thái `conversation_unavailable` nói rõ không tiếp lời được, không bảo người dùng rằng audio/câu gọi món chưa rõ. Kiểm thử text-fixture với Qwen thật: `python scripts/check_order_conversation_live.py`; ASR/phát âm/completion trong phép thử này là fixture được ghi rõ, không phải kết quả đánh giá audio.

Pipeline CPU local: **wav2vec2-base-960h ASR → wav2vec2 B dự đoán âm → đối chiếu CMUdict → Qwen trả lời và gợi ý tiếng Anh**. Qwen gốc sinh câu đối thoại; lựa chọn Original/Fine-tuned điều khiển model tạo gợi ý phát âm bằng nhãn âm ước lượng (không dùng Projector). Các bản thu, transcript, sai khác âm và gợi ý đều thuộc lượt chat riêng. ASR và B chạy trước, nên có thể xem transcript/âm ước lượng trong lúc Qwen còn xử lý; gợi ý không hợp lệ được đánh dấu unavailable, không dựng kết quả model giả. Có thể hover/tap từ được đánh dấu để xem âm kỳ vọng, âm nhận được, chuỗi ARPAbet và nghe từ mẫu.

**Conversation** đối chiếu âm với transcript ASR suy ra, nên có thể sai do ASR nghe nhầm hoặc sửa từ. **Read a sentence** dùng câu mẫu độc lập với audio; câu chuẩn và nút nghe mẫu vẫn hiện trên khung chat khi đóng Settings. Hai chế độ đều chỉ trả lời/đọc bằng tiếng Anh; model vẫn có thể đánh dấu sai, không diễn giải thành điểm chính xác hay kết luận của giáo viên. Chức năng mic dùng Web Audio/AudioWorklet, cần HTTPS hoặc localhost và quyền microphone; không dùng SpeechRecognition hoặc gửi audio tới dịch vụ nhận dạng của trình duyệt. Upload dùng được khi mic không khả dụng.

Worker riêng dùng CPU 2 threads, giữ GPU cho tác vụ người dùng tự chạy. Qwen gốc/LoRA dùng chung backbone trong RAM; model nạp khi xử lý audio. Không gọi LLM API và không khởi động train. SessionStorage giữ lịch sử của tab; server giữ tối đa 32 lượt trong bộ nhớ và không lưu audio upload ra đĩa. Giọng mặc định dùng SpeechSynthesis tiếng Anh; chọn ai33 sẽ gửi text trả lời của bot để tạo voice tiếng Anh và có thể tính phí. Không cần API key cho ASR, MDD, Qwen và giọng trình duyệt.

Source của mục mới: [demo/chat.js](demo/chat.js), [demo/chat.css](demo/chat.css), [src/mdd/demo_chat.py](src/mdd/demo_chat.py); các endpoint `/api/chat/*` tách khỏi `/api/analyze*` và `/api/feedback`. Khởi chạy web bằng launcher bên dưới.

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

Sarah là giọng bot mặc định khi ai33 khả dụng và `.env` có `AI33_VOICE_EN_NAME=Sarah - Mature, Reassuring, Confident`; có thể đổi về Browser trong Settings. Tên giọng công khai trả về qua `/api/chat/config`, API key vẫn ở backend. Các nút nghe từ/IPA/câu mẫu thủ công vẫn dùng browser như trước. ai33 chưa trả word timestamps nên highlight theo playhead là ước lượng. Giọng mới áp dụng cho lượt mới; audio cache cũ không bị đổi tên thành Sarah.

Đã kiểm tra theme ở 1440/768/390/360 px bằng job/voice giả lập: `node scripts/check_purple_sarah_browser.cjs` (mặc định preview 8767; đặt `MDD_QA_URL=http://127.0.0.1:8765` khi kiểm tra web đang chạy). Screenshot, kết quả và đoạn nghe thử Sarah tại `data/generated/purple-sarah-qa/`. Trang tham chiếu `senglish.net/app` yêu cầu đăng nhập; bố cục dựa trên luồng chọn chủ đề/hội thoại và ảnh mẫu người dùng đã cung cấp.

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

Adapter [scripts/llm_client.py](scripts/llm_client.py) có hai backend: `LLM_BACKEND=api` dùng gateway hiện có; `LLM_BACKEND=local` chạy **Qwen3-0.6B** trực tiếp bằng Transformers, không gọi gateway và không cần LLM API key. Cả hai nhận assessment từ wav2vec2, kiểm tra JSON/liên kết lỗi và chuyển văn bản hợp lệ sang **ai33 TTS** như trước. Khi local lỗi, hệ thống trả hướng dẫn mẫu, không tự chuyển sang API.

Chọn Qwen3-0.6B theo GPU đã ghi trong notebook: **RTX 3050 Laptop 4 GB**, khoảng 3,2 GB trống tại thời điểm đo. [Model card Qwen](https://huggingface.co/Qwen/Qwen3-0.6B) công bố hỗ trợ đa ngôn ngữ, giấy phép Apache 2.0 và tắt thinking bằng chat template. Đây là model sinh hướng dẫn; nó không nhận audio hoặc sửa độ chính xác nhận dạng của wav2vec2. Khả năng tiếng Việt và hướng dẫn ngữ âm của checkpoint gốc cần được đánh giá trên tác vụ này.

Tải checkpoint công khai (~1,5 GB), khóa revision và kiểm SHA-256 với Hub:

```powershell
.venv/Scripts/python.exe scripts/download_feedback_model.py
.venv/Scripts/python.exe scripts/download_feedback_model.py --verify-only
.venv/Scripts/python.exe scripts/llm_client.py --env-file configs/feedback_local.env.example --assessment contracts/examples/assessment.json --output data/generated/llm/qwen-baseline.json --trace
```

Linux dùng `.venv/bin/python` thay cho `.venv/Scripts/python.exe`. File [feedback_local.env.example](configs/feedback_local.env.example) dùng CPU inference để giữ VRAM cho train; giới hạn context 4.096 token, tắt thinking. Có thể chọn `LLM_LOCAL_DEVICE=cuda` sau khi đo bộ nhớ thực. Lượt thử trên assessment minh họa chỉ kiểm tra tích hợp, không chứng minh chất lượng hướng dẫn. Bản gốc đã cho lời giải thích sai/chung chung ở âm TH dù JSON hợp lệ; **chưa kích hoạt backend này thay gateway trên demo**. Không bật cho người học chỉ vì schema hợp lệ.

Pipeline Qwen hiện nhận dữ liệu **assessment → feedback đã được người có chuyên môn duyệt**. L2-ARCTIC gốc có audio/nhãn âm vị; bản mở rộng **L2-Arctic-plus** có thêm nhãn lời hướng dẫn công bố (xem bên dưới), có thể dùng để xây nhánh train từ dữ liệu công khai. [review.example.jsonl](data/feedback/review.example.jsonl) là mẫu tổng hợp **chưa duyệt**, không phải dữ liệu train. Có thể nhập JSON tải từ demo thành hàng đợi duyệt:

```powershell
.venv/Scripts/python.exe scripts/prepare_feedback_data.py draft path/to/downloaded-result.json --output data/feedback/drafts.jsonl
```

Người duyệt sửa nội dung hướng dẫn, xác nhận âm mong đợi/âm model nhận diện, `error_id`, từ, cách đặt môi/lưỡi và bài luyện; sau đó điền `review.approved=true`, `review.reviewer`, `review.reviewed_at` (ISO date). `group_id` gom các biến thể cùng bài luyện hoặc cùng người học để không rò rỉ giữa các tập. Script còn gộp các nhóm dùng chung câu tham chiếu. Không đánh dấu dữ liệu tổng hợp thành suy luận audio thật. Thu thập khoảng 200–500 cặp đa dạng làm tập khởi đầu; đây là ngân sách đề xuất, chưa có bằng chứng số lượng này đủ cải thiện.

```powershell
.venv/Scripts/python.exe -m pip install -r requirements/feedback.txt
.venv/Scripts/python.exe scripts/prepare_feedback_data.py prepare data/feedback/reviewed.jsonl --output data/processed/feedback-v1
.venv/Scripts/python.exe scripts/train_feedback_model.py --check-data-only
.venv/Scripts/python.exe scripts/train_feedback_model.py --config configs/feedback_qlora_rtx3050.json
```

Recipe [QLoRA cho RTX 3050](configs/feedback_qlora_rtx3050.json) dùng NF4 4-bit, LoRA rank 8 ở `q_proj`/`v_proj`, batch 1, tích lũy gradient 8, context tối đa 3.072 token và gradient checkpointing. Chỉ token câu trả lời assistant đóng góp loss; prompt được mask `-100`. Model nền giữ nguyên; train các adapter LoRA. Chọn checkpoint bằng validation loss, không dùng test để train. Chạy riêng với tiến trình train MDD; khả năng vừa VRAM và tốc độ trên RTX 3050 cần xác nhận bằng lượt train thật. Hiện chưa có dữ liệu đã duyệt hoặc lượt QLoRA khoa học hoàn tất.

Setup Windows ngày 21/10/2026 đã cài `peft`, `accelerate`, `bitsandbytes`, tải và xác minh Qwen3-0.6B. Kiểm tra kỹ thuật bằng Trainer đã hoàn thành 1 optimizer update, accumulation 8, validation và lưu checkpoint trên RTX 3050 4 GB; tensor thử dài 3.072 token (2.308 token thật, còn lại padding), peak allocated khoảng 2,12 GiB. Collator dùng `logits_to_keep` và nhãn đã shift để bỏ projection ở vị trí không đóng góp loss, vẫn giữ toàn bộ prompt; test so sánh loss và gradient với cách tính đầy đủ đã đạt. Xem [API Qwen3](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen3) và report tại `runs/local-checks/feedback-qlora-20261021-trainer/report.json`. Đây là kiểm tra kỹ thuật dùng mẫu chưa duyệt, chưa phải train hay đánh giá chất lượng adapter chính.

[Notebook duyệt dữ liệu và train LLM](notebooks/feedback_review_and_training.ipynb) đã có output, audio và công cụ sửa assessment/feedback, nhập người duyệt rồi bấm lưu. Có 300 bản nháp template offline tại `data/feedback/drafts-20261021-cef406b.jsonl`, tất cả còn `approved=false`; nguồn là kết quả validation đã lưu của A/B, không dùng test. Nếu mọi mẫu được duyệt, split dự kiến là 242/29/29 train/validation/test. Hướng dẫn template cần được sửa và kiểm tra chuyên môn. Dữ liệu được duyệt lưu riêng ở `data/feedback/reviewed.jsonl`; các cell prepare/train mặc định tắt và giữ yêu cầu tối thiểu 100 train, 20 validation. Tạo lại bộ nháp mới bằng `python -X utf8 scripts/create_feedback_review_drafts.py --output data/feedback/ten-moi.jsonl`.

Để dùng nhãn hướng dẫn đã công bố, [notebook L2-Arctic-plus](notebooks/l2arctic_plus_dataset_audit.ipynb) kiểm tra release chính thức ghim commit `e8c7684c545d94f2419fc4621e745236c192a353`, đã tải vào `data/external/l2arctic-plus/`. File thực tế có **2.699 train / 900 test**, khác số train 1.699 ghi trong README của repo tác giả. [Bài báo Findings of EACL 2026](https://aclanthology.org/2026.findings-eacl.190.pdf), mục 3.2, mô tả GPT-4o tạo lời hướng dẫn và ba người kiểm tra bộ đánh giá; mục 6.1 nói rõ lời hướng dẫn train được GPT-4o tạo **không qua kiểm tra thủ công**. Không cần người dùng duyệt lại toàn bộ nhãn công bố như các template tự tạo, nhưng phải giữ đúng provenance. Split chính thức không trùng utterance/người nói; có 300 câu tham chiếu chuẩn hóa chung, nên không phải đánh giá trên câu hoàn toàn mới. Nhãn gốc bằng tiếng Anh, chưa chuyển sang contract/tiếng Việt của demo và chưa train bằng Plus. Script `python -X utf8 scripts/audit_l2arctic_plus.py` kiểm tra checksum, cấu trúc và split; report ở `data/generated/setup/l2arctic_plus_audit.json`.

Đánh giá model gốc và adapter trên cùng validation; sau khi khóa recipe mới dùng test. Hai báo cáo lưu cả phản hồi để giáo viên so sánh tính đúng, cụ thể và dễ thực hành:

```powershell
.venv/Scripts/python.exe scripts/evaluate_feedback_model.py --split validation --output data/generated/llm/qwen-validation-base.json
.venv/Scripts/python.exe scripts/evaluate_feedback_model.py --split validation --adapter runs/feedback-qwen3-06b-v1/adapter --output data/generated/llm/qwen-validation-adapter.json
```

Báo cáo tự động chỉ đo tỷ lệ phản hồi hợp lệ/liên kết đúng assessment, kể cả tỷ lệ đúng ngay lần đầu; không đo tính đúng của hướng dẫn hoặc sự tiến bộ của người học. Muốn đo cải thiện phát âm, cần thu lại bài đọc trước/sau luyện và chấm độc lập. Chỉ sau khi đạt chất lượng mới chép các thiết lập local vào `.env`, điền `LLM_LOCAL_ADAPTER_PATH` và khởi động lại server demo. Giữ nguyên các thiết lập `AI33_*`.

Backend API hiện có dùng gateway `https://llm.mrdnd.dev/v1`, endpoint `/chat/completions`, model ID `claude-sonnet-5`. Gateway nhận key bằng header `Authorization: Bearer ...`.

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

   Lấy voice ID từ kết quả, điền vào `AI33_VOICE_EN`. Có thể đổi `--provider` sang `elevenlabs` hoặc `minimax` để tìm giọng khác; `--page 2` lấy trang tiếp theo. Cấu hình local hiện chọn **Sarah - Mature, Reassuring, Confident**, ID `elevenlabs_EXAVITQu4vr4xnSDxMaL`, cùng `edge_vi-VN-HoaiMyNeural`. Sarah đã được xác nhận qua danh sách giọng của tài khoản và tạo audio thật. Máy/tài khoản khác cần tra lại giọng; accent thực tế phụ thuộc voice ID.

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

## Kiểm thử độc lập trên SpeechOcean762

[Notebook kiểm thử ngoài L2-ARCTIC](notebooks/mdd_external_validation.ipynb) lưu kết quả thật, biểu đồ và ví dụ audio theo từng cell. Dataset [SpeechOcean762, Interspeech 2021](https://www.isca-archive.org/interspeech_2021/zhang21x_interspeech.pdf) có nhãn từ 5 chuyên gia; giấy phép gốc [OpenSLR 101: CC BY 4.0](https://www.openslr.org/101/). Mã nguồn chính thức được ghim tại commit `613968e3b0b789fc33936fb5eba1973176ba7d11`.

Protocol được chốt trước suy luận trong [configs/external_speechocean762.json](configs/external_speechocean762.json): B chạy toàn bộ 2.500 câu test; A/C và B được so sánh riêng trên cùng 179 câu mà cả 5 chuyên gia chấm mọi âm đúng, không ghi nhận âm thừa. Không train lại, chỉnh ngưỡng hoặc gọi LLM/ai33. Nhãn accent nặng nhưng đúng được chấp nhận; lỗi theo vị trí âm chuẩn cần ít nhất 3/5 chuyên gia chấm sai/thiếu. Chỉ số này khác rubric của L2-ARCTIC; không tính PER khi thiếu phiên âm thực tế đầy đủ. Bộ này gồm người nói tiếng Trung học tiếng Anh, có cả trẻ em, nên chưa xác lập chất lượng trên TTS hoặc giọng bản ngữ.

```powershell
.venv\Scripts\python.exe -X utf8 scripts/prepare_external_speechocean.py --download
.venv\Scripts\python.exe -m unittest discover -s tests -p test_external_speechocean.py -v
.venv\Scripts\python.exe -X utf8 scripts/evaluate_external_speechocean.py --checkpoint windows-notebook-seed42-B-fold1 --selection full-test --threads 2 --output data/generated/external/new-run/B-full
.venv\Scripts\python.exe -X utf8 scripts/evaluate_external_speechocean.py --checkpoint windows-notebook-seed42-A --selection perfect-control --threads 2 --output data/generated/external/new-run/A-control
.venv\Scripts\python.exe -X utf8 scripts/evaluate_external_speechocean.py --checkpoint windows-notebook-seed42-C-fold1 --selection perfect-control --threads 2 --output data/generated/external/new-run/C-control
.venv\Scripts\python.exe scripts/report_external_speechocean.py --results data/generated/external/new-run
```

Dùng thư mục kết quả mới cho mỗi lượt chạy. Thêm `--resume` vào lệnh đánh giá để tiếp tục lượt cũ với cùng checkpoint, code, protocol và dữ liệu. ZIP/WAV nằm trong `data/external/`, không được đưa vào Git. Script chỉ giải nén WAV thuộc test chính thức; kiểm tra hash metadata, WAV, manifest và checkpoint; ghi rõ lỗi và độ phủ thay vì loại mẫu âm thầm. Kết quả gồm `predictions.jsonl`, `metrics.json`, manifest và CSV theo âm/người nói/nhóm tuổi/giới tính. Notebook mặc định đọc cache; các biến chạy tải/suy luận mặc định tắt.

## Kiểm thử checkpoint Projector hiện có

[Notebook pipeline và kiểm thử](notebooks/mdd_projector_validation.ipynb) lưu riêng dữ liệu, loss, kết quả A/B, Projector, cascade Qwen và raw output theo từng cell. Run All chỉ đọc kết quả; các cell chạy suy luận/đánh giá có cờ mặc định `False`. Báo cáo có biểu đồ, CSV và ví dụ ở `data/generated/projector-validation-v1/report/report.html`.

Pipeline mới trong `scripts/projector_inference.py`: WAV → resample 16 kHz → encoder B đóng băng → MLP Projector → embedding Qwen3-0.6B → text. `--stage auto` chỉ chọn checkpoint đã train hoàn tất và qua kiểm tra hash. Nếu chỉ có alignment, output ghi rõ chưa được học task feedback; không coi JSON đúng cấu trúc là chẩn đoán phát âm đúng. Nhánh cascade và web hiện tại được giữ riêng.

```powershell
# Chỉ kiểm thử; không có lệnh train. Đọc lại cache hợp lệ hoặc chạy tiếp khi chưa xong.
.venv\Scripts\python.exe -X utf8 -u scripts/run_projector_validation.py --phase all
.venv\Scripts\python.exe -X utf8 scripts/run_projector_validation.py --phase status
Get-Content data\generated\projector-validation-v1\checkpoint_benchmark.log -Tail 25 -Wait

# Tự thử một WAV; đổi đường dẫn file và câu chuẩn.
.venv\Scripts\python.exe -X utf8 scripts/projector_inference.py --audio "D:\path\recording.wav" --task alignment --max-new-tokens 128
.venv\Scripts\python.exe -X utf8 scripts/projector_inference.py --audio "D:\path\recording.wav" --task feedback --reference "The sentence you read." --output data/generated/my_projector_result.json
.venv\Scripts\python.exe -X utf8 scripts/projector_composed_inference.py --audio "D:\path\recording.wav" --reference "The sentence you read." --output data/generated/my_composed_result.json
```

Protocol tại `configs/projector_validation.json`: A/B cùng 900 test; NLL transcript đủ 900 với Projector ngẫu nhiên, sau alignment và tắt latent audio; sinh transcript pilot 30 mẫu và feedback pilot 12 mẫu (cân bằng 6 speaker, chọn bằng hash trước khi suy luận). Các trường hợp feedback gồm Projector P0, cascade B+Qwen gốc/B+LoRA và oracle Qwen nhận gold phones để chẩn đoán riêng. Cùng pilot dùng greedy decoding, giới hạn 512 token; output sai JSON/schema hoặc chưa có EOS vẫn nằm trong mẫu số. Kiểm tra 6 WAV thật phải cho latent trùng cache. Không train lại, sửa output bằng template, gọi API ngoài hoặc dùng test để chọn checkpoint.

Có thử thêm tổ hợp **Projector alignment + LoRA Qwen cascade** bằng nguyên trọng số đã có, lưu riêng tại `feedback_projector_cascade_adapter_transfer`. Adapter này học từ nhãn âm dạng text; chưa học nhận embedding audio, nên tổ hợp thử nghiệm không được gọi là P1. CLI `projector_composed_inference.py` cho phép thử WAV riêng theo đúng tổ hợp này.

Baseline Projector ngẫu nhiên là khởi tạo theo seed cố định, không phải checkpoint step 0 đã lưu. NLL là teacher forcing, WER pilot là lỗi sinh transcript, word-key F1/ROUGE là khớp feedback; ba nhóm chỉ số có ý nghĩa khác nhau. WER giữ từ Unicode thay vì bỏ qua output ngôn ngữ khác. Chưa có checkpoint feedback P1 thì chưa có bằng chứng trước/sau task tuning. Để mở rộng generation lên 900 mẫu, tạo config/output mới và đặt hai `*_per_speaker` thành 150; chi phí lớn hơn pilot, vẫn giữ budget đã khai báo riêng với benchmark P0/P1 gốc 1.536 token.

## Kiểm tra local

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/validate_contracts.py contracts/examples/assessment.json contracts/examples/feedback.json
```

Kiểm thử thứ tự cập nhật giao diện bằng Node (chỉ dùng cho tests, không cần cho server demo):

```powershell
node --test tests/demo_frontend.test.cjs tests/chat_stream.test.cjs
```

Bộ kiểm thử giữ phản hồi LLM/ai33 ở trạng thái chờ để xác nhận bảng lỗi và hướng dẫn xuất hiện đúng thứ tự, kiểm tra đổi model trong lúc chờ và xử lý dịch vụ lỗi. Provider được giả lập trong unit tests.

`tests/chat_stream.test.cjs` kiểm tra hiện chữ, xếp hàng câu đọc, nhịp word boundaries, gợi ý đến muộn, dừng và bỏ callback cũ. Khi đã có Playwright tại `.tools/ui-qa`, chạy `node scripts/check_chat_stream_browser.cjs` để kiểm tra UI bằng job/voice giả lập; `node scripts/check_chat_stream_live.cjs` chạy một WAV qua model local thật. Cả hai dùng mặc định `http://127.0.0.1:8765`, không gọi ai33 trả phí; file/screenshot lưu ở `data/generated/chat-stream-qa/`. Voice trong kiểm thử headless được giả lập, cần mở trình duyệt có loa để nghe thực tế.

`tests/test_speech_topics.py` kiểm tra catalog, nguồn lời mở đầu, context chủ đề, câu hỏi mở/fallback và route `/api/chat/topic`. `node scripts/check_topic_chat_browser.cjs` kiểm tra chọn/chuyển chủ đề, từ khóa, đánh giá từng lượt, lịch sử theo topic/coach và layout điện thoại. Để thử model thật theo một chủ đề bằng PowerShell: `$env:MDD_QA_TOPIC='travel'` rồi `node scripts/check_chat_stream_live.cjs`; các kết quả này nằm riêng trong `data/generated/topic-chat-qa/`. Lấy cảm hứng từ luồng [luyện nói theo chủ đề của S English](https://senglish.net/luyen-noi-tieng-anh); dùng giao diện/nội dung riêng của project.

`tests/test_sentence_correction.py` kiểm tra parser hoàn thiện câu, giữ nguyên ngữ pháp/từ gốc, thứ tự xử lý, hủy tác vụ và target đọc lại. `node scripts/check_sentence_correction_browser.cjs` dùng phản hồi giả lập để kiểm tra UI; `node scripts/check_sentence_correction_live.cjs` dùng WAV tổng hợp bằng giọng Windows local để chạy ASR/B/Qwen thật và kiểm tra đọc lại. Tạo WAV bằng `powershell.exe -NoProfile -File data/generated/sentence-correction-qa/create_synthetic.ps1`; kết quả ở `data/generated/sentence-correction-qa/`. Đây là kiểm thử tích hợp, không phải benchmark độ chính xác phát âm; voice đầu ra trong Chrome headless được giả lập và không gọi ai33 trả phí.

`python scripts/check_conversation_context_live.py` thử Qwen gốc thật trên transcript cố định và hội thoại hai lượt, lưu raw output tại `data/generated/conversation-continuity-qa/live-summary.json`. Lượt thử dùng CPU, không train hay gọi API; đây là kiểm thử định tính nhỏ, không đo ASR/voice hoặc điểm chất lượng hội thoại. `tests/test_chat_context.py` kiểm tra rút lịch sử không tách rời câu trả lời của bot khỏi câu người dùng.

Thêm `--natural-only --output data/generated/natural-conversation-qa/live-summary.json` để thử ba tình huống bằng Qwen thật: đổi chủ đề, transcript chưa rõ và câu cảm ơn. `tests/test_chat_reference.py` / `tests/test_chat_reference_api.py` kiểm tra bản nháp chính tả, phoneme gốc và route xác nhận. `node scripts/check_chat_reference_browser.cjs` kiểm tra xác nhận câu trên UI bằng dữ liệu giả; ảnh/báo cáo tại `data/generated/reference-ui-qa/`.

`node scripts/check_dark_chat_browser.cjs` kiểm tra bố cục phòng hội thoại, vị trí tin nhắn/thanh đánh giá, hoàn thiện câu và luyện đọc lại, menu, quay lại/kết thúc và responsive 360–1440 px. Job và voice được giả lập; screenshot và kết quả lưu tại `data/generated/light-chat-qa/`.

Test LLM/TTS dùng HTTP giả lập; parser/scorer/runtime và các launcher kiểm tra những trường hợp quan trọng. Unit tests không gọi API và không cập nhật weights pretrained. Kết quả train/test A/B/C thật trên Windows CUDA được lưu riêng trong notebook; GPU Colab vẫn cần xác nhận bằng lượt chạy thật nếu sử dụng.
