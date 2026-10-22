# Huấn luyện local trên CPU

Project hiện hỗ trợ train trực tiếp trên máy Ubuntu i5-10400, RAM 16 GB, dùng **CPU**. Intel UHD 630 không cung cấp CUDA cho cấu hình PyTorch này. Các preset local dùng FP32, 4 threads, batch 1 và gradient checkpointing để giảm bộ nhớ. Huấn luyện đầy đủ trên CPU sẽ chậm; chưa có benchmark thời gian cho toàn bộ A/B/C.

Giữ protocol **l2arctic-v1** đang hoạt động. V2 cân đối metadata vẫn là đề xuất, chưa kích hoạt. Lý thuyết nằm trong [SPEECH_THEORY.md](docs/SPEECH_THEORY.md); lý do chia tập trong [DATA_SPLIT.md](docs/DATA_SPLIT.md).

## Setup và bắt đầu

Chạy từ thư mục gốc project:

```bash
bash scripts/setup_local_training.sh
bash scripts/run_local_training.sh smoke --run-id local-smoke-01
bash scripts/run_local_training.sh A --run-id local-A-seed42
```

Setup tạo hoặc dùng lại `.venv`, cài đúng dependency còn thiếu, kiểm tra PyTorch CPU, model BASE, checksum dữ liệu và tạo target phoneme. Nếu phiên bản thư viện đã đúng thì không truy cập package index. Setup **không tự bắt đầu train** và không cần API key LLM/TTS.

Lệnh `smoke` thực hiện một optimizer update trên tập kỹ thuật nhỏ, có cập nhật Transformer và head, sau đó chấm hai mẫu validation và lưu checkpoint. Nó kiểm tra đường train thật nhưng không phải baseline khoa học. Không lấy checkpoint smoke để khởi tạo thí nghiệm B/C.

Smoke đã chạy thành công ở `runs/local-checks/setup-cpu-smoke-20261020/`: 1 update với khoảng 90,2 triệu tham số trainable, loss hữu hạn, tổng thời gian tiến trình 12,51 giây, peak RSS 2,24 GiB. Audio train là một clip 1,03 giây; không suy ra thời gian train đầy đủ hay peak RAM clip dài từ phép kiểm tra này. Báo cáo ở `data/generated/setup/local_cpu_smoke_resources.json` và `local_cpu_smoke.log`.

Lệnh A bắt đầu lượt nghiên cứu trên toàn bộ source train. Chọn run ID mới cho một thí nghiệm mới; script từ chối ghi đè run đã tồn tại. Có thể kiểm tra command mà không train:

```bash
bash scripts/run_local_training.sh A --run-id local-A-seed42 --dry-run
```

Khi gọi Python trực tiếp, dùng `.venv/bin/python` hoặc `source .venv/bin/activate`. Launcher tự chọn Python trong `.venv`, đường dẫn local và device CPU.

## Cấu hình CPU

| Preset | Tác vụ | Successful updates | Validation mỗi | Checkpoint mỗi |
| --- | --- | ---: | ---: | ---: |
| [local_source.json](configs/local_source.json) | A: học phone từ source | 2.000 | 250 | 50 |
| [local_adapt_vi.json](configs/local_adapt_vi.json) | B: thích ứng Việt, source replay 50% | 400 | 100 | 50 |
| [local_control.json](configs/local_control.json) | C: đối chứng source-only | 400 | 100 | 50 |
| [local_smoke.json](configs/local_smoke.json) | Kiểm tra kỹ thuật, 2 train/2 dev | 1 | 1 | 1 |

A/B/C đều dùng:

- `device=cpu`, `mixed_precision=no`, `cpu_threads=4`.
- Batch vật lý 1, accumulation 4: bốn lượt lấy mẫu mỗi optimizer update.
- Freeze CNN; feature projection, Transformer và head phoneme vẫn được cập nhật.
- Gradient checkpointing bật; AdamW và learning rate giữ như cấu hình nghiên cứu trước.
- Dùng toàn bộ mẫu của từng vai trò; không lọc hoặc cắt audio trong preset khoa học.
- Lưu optimizer, RNG và model để resume. Best checkpoint được chọn bằng F1 validation.

Sampling có hoàn lại nên số update không phải số epoch. CPU dùng FP32; các file `source.json`, `adapt_vi.json`, `control.json` vẫn là preset CUDA/FP16 cho Colab. Launcher local chọn các file `local_*.json` để tránh gọi nhầm CUDA.

Không tự giảm số bước hoặc chỉ train head để làm lượt nghiên cứu chạy nhanh hơn. Nếu thay ngân sách, ghi thành cấu hình/run riêng và giữ B/C cùng điều kiện. Không dùng test để quyết định số bước hay learning rate.

## Dữ liệu đã chuẩn bị

| Vai trò | Người nói | Số câu annotated |
| --- | ---: | ---: |
| Source train | 15 người ngoài Việt Nam | 2.249 |
| Source validation | 5 người ngoài Việt Nam | 750 |
| Việt trong mỗi fold: train / validation / test | 2 / 1 / 1 | 300 / 150 / 150 |

Tổng cộng 3.599 mẫu có annotation thủ công, khoảng 3,66 giờ; người Việt có 600 mẫu thuộc bốn người. Audio gốc 44,1 kHz được đọc, chuyển mono và resample thực sự về 16 kHz lúc load, không cần tạo bản sao WAV.

Target CTC là `perceived_phones`: âm người học thực sự phát ra. Canonical phones và gold error events phục vụ căn chỉnh/chấm điểm. Vocabulary 83 token gồm 40 phone, 40 biến thể `*`, `err`, `[UNK]`, `[PAD]`. Nhãn thô, các typo và quy tắc chuẩn hóa được giữ trong audit; không suy vocabulary từ test.

Các fold tách người và audio. Cả 150 mã câu test mỗi fold đã xuất hiện ở người train, nên kết quả đo người mới đọc câu trong miền đã học, chưa chứng minh khả năng trên câu hoàn toàn mới. [Audit dữ liệu](data/generated/setup/split_audit.md)

## Train B và C sau khi A hoàn tất

Ví dụ fold 1:

```bash
bash scripts/run_local_training.sh B --run-id local-B-fold1-seed42 --fold 1 --init-checkpoint runs/local-A-seed42/best_checkpoint
bash scripts/run_local_training.sh C --run-id local-C-fold1-seed42 --fold 1 --init-checkpoint runs/local-A-seed42/best_checkpoint
```

Lặp lại với `--fold 2`, `3`, `4` và run ID tương ứng. Mỗi B/C khởi tạo từ cùng A, không train tiếp từ B/C của fold trước. A phải là lượt khoa học hoàn tất; runtime không nhận A smoke hoặc chưa đủ ngân sách làm nguồn cho B/C khoa học.

A chọn model bằng source validation. B/C chọn bằng Vietnamese validation của fold; C chỉ cập nhật gradient bằng source nhưng vẫn dùng nhãn validation Việt để chọn checkpoint. Một seed cần 9 lượt train; có thể làm lần lượt, không cần chạy đồng thời trên máy 16 GB RAM.

## Kết quả lưu ở đâu

Mặc định mọi kết quả nằm trên ổ local:

```text
runs/local-A-seed42/
  config.json
  environment.json
  protocol_hashes.json
  data_selection.json
  initialization.json
  parameter_counts.json
  training_log.jsonl
  validation_log.jsonl
  predictions_validation_best.jsonl
  metrics_validation_best.json
  checkpoint/         # model, optimizer, scaler/RNG và metadata để resume
  best_checkpoint/    # model được chọn bằng validation
  completed.json
```

Không cần Google Drive. Có thể đổi vị trí bằng `--runs-root PATH`; dùng cùng vị trí khi resume và tham chiếu checkpoint. Lưu giữ nguyên thư mục run để có bằng chứng, cả trường hợp model không cải thiện.

Với model khoảng 94,4 triệu tham số và AdamW, dự trù khoảng 1,4 GiB mỗi run hoàn tất gồm best model và resumable checkpoint. Chín run cần khoảng 12–13 GiB, cộng vùng tạm khi lưu, dữ liệu và các bản sao. Đây là ước lượng dung lượng, không phải số đo từ toàn bộ lượt nghiên cứu. Hiện máy có khoảng 50 GiB trống khi kiểm tra setup.

## Dừng và resume

Sau khi đã có `checkpoint/training_state.pt`, tiếp tục cùng run bằng:

```bash
bash scripts/run_local_training.sh A --run-id local-A-seed42 --resume runs/local-A-seed42/checkpoint
bash scripts/run_local_training.sh B --run-id local-B-fold1-seed42 --fold 1 --resume runs/local-B-fold1-seed42/checkpoint
```

Khi resume không truyền thêm `--init-checkpoint`. Giữ đúng stage, fold, seed và config; không đổi device/precision giữa chừng. Các update sau checkpoint cuối có thể phải tính lại, tối đa gần 50 bước với preset local. Trước lần lưu đầu tiên chưa có trạng thái để resume; `Ctrl+C` không tự lưu tức thì.

Runtime có bản `checkpoint.previous` để phục hồi khi bị ngắt đúng lúc đổi tên thư mục. Nếu best checkpoint mới hơn optimizer checkpoint phục hồi, runtime dừng để tránh ghép trạng thái khác bước; giữ artifact và phục hồi snapshot tương ứng hoặc tạo run mới.

## Đánh giá và so sánh

Chỉ chạy test sau khi đã khóa cấu hình bằng validation. Ví dụ fold 1, sau khi hoàn tất A/B/C:

```bash
.venv/bin/python scripts/evaluate_mdd.py --checkpoint runs/local-A-seed42/best_checkpoint --split test --fold 1 --output-dir runs/local-A-seed42/evaluation/fold1-test --device cpu
.venv/bin/python scripts/evaluate_mdd.py --checkpoint runs/local-B-fold1-seed42/best_checkpoint --split test --fold 1 --output-dir runs/local-B-fold1-seed42/evaluation/fold1-test --device cpu
.venv/bin/python scripts/evaluate_mdd.py --checkpoint runs/local-C-fold1-seed42/best_checkpoint --split test --fold 1 --output-dir runs/local-C-fold1-seed42/evaluation/fold1-test --device cpu
.venv/bin/python scripts/compare_mdd_runs.py --a runs/local-A-seed42/evaluation/fold1-test --b runs/local-B-fold1-seed42/evaluation/fold1-test --c runs/local-C-fold1-seed42/evaluation/fold1-test --output runs/comparisons/local-fold1-seed42
```

Các lệnh tạo output mới và từ chối ghi đè. Comparator kiểm tra cùng ID/audio/gold/scorer/protocol, B/C cùng A khởi tạo và cùng ngân sách, rồi tính lại điểm từ dự đoán để xuất `summary.json` và `comparison.csv`.

| Chỉ số | Ý nghĩa |
| --- | --- |
| Precision / recall / F1 | Phát hiện lỗi phone; F1 là chỉ số chính |
| FAR | Lỗi bị bỏ sót |
| FRR | Phone đúng bị báo sai |
| PER | S+D+I so với chuỗi phone thực tế |
| DER | Chẩn đoán sai trong lỗi đã bắt được và đủ gold |
| Word F1 | Từ chứa lỗi được đánh dấu đúng |

Rates trong JSON dùng tỷ lệ, mẫu số 0 trả `null`; PER có thể vượt 1. Giữ `*`; `err`/`[UNK]` gold không đủ thông tin bị loại khỏi exact diagnosis nhưng vẫn chấm detection/PER. Alignment có thể khác vị trí chuyên gia, đã ghi trong [scorer calibration](data/generated/setup/scorer_calibration.json). Không so điểm trực tiếp với paper dùng split, inventory hoặc scorer khác.

Hai best checkpoints B/C có thể ở bước khác nhau dù cùng ngân sách 400. Nếu cần so sánh đúng cùng số update, chấm thêm `checkpoint/` cuối 400 bước vào output mới theo [DATA_SPLIT.md](docs/DATA_SPLIT.md). Báo cả từng người và kết quả gộp; bốn người Việt hỗ trợ kết luận hạn chế về tổng quát hóa.

## Git ignore và bằng chứng local

[.gitignore](.gitignore) bỏ qua dataset L2-ARCTIC gốc, `annotations.jsonl` tái tạo được, weights pretrained, file tải dở, `.venv`, `.env`, `data/generated/` và `runs/`. File vẫn tồn tại trên máy để train/resume; ignore không xóa dữ liệu.

PDF bài báo, source code, configs, manifest/splits, vocabulary, normalization, audit và metadata model vẫn được giữ cho Git. Lưu hoặc sao lưu riêng `runs/` nếu cần chia sẻ checkpoint/bằng chứng; không coi Git ignore là cơ chế sao lưu.

## Kiểm tra local

### Thử nghiệm chấm phát âm và gợi ý câu trong hội thoại

Notebook [mdd_pronunciation_calibration.ipynb](notebooks/mdd_pronunciation_calibration.ipynb) có các cell riêng cho audit dữ liệu, nghe mẫu, các biến thể CMU, phân bố score, hiệu chỉnh ngưỡng và so sánh kết quả. Run All mặc định đọc kết quả đã lưu; các cờ chạy inference/calibration mặc định `False`.

Thử nghiệm này giữ nguyên trọng số B và Qwen. Pilot 40 bản thu và toàn bộ 450 validation đã chạy; 300 audio HJK/LXC dùng chọn ngưỡng, 150 PNV dùng kiểm tra ngưỡng. PNV từng tham gia chọn checkpoint B nên không phải tập test độc lập. Bộ test 900 và SpeechOcean không dùng chọn ngưỡng. Báo cáo đầy đủ ở [full-validation/report.html](data/generated/pronunciation-calibration-v1/full-validation/report.html).

Kết quả ngày 22/10/2026: trong 496 cặp ngưỡng của lưới thử đã đăng ký, chưa có cặp nào đạt đồng thời FPR ≤ 10%, FNR ≤ 20%, coverage ≥ 50% trên nhóm fit. Artifact `ineligible`; web giữ trạng thái **chưa đủ bằng chứng**, không coi abstention là phát âm đúng. Phương pháp thử là margin posterior theo âm với căn chỉnh CTC và biến thể từ điển; chưa phải triển khai đầy đủ công thức CTC-GOP trong bài báo. Đây là so sánh cách chấm trên cùng weights, không phải kết quả train mới.

Chạy lại trên PowerShell tại thư mục project (cache posterior được kiểm tra identity trước khi dùng lại):

```powershell
.\.venv\Scripts\python.exe scripts/experiment_pronunciation_calibration.py --self-test
.\.venv\Scripts\python.exe scripts/experiment_pronunciation_calibration.py --pilot --run
.\.venv\Scripts\python.exe scripts/experiment_pronunciation_calibration.py --full-validation --run
```

Trong web, mở mục thu gọn của lượt nói, xác nhận **đúng câu thực sự đã nói** để đối chiếu cùng bản thu. Mục **Ngữ pháp & cách nói tự nhiên** dùng Qwen3-0.6B local, tách khỏi câu tham chiếu phát âm; các câu gợi ý và từ mẫu chỉ được đọc khi bấm nghe. Các smoke test Qwen thật lưu tại [grammar-live.json](data/generated/pronunciation-calibration-v1/grammar-live.json); chưa phải benchmark chất lượng sửa ngữ pháp. Một từ ASR không rõ có thể khiến gợi ý bị giữ lại thay vì đoán thay người nói.

Web hiển thị gợi ý phát âm ngay theo transcript ASR khi có từ điển và alignment hợp lệ. Gợi ý này được ghi là tạm tính vì ASR có thể nghe nhầm; người dùng chỉ cần mở mục sửa câu khi nhận thấy text sai. Không bắt xác nhận trước khi xem gợi ý, và không coi việc không có gợi ý là phát âm đúng.

Sau khi bấm **Đối chiếu audio này**, chỉ khi các từ thay đổi thì web mới xóa câu trả lời/voice cũ và stream câu Qwen mới trong cùng lượt. Câu mới bắt đầu bằng **Sorry for the misunderstanding.** đúng một lần, rồi tiếp tục cuộc trò chuyện theo text đã xác nhận. Lời xin lỗi được ứng dụng thêm sau khi kiểm tra phần nội dung, không giúp một output chỉ lặp câu vượt qua bước kiểm tra. Chỉ sửa chữ hoa/dấu câu/khoảng trắng giữ nguyên câu trả lời, voice và revision. Backend giữ lịch sử trước lượt cần sửa, transcript ASR gốc và posterior của bản thu; không chạy lại ASR/B. Revision riêng chặn phản hồi/voice cũ ghi đè lên lượt mới. Từ riêng chưa có trong CMUdict vẫn được dùng cho hội thoại, nhưng toàn bộ phép đối chiếu âm báo chưa có tham chiếu thay vì gán đúng/sai một phần.

Khi mở collapse hoặc cuộn lên đọc lịch sử, polling/stream/highlight không tự kéo xuống cuối. Việc theo dõi câu mới chỉ tiếp tục khi người dùng ở cuối và không mở phần đọc chi tiết.

[Notebook kiểm thử](notebooks/mdd_chat_error_checks.ipynb) có từng cell cho kết quả web, Qwen thật và browser. Báo cáo contract mới tại [live-pipeline.json](data/generated/optional-reference-qa/live-pipeline.json) và [browser-summary.json](data/generated/optional-reference-qa/browser-summary.json); replay các posterior audio đã lưu tại [diagnostics.json](data/generated/optional-reference-qa/diagnostics.json). Báo cáo Qwen lịch sử [real-qwen.json](data/generated/reference-regeneration-qa/real-qwen.json) giữ trường hợp lặp câu sau cả hai lần sinh; pipeline chặn câu lặp và báo chưa có phản hồi. Việc truyền đúng text và chặn lặp nguyên câu không bảo đảm Qwen3-0.6B luôn đối thoại tự nhiên hoặc phát hiện mọi cách diễn đạt lại cùng ý.

```powershell
.\.venv\Scripts\python.exe scripts/check_reference_regeneration_live.py
node scripts/check_reference_regeneration_browser.cjs
```

Lệnh Python trên gửi một audio validation tới backend local đang chạy và thực hiện hai lần sinh lại bằng Qwen, không gọi TTS/API trả phí. Lệnh browser dùng model/voice giả lập. Nếu bot cần làm rõ hoặc chưa tạo được phản hồi, nút **Sửa câu hoặc nói lại** mở phần nhập lại câu; phản hồi cố định được ghi nguồn riêng, không coi là kết quả model thành công.

Kiểm thử audio lỗi và hội thoại trong [mdd_chat_error_checks.ipynb](notebooks/mdd_chat_error_checks.ipynb) có 4 bản thu validation chạy lại qua ASR+B, audio nghe được, nhãn thủ công theo từ và biểu đồ TP/FP/FN. [Báo cáo nghe audio](data/generated/pronunciation-error-qa/report.html) giữ cả lỗi phát hiện được, lỗi bỏ sót và bản đúng bị báo nhầm. Chạy lại bằng:

```powershell
.\.venv\Scripts\python.exe scripts/check_pronunciation_error_examples.py --real-audio
```

Khi calibration chưa đạt, web vẫn cho xem **gợi ý kiểm tra âm** màu vàng từ các khác biệt decoder với biến thể CMU, có nhãn chưa phải kết luận lỗi. Gợi ý dựa trên ASR xuất hiện ngay; sửa câu đối chiếu là tùy chọn khi text nhận sai. Không có gợi ý cũng không có nghĩa phát âm đúng. Không đổi artifact thất bại thành eligible hoặc dùng gợi ý này làm điểm calibrated. Trên 450 validation, cách chọn biến thể có precision 68,03%, recall 63,16%, FPR 20,65% ở mức từ; đây là thống kê diagnostic trên validation, không phải kết quả test độc lập.

Lỗi Qwen chỉ nói `Hi!` được xử lý bằng kiểm tra phần đóng góp hội thoại, retry tối đa một lần và nhánh hỏi rõ khi ASR có nhiều từ không xác định. Báo cáo raw trước/sau tại [chat-recovery-qa](data/generated/chat-recovery-qa/live-fewshot.json). Qwen thật đã hỏi rõ với transcript lỗi được báo; các trường hợp công việc vẫn có câu hỏi lại thông tin đã biết, chưa chứng minh hội thoại tự nhiên đã hoàn thiện. Nếu hai lần sinh đều không dùng được và ASR chưa rõ, câu hỏi fallback có nguồn riêng và nhãn **Cần làm rõ**, không ghi thành kết quả Qwen thành công.

Kiểm thử riêng câu `i usually wakes up at seven am then i brush my teeth and go to work by bus` cho thấy Qwen đã sửa `wakes → wake`, nhưng bộ lọc cũ loại kết quả vì bỏ đại từ `I` lặp sau `then`. Cách viết giờ `am → a.m.` cũng bị loại nhầm. `sentence_rewrite.py` hiện chuẩn hóa giờ 12 tiếng có hour/period rõ và cho phép lược cùng chủ ngữ trong phối hợp trước động từ được nhận diện; không áp dụng cho chủ thể khác nhau hoặc bỏ phủ định. Hai đề xuất được kiểm tra riêng, nên một trường không được hỗ trợ không làm mất trường hợp lệ còn lại. Thay đổi viết hoa/dấu câu cũng được ghi nhận, còn giải thích hiển thị được suy ra từ edit thực tế.

Raw Qwen trước/sau cùng prompt ở [grammar-routine-qa](data/generated/grammar-routine-qa/after.json); so sánh lại chính raw output cũ bằng hai parser ở [comparison.json](data/generated/grammar-routine-qa/comparison.json). Notebook kiểm thử có cell và biểu đồ riêng. Ba câu này là regression phát triển, không phải benchmark chất lượng ngữ pháp; không train, đổi prompt hoặc đổi trọng số.

Luồng hội thoại hiện hoàn tất gợi ý trong collapse trước khi bot bắt đầu sinh chữ/voice: phân tích âm → hoàn thiện câu/gợi ý ngữ pháp → coach nếu áp dụng → snapshot feedback hoàn tất → sinh phản hồi. Hủy ở giai đoạn feedback không chạy bước hội thoại. Bộ lọc viết lại còn hỗ trợ các nhóm chỉnh sửa có giới hạn cho câu người học nhiều lỗi: biến thể động từ bất quy tắc, `subject + no + verb`, chủ thể đồng nhất bị thiếu trong mệnh đề thời gian, giới từ còn thiếu ở một số khung động từ và định từ sở hữu trong một số cấu trúc vệ sinh cơ thể. Không bỏ qua chủ thể/sở hữu đã nói; có tên riêng hoặc đại từ chỉ người khác thì không suy ra chủ thể/sở hữu còn thiếu.

Kiểm thử câu `Yesterday I wake up at 7, I no brush teeth...` ở [grammar-past-qa](data/generated/grammar-past-qa/after.json) giữ raw output và so sánh parser. Prompt bổ sung yêu cầu sửa câu nối bằng dấu phẩy, nhưng Qwen0.6B vẫn để lại comma splice trong lượt thử: không gọi đây là bản sửa ngữ pháp hoàn hảo. Gợi ý không tự thay câu dùng chấm audio hoặc lịch sử. Kiểm tra thứ tự bằng một audio thật lưu tại [live-order.json](data/generated/grammar-past-qa/live-order.json); notebook có cell riêng.

```bash
.venv/bin/python -m pip check
.venv/bin/python scripts/check_training_environment.py --output data/generated/setup/environment.json
.venv/bin/python scripts/download_models.py --verify-only
.venv/bin/python scripts/prepare_l2arctic.py --verify-audio
.venv/bin/python -m unittest discover -s tests -v
```

Các lệnh trên không chạy train trên BASE. Riêng `run_local_training.sh smoke` có một update kỹ thuật trên model copy, được ghi rõ `technical_smoke_only`; chưa phải kết quả trước/sau thích ứng Việt.

## LoRA riêng cho sửa ngữ pháp và cách nói tự nhiên

Notebook [qwen_grammar_training.ipynb](notebooks/qwen_grammar_training.ipynb) có các cell riêng cho nguồn,
split, kiểm tra trùng, biểu đồ task/unchanged/token/error type, lệnh train và so sánh trước/sau.
Mặc định Run All chỉ đọc dữ liệu/vẽ biểu đồ; không train, inference hay gọi API.

Nguồn được chọn từ bài báo khoa học và trang phát hành của tác giả:

| Nguồn | Công bố | Vai trò |
| --- | --- | --- |
| Cambridge Write & Improve / LOCNESS v2.1 | [BEA-2019, ACL Anthology](https://aclanthology.org/W19-4406/), [Cambridge phát hành](https://www.cl.cam.ac.uk/research/nl/bea2019st/) | W&I cho nhãn grammar/spelling/lexical; W&I+LOCNESS dev cho validation |
| JFLEG | [EACL-2017, ACL Anthology](https://aclanthology.org/E17-2037/), [repository chính thức](https://github.com/keisks/jfleg) | Bốn bản sửa fluency của người cho mỗi câu; học task natural |

Không lấy dữ liệu scrape từ bài báo tin tức, output demo hoặc Qwen tự sinh làm nhãn đúng.
SHA256 archive Cambridge và từng file JFLEG tại commit cố định được lưu trong
[grammar_sources.lock.json](configs/grammar_sources.lock.json). Corpus chỉ dùng phi thương mại theo giấy phép gốc:
W&I cho nghiên cứu/giáo dục phi thương mại, LOCNESS không được phát tán cho bên thứ ba nếu chưa có phép,
JFLEG CC BY-NC-SA 4.0. Toàn bộ dữ liệu tải/processed và HTML xuất ở máy local được ignore khỏi Git;
trên máy khác dùng downloader chính thức, không public ZIP/corpus. Giấy phép đầy đủ nằm trong dataset/licenses.

Dataset local `data/processed/grammar-wi-jfleg-v1` đã chuẩn bị:

| Tập | Mẫu train/eval | Câu nguồn khác nhau | Grammar | Natural |
| --- | ---: | ---: | ---: | ---: |
| Train | 28.501 | 27.126 | 26.534 | 1.967 |
| Validation | 4.292 | 4.292 | 4.132 | 160 |
| Test | 3.746 | 3.746 | 2.999 | 747 |

Train giữ 8.773 target không thay đổi source. Một câu JFLEG train có thể có nhiều target khác nhau của người;
validation/test mỗi câu chỉ một record với đầy đủ references. Không tự nhân số câu test bằng số annotator.
W&I có 34.308 câu train gốc, dev có 4.384; JFLEG có 754 dev và 747 test.
W&I được đối chiếu chính xác từng câu M2 với essay JSON để lấy author/document ID;
các bài train của người viết cũng xuất hiện ở dev bị loại. Dành 10% author train làm internal grammar test;
W&I dev là validation. JFLEG dev chia theo nhóm câu 80/20 train/validation, JFLEG test giữ độc lập.
Đây là protocol nghiên cứu riêng: không gọi internal grammar test là BEA official test,
không dùng điểm JFLEG dev như benchmark độc lập vì một phần đã dùng train.
Archive BEA có 4.477 input test chưa có gold; chúng được giữ riêng và dùng chống trùng, không bịa nhãn.

Loại câu có UNK/Um, target rỗng, trùng câu nguồn và trùng source/reference giữa các split sau chuẩn hóa lexical.
Audit hiện có 0 text overlap và 0 group overlap giữa các tập. Đây là chống trùng exact sau chuẩn hóa,
chưa phải chống semantic near-duplicate hoặc bằng chứng rằng base Qwen chưa từng thấy corpus khi pretrain.
JFLEG không cung cấp author ID nên chỉ khẳng định câu và các reference đi cùng nhau, không khẳng định author-disjoint.
Giới hạn context 768; mẫu dài nhất hiện 607 tokens, không cắt target. Exclusions và checksum đầy đủ ở manifest.json.

Adapter dùng hai task-specific single-field JSON: W&I → `{"grammar_corrected":"..."}`,
JFLEG → `{"natural":"..."}`. Không bịa natural label cho W&I hoặc explanation của chuyên gia.
Đây là dữ liệu văn viết, chưa đủ để chứng minh sửa mọi câu hội thoại/ASR lỗi hoặc luôn giữ đúng ý định người nói.
Web hiện vẫn dùng base Qwen với contract ba trường; sau benchmark cần nối adapter bằng request đúng từng task.
Không tự kích hoạt adapter mới trong web khi chưa đánh giá.

Cấu hình [grammar_qwen3_06b_qlora.json](configs/grammar_qwen3_06b_qlora.json): QLoRA NF4 rank 8,
q_proj/v_proj, 2 epochs, batch 1 × accumulation 16, gradient checkpointing và bounded projection loss.
Chỉ adapter Qwen được cập nhật; không nạp hay thay weights wav2vec2/projector, không ghi đè LoRA phát âm.
Chưa chạy train để xác nhận VRAM/thời gian. Trên máy này dependencies đã có; trên máy mới cài
`requirements/grammar.txt` vào môi trường PyTorch CUDA hiện có và thêm requirements/notebook.txt nếu dùng notebook.

```powershell
# Máy mới: tái tạo dữ liệu từ nguồn chính thức (output phải chưa tồn tại).
.\.venv\Scripts\python.exe -X utf8 scripts/grammar_data.py prepare --download

# Máy hiện tại: đã có dataset, kiểm tra toàn bộ trước khi train.
.\.venv\Scripts\python.exe -X utf8 scripts/train_grammar_model.py --check-data-only

# Bạn chủ động bắt đầu train; không có --run thì entry point chỉ kiểm tra.
.\.venv\Scripts\python.exe -X utf8 scripts/train_grammar_model.py --run

# Tiếp tục checkpoint dở; không ghi đè lượt đã completed.
.\.venv\Scripts\python.exe -X utf8 scripts/train_grammar_model.py --run --resume

# Theo dõi sau khi log đầu tiên được tạo.
Get-Content .\runs\grammar-qwen3-06b-wi-jfleg-v1\training_history.jsonl -Tail 20 -Wait
```

Đánh giá bản gốc trước train và adapter sau train với cùng split/prompt/generation/precision/device.
Validation dùng phát triển, khóa recipe rồi mới chạy full test bằng cách đổi `--split test` ở cả hai lượt
và dùng output mới. `--limit 50` ở cả hai chỉ là development probe, không gọi là full benchmark.
Các lệnh inference sau **chỉ chạy khi bạn thực thi**, chưa chạy trong bước chuẩn bị dataset:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_grammar_model.py --split validation --output data/generated/grammar-benchmark-v1/before-validation --run
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_grammar_model.py --split validation --adapter runs/grammar-qwen3-06b-wi-jfleg-v1/adapter --output data/generated/grammar-benchmark-v1/after-validation --run
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_grammar_model.py --compare data/generated/grammar-benchmark-v1/before-validation data/generated/grammar-benchmark-v1/after-validation --output data/generated/grammar-benchmark-v1/comparison-validation.json
```

Evaluator giữ raw output; JSON lỗi vẫn là failure, không fallback về source. Lượt resume yêu cầu fingerprint khớp;
adapter phải completed, khớp frozen dataset và base weights. Báo cáo hiện hỗ trợ valid JSON, exact match với
bất kỳ reference, chrF và unchanged-preservation. Đây là diagnostics, không gọi là accuracy ngữ pháp,
không phải ERRANT F0.5 hoặc JFLEG GLEU, không thay thế chấm naturalness/giữ nguyên ý. Chưa có kết quả sau train.
Notebook tự đọc bảng so sánh và vẽ loss khi bạn tạo đủ artifacts.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest tests.test_grammar_data tests.test_grammar_evaluation tests.test_feedback_training tests.test_feedback_memory -v
.\.venv\Scripts\python.exe -X utf8 scripts/build_grammar_training_notebook.py --execute
```

## Colab là lựa chọn bổ sung

Notebook [train_mdd_colab.ipynb](notebooks/train_mdd_colab.ipynb) vẫn có thể dùng khi cần GPU. Chọn các preset CUDA có sẵn và giữ PyTorch CUDA của Colab; không cài `requirements/cpu-torch.txt` vào runtime GPU. Workflow local không cần upload ZIP hoặc mount Drive.

Gói ZIP đã tạo trước đây là snapshot Colab tại thời điểm đóng gói, không tự cập nhật khi code/docs/config local đổi. Nếu cần đóng gói lại, dùng `scripts/export_colab_bundle.py --output PATH_MOI.zip` và chọn tên gói tương ứng trong notebook. Không ghép checkpoint từ các protocol/code hash khác nhau để né kiểm tra provenance.
