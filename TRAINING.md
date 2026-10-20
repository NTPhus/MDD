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

```bash
.venv/bin/python -m pip check
.venv/bin/python scripts/check_training_environment.py --output data/generated/setup/environment.json
.venv/bin/python scripts/download_models.py --verify-only
.venv/bin/python scripts/prepare_l2arctic.py --verify-audio
.venv/bin/python -m unittest discover -s tests -v
```

Các lệnh trên không chạy train trên BASE. Riêng `run_local_training.sh smoke` có một update kỹ thuật trên model copy, được ghi rõ `technical_smoke_only`; chưa phải kết quả trước/sau thích ứng Việt.

## Colab là lựa chọn bổ sung

Notebook [train_mdd_colab.ipynb](notebooks/train_mdd_colab.ipynb) vẫn có thể dùng khi cần GPU. Chọn các preset CUDA có sẵn và giữ PyTorch CUDA của Colab; không cài `requirements/cpu-torch.txt` vào runtime GPU. Workflow local không cần upload ZIP hoặc mount Drive.

Gói ZIP đã tạo trước đây là snapshot Colab tại thời điểm đóng gói, không tự cập nhật khi code/docs/config local đổi. Nếu cần đóng gói lại, dùng `scripts/export_colab_bundle.py --output PATH_MOI.zip` và chọn tên gói tương ứng trong notebook. Không ghép checkpoint từ các protocol/code hash khác nhau để né kiểm tra provenance.
