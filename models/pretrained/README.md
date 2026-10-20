# Checkpoint wav2vec2 local cho MDD

Pipeline mặc định dùng duy nhất **wav2vec2-base** để fine-tune phoneme CTC/MDD. Inventory local hiện có:

| Thư mục | Nguồn | Vai trò | File weights |
| --- | --- | --- | --- |
| [wav2vec2-base](wav2vec2-base/) | [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) | Backbone khởi đầu để fine-tune MDD | `pytorch_model.bin` |
| [wav2vec2-base-960h](wav2vec2-base-960h/) | [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) | Đã tải, tùy chọn ASR, không thuộc pipeline MDD mặc định | `model.safetensors` |

Cả hai có 12 Transformer layer, hidden size 768 và nhận audio mono 16 kHz. BASE có kiến trúc checkpoint `Wav2Vec2ForPreTraining`; chưa được fine-tune dự đoán lỗi phát âm. Các file tokenizer/vocab upstream đi kèm BASE không phải inventory phoneme MDD đã thiết kế cho L2-ARCTIC. Sau khi fine-tune phone CTC, đầu ra là chuỗi phone để đối chiếu với câu tham chiếu; không tự có transcript chữ. Bản 960h có CTC head nhận dạng chữ, phục vụ nhánh ASR tùy chọn nếu mở rộng về sau.

[manifest.json](manifest.json) liệt kê inventory đã tải, revision, dung lượng và `default_models` chỉ gồm BASE. Mỗi thư mục model có `download_revision.txt` khóa commit SHA và `download_manifest.json` ghi kích thước, SHA-256 local, checksum Hub, nguồn tải của từng file và cấu hình đã kiểm tra. Lần chạy lại tiếp tục dùng revision đã khóa. Script tải đúng một định dạng weights cho mỗi model.

## Kiểm tra file local

Từ thư mục gốc project:

```bash
python3 scripts/download_models.py --verify-only
```

Lệnh mặc định chỉ kiểm tra BASE, không cần mạng, PyTorch hoặc Transformers. Khi tải lần đầu, script đối chiếu SHA-256 của file LFS với Hub và Git blob SHA-1 của các file thường. Nếu tải bị gián đoạn, chạy lại không có `--verify-only` để tiếp tục file `.part`. Có thể kiểm tra checkpoint ASR đã tải bằng lựa chọn tường minh:

```bash
python3 scripts/download_models.py --verify-only --models wav2vec2-base-960h
```

## Nạp model local

Môi trường `.venv` local đã có PyTorch CPU, Transformers và safetensors. Checkpoint BASE đã được nạp với head phoneme mới 83 token bằng `Wav2Vec2ForCTC`, chạy forward không tính gradient trên một audio L2-ARCTIC thật, thu được logits và CTC loss hữu hạn. Báo cáo ở `data/generated/setup/model_forward.json`; kiểm tra này có **0 optimizer update**, không đo độ chính xác MDD.

```python
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
)

base_path = "models/pretrained/wav2vec2-base"
base = Wav2Vec2ForPreTraining.from_pretrained(base_path, local_files_only=True)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_path, local_files_only=True)

```

Đoạn code trên nạp kiến trúc pretraining gốc. Pipeline MDD thực tế dùng `Wav2Vec2ForCTC` với vocabulary riêng ở `data/processed/l2arctic-v1/vocab.json`; parser, collator và scorer đã có. Các preset CPU hỗ trợ train A/B/C local; Colab là lựa chọn bổ sung. Checkpoint sau train lưu riêng khỏi pretrained trong `runs/`. Xem [hướng dẫn train](../../TRAINING.md) và [kế hoạch nghiên cứu](../../docs.md).
