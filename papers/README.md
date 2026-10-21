# Các paper cho project MDD tiếng Anh của người Việt

Nhóm MDD gốc có bảy file PDF tương ứng với các tài liệu `p1` đến `p7` trong [kế hoạch](../docs.md). Đã kiểm tra định dạng PDF, số trang, tiêu đề ở trang đầu và SHA-256. Tổng dung lượng nhóm này khoảng 5,34 MiB.

| Mã | Paper và bản PDF local | Số trang | Nguồn tải |
| --- | --- | --- | --- |
| p1 | [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](2020_baevski_wav2vec2.pdf) | 19 | [arXiv](https://arxiv.org/pdf/2006.11477) |
| p2 | [L2-ARCTIC: A Non-native English Speech Corpus](2018_zhao_l2_arctic.pdf) | 5 | [ISCA Archive](https://www.isca-archive.org/interspeech_2018/zhao18b_interspeech.pdf) |
| p3 | [A Study on Fine-Tuning wav2vec2.0 Model for the Task of Mispronunciation Detection and Diagnosis](2021_peng_wav2vec2_mdd.pdf) | 5 | [ISCA Archive](https://www.isca-archive.org/interspeech_2021/peng21e_interspeech.pdf) |
| p4 | [Explore wav2vec 2.0 for Mispronunciation Detection](2021_xu_explore_wav2vec2_mispronunciation.pdf) | 5 | [ISCA Archive](https://www.isca-archive.org/interspeech_2021/xu21k_interspeech.pdf) |
| p5 | [Text-Aware End-to-end Mispronunciation Detection and Diagnosis](2022_peng_text_aware_mdd.pdf) | 5 | [arXiv](https://arxiv.org/pdf/2206.07289) |
| p6 | [Evaluating OpenAI’s Whisper ASR: Performance Analysis Across Diverse Accents and Speaker Traits](2024_whisper_diverse_accents.pdf) | 23 | [University of Cambridge Repository](https://api.repository.cam.ac.uk/server/api/core/bitstreams/4ef0ef50-0f8a-4325-82bf-7e333cb23565/content) |
| p7 | [Unlocking Large Audio-Language Models for Interactive Language Learning](2026_liu_interactive_language_learning.pdf) | 24 | [ACL Anthology](https://aclanthology.org/2026.findings-eacl.190.pdf) |

Thứ tự đọc theo kế hoạch: **p2 → p3 → p1 → p5**. Đọc p4 để tham khảo hướng phát hiện đúng/sai trên đoạn âm; p6 phục vụ đánh giá STT theo accent; p7 phục vụ thiết kế phản hồi hội thoại.

File p6 là bản lưu công khai tại Cambridge; số trang có thể khác bản dàn trang của tạp chí. File p5 là preprint arXiv. Nguồn tải, dung lượng, số trang và checksum của từng file được lưu trong [manifest.json](manifest.json).

## Huấn luyện model nhỏ để hướng dẫn sửa phát âm

Đã bổ sung năm PDF trong [feedback](feedback/), phục vụ nhánh **assessment MDD → Qwen → hướng dẫn luyện phát âm**. Bài sát tác vụ nhất là **p7**, đã có trong nhóm MDD: [Unlocking Large Audio-Language Models for Interactive Language Learning](2026_liu_interactive_language_learning.pdf), Findings of EACL 2026. Bài này giới thiệu L2-Arctic-plus, giải thích cách tạo và kiểm tra lời hướng dẫn, instruction-tuning và đánh giá phản hồi. [Code và annotation của tác giả](https://github.com/zoeyada/ALMs4Learning).

| Mã | Paper và PDF local | Dùng để nghiên cứu | Nguồn |
| --- | --- | --- | --- |
| feedback1 | [Qwen3 Technical Report](feedback/2025_qwen3_technical_report.pdf) | Kiến trúc, các giai đoạn train, distillation và benchmark của họ Qwen3, gồm bản 0.6B | [arXiv 2025](https://arxiv.org/abs/2505.09388) |
| feedback2 | [LoRA: Low-Rank Adaptation of Large Language Models](feedback/2021_hu_lora.pdf) | Giữ nguyên trọng số nền, học thêm các ma trận adapter có rank nhỏ | [arXiv 2021, v2](https://arxiv.org/abs/2106.09685v2) |
| feedback3 | [QLoRA: Efficient Finetuning of Quantized LLMs](feedback/2023_dettmers_qlora.pdf) | Nén model nền xuống 4-bit, NF4 và fine-tune adapter để giảm bộ nhớ | [arXiv 2023](https://arxiv.org/abs/2305.14314) |
| feedback4 | [Self-Instruct: Aligning Language Models with Self-Generated Instructions](feedback/2023_wang_self_instruct.pdf) | Sinh và lọc các cặp instruction/response cho instruction-tuning | [ACL 2023](https://aclanthology.org/2023.acl-long.754/) |
| feedback5 | [Distilling Step-by-Step!](feedback/2023_hsieh_distilling_step_by_step.pdf) | Chuyển kiến thức tác vụ từ model lớn sang model nhỏ bằng dữ liệu giám sát bổ sung | [Findings of ACL 2023](https://aclanthology.org/2023.findings-acl.507/) |

Thứ tự đọc đề xuất cho nhánh này: **p7 → feedback1 → feedback2 → feedback3 → feedback4 → feedback5**. Dataset âm thanh nền được mô tả trong **p2**; cách fine-tune wav2vec2 cho nhận diện lỗi phát âm nằm ở **p3**.

Các bài LoRA, QLoRA, Self-Instruct và Distilling Step-by-Step mô tả phương pháp huấn luyện tổng quát; chúng không chứng minh Qwen3-0.6B hướng dẫn phát âm hiệu quả. Kết quả của các ALM trong p7 cũng cần được phân biệt với pipeline Qwen nhận assessment bằng văn bản của project. Dữ liệu hướng dẫn tiếng Việt và hiệu quả sau fine-tune vẫn cần đánh giá riêng.

Nguồn tải chính xác, phiên bản PDF, số trang, dung lượng và SHA-256 của năm file mới nằm trong [feedback/manifest.json](feedback/manifest.json). Bản p7 hiện có được dùng lại, không lưu một bản PDF trùng.

## Nền tảng TTS

Đã bổ sung **7 PDF về TTS** trong [tts](tts/README.md): Tacotron 2, FastSpeech 2, HiFi-GAN, VITS, VALL-E, SpeechT5 và F5-TTS. Tài liệu đi kèm giải thích input, các bước xử lý, output, khác biệt giữa model với API và khả năng đa tác vụ. Manifest riêng của nhóm này ở [tts/manifest.json](tts/manifest.json); bảy PDF MDD ở trên được giữ nguyên.
