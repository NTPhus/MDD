# Các paper cho project MDD tiếng Anh của người Việt

Bảy file PDF tương ứng với các tài liệu `p1` đến `p7` trong [kế hoạch](../test.md). Đã kiểm tra định dạng PDF, số trang, tiêu đề ở trang đầu và SHA-256. Tổng dung lượng khoảng 5,34 MiB.

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

## Nền tảng TTS

Đã bổ sung **7 PDF về TTS** trong [tts](tts/README.md): Tacotron 2, FastSpeech 2, HiFi-GAN, VITS, VALL-E, SpeechT5 và F5-TTS. Tài liệu đi kèm giải thích input, các bước xử lý, output, khác biệt giữa model với API và khả năng đa tác vụ. Manifest riêng của nhóm này ở [tts/manifest.json](tts/manifest.json); bảy PDF MDD ở trên được giữ nguyên.
