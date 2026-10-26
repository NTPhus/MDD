"""Current, evidence-backed chapters; reads saved artifacts without model calls."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def block(kind, text):
    return {"type": kind, "text": text}


def table(headers, rows, widths=None, source=None):
    return {"type": "table", "headers": headers, "rows": rows,
            "widths": widths or [1] * len(headers), "source": source}


def artifact_figure(path, label, caption, source):
    return {"type": "artifact_figure", "file": path, "label": label,
            "caption": caption, "source": source, "max_height_mm": 100}


def percent(value):
    return "—" if value is None else f"{100 * value:.2f}"


def update_chapters(chapters):
    p = lambda t: block("p", t)
    h = lambda t: block("h2", t)
    note = lambda t: block("note", t)
    add = lambda title, blocks: chapters.append({"title": title, "blocks": blocks})
    asr = read("data/generated/asr-backends-20261023/results.json")
    cro = read("data/generated/crottc-pilot-20261023/results.json")
    cro_manifest = read("data/generated/crottc-pilot-20261023/manifest.json")
    model_ids = ["wav2vec2-base-960h", "wav2vec2-vi-en-v1", "whisper-small.en"]
    group_labels = [("tlv_previously_observed_heldout", "TLV · 111 câu"),
                    ("saa_external_same_prompt", "SAA · 4 người Việt"),
                    ("user_development", "3 recording phát triển")]
    chapters[1]["blocks"].append(p("Cơ sở nghiên cứu: wav2vec 2.0 học biểu diễn tự giám sát rồi thích ứng có nhãn [R-WAV2VEC2]; Qwen3 là decoder ngôn ngữ [R-QWEN3]. LoRA bổ sung cập nhật hạng thấp vào trọng số đã đóng băng [R-LORA], còn QLoRA kết hợp base lượng tử với adapter [R-QLORA]. Recipe dự án chỉ mở q_proj/v_proj, không phải toàn bộ attention/MLP. L2-ARCTIC cung cấp nhãn phát âm [R-L2ARCTIC]; L2-Arctic-plus mở rộng phần giải thích/lời khuyên [R-ALM]."))
    chapters[7]["blocks"].append(p("Nguồn W&I+LOCNESS được giới thiệu trong BEA-2019 với đánh giá ERRANT F0.5 [R-WI]; JFLEG tập trung sửa fluency của văn viết [R-JFLEG]. Vì vậy chrF/JSON trên 16 probe nội bộ chưa thay thế được benchmark theo hai tài liệu này, và fluency văn viết khác tốc độ/ngắt nghỉ khi nói."))
    chapters[10]["blocks"].append(p("Nghiên cứu CTC-GOP đề xuất điểm phát âm từ mô hình CTC [R-CTCGOP]. Hàm ctc_variant_posterior_margin_v1 trong dự án là thử nghiệm riêng; báo cáo không nhận là bản tái hiện phương pháp hoặc điểm công bố của CTC-GOP. CMUdict [R-CMU] cung cấp biến thể từ điển, không tự quyết định mọi biến thể thực nói có hợp lệ hay không."))

    add("14. Từ checkpoint đã train tới pipeline web hiện tại", [
        p("Các lượt train ở chương trước là tài sản thực nghiệm cục bộ. Tệp checkpoint tồn tại chưa có nghĩa mọi request web đều chạy checkpoint đó. Bảng sau đối chiếu đường gọi hiện tại trong local_speech_engine.py, demo_model_registry.py và worker với các artifact đã kiểm tra [S19–S22]."),
        table(["Vai trò", "Model / nơi lưu", "Thực tế sử dụng"], [
            ["Nghe tiếng Anh", "models/pretrained/faster-whisper-small.en", "Whisper small.en; CPU INT8, 2 threads, beam 5; mặc định chat"],
            ["Đối thoại", "models/pretrained/Qwen3-0.6B", "Qwen base CUDA BF16, lịch sử thật; không phải LoRA grammar hay LoRA feedback"],
            ["Ngữ pháp / natural", "runs/grammar-qwen3-06b-wi-jfleg-v1/adapter", "Adapter qwen-grammar; chạy riêng trong collapse"],
            ["Lỗi phát âm", "runs/cascade-l2arctic-plus-seed42/B/best_checkpoint", "wav2vec2 phoneme B; CPU; so các biến thể CMUdict"],
            ["Phát âm thử nghiệm", ".tools/crottc/bundle", "CROTTC-IF/WavLM Large; CPU FP32; chỉ khi được chọn"],
            ["BERT", "models/pretrained/bert-base-uncased", "Giữ cho ASR wav2vec2 cũ; bypass, không tải trong luồng Whisper"],
            ["Qwen feedback", "runs/cascade-l2arctic-plus-seed42/Q/adapter", "Có checkpoint và pilot; luồng analyze hiện luôn có khóa assessment nên bỏ qua đường coaching cũ"],
            ["Projector", "runs/projector-l2arctic-plus-seed42/alignment", "Nhánh offline P0; không nằm trong luồng web"],
            ["Giọng trả lời", "Kokoro local; tùy chọn ai33 Sarah", "Kokoro mặc định; ai33 là API ngoài máy nếu chủ động chọn"],
        ], [1.1, 2.15, 2.5]),
        h("Trình tự một lượt nói"),
        p("**Mic/upload → mono 16 kHz → Whisper → transcript → Qwen base → câu trả lời → TTS.** Whisper không nhận câu mục tiêu, lịch sử, hotwords hay bản sửa Qwen làm prompt. Transcript gốc được giữ trong artifact để kiểm tra. Luồng kiểm tra phát âm dùng audio gốc cùng transcript đang được chọn, không dùng câu đã sửa ngữ pháp làm nhãn phát âm."),
        p("Hội thoại có thể xuất hiện khi nội dung sẵn, còn ngữ pháp và phát âm tiếp tục xử lý trong collapse. Với CROTTC, tác vụ phone CPU được tách để có thể chạy cùng sinh câu trả lời trên GPU. Tùy chọn chờ voice câu mở đầu chỉ là điều khiển trình bày; không chứng minh tốc độ sinh voice hoặc model đã tăng."),
        p("Khi người dùng sửa transcript, hệ thống so văn bản mới với cũ: chỉ đổi nội dung thật mới tạo lại câu trả lời; tái chấm dùng bằng chứng âm học đã giữ từ bản thu. Sửa transcript không thay đổi audio đã nói. Chỉ các phép chuyển số một token rõ nghĩa mới được dùng để tra từ điển; không suy ra câu mới từ ngữ cảnh rồi coi đó là lời nói chắc chắn."),
        note("Dropdown Original/Fine-tuned của coaching không đổi Whisper thành wav2vec2 và không tự biến Qwen hội thoại thành adapter đã train. coaching_model_id trong metadata cũng chưa đủ chứng minh adapter đã được gọi. Muốn kết luận phải đọc đường gọi và runtime check thực sự."),
        h("Những thành phần local và thành phần ngoài máy"),
        p("Whisper, Qwen base/adapter, B phoneme, CROTTC, BERT và Kokoro đều có tệp chạy ở máy. Whisper/CROTTC tải checkpoint công bố rồi inference, chưa fine-tune trong dự án. Kokoro dùng trọng số local và runtime ONNX [R-KOKORO; R-KOKORO-ONNX]. ai33 tạo voice qua API; ngrok chỉ là đường truy cập web. Việc tunnel chạy ngoài máy không có nghĩa ASR/Qwen đang dùng API ngoài."),
        table(["Danh tính", "Revision / SHA256 trọng số"], [
            ["Whisper small.en", "revision d1d751a5f8271d482d14ca55d9e2deeebbae577f; model.bin SHA256 62b2a45b05ee59acb4a5341b33ee35e041395d378d418a18acfe4c9e768ee37a"],
            ["Qwen3-0.6B", "revision c1899de289a04d12100db370d81485cdf75e47ca; SHA256 f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"],
            ["Phoneme B", "SHA256 da000521b31ad4d7445b8eb13d82d8a767d6535e678d727aae9baa8a8b177110"],
            ["Grammar adapter", "SHA256 f865033a3053229864eb1fee1957aa5488ea5376d4d3a3279b4cb6d68d58632e"],
            ["CROTTC-IF", "revision 9149e5107b3332b9aa3f5573f6328a4e312006ad; model.ckpt SHA256 3a9a4a03d2af6c90f71db8972a51dd4ce4559d1e9801f078904a65ce4aa113ef"],
        ], [1.1, 4.6], "Tệp setup/provenance và manifest đã lưu [S19–S22]. Revision nguồn và SHA tệp là hai loại định danh khác nhau."),
    ])

    rows = []
    latency = []
    for group, label in group_labels:
        models = asr["groups"][group]["models"]
        rows.append([label] + [percent(models[m]["strict"]["wer"]) for m in model_ids])
        latency.append([label] + [f"{models[m]['inference_seconds'] / models[m]['cases']:.3f}" for m in model_ids])
    add("15. Đối chứng ASR: Whisper small.en và wav2vec2", [
        p("Thực nghiệm cố định gồm 118 recording, tổng 617,55 giây: 111 câu TLV đã dùng làm holdout của nhánh char-ASR; bốn người Việt từ Speech Accent Archive [R-SAA] đọc cùng một đoạn; ba recording người dùng đã xem trong giai đoạn phát triển. Mọi audio có kết quả hoàn tất cho đủ ba backend. Không BERT, Qwen hoặc TTS tham gia chấm ASR [S19]."),
        p("Tất cả chạy CPU hai threads: wav2vec2 base và fine-tuned dùng FP32 greedy CTC; Whisper dùng INT8 beam 5 và native word timestamps. Cùng audio nguyên vẹn và cùng strict tokenization khi tính WER. Reference chỉ dùng chấm sau suy luận, không đưa vào decoder. So base → fine-tuned là đối chứng cập nhật trọng số cùng họ model; so với Whisper đồng thời thay kiến trúc, pretraining và decoder nên không cô lập nguyên nhân cải thiện."),
        table(["Nhóm · strict WER (%) ↓", "wav2vec2 base", "wav2vec2 fine-tuned", "Whisper small.en"], rows, [2.1, 1.15, 1.4, 1.35], "Nguồn: results.json/groups; micro edit counts; strict WER là thước đo chính [S19]."),
        artifact_figure("data/generated/asr-backends-20261023/wer_by_group.png", "A1", "Strict WER và WER chuẩn hóa số đồng nhất theo nhóm; các nhóm không gộp thành một bảng xếp hạng chung.", "asr-backends-20261023; 118 audio × 3 backend đã hoàn tất."),
        p("Trên TLV: 436/1.023 → 385/1.023 → 284/1.023 word edits, tương ứng WER 42,62 → 37,63 → 27,76%. Fine-tune giảm 4,99 đpt, còn Whisper thấp hơn fine-tuned 9,87 đpt. Có 42 câu fine-tuned tốt hơn base, 10 câu kém hơn, 59 bằng; Whisper tốt hơn fine-tuned ở 61 câu, kém ở 17 và bằng ở 33. Số câu khớp tuyệt đối lần lượt 3, 4 và 17/111."),
        p("Số lịch sử ở chương 9 là 440 → 388 edits trên GPU BF16 theo scorer của lượt train. Đợt đối chứng mới dùng CPU FP32 và tokenization hiện tại nên cho 436 → 385. Hai artifact cùng được giữ; không cộng các counter giữa hai phép đo, không sửa log train để khớp bảng mới."),
        table(["ASR giây/audio ↓", "wav2vec2 base", "wav2vec2 fine-tuned", "Whisper small.en"], latency, [2.1, 1.15, 1.4, 1.35], "Loại thời gian tải model và decode/resample audio; Whisper bao gồm tạo word timings. Không phải latency toàn hệ thống."),
        artifact_figure("data/generated/asr-backends-20261023/asr_latency.png", "A2", "Thời gian ASR trung bình và real-time factor; Whisper chính xác hơn về WER trong các nhóm nhưng chậm hơn hai CTC backend trên máy này.", "Số đo thực trên cùng đợt 23/10; chưa bao gồm Qwen, pronunciation và TTS."),
        h("Kết quả có ích nhưng chưa đủ cho mọi giọng Việt"),
        p("TLV chỉ có một người và holdout đã quan sát trước, không phải test hoàn toàn mới sau mọi quyết định. SAA có bốn giọng nhưng cùng một bài đọc. Ba recording người dùng là development cases. Split audit không thấy giao audio/speaker/prompt với train/validation char-ASR local, nhưng không biết corpus công khai có xuất hiện trong pretraining Whisper hay không."),
        p("Whisper vẫn sinh sai ở một số audio phát âm khó: câu đích ‘End of the road for some players’ được nghe thành ‘And also load for some players.’; câu ‘Zero five eight three’ thành ‘I don't know what to do.’. Reference là câu định đọc, chưa có nhãn độc lập về từng từ thực sự phát ra. WER thấp hơn không cho phép dùng Whisper để kết luận một âm đã phát âm đúng."),
        p("Whisper là mô hình học audio → transcript theo weak supervision quy mô lớn [R-WHISPER]; faster-whisper/CTranslate2 cung cấp bản chạy tối ưu cục bộ [R-FASTER-WHISPER]. Dự án đã chọn nó cho ASR từ kết quả quan sát, không fine-tune Whisper và không lấy điểm trong paper thay cho bảng đo ở đây."),
        h("BERT đã cải thiện gì trước khi đổi ASR"),
        p("Đợt phát triển 10 audio giọng Việt trước đây dùng wav2vec2 char-ASR cho 109/335 word edits (WER 32,54%). Sau bước chọn/chuẩn hóa ứng viên bằng BERT-base-uncased, conversation_text còn 104/335 edits (31,04%): giảm 1,49 đpt; chưa có câu khớp tuyệt đối và chưa có human semantic review hoàn tất. Tác vụ này không train lại BERT. Kết quả chỉ so trước/sau trong cùng probe, không ghép vào 118 audio hoặc suy thành benchmark BERT với Whisper [S23]."),
        p("Theo bài BERT [R-BERT], masked language modeling giúp học ngữ cảnh chữ. Trong pipeline đã thử, nó chỉ chấm ứng viên chữ từ transcript/phone, không nghe waveform và không bảo đảm câu khôi phục là câu người dùng thực sự nói. Việc bước này hiện được bypass với Whisper giảm thêm một chặng suy luận; cần benchmark riêng nếu muốn bật lại."),
    ])

    m = cro["summary"]["models"]
    cro_rows = []
    for key, title, formatter in [
        ("f1", "Phone F1 (%) ↑", percent), ("per", "PER (%) ↓", percent),
        ("precision", "Phone precision (%) ↑", percent), ("recall", "Phone recall (%) ↑", percent)]:
        cro_rows.append([title] + [formatter(m[x]["raw"]["metrics"][key]) for x in ("cascade-B", "crottc-if")])
    reviews = []
    for key, title in [("hint_precision", "Hint precision (%)"), ("hint_recall", "Hint recall (%)"),
                       ("hint_false_positive_rate", "Từ đúng bị đưa vào review (%)")]:
        reviews.append([title] + [percent(m[x]["ui_review"][key]) for x in ("cascade-B", "crottc-if")])
    add("16. CROTTC-IF, wav2vec2 B và đánh giá trôi chảy", [
        p("CROTTC-IF được lấy từ checkpoint tác giả công bố [R-CROTTC; R-CROTTC-BUNDLE], gồm WavLM Large [R-WAVLM] cùng lớp Conformer/decoder. Dự án chỉ inference, không train thêm. Pilot có 24 câu nhãn chuyên gia từ sáu speaker test Plus: mỗi người hai câu nhóm ít lỗi và hai câu nhiều lỗi, chọn hash cố định trước suy luận; thêm ba audio người dùng không có nhãn phone gold. Đã hoàn thành đủ 54 lượt model/audio [S20]."),
        p("Đây là pilot phân tầng bằng nhãn, chưa ước lượng tần suất lỗi toàn tập. Decoder CROTTC dùng đường AR/CTC chính thức, nhận audio không có canonical prompt. Các phone lặp được giữ vì đây đã là chuỗi autoregressive được giải mã, không phải frame labels cần CTC-collapse lần nữa. Phiên bản phát hành không trả posterior frame/phone timestamps; backend chỉ đối chiếu chuỗi, không tự tạo GOP hoặc thời điểm phone giả."),
        table(["24 câu có nhãn · inventory gốc", "wav2vec2 B", "CROTTC-IF"], cro_rows + [
            ["PER insertions / 819 gold phones", str(m["cascade-B"]["raw"]["counts"]["PER_I"]), str(m["crottc-if"]["raw"]["counts"]["PER_I"])],
            ["Inference tổng / trung bình (s)", f"{m['cascade-B']['inference_sec']:.3f} / {m['cascade-B']['inference_sec']/24:.3f}", f"{m['crottc-if']['inference_sec']:.3f} / {m['crottc-if']['inference_sec']/24:.3f}"],
        ], [2.5, 1.5, 1.5], "Cùng scorer canonical-phone-and-ordered-insertion-slots-v1. Không phải scorer công bố của paper CROTTC [S20]."),
        artifact_figure("data/generated/crottc-pilot-20261023/research_metrics.png", "A3", "Phone detection F1/PER trên 24 câu; inventory gốc là chính, chuẩn hóa chung 39-phone là sensitivity analysis.", "crottc-pilot-20261023; raw và shared_inventory chấm sau inference, không dùng để tune."),
        p("Chuẩn hóa chung bỏ stress/dấu distortion, AX → AH và giữ unknown trong PER/detection. B có F1 49,65%, PER 18,07%; CROTTC có F1 19,01%, PER 53,11%. Thứ tự quan sát ở pilot không đảo chiều, nhưng cách chuẩn hóa làm mất thông tin distortion. Không xóa unknown rồi coi phần còn lại là đánh giá đầy đủ cả câu."),
        table(["Review theo word_index gold", "wav2vec2 B", "CROTTC-IF"], reviews + [["Quyết định đã hiệu chuẩn", "0", "0"]], [2.5, 1.5, 1.5], "Đây là hành vi danh sách từ cần xem lại của UI, khác research phone F1. Các quyết định hiệu chuẩn đều abstain [S20]."),
        note("CROTTC bắt được nhiều từ lỗi hơn trong danh sách review (82% so với 63%) nhưng cũng báo nhiều từ đúng hơn (55/128 so với 22/128). Hint F1 gần nhau không triệt tiêu đánh đổi này. B tiếp tục là mặc định; CROTTC là tùy chọn nghiên cứu, chưa có cơ sở nâng làm model chính."),
        p("Audio Stella 24,16 giây mất khoảng 223,75 giây khi chạy CROTTC offline, sinh 379 phones có đoạn lặp và đưa 60/69 từ vào review. Hai mẫu road/digits cũng có bất ổn. Không có nhãn chuyên gia ở ba audio người dùng nên chỉ báo đầu ra/latency, không gán precision hay accuracy. Model/worker CROTTC tốn khoảng 2,1 GiB RSS trong phiên chạy; lỗi MemoryError ban đầu và việc resume đã được lưu trong execution_notes.json."),
        h("Điểm paper và điểm local đo những gì"),
        p("Paper CROTTC-IF báo F1 71,77% trên L2-ARCTIC [R-CROTTC]. Điểm này dùng protocol của tác giả; scorer dự án gắn lỗi theo canonical positions và ordered insertion slots. Pilot local chỉ 24 câu đã phân tầng; overlap training checkpoint công bố với split local chưa được xác minh. Vì khác scorer, split, decoder và dữ liệu, 19,01% local không phải kết quả tái hiện hoặc phủ định trực tiếp 71,77% của paper."),
        h("Trôi chảy: thống kê mô tả từ Whisper"),
        p("Module speech_fluency.py dùng native word times của Whisper và độ dài bản thu để báo tốc độ từ/phút, thời gian khoảng nói, số pause nội bộ từ 0,25 giây, pause dài nhất và khoảng trống timestamp đầu/cuối. Mốc 0,25 giây là quy ước thống kê hiện tại, chưa được hiệu chỉnh bằng chuyên gia. Chỉ số phụ thuộc transcript/timing ASR; khi sửa câu đối chiếu, measured_transcript vẫn chỉ rõ transcript thực sự được đo."),
        p("Live check đã lưu chứng minh cả B/CROTTC có thể đi cùng Whisper và module fluency [S21]. Ví dụ road có khoảng 94,99 từ/phút và 0 pause nội bộ theo ASR 3,79 giây; TLV/a0018 khoảng 178,57 từ/phút trên 1,68 giây. Đây là ví dụ kết nối module, không phải thang IELTS hay điểm fluency đã train. Chưa chấm prosody, lexical stress hoặc xác thực correlation với người chấm."),
    ])

    add("17. Đối chiếu nghiên cứu và thiết kế thực nghiệm tiếp theo", [
        p("Tài liệu ngoài dự án hỗ trợ lựa chọn phương pháp, dữ liệu và thước đo. Báo cáo tách ba mức: mô tả phương pháp được công bố; checkpoint đã tải/chạy local; và kết quả đủ bằng chứng trong thực nghiệm này. Không xếp tất cả mô hình thành một ‘accuracy’ vì ASR, MDD, sửa câu và voice giải quyết những bài toán khác nhau."),
        table(["Nghiên cứu / mô hình", "Cơ sở liên quan", "Bằng chứng local / giới hạn"], [
            ["wav2vec 2.0 [R-WAV2VEC2]", "Biểu diễn audio tự giám sát + fine-tune có nhãn", "Có phoneme A/B/C và char-ASR; cải thiện so cùng base trên split riêng"],
            ["Whisper [R-WHISPER]", "Audio → transcript; weak supervision", "Pretrained small.en thắng WER ba nhóm; chưa đánh giá MDD bằng Whisper"],
            ["Qwen3 + LoRA/QLoRA [R-QWEN3; R-LORA; R-QLORA]", "Sinh text; thích ứng ít tham số", "Hai adapter có train; chưa có benchmark hội thoại độc lập"],
            ["L2-Arctic-plus [R-ALM]", "Nhãn giải thích lỗi, suggestion và ALM tuning", "Giữ cascade đối chứng; số ALM lớn trong paper không phải điểm Qwen0.6 local"],
            ["BERT [R-BERT]", "Encoder masked language modeling", "Không fine-tune; hỗ trợ lexical candidates nhưng không bảo đảm khôi phục đúng ý; bypass Whisper"],
            ["CTC-GOP [R-CTCGOP]", "Điểm phát âm dựa trên CTC", "Posterior margin local là phương pháp thử riêng; calibration chưa đạt"],
            ["CROTTC-IF [R-CROTTC; R-CROTTC-BUNDLE]", "Prompt-free MDD; acoustic/linguistic bias", "Đã tải và thử 27 audio; chưa đạt lựa chọn mặc định theo pilot hiện tại"],
            ["W&I/JFLEG [R-WI; R-JFLEG]", "Grammar và fluency văn viết có nhãn công khai", "Đã train grammar LoRA; chưa chạy đủ ERRANT F0.5/GLEU test"],
            ["SpeechOcean762 [R-SPEECHOCEAN]", "Phát âm tiếng Anh; nhiều người chấm", "Kiểm chứng ngoài miền cho thấy false alarm cao; khác L1 và rubric"],
        ], [1.5, 1.8, 2.3]),
        h("Giải thích vì sao fine-tune vẫn có thể kém pretrained khác"),
        p("Fine-tune tối ưu loss trên dữ liệu, nhãn và ngân sách cụ thể; không tạo bảo đảm vượt mọi model. ASR Việt có khoảng 2,49 giờ train gồm source replay, trong khi Whisper bắt đầu từ pretraining khác. Trên 111 TLV, fine-tuned wav2vec2 vẫn cải thiện so base của chính nó. Việc Whisper có WER thấp hơn không chứng minh training đã vô ích hoặc chỉ khác Transformer là nguyên nhân."),
        p("BERT chọn từ hợp văn cảnh nhưng không nhận trực tiếp audio trong nhánh hiện tại; Qwen cũng có thể sinh câu trôi chảy khác nội dung đã nói. Vì vậy câu có nghĩa chưa đủ chứng minh đúng từng chữ. Cần giữ bản thu, transcript, câu người dùng xác nhận và câu gợi ý thành các dữ liệu có nguồn riêng; đánh giá edit fidelity và unsupported changes thay vì chỉ độ mượt."),
        p("GOPT [R-GOPT] là hướng tham khảo cho đánh giá phát âm nhiều mức. Chưa có checkpoint GOPT chạy trong demo và chưa đo correlation với người chấm ở dự án; không đưa điểm công bố của nó vào bảng model local. Đây là lựa chọn nghiên cứu sau khi khóa được rubric và tập kiểm chứng độc lập."),
        h("Các bước tiếp theo có thể kiểm chứng được"),
        table(["Mục tiêu", "Thực nghiệm đề xuất", "Tiêu chí báo cáo"], [
            ["Giảm báo nhầm MDD", "Thu thêm mẫu người Việt đúng/sai với nhãn phone; fit ngưỡng trên validation riêng, khóa test trước", "Precision/recall/F1, FPR, FNR, coverage và tỷ lệ abstain; phân tích theo phone/speaker"],
            ["Nâng wav2vec2 B", "Đối chứng giữ model/decoder; thử thêm dữ liệu, sampling/replay và fine-tune head phát hiện trực tiếp có nhãn", "Cùng scorer và test; nhiều seed/speaker; không chọn ngưỡng bằng test"],
            ["Grammar/natural", "Đánh giá đủ 2.999 grammar + 747 natural heldout, multi-reference; kiểm cả câu đúng và câu vô nghĩa", "ERRANT F0.5, GLEU phù hợp nguồn, giữ nguyên control, semantic fidelity và JSON contract tách riêng"],
            ["Hội thoại Qwen", "Tập multi-turn có sửa ASR, ngoài chủ đề, nói thiếu và lặp; đánh giá độc lập với grammar", "Tính liên quan câu trước, tỷ lệ lặp/không hiểu, hallucination và latency"],
            ["CROTTC", "Tái hiện scorer/split tác giả trước; kiểm chuỗi lặp dài, memory/timeout; khóa version", "Paper reproduction riêng; sau đó mới benchmark cùng policy UI local"],
            ["Fluency / tốc độ", "Đối chiếu word timings và pause bằng nhãn; đo cold/warm theo từng chặng", "Sai số timing, correlation/rubric nếu có người chấm; p50/p95/RSS/VRAM thay vì một số trung bình"],
        ], [1.1, 2.5, 2.2]),
        note("Đây là kế hoạch nghiên cứu tiếp theo, chưa phải kết quả đã đạt và không tự khởi động training. Bản báo cáo này chỉ đọc lại output có sẵn; người dùng chủ động chạy các lệnh train khi quyết định."),
    ])
    paths = ["data/generated/asr-backends-20261023/manifest.json",
             "data/generated/asr-backends-20261023/results.json",
             "data/generated/crottc-pilot-20261023/manifest.json",
             "data/generated/crottc-pilot-20261023/results.json",
             "data/generated/crottc-live-20261023/verification.json",
             "data/generated/vi-voice-pipeline-v1/bert-v2/results.json",
             "scripts/local_speech_engine.py", "src/mdd/demo_model_registry.py",
             "src/mdd/chat_assessment.py", "src/mdd/speech_fluency.py"]
    snapshot = {"date": "2026-10-23", "scope": "Saved artifacts only; no training/inference/API",
                "input_files": [{"path": path, "sha256": hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),
                                 "bytes": (ROOT/path).stat().st_size} for path in paths],
                "asr_manifest_canonical_sha256": asr["manifest_sha256"],
                "crottc_manifest_canonical_sha256": cro["manifest_sha256"],
                "crottc_model_source": cro_manifest["models"]}
    (HERE/"evidence_current_comparisons.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def bibliography_blocks():
    path = ROOT / "docs/references_20261023.json"
    if not path.is_file():
        raise FileNotFoundError("Wait for the verified shared primary bibliography: " + str(path))
    value = json.loads(path.read_text(encoding="utf-8"))
    entries = value["references"] if isinstance(value, dict) else value
    output = [block("h2", "Tài liệu nghiên cứu và tài liệu triển khai gốc"),
              block("p", "Trích dẫn [R-TÊN] bên trong các chương chỉ phương pháp hoặc nguồn dữ liệu liên quan. Chúng không thay thế bằng chứng thực chạy [Sxx]. Các đường dẫn dưới đây là bài báo, trang dữ liệu hoặc mã/tài liệu chính thức; ngày tra cứu 23/10/2026.")]
    for entry in entries:
        sid = entry.get("id", entry.get("key"))
        title = entry.get("title", "")
        authors = entry.get("authors", "")
        if isinstance(authors, list):
            authors = ", ".join(authors)
        year = entry.get("year") or "tài liệu cập nhật"
        url = entry.get("url", "")
        output.append(block("small", f"**[{sid}] {authors} ({year}). {title}.** [{url}]({url})"))
    return output
