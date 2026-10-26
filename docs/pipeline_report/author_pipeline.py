"""Current Vietnamese system report from saved evidence only; no inference."""
from pathlib import Path
from datetime import datetime
import csv
import json
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
def read(path): return json.loads((ROOT / path).read_text(encoding='utf-8'))
def rows(path):
    with (ROOT / path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))
snapshot = read('docs/pipeline_report/source_snapshot.json')
chapters = []
def p(t): return {'type': 'p', 'text': t}
def h(t): return {'type': 'h2', 'text': t}
def note(t): return {'type': 'note', 'text': t}
def code(t): return {'type': 'code', 'text': t}
def bullets(*items): return {'type': 'bullets', 'items': list(items)}
def tab(headers, values, widths=None, source=None):
    out = {'type': 'table', 'headers': headers, 'rows': values}
    if widths: out['widths'] = widths
    if source: out['source'] = source
    return out
def diagram(key, caption, source): return {'type': 'diagram', 'id': key, 'caption': caption, 'source': source}
def chapter(title, *blocks): chapters.append({'title': title, 'blocks': list(blocks)})
def pct(v): return f'{float(v)*100:.2f}'

chapter('1. Tổng quan và kết luận có bằng chứng',
 p('Dự án xây dựng môi trường luyện nói tiếng Anh có hội thoại, phản hồi ngữ pháp và đối chiếu phát âm. Báo cáo được viết lại theo mã nguồn và artifact ngày 23/10/2026: hệ thống chạy thế nào, đã train thành phần nào, model nào thực sự được gọi và kết quả nào đã đo.'),
 note('**Luồng mặc định:** audio → Whisper small.en → văn bản tiếng Anh → Qwen3-0.6B base trả lời; Qwen LoRA ngữ pháp bổ sung feedback. Nhánh âm học wav2vec2 B đối chiếu phát âm; Kokoro local tạo voice. CROTTC-IF là lựa chọn thử nghiệm. BERT và Projector không nằm trong luồng Whisper mặc định.'),
 tab(['Câu hỏi', 'Kết luận hiện có'], [
 ['ASR nào tốt hơn?', '111 audio TLV: WER wav2vec2 base 42,62%; sau fine-tune 37,63%; Whisper 27,76%. Whisper tốt hơn trong phép thử này nhưng chậm hơn.'],
 ['Fine-tune có tác dụng không?', 'Có cải thiện trên test chung của từng protocol. Nhận dạng chữ và phát hiện lỗi âm dùng hai checkpoint/head khác nhau.'],
 ['Phát âm đã đáng tin chưa?', 'Calibration chưa đạt policy. Sound details là gợi ý cần xem lại; thiếu cảnh báo không có nghĩa phát âm đúng.'],
 ['CROTTC thay B chưa?', 'Chưa. Pilot 24 audio chuyên gia cho nhiều cảnh báo nhầm hơn B; giữ B mặc định.'],
 ['Qwen đã train phần nào?', 'Hai adapter: feedback từ phone-text và grammar/natural wording. Dialogue vẫn dùng base; chưa có benchmark hội thoại do người đánh giá.'],
 ['Local hết chưa?', 'Mặc định ASR, Qwen, scorer, Kokoro chạy local. ai33 là API ngoài tùy chọn; ngrok là đường truy cập qua mạng.'],
 ], [1.2, 3.8], 'Nguồn cục bộ [L01–L10]; phép đo chi tiết ở Chương 8–11.'),
 p('WER đo sai khác chữ; phone F1 đo phát hiện lỗi âm; JSON hợp lệ đo đúng định dạng. Các chỉ số này không thay thế nhau và không được gọi chung là độ chính xác hệ thống. Số công bố trong bài báo được tách khỏi số đo local.'),
)

chapter('2. Cơ sở khoa học và lựa chọn kiến trúc',
 tab(['Thành phần', 'Tài liệu gốc', 'Áp dụng trong dự án'], [
 ['wav2vec 2.0', 'Học biểu diễn âm tự giám sát rồi fine-tune CTC [R-WAV2VEC2].', 'Head phone MDD và head ký tự ASR tách biệt.'],
 ['Whisper', 'Encoder–decoder ASR, weak supervision quy mô lớn [R-WHISPER].', 'small.en qua faster-whisper/CTranslate2; không nhận đáp án chuẩn.'],
 ['Qwen3-0.6B', 'Decoder ngôn ngữ; technical report và model card [R-QWEN3].', 'Text-only; base dialogue và LoRA riêng theo tác vụ.'],
 ['BERT-base-uncased', 'Masked language modeling hai chiều [R-BERT].', 'Xếp hạng ứng viên ASR cũ; không tự nghe audio.'],
 ['CROTTC-IF', 'MDD không dùng canonical prompt khi inference [R-CROTTC].', 'Checkpoint công bố, chưa train local; pilot và UI thử nghiệm.'],
 ['LoRA / QLoRA', 'Adapter hạng thấp, base frozen; QLoRA dùng base lượng tử hóa [R-LORA], [R-QLORA].', 'Train Qwen trên GPU 4 GiB; inference có thể nạp base BF16.'],
 ], [1.05, 2.4, 2.2]),
 p('ASR tìm từ người nói định truyền đạt; MDD giữ bằng chứng âm thực phát ra. ASR có thể nhận đúng từ dù âm chưa chuẩn. Whisper nhận đúng “three” không chứng minh /θ/ đúng. Qwen cascade nhận chữ và metadata nên không nghe lại được tín hiệu đã mất sau ASR.'),
 h('Vì sao BERT chưa giải quyết triệt để ASR?'),
 p('BERT xếp hạng từ theo ngữ cảnh chữ. Transcript thiếu nhiều âm hoặc tách từ sai có thể khiến model chọn câu phổ biến nhưng khác câu đã nói. Whisper mặc định bỏ bước phục hồi này. Câu khôi phục bằng model phải ghi inferred, không coi là user-confirmed. Đây là suy luận thiết kế từ đặc tính MLM và probe local, không phải kết luận BERT kém trong mọi tác vụ [R-BERT].'),
 h('Hướng nghiên cứu liên quan'),
 p('CTC-GOP cung cấp hướng điểm âm học và calibration [R-CTCGOP]; scorer margin/variant hiện tại chưa tái lập đầy đủ công thức đó. GOPT nghiên cứu điểm phát âm nhiều mức trên SpeechOcean762 [R-GOPT], nhưng chưa cài/đo ở máy này. Tương quan PCC trong bài báo không phải phần trăm phát hiện lỗi.'),
)

chapter('3. Kiến trúc và các model đang ở local',
 diagram('architecture', 'Hình 1. Kiến trúc hiện tại: ASR tạo chữ; acoustic branch giữ bằng chứng âm; Qwen sinh text; TTS tạo voice.', 'Nguồn: registry, speech engine, worker và TTS registry [L01], [L02].'),
 tab(['Vai trò', 'Artifact local', 'Trạng thái / thiết bị'], [
 ['ASR mặc định', 'models/pretrained/faster-whisper-small.en', 'CPU INT8, 2 threads, beam 5; chưa fine-tune local.'],
 ['ASR đối chứng', 'models/pretrained/wav2vec2-base-960h\nruns/asr-vietnamese-english-v1/best_checkpoint', 'Base và thích ứng Việt; CPU FP32 trong so sánh mới.'],
 ['Phoneme mặc định', 'runs/cascade-l2arctic-plus-seed42/B/best_checkpoint', 'B Plus; CTC phone; CPU; best step 200.'],
 ['Phoneme thử nghiệm', '.tools/crottc/bundle', 'CROTTC-IF + WavLM Large; process CPU FP32 riêng.'],
 ['Text backbone', 'models/pretrained/Qwen3-0.6B', 'Reply base; live job cuda:0 BF16.'],
 ['Grammar LoRA', 'runs/grammar-qwen3-06b-wi-jfleg-v1/adapter', 'Đã train; grammar_corrected và natural.'],
 ['Feedback LoRA', 'runs/cascade-l2arctic-plus-seed42/Q/adapter', 'Đã train; coach legacy không gọi khi có assessment.'],
 ['Masked LM', 'models/pretrained/bert-base-uncased', 'Đã tải; nhánh ASR cũ; Whisper path không nạp.'],
 ['TTS mặc định', 'models/pretrained/kokoro-82m + .venv-kokoro', 'Kokoro ONNX; CUDAExecutionProvider đã kiểm thử.'],
 ['Projector', 'runs/projector-l2arctic-plus-seed42/alignment', 'P0 hoàn tất; ngoài đường chat web.'],
 ], [1.05, 2.5, 2], 'Nguồn [L01–L04], [L10]. Ready qua metadata khác với model được gọi ở từng job.'),
 p('Revision cố định: Whisper d1d751a5f8271d482d14ca55d9e2deeebbae577f; Qwen c1899de289a04d12100db370d81485cdf75e47ca; CROTTC 9149e5107b3332b9aa3f5573f6328a4e312006ad; BERT 86b5e0934494bd15c9632b12f734a8a67f723594. Hash trọng số được lưu ở setup/training/benchmark manifests; báo cáo không thay weights.'),
 note('Qwen2-Audio chỉ có metadata/tokenizer/config, thiếu weight shards để inference. Qwen3-0.6B là model text. Projector là nhánh ghép audio embeddings; current cascade chưa phải audio language model end-to-end.'),
 code('demo/chat.js                         mic/upload, polling, collapse, voice\nsrc/mdd/demo_chat.py                  jobs, revision, voice, validation\nsrc/mdd/demo_model_registry.py        ASR / pronunciation / adapters\nscripts/local_chat_process.py        persistent JSONL worker\nscripts/local_chat_worker.py         Qwen load / generate / adapters\nscripts/local_speech_engine.py       orchestration ASR → reply → feedback\nscripts/whisper_asr.py               offline faster-whisper wrapper\nsrc/mdd/pronunciation_assessment.py   CTC posterior / CMU variants\nsrc/mdd/phone_sequence_assessment.py  CROTTC sequence review\nsrc/mdd/speech_fluency.py             descriptive word-time measures\nscripts/kokoro_tts.py                local TTS\nscripts/ai33_tts.py                  optional API TTS'),
)

chapter('4. Dữ liệu, nhãn và ranh giới test',
 tab(['Nguồn', 'Nhãn / tác vụ', 'Cách dùng và giới hạn'], [
 ['L2-ARCTIC [R-L2ARCTIC]', '3.599 câu có phone annotation trong bản local.', '24 speaker, 6 L1, 4 người Việt. Phần WAV đầy đủ không tự có gold MDD.'],
 ['L2-Arctic-plus [R-ALM]', 'Reference, phone annotation và feedback.', 'Phone từ L2-ARCTIC; train feedback do GPT-4o tạo; không phải tất cả được chuyên gia duyệt.'],
 ['W&I+LOCNESS [R-WI]', 'Bài viết người học và sửa grammar.', 'Train grammar LoRA; domain bài viết khác transcript hội thoại.'],
 ['JFLEG [R-JFLEG]', 'Câu và nhiều bản sửa fluency văn bản.', 'Task natural; giữ official test 747 câu.'],
 ['SpeechOcean762 [R-SPEECHOCEAN]', '5.000 câu, 250 người giọng Trung, 5 chuyên gia.', 'External test 2.500 câu; khác rubric/domain L2-ARCTIC.'],
 ['SAA [R-SAA]', 'Người nhiều L1 đọc đoạn Stella.', '4 người Việt ở probe ASR; cùng một prompt.'],
 ['3 bản người dùng', 'Câu định đọc được người dùng cung cấp.', 'Development; không nhãn lỗi phone chuyên gia.'],
 ], [1.25, 1.95, 2.4], 'Nguồn corpus chính thức và manifests [L03–L09].'),
 h('Protocol Plus'),
 p('Train/validation/test: 2.249/450/900 utterances, tách speaker/utterance. Validation HJK/LXC/PNV; test NJS/TLV/TNI/TXHC/YKWK/ZHAA, mỗi người 150 câu. HQTV và THV cung cấp 300 câu train Việt; PNV validation Việt, TLV test Việt. Câu đọc có thể trùng giữa speaker; không phải prompt-disjoint.'),
 p('Protocol v1 cũ dùng HQTV test, Plus dùng TLV. Có 750/900 audio test Plus từng ở source train v1; A/B Plus đã train lại. Không so F1 70,85% trên HQTV với 49,30% trên sáu speaker để kết luận model giảm chất lượng.'),
 h('ASR test và calibration'),
 p('111 TLV ở so sánh ASR được kiểm tách audio/speaker/prompt khỏi train/validation của lượt fine-tune ASR chữ. Đây là previously observed held-out, chưa phải test kín mới thu. Bốn SAA cùng prompt và ba bản người dùng là nhóm bổ sung; không trộn để tuyên bố tổng quát mọi giọng Việt. Overlap với pretraining checkpoint công bố chưa biết.'),
 p('Calibration phát âm fit 300 câu HJK/LXC, holdout 150 PNV. PNV không fit ngưỡng nhưng đã tham gia chọn checkpoint B trước đó; chưa độc lập hoàn toàn với quy trình chọn model.'),
)

chapter('5. Đã train gì và trọng số nào thay đổi?',
 tab(['Lượt train', 'Supervision / loss', 'Trọng số thay đổi', 'Artifact hoàn tất'], [
 ['MDD A', 'Audio → perceived phones; CTC.', 'CNN frozen; Transformer + phone head.', '2.000 updates; source baseline.'],
 ['MDD B', '300 câu Việt + source replay 50%; CTC.', 'Best A → Transformer + phone head.', '400 updates; best Plus step 200.'],
 ['MDD C v1', 'Source-only tiếp tục; cùng budget B.', 'Best A → Transformer + phone head.', '400 updates v1; chưa C Plus hoàn tất.'],
 ['ASR Việt', '2.445 clip, gồm 611 source replay; char CTC.', 'CNN frozen; Transformer + char head.', '2 epochs, 1.223 updates; best 1.050.'],
 ['Qwen feedback', '2.249 text phone/error → feedback; assistant CE.', 'QLoRA rank8, alpha16, q_proj/v_proj; base frozen.', '3 epochs, 846 steps; best800; val loss0,378810.'],
 ['Qwen grammar', '28.501 records grammar/natural; assistant CE.', 'Adapter riêng rank8, alpha16, q_proj/v_proj.', '2 epochs, 3.564 steps; best3.500; val loss0,184105.'],
 ['Projector P0', 'Audio features → transcript; autoregressive CE.', '2 Linear + GELU; encoder/Qwen frozen.', '1 epoch, 282 updates; P1 chưa train.'],
 ], [1, 1.8, 1.9, 1.8], 'Nguồn provenance/config/completed thực chạy [L03]. Báo cáo training đồng hành trình bày optimizer, learning curves và từng protocol.'),
 p('Mỗi adapter Qwen có 1.146.880 tham số trainable, dropout0,05. Hai adapter là hai tác vụ; chưa có adapter hội thoại tự do được train. Loss feedback thấp không tự chứng minh hội thoại tốt hoặc chấm âm đúng. LoRA cập nhật các ma trận thích ứng trong Transformer, không toàn bộ base weights [R-LORA], [R-QLORA].'),
 h('Vai trò Projector'),
 p('Projector ánh xạ hidden vectors wav2vec2 768 chiều qua Linear → GELU → Linear thành embedding Qwen 1.024 chiều, ghép với text embeddings để Qwen sinh chữ. Ánh xạ ngẫu nhiên phải được học alignment P0. P1 dự kiến học feedback với Projector + Qwen LoRA nhưng chưa có checkpoint hoàn tất. Ghép P0 với adapter vốn train trên text-phone là transfer thử nghiệm, không phải đã train P1.'),
 p('Whisper, BERT, CROTTC-IF và Kokoro chỉ tải checkpoint, chưa fine-tune local. Lượng tử hóa inference và thêm wrapper không phải training. Biên soạn báo cáo không khởi động lượt train mới.'),
)

chapter('6. Workflow từ bản thu đến reply và collapse',
 diagram('workflow', 'Hình 2. Reply công bố trước grammar. CROTTC có thể chạy nền; B mặc định vẫn chấm âm trước reply.', 'Nguồn LocalSpeechEngine.analyze/run/_respond/_finish_pronunciation [L01].'),
 p('Browser thu mic/đọc file, trộn mono, resample16kHz và đóng gói PCM16 WAV. API nhận audio0,25–30 giây, một kênh, sample width2bytes; audio decoded tối đa1MiB, body speech tối đa2MiB. Mic chunks giữ ở browser, không stream liên tục đến model.'),
 p('POST /api/chat/speech trả job cho polling. Whisper chỉ nhận waveform, không nhận reference/history/đáp án; trả raw text và word intervals. BERT recovery và Qwen normalization bị tắt khi dùng Whisper. Bước chuẩn hóa số đơn giản giữ giá trị phục vụ từ điển, không dựng lại nội dung câu.'),
 bullets('B mặc định: posterior_evidence đồng bộ sau ASR, CMU variants và scorer chạy trước Qwen reply.',
 'CROTTC tùy chọn với Whisper: inference âm CPU ở thread/process riêng; Qwen GPU sinh reply, grammar xử lý tiếp rồi nhận kết quả âm. Lỗi/timeout không làm mất reply.',
 'Fluency lấy transcript và word times của bản thu ban đầu; không lấy câu grammar để tính lại tốc độ nói.'),
 p('Qwen base nhận transcript hoặc user-confirmed text cùng history thật, sinh1–2 câu tiếng Anh. Prompt yêu cầu bám lượt cuối, không lặp, không sửa grammar trong reply. Echo filter, policy, một lần retry và fallback đều có nguồn riêng. Câu mở đầu soạn sẵn; café có state/rules hỗ trợ, không gán toàn bộ hành vi đó cho model.'),
 tab(['Mốc', 'Hành vi', 'Ý nghĩa'], [
 ['ASR', 'transcribing; text + word times.', 'Chưa có reply.'],
 ['Reply', 'generating → reply_ready=true.', 'Text đã chốt; voice có thể dùng.'],
 ['Grammar', 'Greedy riêng grammar_corrected và natural.', 'Cập nhật collapse, không đổi reference âm.'],
 ['CRO pending', 'Join sau reply/grammar; có timeout.', 'Giữ reply nếu phần âm unavailable.'],
 ['Cuối lượt', 'ready / cancelled / error.', 'Không tự train hoặc gửi lại TTS khi lỗi.'],
 ], [1, 2.2, 2.2]),
 p('reply_ready khác voice đã phát: TTS còn tổng hợp, tải/buffer/play. Câu mở đầu bật auto-read đợi voice sẵn rồi mới hiện và phát. Reply hiển thị toàn câu, highlight khi đọc; feedback đến sau không thay voice text.'),
)

chapter('7. Phát âm, fluency và xác nhận câu nói',
 p('Reference mặc định là text Whisper nghe được; câu đọc cố định lấy từ server. Người học sửa text trong collapse khi nghe nhầm. Grammar/natural suggestion giữ riêng, không tự dùng để kết luận audio cũ sai những từ chưa nói. Reference sai có thể làm alignment và cảnh báo sai.'),
 p('B dùng ctc_variant_posterior_margin_v1 với CMUdict variants; greedy edit comparison còn dùng gợi ý. CROTTC xuất perceived phones đã decode; không gộp phone lặp như CTC. Sequence scorer căn chỉnh edit-cost với CMU variants, không mượn ngưỡng posterior của B. Wrapper checkpoint công bố không có frame posterior hoặc phone timestamps tin cậy [R-CROTTC], [R-CMU].'),
 note('Calibration B ineligible: không ngưỡng nào đạt đồng thời FPR≤10%, FNR≤20%, coverage≥50% và số quyết định tối thiểu. Accept/error thresholds=null. Cả B và CROTTC chưa có pass/fail calibrated trong chat; sound candidates là gợi ý xem lại, không phải nhãn chuyên gia.'),
 tab(['Fluency measure', 'Công thức / quy ước', 'Giới hạn'], [
 ['Recording rate', '60 × số từ / thời lượng bản thu.', 'Tính cả khoảng đầu/cuối.'],
 ['Speech-window rate', '60 × số từ / (cuối từ cuối − đầu từ đầu).', 'Phụ thuộc word timestamps.'],
 ['Articulation duration', 'Speech window trừ internal gaps≥0,25s; rate=60×số từ/duration này.', 'Không phải forced alignment/silence gold.'],
 ['Pause / filler', 'Gap count, tổng, dài nhất; filler/từ lặp trong transcript.', 'ASR có thể bỏ filler; lặp có thể có chủ đích.'],
 ], [1.2, 2.2, 2], 'Nguồn speech_fluency.py [L01]. Không có điểm0–100, IELTS, stress/intonation hoặc human-calibrated fluency.'),
 diagram('reference', 'Hình 3. Sửa text tái dùng bằng chứng âm gốc; chỉ regenerate reply khi nội dung từ thay đổi.', 'Nguồn chat_reference.py, acoustic_evidence.py và ChatService.reference [L01].'),
 p('Private evidence theo job: B là log_probs nén + inventory, CROTTC là phone sequence + model/revision/hash. Public polling bỏ _acoustic_evidence. Sửa reference không chạy lại ASR hoặc acoustic model. Chỉ đổi dấu câu/hoa thường giữ reply/voice; đổi chuỗi từ tăng revision, sinh reply mới kèm lời xin lỗi do app thêm. Fluency giữ measured_transcript ban đầu.'),
)

asr = {(r['group'], r['model']): r for r in rows('data/generated/asr-backends-20261023/metrics.csv')}
groups = [('tlv_previously_observed_heldout', 'TLV:111'), ('saa_external_same_prompt', 'SAA:4'), ('user_development', 'User:3')]
models = ['wav2vec2-base-960h', 'wav2vec2-vi-en-v1', 'whisper-small.en']
chapter('8. So sánh ASR: base, fine-tuned và Whisper',
 p('Hoàn tất118 audio, khoảng617,55 giây:111 TLV,4 SAA và3 người dùng. Cùng audio cho mọi model; không đưa reference/history vào decoder, không BERT/Qwen hậu xử lý để tăng điểm. WER=(substitutions+deletions+insertions)/số từ reference; thấp hơn tốt hơn.'),
 tab(['Nhóm', 'Base WER(%)', 'Fine-tuned WER(%)', 'Whisper WER(%)'], [[label]+[pct(asr[(g,m)]['strict_wer']) for m in models] for g,label in groups], [1.2,1.2,1.5,1.4], 'Nguồn metrics.csv/results.json [L04]. Strict chuẩn hóa case/punctuation đồng đều; giữ cách viết số.'),
 tab(['TLV:1.023 từ', 'Base', 'Fine-tuned', 'Whisper'], [
 ['Word edits']+[asr[('tlv_previously_observed_heldout',m)]['strict_word_errors'] for m in models],
 ['Câu exact /111']+[asr[('tlv_previously_observed_heldout',m)]['exact_matches'] for m in models],
 ['Mean ASR giây']+[f"{float(asr[('tlv_previously_observed_heldout',m)]['mean_inference_seconds']):.3f}" for m in models],
 ], [1.5,1,1,1], 'Latency chỉ inference; không gộp model load, Qwen hay TTS.'),
 p('Fine-tune giảm51 word edits trên TLV (436→385), WER giảm4,99 điểm phần trăm. Whisper còn284 edits, hơn bản fine-tuned101 edits;61 câu tốt hơn,17 xấu hơn,33 bằng nhau. Whisper mean2,363s so0,447s của wav2vec2 fine-tuned. Chọn Whisper ưu tiên transcript, không vì nó nhanh hơn.'),
 p('Provenance ASR lịch sử GPU BF16 ghi43,01%→37,93%. So sánh mới CPU FP32 dùng cùng evaluator ghi42,62%→37,63%. Đây là phép đo khác execution/scoring; không sửa ngược số lịch sử để ép khớp.'),
 h('Số và các ca thất bại'),
 p('Secondary metric chuẩn hóa số tiếng Anh đồng đều cho mọi model: Whisper WER27,65% TLV,7,61% SAA,23,38% user. Strict WER vẫn là chỉ số chính. Không chọn chuẩn hóa có lợi riêng cho một model.'),
 p('Trong benchmark, Whisper nghe “End of the road for some players” thành “And also load for some players.” và “Zero five eight three” thành “I don\'t know what to do.” Câu trôi chảy vẫn có thể sai ý. Live job khác có output road tương tự nhưng không trùng từng ký tự. Người học cần xác nhận câu hoặc nói lại khi chưa hiểu.'),
 note('So sánh triển khai: wav2vec2 CPU FP32 greedy với Whisper CPU INT8 beam5, khác kiến trúc/pretraining/decoder. Cặp base→fine-tuned wav2vec2 đo tác động thích ứng trong cùng họ model. Chưa đủ để kết luận mọi wav2vec2 kém Whisper hoặc mọi giọng Việt đều được nhận đúng.'),
)

cro = {r['model']:r for r in rows('data/generated/crottc-pilot-20261023/research_metrics.csv') if r['inventory']=='raw'}
ui = {r['model']:r for r in rows('data/generated/crottc-pilot-20261023/ui_review_metrics.csv')}
chapter('9. B, CROTTC-IF và external pronunciation test',
 h('B Plus trên đủ900 câu'),
 tab(['Chỉ số', 'B', 'Phạm vi'], [
 ['Phone precision/recall/F1', '48,77/49,84/49,30%', 'Micro test6speaker900utterances.'],
 ['PER', '16,71%', 'So perceived-phone gold.'],
 ['Word F1', '62,78%', 'Research scorer; khác UI variant-review.'],
 ['TLV phone F1', '69,77%', '150 câu một người Việt.'],
 ], [1.6,1.6,2], 'Nguồn Plus audio/B/metrics.json [L05]. A trên cùng test F1 48,38%; TLV64,80%.'),
 h('Pilot ghép cặp CROTTC'),
 p('27 audio×2model=54 predictions hoàn tất. Chỉ số đối chiếu lỗi âm chỉ tính24 audio có nhãn chuyên gia:4 audio/speaker×6test speaker, chia tầng mật độ lỗi trước inference.3 audio user chỉ quan sát output/latency. Pilot đã được quan sát và chọn theo nhãn để có ít/nhiều lỗi; không đại diện900 câu hay test kín.'),
 tab(['24 audio chuyên gia', 'B', 'CROTTC-IF'], [
 ['Phone F1 scorer dự án(%)']+[pct(cro[m]['f1']) for m in ['cascade-B','crottc-if']],
 ['PER inventory gốc(%)']+[pct(cro[m]['per']) for m in ['cascade-B','crottc-if']],
 ['UI hint precision(%)']+[pct(ui[m]['hint_precision']) for m in ['cascade-B','crottc-if']],
 ['UI hint recall(%)']+[pct(ui[m]['hint_recall']) for m in ['cascade-B','crottc-if']],
 ['UI hint FPR(%)']+[pct(ui[m]['hint_false_positive_rate']) for m in ['cascade-B','crottc-if']],
 ['Tổng phoneme inference(s)']+[f"{float(ui[m]['inference_sec']):.3f}" for m in ['cascade-B','crottc-if']],
 ], [2.2,1.2,1.2], 'Nguồn research_metrics.csv/ui_review_metrics.csv [L06].0 calibrated decisions ở cả hai; UI hints khác research word-event F1.'),
 p('CROTTC có370 phone insertions trên819 gold phones. Chuẩn hóa shared inventory bỏ stress/distortion, AX→AH: B F1/PER49,65/18,07%; CRO19,01/53,11%. Khác inventory chưa giải thích hết chênh lệch. Không tự gộp phone lặp từ decoder autoregressive vì sẽ đổi hậu xử lý không được checkpoint quy định.'),
 p('B báo nhầm22/128 từ đúng; CRO55/128. CRO bắt82/100 từ lỗi so63/100 của B nhưng precision thấp hơn. Giữ B mặc định, CRO thử nghiệm. Bản Stella24,16s mất223,75s CROTTC offline và sinh nhiều chunk lặp; web có timeout45s cho mỗi request load/infer.'),
 h('Đối chiếu với công bố'),
 p('Bài báo CROTTC-IF công bố F1 71,77% trên L2-ARCTIC [R-CROTTC]. Đó là evaluator/protocol tác giả, khác24-case local canonical-plus-insertion-slot scorer. Chưa audit trọn split training của checkpoint để loại overlap với pilot. Số local thấp không chứng minh bài báo sai; paper-score không chứng minh hệ thống đạt71,77%.'),
 h('External test và calibration'),
 p('B chạy2.500 câu SpeechOcean762: phone F1 21,04%, word F1 22,89%; gold-majority-correct word FPR57,68%. Trong179 câu5chuyên gia chấm hoàn toàn đúng,173 có flag. Đây là canonical-position baseline cũ, khác UI posterior/variant hiện tại và domain giọng Trung có trẻ em. Kết quả không được gán trực tiếp cho scorer UI mới [L07], [R-SPEECHOCEAN].'),
 p('Calibration mới không đạt policy:1.389 từ holdout có gold nhưng0quyết định, coverage0%. Precision/FPR trên quyết định là undefined, không phải100%đúng hay0%sai [L08].'),
)

chapter('10. Qwen, grammar và chất lượng hội thoại',
 diagram('routing', 'Hình 4. Một backbone Qwen dùng chung; base reply, grammar LoRA cho collapse, coach legacy bị bỏ qua.', 'Nguồn QwenChatEngine, dialogue_request và điều kiện coaching [L01].'),
 tab(['Development probe', 'Trước train', 'Sau train'], [
 ['Grammar JSON /8', '0/8', '8/8'], ['Natural JSON /8', '0/8', '8/8'],
 ['Grammar exact /8', '0/8 trong JSON contract', '4/8'], ['Natural exact /8', '0/8 trong JSON contract', '2/8'],
 ['Giữ grammar controls /4', '0/4 trong JSON contract', '4/4'], ['Giữ natural controls /4', '0/4 trong JSON contract', '1/4'],
 ], [2,1.3,1.3], '16 validation records, CPU FP32 greedy; saved paired comparison [L09].'),
 p('Grammar chrF86,95, natural chrF93,78 sau train ở probe nhỏ. JSON100% chỉ chứng minh contract. Exact match bỏ sót đáp án hợp lệ; chrF cao vẫn có thể đổi nghĩa. Chưa có đầy đủ official ERRANT F0.5, JFLEG GLEU hoặc human faithfulness trên test; không công bố grammar100%chính xác.'),
 p('Config/live jobs xác nhận qwen-grammar được nạp cho rewrite. Worker tắt adapter khi reply, bật grammar cho2generation sau. Adapter feedback đã train nhưng legacy coach chỉ chạy khi không có key assessment; current analyze luôn có key này. Nhãn Fine-tuned/coaching_model_id không chứng minh adapter đó sinh reply.'),
 p('Chưa có benchmark hội thoại do người đánh giá độc lập. Opening soạn sẵn, echo filter, state café, retry, clarification fallback là logic ứng dụng cần tách khỏi raw model output. ASR nghe sai nhưng câu vẫn trôi chảy có thể làm Qwen trả lời sai chủ đề; prompt không bảo đảm phục hồi âm thanh đã nghe nhầm.'),
 p('Ví dụ live road: transcript “And also load for some players.” khiến Qwen nói “If you\'re adding load for some players, it\'s important to consider their needs.” Câu trả lời có cấu trúc nhưng không phản ánh đúng câu mục tiêu “End of the road for some players”. Đây là giới hạn hội thoại thực tế còn tồn tại, không nên mô tả đã giải quyết xong [L02].'),
 p('Development checks ghi adapter sửa “I usually wakes up…”→“I usually wake up…”, giữ “Currently I am a software engineer”, sửa nhiều lỗi câu “Yesterday I wake up…”. Nhưng với “Banana software yesterday purple.”, thêm động từ vẫn chưa tạo ý nghĩa rõ. Case minh họa khác benchmark tổng quát [L09].'),
 p('BERT trước Whisper trên10audio giọng Việt giảm WER109/335(32,54%)→104/335(31,04%):3tốt,1xấu,6bằng,0exact. Stella có30word errors tăng31sau BERT. Vì vậy giữ BERT đối chứng, mặc định dùng Whisper, không dùng cảm nhận vài ca để thay đánh giá ghép cặp [L11].'),
)

chapter('11. Voice, highlight, latency và phần local',
 tab(['Khâu', 'Mặc định', 'Tùy chọn'], [
 ['ASR', 'Whisper local CPU.', 'wav2vec2 khi override.'], ['Dialogue/grammar', 'Qwen local; live CUDA:0 BF16.', 'Không cần LLM API cho speech chat.'],
 ['Phát âm/fluency', 'B, CMUdict, timing local.', 'CROTTC local thêm RAM/CPU.'], ['Voice', 'Kokoro ONNX local CUDA.', 'ai33 Sarah là API ngoài; browser do OS/browser quyết định.'],
 ['Truy cập', 'App localhost/reverse proxy.', 'Ngrok chuyển request/audio qua tunnel, model vẫn local.'],
 ], [1.1,2.1,2.2], 'Nguồn code, setup manifests và live verification [L01], [L02], [L10].'),
 p('Kokoro82M theo model card được export ONNX, runtime riêng [R-KOKORO], [R-KOKORO-ONNX]. Catalog web gọn:Heart/Michael/Emma/George. Kokoro af_sarah và Sarah ElevenLabs qua ai33 là hai voice khác nhau. Trùng tên không đồng nghĩa cùng ID hoặc model.'),
 p('UI prefetch câu mẫu trong lúc feedback xử lý, dedup theo job/revision/target/voice/speed/text; không tự phát. Sound details có nghe từng từ/cả reference sentence. Reply voice chỉ đọc reply đã chốt, không đọc lại user hoặc toàn bộ feedback.'),
 p('Toàn reply được hiện sẵn, highlight tiến trái→phải theo playhead. Provider audio hiện thiếu timestamp chính xác từng từ nên tracker là ước lượng; browser speech có boundary nếu hỗ trợ. Đây chưa phải forced alignment text–voice. Không dùng word times audio người học để đồng bộ voice bot.'),
 tab(['Phép đo', 'Số đã lưu', 'Điều kiện'], [
 ['Whisper111TLV', 'Mean ASR2,363s.', 'CPU INT8 beam5; không Qwen/TTS.'],
 ['B/CRO24pilot', 'Tổng9,408/106,233s.', 'CPU, cùng audio; không ASR/Qwen/TTS.'],
 ['Kokoro GPU opening', 'Warm uncached~0,433s; cold3,168s.', 'Một câu;3warm runs.'],
 ['Kokoro CPU trước', 'Warm~2,532s.', 'Artifact khác thời điểm.'],
 ['ai33 trước', 'Uncached~7,75s,n=1.', 'Khác voice/thời điểm; không benchmark đồng thời.'],
 ], [1.3,1.8,2.2], 'Nguồn [L04], [L06], [L10]. Không cộng mean khác thí nghiệm thành end-to-end latency.'),
 p('Ba live jobs mới có reply_ready khoảng3,33–8,03s, hoàn tất4,77–24,08s; không gọi TTS. Input/cache/load/sampling khác nhau nên chỉ xác nhận integration, không xếp hạng tốc độ B/CRO. Warmup nạp model/từ điển, không train hoặc forward giả [L02].'),
)

chapter('12. API, triển khai và tái tạo tài liệu',
 tab(['API', 'Mục đích', 'Giới hạn'], [
 ['GET /api/chat/config', 'Catalog, policy, readiness.', 'Không chứng minh từng model đã generate.'],
 ['POST /api/chat/topic', 'Opening soạn sẵn.', 'Không Qwen generation mới.'],
 ['POST /api/chat/speech', 'Audio/history/model/pronunciation ID.', 'Server map ID; một generation slot.'],
 ['GET /api/chat/jobs/{id}', 'Polling partial text/snapshot.', 'Không WebSocket/SSE.'],
 ['POST /api/chat/reference', 'Reassess private evidence.', 'Regenerate khi đổi chuỗi từ.'],
 ['POST /api/chat/voice', 'Kokoro/ai33 reply/sentence.', 'TTS cache/slot riêng.'],
 ['POST /api/chat/cancel', 'Cooperative cancel.', 'Không dừng training ngoài app.'],
 ], [1.7,1.9,1.7]),
 p('Practice giữ /api/analyze, /api/analyze-live, /api/reference, /api/feedback riêng. A/B/C practice không đổi backend chat. Analyze-live gửi audio prefix nhiều lần, không phải streaming Whisper. sessionStorage giữ history chữ, không phục hồi acoustic evidence sau server restart.'),
 code('# Người dùng chủ động chạy demo\n$env:MDD_CHAT_ASR = "whisper-small-en"\n$env:MDD_CHAT_PRONUNCIATION = "wav2vec2-b"\n.venv/Scripts/python.exe -X utf8 scripts/serve_demo.py --port 8765\n\n# Build báo cáo từ artifact; KHÔNG chạy model\n.venv/Scripts/python.exe -X utf8 docs/pipeline_report/capture_snapshot.py\n.venv/Scripts/python.exe -X utf8 docs/pipeline_report/author_pipeline.py\n.venv/Scripts/python.exe -X utf8 docs/pipeline_report/build_pipeline_documents.py\n.venv/Scripts/python.exe -X utf8 docs/pipeline_report/validate_documents.py'),
 p('MDD_CHAT_* là process environment; provider .env loader chỉ đọc một số khóa LLM/ai33, không tự áp dụng mọi MDD setting. Whisper strict mặc định: thiếu model báo not ready; lựa chọn auto mới cho phép fallback. Báo cáo/snapshot không chứa API keys.'),
 p('Local app cổng8765. Nginx chuyển về loopback; ngrok HTTPS hỗ trợ mic trên browser. LAN HTTP không bảo đảm cấp mic. Đây là topology được hỗ trợ; địa chỉ tunnel có thể thay đổi. Training scripts chỉ chạy khi người dùng chủ động thao tác.'),
)

chapter('13. Giới hạn và cải thiện ưu tiên',
 tab(['Ưu tiên', 'Cần làm tiếp', 'Bằng chứng để chấp nhận'], [
 ['1.Phát âm', 'Filter hints bằng acoustic evidence; fit ngưỡng trên cả đúng/sai và accent variants.', 'FPR/FNR/coverage đăng ký trước; holdout mới chưa chọn checkpoint.'],
 ['2.Cải thiện B', 'Thử direct MDD head/multi-task CTC+detection, cân bằng đúng/sai, thêm speaker Việt có gold phones.', 'Cùng split/seed/budget; per-speaker precision/recall/FPR/diagnosis.'],
 ['3.ASR Việt', 'Giữ Whisper baseline; muốn fine-tune cần nhiều speaker/thiết bị/nói tự do với text xác nhận.', 'WER/insertions/deletions trên test mới; regression accent khác.'],
 ['4.Grammar', 'Official test W&I/JFLEG và kiểm giữ nghĩa transcript.', 'ERRANT F0.5/GLEU + human review; JSON probe không đủ.'],
 ['5.Hội thoại', 'Multi-turn cases đổi chủ đề, nghe nhầm, clarification; log raw/filter/fallback.', 'Human relevance/echo/naturalness; không tính fallback là Qwen thành công.'],
 ['6.Nghiên cứu', 'Audit evaluator/checkpoint CRO; P1 là lượt train mới do user khởi chạy.', 'Đối chứng rõ và latency/RAM; paper-score không thay local-score.'],
 ], [1.1,2.3,2.3]),
 p('CTC-GOP và GOPT là hướng tham khảo cho bằng chứng âm và đánh giá nhiều mức [R-CTCGOP], [R-GOPT]. Chỉ chuyển thành mặc định sau phép đo cùng dữ liệu/scorer. L2-ARCTIC uy tín giúp kiểm soát nhãn nhưng không tự bảo đảm domain spontaneous Vietnamese-English/microphone cá nhân.'),
 note('Có bằng chứng thích ứng wav2vec2 cải thiện các test đã nêu, Whisper giảm WER và grammar adapter hoạt động. Chưa đủ bằng chứng scorer phát âm calibrated, grammar hoàn hảo hoặc hội thoại tự nhiên mọi tình huống. Đây là phạm vi kết luận hiện tại.'),
)

local = [
 ('L01','Code điều phối','scripts/local_speech_engine.py; scripts/local_chat_worker.py; src/mdd/demo_chat.py; src/mdd/demo_model_registry.py; src/mdd/chat_reference.py; src/mdd/acoustic_evidence.py; src/mdd/speech_fluency.py; demo/chat.js'),
 ('L02','Live integration','data/generated/crottc-live-20261023/verification.json; reference-reassessment.json; các job snapshot cùng thư mục'),
 ('L03','Provenance training','runs/cascade-l2arctic-plus-seed42/{A,B}/completed.json; Q/training_provenance.json; runs/grammar-qwen3-06b-wi-jfleg-v1/training_provenance.json; runs/asr-vietnamese-english-v1/training_provenance.json; runs/projector-l2arctic-plus-seed42/alignment/training_provenance.json'),
 ('L04','ASR118','data/generated/asr-backends-20261023/{manifest,results}.json; metrics.csv; report.html; notebooks/whisper_vs_wav2vec2.ipynb'),
 ('L05','MDD Plus900','data/generated/cascade-l2arctic-plus-seed42/audio/{A,B}/metrics.json'),
 ('L06','CRO pilot27','data/generated/crottc-pilot-20261023/{manifest,results,execution_notes}.json; research_metrics.csv; ui_review_metrics.csv; report.html; notebooks/crottc_pronunciation_pilot.ipynb'),
 ('L07','SpeechOcean762','data/generated/external/speechocean762/20261021T024600Z/B-full/metrics.json; configs/external_speechocean762.json'),
 ('L08','Calibration','data/generated/pronunciation-calibration-v1/full-validation/{calibration,report}.json'),
 ('L09','Grammar probes','data/generated/grammar-adapter-demo-v1/comparison.json; runs/grammar-qwen3-06b-wi-jfleg-v1/demo_validation.json; data/generated/sentence-prefetch-qa/grammar-runtime-check.json'),
 ('L10','TTS/latency','models/pretrained/kokoro-82m/setup_manifest.json; data/generated/kokoro-gpu-validation-v1/summary.json; src/mdd/tts_registry.py; scripts/kokoro_tts.py; scripts/ai33_tts.py'),
 ('L11','BERT probes','data/generated/vi-voice-pipeline-v1/bert-v2/results.json; data/generated/user-stella-audio-20261023/result.json; notebooks/vi_voice_bert_validation.ipynb'),
]
refs = read('docs/references_20261023.json')
entries = refs['references'] if isinstance(refs,dict) else refs
chapter('14. Tài liệu tham khảo và dấu vết tái lập',
 p('Các mã [R-…] dẫn bài báo/model card/repository chính thức đã tra ngày23/10/2026. Tài liệu nghiên cứu giải thích phương pháp; số local lấy từ [L…]. Thông số local ưu tiên config/checkpoint thực có, không suy từ tên model.'),
 h('Bài báo và tài liệu gốc'),
 *[{'type':'small','text':f"**[{r['id']}]** {r.get('authors','')} ({r.get('year') or 'tài liệu cập nhật'}). [{r['title']}]({r['url']})."} for r in entries],
 h('Artifact và mã nguồn'),
 *[{'type':'small','text':f'**[{sid}] {title}.** `{path}`'} for sid,title,path in local],
 h('Phạm vi snapshot'),
 p('source_snapshot.json lưu SHA256/size code và artifact đã đọc cùng config allowlist. Không đọc credentials, gửi audio hoặc chạy inference/training/TTS khi biên soạn. build_manifest.json lưu hash PDF/DOCX; qa/document_validation.json lưu kiểm tra cấu trúc/bố cục/citations. Hai báo cáo dùng cùng bibliography, mỗi số local vẫn gắn protocol riêng.'),
 bullets('Tài liệu này mô tả pipeline, local model, runtime và kết quả so sánh chính.',
 'Báo cáo training giữ optimizer, learning curves, A/B/C v1 và Plus, Qwen/Projector, hình notebook có nguồn cell.',
 'HTML/notebook benchmark giữ từng case, counter, manifest và đồ thị. Build tài liệu không chạy lại model.'),
)
def typography(text):
    """Space numeric prose while preserving IDs, model names, paths and links."""
    saved = []
    def protect(match):
        saved.append(match.group(0))
        return '\ue000' + str(len(saved)-1) + '\ue001'
    pattern = (r'`[^`]*`|\[[^\]]*\](?:\(https?://[^)]+\))?'
               r'|(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.{},/-]+'
               r'|\b(?:Qwen3|wav2vec2|wav2vec|SpeechOcean762|F0\.5|F1|P0|P1|BF16|FP32|INT8|v1|v2|q_proj|v_proj)\b')
    text = re.sub(pattern, protect, str(text))
    text = re.sub(r'(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])', ' ', text)
    text = re.sub(r'(?<=%)(?=[^\W\d_])', ' ', text)
    text = re.sub(r'\ue000(\d+)\ue001', lambda m: saved[int(m.group(1))], text)
    return text

for section in chapters:
    for block in section['blocks']:
        if block['type'] in {'p','note','h2'}: block['text'] = typography(block['text'])
        if block['type'] == 'table':
            block['headers'] = [typography(value) for value in block['headers']]
            block['rows'] = [[typography(value) for value in row] for row in block['rows']]

(HERE/'pipeline_content.json').write_text(json.dumps({'title':'MDD – Pipeline, model local và đánh giá','snapshot_time':datetime.fromisoformat(snapshot['captured_at_bangkok']).strftime('%d/%m/%Y %H:%M:%S (UTC+7)'),'chapters':chapters},ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Authored {len(chapters)} chapters / {sum(len(c["blocks"]) for c in chapters)} blocks')
