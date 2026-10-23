"""Vietnamese current-pipeline prose and tables, backed by read-only evidence."""
from pathlib import Path
from datetime import datetime
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
snapshot=json.loads((HERE/'source_snapshot.json').read_text(encoding='utf-8'))
chapters=[]
def p(t):return {'type':'p','text':t}
def h(t):return {'type':'h2','text':t}
def note(t):return {'type':'note','text':t}
def code(t):return {'type':'code','text':t}
def src(t):return {'type':'source','text':t}
def bullets(*items):return {'type':'bullets','items':list(items)}
def table(headers,rows,widths=None,source=None):
    out={'type':'table','headers':headers,'rows':rows}
    if widths:out['widths']=widths
    if source:out['source']=source
    return out
def diagram(key,caption,source):return {'type':'diagram','id':key,'caption':caption,'source':source}
def chapter(title,*blocks):chapters.append({'title':title,'blocks':list(blocks)})

chapter('1. Pipeline hiện tại trong một trang',
    p('Hệ thống MDD hiện có hai workspace chính: **Luyện phát âm** để đọc câu/từ và xem đối chiếu âm; **Đối thoại AI** để thu audio, nhận câu trả lời bằng tiếng Anh và xem feedback khi cần. Tài liệu tập trung đường đối thoại giọng nói đang có trong code, rồi mô tả riêng đường practice, phần triển khai và các nhánh nghiên cứu.'),
    note('**Luồng chat chính:** audio → wav2vec2 ASR → wav2vec2 phoneme B + assessment → Qwen base chuẩn hóa hình thức → Qwen base trả lời → công bố reply_ready → Qwen grammar viết lại → hoàn tất job. Voice có thể đọc reply đã chốt khi feedback còn chờ.'),
    table(['Thành phần','Hiện làm việc gì','Trạng thái trong chat'],[
        ['wav2vec2 ASR đã thích ứng Việt','Nghe audio thành văn bản tiếng Anh, ước lượng thời gian từng từ','Đang dùng; CPU'],
        ['wav2vec2 phoneme cascade-B','Ước lượng chuỗi phone và posterior để đối chiếu phát âm','Đang dùng; CPU'],
        ['CMUdict + scorer/rules','Tạo âm chuẩn, biến thể phát âm, alignment và gợi ý tạm tính','Đang dùng; không phải LLM'],
        ['Qwen3-0.6B base','Chuẩn hóa hình thức transcript và trả lời hội thoại','Đang dùng'],
        ['Qwen + LoRA grammar','Sinh grammar_corrected và natural bằng hai lượt riêng','Đang dùng khi không có câu đọc cố định'],
        ['Qwen + LoRA feedback','Coach phát âm từ nhãn lỗi dạng text','Có sẵn; nhánh legacy bị bỏ qua khi có assessment mới'],
        ['Browser voice hoặc ai33','Đọc câu trả lời đã chốt; nghe mẫu cần thao tác riêng','Tùy lựa chọn/cấu hình'],
    ],[1.3,2.3,1.6],'Nguồn: [P01–P07], [P12].'),
    h('Ba phân biệt quan trọng'),
    bullets('Hai wav2vec2 là hai checkpoint/head khác nhau. Model ASR xuất chữ; model MDD B xuất phone. Không dùng transcript ASR như kết luận phát âm đúng.',
        'Qwen dùng chung một backbone trong worker và bật/tắt adapter theo tác vụ. Không cần ba bản Qwen backbone độc lập cho base, grammar và feedback.',
        'Đã train và ready không đồng nghĩa đang được gọi. Projector, Qwen2-Audio và thí nghiệm benchmark không nằm trong đường chat audio hiện tại.'),
    p('Trạng thái đã đối chiếu bằng GET /api/chat/config: ASR=wav2vec2-vi-en-v1, grammar=qwen-grammar, reply=qwen-base, acoustic_device=cpu, device policy=auto; speech_ready=true. Base và feedback adapter đều được công bố ready. Đây là snapshot config, không phải một phép chạy inference mới.'),
)

chapter('2. Kiến trúc hệ thống và cấu trúc code',
    diagram('architecture','Hình 1. Kiến trúc đường chat: browser → HTTP/job controller → worker riêng. Các khối bên trong thể hiện trách nhiệm; thứ tự thực thi xem Hình 2.','Nguồn: demo/chat.js; demo.py; demo_chat.py; local_chat_process.py; local_chat_worker.py [P01], [P02], [P10].'),
    p('Browser quản lý mic/upload, lịch sử hiển thị, polling và phát voice. HTTP server phục vụ static UI và kiểm tra request, tạo job, giữ snapshot cùng revision. Tính toán chat được chuyển qua stdin/stdout JSONL đến worker Python tồn tại lâu, nơi cache ASR, phoneme model và Qwen. Worker xử lý lần lượt, không chạy ba tác vụ Qwen song song.'),
    h('Bản đồ thư mục cần đọc'),
    code('MDD/\n  demo/                         Giao diện HTML/CSS/JavaScript\n    chat.js                     Đối thoại, mic, polling, reference, voice\n    app.js                      Workspace luyện phát âm\n  src/mdd/\n    demo.py                     HTTP routes, static files, practice API\n    demo_chat.py                ChatService, jobs, readiness, revisions\n    demo_model_registry.py      Chọn ASR/grammar checkpoint\n    chat_runtime.py             Chính sách CPU/CUDA cho Qwen\n    demo_inference.py           Nạp/chạy wav2vec2 và CTC\n    pronunciation_assessment.py Scorer posterior/variant cho chat\n    chat_assessment.py          Calibration gate và output assessment\n    chat_reference.py           Sửa reference và reassess\n  scripts/\n    serve_demo.py               Entry point server\n    local_chat_process.py       Vòng đời worker và JSONL IPC\n    local_chat_worker.py        Qwen engine, adapter switching, event loop\n    local_speech_engine.py      Điều phối toàn lượt audio chat\n    grammar_inference.py        Hai task của grammar adapter\n    transcript_normalization.py Guard chuẩn hóa hình thức\n  models/pretrained/            Backbone và tokenizer\n  runs/                         Checkpoint/adapter đã train\n  data/generated/               Calibration/report/voice cache/log\n  notebooks/                    Thực nghiệm, kiểm chứng, visualization\n  deploy/nginx/                 Template reverse proxy\n\nPosterior runtime nằm trong ChatJob memory ở server.'),
    h('Ranh giới giữa các lớp'),
    table(['Lớp','Sở hữu / quyết định','Không suy ra từ lớp này'],[
        ['UI','Thu audio, hiển thị, history theo topic/model, yêu cầu sửa reference và voice','Không tự chọn checkpoint path hoặc ngưỡng calibration'],
        ['ChatService','Validate, request idempotency, job state, slot, reference revision, metadata','Không coi model_id được chọn là bằng chứng model đã generate'],
        ['Worker','Inference, chọn adapter/device, tạo text và acoustic outputs','Không tự mở training hoặc thay weights'],
        ['Registry/artifact','Model path hợp lệ, completed/validated/hash','Ready không phải benchmark chất lượng'],
        ['Notebook/training runner','Tạo weights, đánh giá và report nghiên cứu','Không được tự chạy khi một request chat tới'],
    ],[1,2.1,2]),
    p('Sự tách worker áp dụng cho chat. Workspace practice có đường LocalInference riêng dưới DemoApplication; không nên khái quát rằng mọi inference của toàn ứng dụng đều nằm trong cùng worker chat.'),
)

chapter('3. Model nào phụ trách việc gì?',
    table(['Vai trò / model ID','Checkpoint hoặc adapter hiện tại','Đầu vào → đầu ra'],[
        ['ASR\nwav2vec2-vi-en-v1','runs/asr-vietnamese-english-v1/best_checkpoint','Waveform → transcript chữ + word intervals'],
        ['Phoneme MDD\ncascade-B','runs/cascade-l2arctic-plus-seed42/B/best_checkpoint','Audio → CTC phone tokens + log probabilities'],
        ['Hội thoại/format\nqwen-base','models/pretrained/Qwen3-0.6B','Text + history → normalized text hoặc reply'],
        ['Sửa câu\nqwen-grammar','Qwen base + runs/grammar-qwen3-06b-wi-jfleg-v1/adapter','Text → JSON grammar_corrected; text → JSON natural'],
        ['Coach legacy\nqwen-trained','Qwen base + runs/cascade-l2arctic-plus-seed42/Q/adapter','Sentence + annotation_info → feedback JSON theo word key'],
        ['Từ điển/scorer','CMUdict đã cố định hash + code deterministic','Reference text + posterior → assessment và gợi ý'],
    ],[1.2,2.2,1.8],'Nguồn: [P03–P07], [P09].'),
    h('Lựa chọn checkpoint và readiness'),
    bullets('ASR ưu tiên run completed, validation_improved=true, có trained_sha256 và weights. Engine kiểm SHA khi nạp. Nếu chưa đủ điều kiện, registry chọn wav2vec2-base-960h pretrained.',
        'Phoneme model ở chat là cascade-B Plus best checkpoint, step 200. Không phải B cũ của windows-notebook-seed42; code yêu cầu completed.json của run B.',
        'Grammar adapter cần completed, dataset_kind đúng, demo_validation passed, adapter SHA khớp provenance và file tồn tại. Worker kiểm SHA adapter trước khi nạp.',
        'Feedback adapter được công bố sẵn sàng nếu base/adapter files và provenance scientific_training completed có mặt; các đường submit/load còn kiểm checksum. Có sẵn không làm thay đổi điều kiện gọi coach legacy.'),
    h('UI selection khác model thực trả lời'),
    p('Server khai báo default_model_id=qwen-base, nhưng frontend khởi tạo state.model=qwen-trained, sau đó khôi phục lựa chọn phiên nếu có; lựa chọn không ready mới fallback base. Vì vậy nhãn “Fine-tuned” trên UI có thể là coach được chọn, không phải model đang sinh câu đối thoại. Hàm dialogue_request luôn đặt model_id=qwen-base cho speech chat.'),
    note('Nhánh legacy coach chỉ gọi Qwen nếu có khác biệt, có reference và **không có key assessment**. analyze hiện luôn trả assessment_fields với key này. Vì thế luồng speech hiện tại bỏ qua lời gọi coach phát âm Q0/Q1; coaching mặc định unavailable. metadata coaching_model_id vẫn có thể ghi qwen-trained theo lựa chọn. Đây là mô tả code hiện tại, không phải lỗi đã được sửa trong lượt làm tài liệu.'),
)

chapter('4. Workflow một lượt nói',
    diagram('workflow','Hình 2. Thứ tự thực thi lượt nói trong current speech path. Các bước ASR và phoneme chạy tuần tự; Qwen reply được công bố trước feedback viết lại.','Nguồn: LocalSpeechEngine.run/analyze/_respond; ChatService.submit_speech [P02], [P04], [P05].'),
    h('Từ audio đến job'),
    p('Browser thu mic hoặc giải mã file upload, trộn mono và resample về 16 kHz, đóng gói PCM16 WAV rồi gửi base64. API chat chấp nhận WAV không nén dài 0,25–30 giây, một kênh, sample width 2 bytes; decoded audio tối đa 1 MiB. HTTP body chat speech tối đa 2 MiB. Các audio chunks khi thu chỉ lưu ở browser; không truyền mic liên tục qua mạng.'),
    p('POST /api/chat/speech gửi model_id, request_id, audio_base64, history cùng topic_id hoặc practice_id. Backend kiểm role/history, chọn reference, kiểm model ready/hash và giữ một generation slot. Nếu đang có lượt khác sẽ trả 409 chat_busy. Request_id cùng nội dung trả lại job cũ; cùng ID nhưng đổi nội dung bị từ chối.'),
    h('Hội thoại và feedback hoàn tất ở hai mốc'),
    p('Sau acoustic analysis, normalization của Qwen base chạy trước dialogue khi không có reference đọc cố định. Dialogue nhận user-confirmed text hoặc bản normalization qua lexical guard; grammar rewrite chưa chạy ở bước này. Reply được lọc lặp và policy rồi mới phát hành; khi reply_ready=true, text/spoken_response được chốt. Tiếp theo worker mới sinh grammar/natural feedback trên cùng lượt.'),
    table(['Mốc','Trạng thái thường thấy','UI có thể làm gì'],[
        ['Nhận/đang phân tích audio','transcribing','Giữ loading; chưa coi đã có câu trả lời'],
        ['Chuẩn hóa hình thức','correcting','Có acoustic snapshot nhưng reply chưa ready'],
        ['Sinh đối thoại','generating','Polling nhận text từng phần; chưa tự đọc phần chưa chốt'],
        ['Reply đã chốt','speech.reply_ready=true, job có thể chưa ready','Hiện đầy đủ reply; có thể bắt đầu voice'],
        ['Sinh grammar feedback','correcting; rewrite pending','Reply vẫn giữ nguyên; feedback trong collapse cập nhật sau'],
        ['Xong/hủy/lỗi','ready / cancelled / error','Chốt trạng thái; không tạo thêm lượt để bổ sung feedback'],
    ],[1.3,1.7,2.1]),
    p('Tên stage correcting có thể xuất hiện cả trước lẫn sau generating vì dùng cho normalization và rewrite. Không thể chỉ nhìn tên stage để kết luận hệ thống đang sửa grammar trước reply. Metadata cuối có feedback_order=reply_first; các timing riêng ghi audio_decode, ASR, phoneme, assessment, rewrite và tổng lượt.'),
)

chapter('5. Nhánh âm thanh và chấm phát âm',
    h('Hai model âm thanh độc lập'),
    p('ASR dùng Wav2Vec2ForCTC với head ký tự từ wav2vec2-base-960h đã thích ứng Việt. Argmax CTC và processor decode tạo raw transcript; frame IDs còn dùng ước lượng word intervals. Phoneme B dùng head 83 tokens trên backbone wav2vec2-base, trả phone sequence và toàn ma trận log_probs. Cả hai chạy CPU, eval và inference_mode; không có Qwen hay Projector trực tiếp nhận audio trong luồng này.'),
    p('Ở implementation hiện tại, analyze decode audio để chạy ASR; posterior_evidence đi qua LocalInference và đọc lại cùng audio bytes để chạy B. Đây là hai lần chuẩn bị tín hiệu cho hai model, không phải hai bản thu khác nhau. Khi chỉ sửa reference, cả hai bước này được bỏ qua nhờ posterior cache.'),
    h('Reference và CMUdict'),
    table(['Chế độ','Nguồn câu đối chiếu','Cách dùng'],[
        ['Hội thoại tự do','Raw ASR transcript ban đầu','Tạo canonical phones và gợi ý tạm tính; có thể bị ảnh hưởng ASR nghe sai'],
        ['Đọc câu mẫu','Câu đã chọn qua practice_id','Reference được cung cấp từ server; không lấy grammar rewrite thay thế'],
        ['Người dùng sửa câu','Text đã xác nhận cho đúng recording/job','Reassess cùng acoustic evidence và cập nhật provenance'],
        ['Grammar/natural suggestion','Một đề xuất viết lại riêng','Không tự trở thành reference phát âm của audio cũ'],
    ],[1.2,1.6,2.4]),
    p('dictionary_reference và target_words tra âm chuẩn trong CMUdict; scorer mới xét các biến thể phát âm, giữ metadata stress cho IPA/nghe mẫu nhưng bỏ stress trong chuỗi phone so sánh. Từ ngoài từ điển hoặc alignment không đủ sẽ unavailable/uncertain; không tự gán âm của từ bên cạnh. Draft spelling/candidate chỉ là gợi ý, không phải gold reference.'),
    h('Assessment mới khác decoder edit-distance baseline'),
    p('Current chat scorer là ctc_variant_posterior_margin_v1: alignment CTC với các biến thể CMU và margin giữa posterior phone tham chiếu với phone cạnh tranh. raw_comparison từ edit distance vẫn tồn tại để diagnostic. Scorer này tách với scorer nghiên cứu frozen trong báo cáo train; không phải triển khai đầy đủ công thức CTC-GOP công bố.'),
    note('Calibration full-validation hiện **ineligible**, lý do không có cặp ngưỡng đạt policy fit; accept/error thresholds=null. Runtime không bật quyết định calibrated đúng/sai. UI vẫn có thể hiện **sound candidates tạm tính** từ khác biệt greedy/biến thể CMU, nhưng không biến chúng thành lỗi đã được xác nhận. Không có gợi ý không có nghĩa phát âm đúng.'),
    p('Các lớp kiểm tra calibration đối chiếu method, checkpoint/vocab/dictionary hash, split, selection và tiêu chí validation. Flag verified do browser gửi không đủ bật chấm. FPR/FNR/coverage và abstention là những khái niệm của phép đánh giá scorer, không được tự đổi thành confidence xác suất cho một từ.'),
    src('Nguồn: [P04], [P07–P09].'),
)

chapter('6. Qwen: routing, grammar và thiết bị',
    diagram('routing','Hình 3. Một backbone Qwen dùng chung, tắt/bật LoRA theo tác vụ. Nhánh feedback nét đứt là code legacy, không được gọi trong current speech path có assessment.','Nguồn: QwenChatEngine.load/generate; dialogue_request; grammar_inference.rewrite [P04–P06], [P10].'),
    h('Normalization không phải sửa grammar'),
    p('Qwen base đề xuất normalized_text để thêm dấu câu/viết hoa và một số apostrophe/compound đã định nghĩa. Lexical guard kiểm chuỗi từ, số, phủ định và dấu hiệu câu hỏi. Đề xuất sửa tense, đoán từ OOV, thêm chủ ngữ hoặc paraphrase bị từ chối. Invalid JSON, output dở hoặc model failure quay về format_rules. Raw transcript và pronunciation reference vẫn giữ riêng.'),
    h('Dialogue và hai task grammar'),
    p('Speech dialogue luôn dùng base Qwen, enable_thinking=False; profile conversation bật sampling temperature=0,7, top_p=0,8, top_k=20, min_p=0. ReplyWithoutEcho lọc nhắc lại lời người học/câu trước; GroundedReply xử lý tính phù hợp của phản hồi và state café có giới hạn. Nếu không có đóng góp dùng được hoặc policy cần retry, model thử thêm tối đa một lần. Fallback được ghi nguồn riêng.'),
    p('Sau reply, grammar_inference gọi qwen-grammar lần lượt cho task grammar_corrected và natural, mỗi task JSON một trường, greedy. Parser task và lexical guard kiểm kết quả độc lập. Lỗi một task ghi unavailable cho task đó, không coi text giữ nguyên do lỗi là một lần sửa thành công. Nếu grammar adapter không ready, workflow dùng base rewrite contract qua sentence_rewrite.'),
    p('Grammar aggregate được tính lại sau khi loại trường không dùng được, kèm field_reasons, partial và feedback_state: unavailable, partial, suggestions_available hoặc no_change_suggested. Một câu bị model chép lại chỉ có nghĩa không đề xuất thay đổi; không phải kết luận câu đúng grammar, đúng nghĩa hoặc tự nhiên.'),
    table(['Tác vụ','Model / adapter','Generation budget trong code'],[
        ['Chuẩn hóa transcript','qwen-base; greedy','192 tokens; timeout 35 giây'],
        ['Dialogue thường','qwen-base; sampling','160 tokens; timeout 75 giây'],
        ['Dialogue state café','qwen-base; sampling','96 tokens; timeout 45 giây'],
        ['Retry dialogue','qwen-base; cùng profile','96 tokens; timeout 35 giây'],
        ['Grammar / natural','qwen-grammar; greedy','320 tokens; 512 nếu transcript >40 từ; mỗi task timeout 45 giây'],
        ['Coach legacy nếu đủ điều kiện','Model_id đã chọn; structured greedy','512 tokens; timeout 110 giây'],
    ],[1.3,1.6,2.1],'Đây là inference budgets; khác max_length/epochs/QLoRA trong tài liệu training [P04–P06].'),
    h('Một worker, cache model, chuyển adapter'),
    p('QwenChatEngine giữ model và tokenizer đã nạp, cache adapter theo identity/mtime/size/SHA. Với request qwen-base, disable_adapter() khi base đã bọc PEFT. Grammar bật adapter grammar; feedback legacy bật adapter chat. Model inference dùng eval, requires_grad=False. Context giới hạn min(4.096, model max_position_embeddings), bao gồm budget output; khi cần bỏ opener assistant hoặc các cặp history cũ, giữ system và lượt hiện tại.'),
    table(['Policy/runtime','Cách thực thi','Phạm vi fallback'],[
        ['cpu','FP32, eager attention','Dùng khi chỉ định hoặc auto không đủ điều kiện CUDA'],
        ['cuda','cuda:0; BF16 nếu hỗ trợ, nếu không FP16; SDPA','Chỉ định cuda mà không dùng được sẽ báo lỗi'],
        ['auto (mặc định)','Chọn CUDA nếu available và free memory ≥2 GiB; nếu không CPU','Base load bị CUDA OOM có nhánh fallback CPU'],
        ['Acoustic','ASR và phoneme model luôn CPU','Policy Qwen không tự chuyển chúng sang GPU'],
    ],[1.1,2.2,2]),
    note('Inference Qwen hiện **không dùng NF4**: base được nạp regular BF16/FP16 trên CUDA hoặc FP32 trên CPU rồi gắn LoRA. NF4/QLoRA thuộc recipe train. GET config trả device=auto chỉ là policy; qwen_runtime.device/dtype/attention trong metadata lượt sinh mới cho biết placement thực. Không khẳng định generation OOM hoặc adapter-load OOM cũng có cùng fallback.'),
)

chapter('7. Khi sửa câu ASR hoặc luyện câu mẫu',
    diagram('reference','Hình 4. Sửa reference: tái dùng acoustic evidence và chỉ regenerate reply nếu nội dung từ thay đổi.','Nguồn: ChatService.reference; reassess_reference; LocalSpeechEngine.regenerate [P02], [P08].'),
    h('Hai trường hợp sửa reference'),
    table(['Thay đổi người dùng','Xử lý acoustic','Xử lý reply / voice'],[
        ['Chỉ case, dấu câu hoặc khoảng trắng','Reassess và ghi reference_source=user_confirmed','Giữ reply và voice trả lời; sentence voice có thể reset nếu exact text đổi; không gọi Qwen thêm'],
        ['Thay đổi nội dung từ','Reassess bằng posterior cache, không ASR/B lại','generation_revision++; xóa reply/voice cũ, sinh lại bằng text đã xác nhận'],
        ['Từ mới ngoài CMUdict','Giữ nội dung cho hội thoại; pronunciation unavailable khi thiếu reference','Không tự invent phones; feedback cũ không được giữ như còn hợp lệ'],
    ],[1.3,1.9,2]),
    p('POST /api/chat/reference cần job ready còn trong phiên. Server so spoken_text_key để quyết định có đổi lời đã nói hay không. Khi đổi từ, reply revision cũ được lưu giới hạn, voice IDs cũ bị vô hiệu, tạo cancel event/revision mới và worker regenerate trên cùng job. Các callback kiểm revision để output cũ không ghi đè mới.'),
    p('Regenerate chỉ nhận text đã được server xác nhận và context trước lượt sửa. Worker không decode audio, không chạy ASR hoặc phoneme B. Base Qwen trả lời theo nghĩa mới rồi grammar rewrite; ứng dụng chèn “Sorry for the misunderstanding.” một lần, được ghi là nội dung ứng dụng, không gán thành câu model tự sinh.'),
    h('Cache để đối chiếu lại'),
    p('pack_evidence lưu log_probs float32 nén zlib/base64, id_to_phone, blank_id, duration, checkpoint/vocabulary SHA trong _acoustic_evidence của ChatJob memory phía server. Giới hạn frame/vocabulary và kích thước được kiểm tra khi unpack. poll bỏ các key bắt đầu bằng “_”, nên ma trận posterior riêng không nằm trong public job snapshot. Đây không phải database lưu posterior lâu dài qua restart.'),
    h('Các trường text phải giữ tách biệt'),
    table(['Trường','Ý nghĩa','Được dùng ở đâu'],[
        ['transcript','Kết quả ASR raw ban đầu','Lưu bằng chứng; so raw/normalized; không ghi đè khi sửa grammar'],
        ['reference_text / reference_source','Câu dùng để đối chiếu phát âm và nguồn','CMU + acoustic assessment; reading_prompt/asr/user_confirmed'],
        ['conversation_text / conversation_source','Nội dung cho đối thoại','Ưu tiên user-confirmed, rồi normalized có guard, rồi raw ASR'],
        ['rewrite.grammar_corrected / natural','Đề xuất học ngôn ngữ','Hiển thị/nghe theo thao tác; không tự đổi reference hoặc history'],
        ['job.text / spoken_response','Câu trả lời của bot đã chốt','UI và voice reply_only'],
    ],[1.7,1.7,2]),
    h('Chế độ đọc câu'),
    p('Với practice_id, server lấy câu từ PRACTICE_PROMPTS làm reference. Acoustic engine vẫn nhận dạng ASR nhưng đối chiếu dùng câu được chọn; bỏ normalization và grammar rewrite của nhánh hội thoại không có reference. dialogue_request nhận ngữ cảnh “I just practised reading…”. Điều này khác lấy câu Qwen đã sửa để chấm lại chính audio cũ.'),
    p('Code còn nhận correction_job_id cho luyện một completion-only sentence đã sẵn sàng, cùng model/topic và có server-owned source job. Luồng grammar_and_natural hiện không tự tạo completion-only output. Vì vậy không nên mô tả mọi grammar suggestion hiện đều là target luyện lại hợp lệ qua cơ chế legacy này.'),
)

chapter('8. API, trạng thái UI, history và voice',
    h('Các API của đường đối thoại'),
    table(['Method / route','Chức năng','Điểm cần phân biệt'],[
        ['GET /api/chat/config','Model/voice readiness, topic, prompt và policy','Không load/infer model chỉ để xem cấu hình'],
        ['POST /api/chat/topic','Tạo lời mở đầu chủ đề','Nội dung biên soạn sẵn, không phải Qwen generate mới'],
        ['POST /api/chat/speech','Gửi một audio, history và mode','HTTP 200 job snapshot; không đợi toàn lượt xong'],
        ['GET /api/chat/jobs/{id}','Polling text/speech/status/metadata/voice','Không SSE hoặc WebSocket'],
        ['POST /api/chat/reference','Xác nhận/sửa câu đối chiếu','Reassess; chỉ regenerate khi từ thay đổi'],
        ['POST /api/chat/cancel','Hủy lượt đang chạy','Cooperative cancel, có thể qua cancelling'],
        ['POST /api/chat/voice','Yêu cầu ai33 cho reply/sentence target hợp lệ','Tách khỏi generation reply'],
        ['GET /api/chat/audio/{id}','Lấy voice audio qua opaque ID','Không trả provider/CDN URL trực tiếp'],
        ['POST /api/chat/send','Endpoint chat text còn trong backend','Dùng model được chọn; khác speech luôn base reply'],
    ],[1.8,1.8,2],'Nguồn: [P01], [P02], [P11], [P12].'),
    p('Cả speech và reference polling dùng 300 ms khi stage=generating, 700 ms ở stage khác; voice polling hiện 700 ms. Worker có chunk events qua JSONL; browser nhận các snapshot HTTP chứa partial text chứ không nghe trực tiếp stdout hay token stream server. Mic chunks chỉ local; đây không phải realtime network audio stream.'),
    h('Hiển thị và lịch sử'),
    p('UI giữ nguyên reply sau reply_ready; grammar/coaching đến sau cập nhật collapse của cùng lượt. Nội dung phản hồi và voice không bị thay bằng câu grammar. Lịch sử chia theo model/topic; sessionStorage lưu text/metadata và lựa chọn, không lưu audioUrl/voice/job quyền điều khiển. Reload có thể khôi phục hội thoại để đọc nhưng không phục hồi job cũ cho reassess/ai33.'),
    p('history() chỉ gửi các cặp user/assistant đã có assistant ready; user text ưu tiên confirmed, rồi normalized bảo toàn từ, rồi raw ASR. Browser giới hạn tối đa 20 messages/12.000 ký tự bằng cách bỏ cả cặp đầu. Backend kiểm role xen kẽ và giới hạn riêng; model worker tiếp tục áp dụng token context budget. Rewrite không tự ghi đè lời người học trong history.'),
    h('Voice và dừng tác vụ'),
    table(['Đường voice','Nguồn nội dung','Thời điểm / giới hạn'],[
        ['Browser speechSynthesis','Câu trả lời bot đã chốt','Bắt đầu sau reply_ready; engine giọng do browser/OS quyết định'],
        ['ai33','Reply đã chốt, locale/voice/speed hợp lệ','Server gọi TTS/cache riêng; có thể dùng API credits'],
        ['Nghe từ / cả reference sentence','Từ hoặc câu trong panel','Playback chỉ khi bấm; ai33 sentence có thể được prefetch trước'],
        ['Stop voice','Chỉ dừng phát tiếng','Không tự hủy job AI'],
        ['Stop AI / End room','Cancel event và dừng phần UI/mic/voice phù hợp','Không khởi động/dừng training hoặc kill GPU process ngoài ứng dụng'],
    ],[1.4,1.8,2]),
    p('Word highlight dùng word boundary của browser khi có; ai33 chưa có timestamp từng từ nên highlight theo playhead là ước lượng. Config snapshot ai33_available=true và default_voice_provider=ai33 phản ánh khả năng hiện tại; lựa chọn lưu trong phiên có thể khác. Tài liệu không tạo TTS hoặc dùng credits để kiểm tra.'),
    p('Code hiện có prefetchSentenceVoice: nếu chọn ai33, tab đang hiển thị, đang ở trong room và đã có transcript, browser có thể yêu cầu tạo/buffer audio câu mẫu trước khi bấm nghe, kể cả auto-read đang tắt. Việc tạo audio có thể dùng credits; không tự phát câu mẫu. Request được dedup theo job/revision/target/voice/speed/text. Server có slot TTS reply và sentence riêng, cho phép một reply và một sentence được tổng hợp đồng thời; sentence có thể yêu cầu khi chưa job ready nhưng đã có text hợp lệ.'),
)

chapter('9. Workspace practice và cấu trúc triển khai',
    h('Practice là đường gọi riêng với chat'),
    table(['Đường','Input / model','Workflow'],[
        ['Luyện phát âm /api/analyze','Audio + model trong catalog + reference','LocalInference → phone comparison/feature → analysis snapshot'],
        ['Luyện theo nhịp /api/analyze-live','Audio prefix gửi lặp trong lúc thu, final flag','So sánh prefix/câu hoặc từ; chưa phải trạng thái cuối khi is_final=false'],
        ['/api/reference','Text và optional sample_id trong practice','Ưu tiên canonical dataset khi sample/prompt khớp, rồi fallback CMUdict'],
        ['/api/feedback','analysis_id do server cấp','FeedbackService dùng snapshot; optional configured LLM/template và TTS'],
        ['Đối thoại AI /api/chat/*','Audio → speech job → worker','Feedback nằm trong job snapshot; không gọi /api/feedback của practice'],
    ],[1.6,1.8,2],'Nguồn: DemoApplication, app.js, FeedbackService; khác ChatService [P01], [P13].'),
    p('Practice model catalog có BASE chỉ trích đặc trưng và các checkpoint A/B/C đã có. BASE chưa có phone head MDD nên can_recognize_phonemes=false. Model được chọn trong practice không thay đường phoneme cascade-B cố định của chat. Các sample practice được chọn từ source train hợp lệ, không tự coi chúng là test benchmark.'),
    h('Topology triển khai có trong project'),
    code('Localhost: browser → http://127.0.0.1:8765 → Python app\nLAN qua Nginx: browser → LAN host:80 → Python127.0.0.1:8765\nNgrok HTTPS: browser → ngrok public HTTPS → Nginx127.0.0.1:8081\n             → Python127.0.0.1:8765\nChat nội bộ: Python app ↔ JSONL worker ↔ model weights/cache\nVoice tùy chọn: Python app → ai33 TTS → cache → browser'),
    p('serve_demo.py mặc định loopback port 8765, HF/Transformers offline cho model weights local. Có --lan với danh sách allow-host rõ ràng; topology reverse proxy dùng backend loopback. Nginx proxy các route được app cho phép, không dùng repo làm static root. Ngrok control port riêng không thuộc app pipeline. Đây là topology/config được hỗ trợ, không khẳng định tất cả đang được mở cùng lúc.'),
    p('Mic cần secure context: localhost hoặc HTTPS; LAN HTTP thường vẫn dùng upload/sample nhưng browser không cấp mic. Lớp Nginx/ngrok thay cách truy cập, không đổi model, task, weights hay tự thêm realtime streaming.'),
    h('Entry point và trạng thái đọc được'),
    code('# Ví dụ entry point, không thực thi trong quá trình làm tài liệu\n.venv/Scripts/python.exe -X utf8 scripts/serve_demo.py --port 8765\n\n# MDD_CHAT_DEVICE chỉ chọn placement cho Qwen worker\n# auto (mặc định), cpu hoặc cuda\n\n# GET /api/status          trạng thái app/model catalog/training artifacts\n# GET /api/chat/config     lựa chọn ASR/grammar và policy chat\n# GET /api/chat/jobs/{id}  metadata thực của lượt còn tồn tại'),
    p('Các tab/status đọc file log/provenance để hiển thị training; inference server không tự chạy train. Cache voice, chat-worker.log và report dưới data/generated là artifact vận hành. Job snapshots và quyền thao tác job thuộc phiên server; text history ở browser không phải database phục hồi mọi trạng thái.'),
    h('Những model/nhánh không nằm trong runtime chat chính'),
    table(['Nhánh','Vai trò hiện có','Trạng thái đối với chat'],[
        ['Projector → Qwen','Nghiên cứu ghép feature audio vào embedding Qwen','Không được LocalSpeechEngine gọi'],
        ['Qwen2-Audio-7B-Instruct','Metadata nguồn; chưa có weights train đã xác nhận','Không dùng cho ASR, reply hoặc grammar ở chat'],
        ['wav2vec2 A/C và B v1 cũ','Đối chứng/benchmark; có thể hiện trong workspace model catalog','Không thay cascade-B cố định của speech chat'],
        ['Training scripts/notebooks','Chuẩn bị dữ liệu, train, đánh giá khoa học','Ngoài đường inference request'],
    ],[1.5,2.1,1.8]),
    src('Nguồn: [P01], [P10], [P13], [P14].'),
)

chapter('10. Ví dụ sử dụng và các trường hợp ngoại lệ',
    h('Ví dụ 1: một câu hội thoại có lỗi grammar'),
    p('Người học nói “Yesterday I go to the park with my friend”. ASR ghi raw text, phoneme B tạo acoustic evidence trên bản thu đó. Normalization chỉ có thể đổi hình thức, không được tự đổi go→went. Base Qwen trả lời theo nội dung lời người học trước. Sau reply, grammar adapter có thể đề xuất “Yesterday I went…”, nhưng đề xuất không tự trở thành câu đối chiếu của audio cũ. Đây là ví dụ minh họa luồng, không phải output mới đã chạy.'),
    h('Ví dụ 2: ASR nghe nhầm một từ'),
    p('ASR trả một từ không đúng ý người nói. Người dùng sửa reference trên cùng recording. Nếu spoken_text_key thay đổi, server reassess từ posterior ban đầu, tăng revision và yêu cầu base Qwen trả lời theo text đã xác nhận. Không chạy lại ASR/B, không cần ghi âm lại. Nếu chỉ đổi viết hoa hoặc dấu câu thì giữ reply/voice và không generate thêm.'),
    h('Ví dụ 3: câu đọc có từ ngoài CMUdict'),
    p('Reference có một tên riêng thiếu từ điển. Hệ thống giữ từ đó cho hội thoại, nhưng pronunciation đánh dấu thiếu reference/uncertain thay vì gán phone của một từ quen thuộc. Spelling draft hoặc sound-neighbour hint chưa được dùng làm ground truth tự động.'),
    table(['Tình huống','Hành vi hiện có','Cách diễn giải'],[
        ['Một lượt khác đang chạy','409 chat_busy, không mở parallel generation','Ứng dụng tối ưu cho inference local tuần tự'],
        ['Normalization sai JSON/đổi từ','Reject/fallback format_rules','Không gọi thay từ là sửa hình thức thành công'],
        ['Grammar task thất bại','Trường task unavailable; task khác vẫn có thể dùng','Text giữ nguyên do lỗi khác với “không cần sửa”'],
        ['Reply chỉ lặp hoặc không hợp lệ','Retry tối đa một lần; fallback có nguồn riêng','Không đếm fixed fallback là output model thành công'],
        ['Calibration chưa đạt','Uncertain và provisional suggestions','Không phát sinh pass/fail calibrated'],
        ['Worker chết hoặc timeout','Lượt báo lỗi; HTTP app vẫn có thể phục vụ','Không tự khẳng định mọi failure đều fallback CPU'],
        ['Sửa reference khi job không còn','Không còn quyền reassess job cũ','Lịch sử khôi phục không thay thế acoustic snapshot server'],
    ],[1.5,1.9,1.8]),
    note('Tài liệu này chụp trạng thái code/config hiện tại. Các ranh giới còn hạn chế gồm calibration ineligible, coach adapter không được gọi trong assessment path, defaults UI/server khác nhau và nhánh completion-only legacy. Không có thay đổi code hoặc runtime nào được thực hiện để viết tài liệu.'),
)

SOURCES=[
    ('P01','HTTP server / entry point','src/mdd/demo.py:159 DemoApplication; :525 _dispatch; scripts/serve_demo.py:16 main'),
    ('P02','Chat jobs, voice và revision','src/mdd/demo_chat.py:93 _models; :110 config; :249 submit_speech; :323 reference; :414 _generate_speech; :501 poll; :512 cancel; :522 voice'),
    ('P03','Model selection và device policy','src/mdd/demo_model_registry.py:10 asr_selection; :25 grammar_selection; src/mdd/chat_runtime.py:7 requested_device; :15 select_runtime'),
    ('P04','Điều phối acoustic và lượt nói','scripts/local_speech_engine.py:125 LocalSpeechEngine; :129 transcribe; :161 analyze; :219 run; :235 regenerate; :252 _respond'),
    ('P05','Normalization và context','scripts/transcript_normalization.py:86 normalization_request; :106 parse_normalization; :128 normalize; scripts/conversation_context.py:22 prepare_dialogue_context'),
    ('P06','Grammar adapter và lexical guard','scripts/grammar_inference.py:7 rewrite; scripts/sentence_rewrite.py:rewrite_request / parse_rewrite; scripts/grammar_data.py:messages'),
    ('P07','Phoneme inference và scorer','src/mdd/demo_inference.py:115 _audio; :152 _load_bundle; :206 analyze; :266 posterior_evidence; src/mdd/chat_assessment.py:118 load_calibration; :154 assessment_fields; src/mdd/pronunciation_assessment.py:assess'),
    ('P08','Reference và private evidence','src/mdd/chat_reference.py:112 reference_resolution; :152 reassess_reference; src/mdd/acoustic_evidence.py:19 pack_evidence; :37 unpack_evidence'),
    ('P09','Từ điển và candidate hints','src/mdd/demo_reference.py:dictionary_reference; src/mdd/chat_pronunciation.py:56 target_words; src/mdd/chat_diagnostics.py:sound_candidates; src/mdd/phonetic_candidates.py:word_intervals / sound_candidates'),
    ('P10','Worker và Qwen inference','scripts/local_chat_process.py:22 _start; :111 _request; scripts/local_chat_worker.py:19 generation_options; :34 bounded_inputs; :61 QwenChatEngine; :71 load; :135 generate; :178 main'),
    ('P11','Frontend chat/audio/history','demo/chat.js:state / sessionKey / save / restore / regenerateReference / history / sendAudio / initialize / prefetchSentenceVoice / requestVoiceResource; demo/recorder-worklet.js'),
    ('P12','Voice và reply policy','src/mdd/chat_voices.py; scripts/ai33_tts.py; scripts/conversation_reply.py:ReplyWithoutEcho; scripts/conversation_policy.py:GroundedReply; scripts/conversation_state.py:cafe_state; src/mdd/speech_topics.py'),
    ('P13','Practice và feedback riêng','demo/app.js; src/mdd/demo_realtime.py; src/mdd/demo_feedback.py:FeedbackService; scripts/llm_client.py'),
    ('P14','Triển khai','deploy/nginx/README.md; deploy/nginx/nginx.conf.template; deploy/nginx/ngrok.conf.template; scripts/deploy_demo_nginx.py; scripts/deploy_demo_ngrok.py'),
    ('P15','Trạng thái artifact hiện tại','runs/asr-vietnamese-english-v1/training_provenance.json; runs/cascade-l2arctic-plus-seed42/B/completed.json; runs/grammar-qwen3-06b-wi-jfleg-v1/{training_provenance,demo_validation}.json; data/generated/pronunciation-calibration-v1/full-validation/calibration.json'),
]
chapter('11. Nguồn đối chiếu và phạm vi snapshot',
    p('Tài liệu dựa code đang có trong workspace và model/artifact metadata, thay vì sao chép mô tả README. source_snapshot.json ghi thời điểm, SHA256 của 43 source/artifact files và một tập field config được chọn từ GET /api/chat/config. Không đọc credentials vào báo cáo, không gửi audio, không chạy model, train hoặc TTS.'),
    *[{'type':'small','text':f'**[{sid}] {title}.** `{path}`'} for sid,title,path in SOURCES],
    h('Các mô tả cũ cần đọc đúng phạm vi'),
    table(['Tài liệu / nhánh','Điểm cần chú ý hiện tại'],[
        ['README phần reference cũ','Câu mô tả không chạy lại Qwen chỉ đúng với no-op lexical; thay đổi từ hiện regenerate reply/rewrites'],
        ['README/workspace feedback','LLMClient/gateway trong practice không phải đường Qwen local speech chat'],
        ['docs.md phần chưa cài Torch/inference','Là trạng thái lịch sử; không đại diện model/readiness hiện có'],
        ['Chỉ số/đồ thị trong training_report','Dùng để biết chất lượng thực nghiệm; không xác định model hiện được gọi ở runtime'],
        ['Config default_model_id / notice text','Là config/nhãn; cần kết hợp frontend và call graph, đặc biệt Qwen auto device và UI trained selection'],
    ],[1.8,3.2]),
    h('Các tệp đồng hành'),
    bullets('evidence_acoustic.json: acoustic flow, references, calibration và cache.',
        'evidence_models.json: model routing, readiness, device, adapter switching và exception paths.',
        'evidence_app.json: UI, API, polling, history/voice và deployment.',
        'pipeline_content.json: nguồn nội dung dùng chung cho PDF/DOCX; figures/: bốn sơ đồ PDF vector và PNG cho Word.',
        'build_manifest.json và qa/: hash đầu ra, kiểm tra tài liệu và preview.'),
    p('Các mã nguồn có thể tiếp tục thay đổi sau snapshot. “Đang dùng” trong tài liệu là call path đã đối chiếu tại thời điểm chốt, không phải bảo đảm mọi model sẽ giữ nguyên khi registry, checkpoint hoặc app được cập nhật. Báo cáo training riêng vẫn là nơi xem dữ liệu train, hyperparameters và test/pilot metrics.'),
)

(HERE/'pipeline_content.json').write_text(json.dumps({'title':'MDD – Pipeline hiện tại','snapshot_time':datetime.fromisoformat(snapshot['captured_at_bangkok']).strftime('%d/%m/%Y %H:%M:%S (UTC+7)'),'chapters':chapters},ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Authored {len(chapters)} chapters / {sum(len(c["blocks"]) for c in chapters)} blocks')
