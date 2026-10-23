"""Reviewable report text, with numeric tables read from saved source artifacts."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def read(p):return json.loads((ROOT/p).read_text(encoding="utf-8"))
w=read("docs/training_report/evidence_wav2vec.json")
q=read("docs/training_report/evidence_qwen.json")
validation=read("data/generated/projector-validation-v1/report/summary.json")
chapters=[]
def p(t):return {"type":"p","text":t}
def h(t):return {"type":"h2","text":t}
def note(t):return {"type":"note","text":t}
def src(t):return {"type":"source","text":t}
def bullets(*items):return {"type":"bullets","items":list(items)}
def tab(headers,rows,widths=None,source=None):
    d={"type":"table","headers":headers,"rows":rows}
    if widths:d["widths"]=widths
    if source:d["source"]=source
    return d
def fig(n,caption,note=None,maxh=95):
    d={"type":"figure","id":n,"caption":caption,"max_height_mm":maxh}
    if note:d["note"]=note
    return d
def chapter(title,*blocks):chapters.append({"title":title,"blocks":list(blocks)})
def pct(v):return "—" if v is None else f"{v*100:.2f}"
def num(v,d=2):return "—" if v is None else f"{v:.{d}f}"

chapter("1. Tổng quan và tiến trình thực nghiệm",
    p("Tài liệu tổng hợp quá trình huấn luyện, cách triển khai, cấu hình và kết quả của các nhánh wav2vec2 và Qwen trong dự án MDD. Bằng chứng được lấy từ 14 notebook, metadata checkpoint, cấu hình thực chạy và các tệp đánh giá cục bộ tại ngày 23/10/2026. Toàn bộ 35 hình PNG đã nhúng trong notebook được đưa vào tài liệu, giữ nguyên nội dung và có nguồn cell/output."),
    note("**Kết quả chính:** thích ứng wav2vec2 bằng dữ liệu Việt cải thiện F1 trên tập test chung của từng protocol; Qwen cascade và Qwen ngữ pháp đã fine-tune xong. Kết quả sinh feedback hiện mới ở pilot; Projector hoàn thành alignment nhưng chưa train giai đoạn feedback P1."),
    tab(["Nhánh","Đã thực hiện","Kết quả có thể báo cáo"],[
        ["MDD v1 A/B/C","2.000 / 400 / 400 updates","150 câu HQTV: F1 66,37 / 70,85 / 65,20%"],
        ["MDD cascade A/B","2.000 / 400 updates","900 câu: F1 48,38 → 49,30%; TLV: 64,80 → 69,77%"],
        ["Qwen cascade Q1","3 epochs; 846 updates","Best validation loss 0,378810; pilot feedback 12 mẫu"],
        ["Projector P0","1 epoch alignment; 282 updates","900 câu NLL; sinh transcript 30 mẫu; chưa có P1"],
        ["Qwen grammar","2 epochs; 3.564 updates","Best validation loss 0,184105; probe 16 câu validation"],
        ["ASR chữ tiếng Anh","2 epochs; 1.223 updates","111 câu TLV: WER 43,01 → 37,93%"],
    ],[1.2,1.3,2.6],"Nguồn: [S03–S13]. Các hàng đo những tác vụ và tập chấm khác nhau."),
    h("Các mốc đã ghi nhận"),
    tab(["Thời gian Bangkok (UTC+7)","Diễn biến"],[
        ["20/10/2026","Hoàn thành chuỗi wav2vec2 A → B → C trên Windows, đánh giá HQTV fold 1."],
        ["21/10/2026","Train lại A/B trên protocol Plus; Qwen cascade chạy 13:22–15:31. Projector alignment bắt đầu 15:56, khoảng 19 phút 20 giây."],
        ["22/10 14:03 → 23/10 00:43","Train adapter Qwen ngữ pháp riêng, khoảng 10 giờ 40 phút."],
        ["23/10/2026 08:39–08:51","Fine-tune ASR chữ wav2vec2-base-960h; hoàn tất 1.223 updates."],
        ["23/10/2026","Rà soát, đối chiếu nguồn và biên soạn báo cáo này."],
    ],[1.4,3]),
    p("Trạng thái completed của một lượt train không đồng nghĩa mọi phép đánh giá đã hoàn tất. Các notebook và report cũ còn có chữ ‘Awaiting’; báo cáo dùng provenance/checkpoint/metrics mới hơn để xác định trạng thái thực tế. Không chạy lại training hoặc inference trong quá trình biên soạn."),
)

chapter("2. Bài toán, kiến trúc và dữ liệu",
    h("Vai trò của từng mô hình"),
    tab(["Thành phần","Đầu vào → đầu ra","Phần được học"],[
        ["wav2vec2 MDD","Audio mono 16 kHz → chuỗi phone thực nghe được; đối chiếu canonical để suy ra lỗi","Transformer + CTC phoneme head; CNN đóng băng"],
        ["Qwen cascade","Câu tham chiếu + thông tin phone/lỗi → JSON word→issue/suggestion tiếng Anh","LoRA q_proj/v_proj; không nhận audio trực tiếp"],
        ["Projector + Qwen","Audio feature 768 → embedding 1024 → Qwen","Alignment học Projector; feedback P1 chưa train"],
        ["Qwen grammar","Câu văn + task → grammar_corrected hoặc natural","Adapter LoRA riêng với cascade"],
        ["wav2vec2 ASR","Audio → văn bản tiếng Anh","Backbone + CTC ký tự; khác head phoneme MDD"],
    ],[1.1,2.6,1.7],"Nguồn: [S01], [S07], [S09], [S11–S13]."),
    p("Pipeline cascade: **audio → wav2vec2 A/B → greedy CTC phones → căn chỉnh với canonical phones → annotation_info dự đoán → Qwen → feedback JSON**. Gold perceived phones dùng giám sát train và chấm kết quả, không đưa vào decoder hay đầu vào suy luận cascade. Nhánh oracle dùng nhãn phone gold để đo mức ảnh hưởng của đầu vào, không phải pipeline triển khai độc lập."),
    h("Thông số kiến trúc"),
    tab(["Thông số","wav2vec2 MDD","Qwen3-0.6B"],[
        ["Checkpoint gốc","facebook/wav2vec2-base","Qwen/Qwen3-0.6B"],
        ["Transformer layers / hidden","12 / 768","28 / 1.024"],
        ["Attention heads","12","16 query heads / 8 KV heads"],
        ["FFN intermediate","3.072","3.072"],
        ["Vocabulary","83 phone tokens, gồm blank/UNK/err","151.936 tokens"],
        ["Tham số","94.435.539 tổng; 90.235.091 trainable","Tên model 0.6B; 1.146.880 tham số LoRA trainable"],
        ["CNN / position","7 CNN layers, mỗi lớp 512 channels","RoPE theta 1.000.000; config max_position_embeddings 40.960"],
    ],[1.2,1.9,2.2],"Nguồn: config model local và parameter_counts; context train thực tế nhỏ hơn giới hạn kiến trúc [S01]."),
    p("CNN wav2vec2 dùng kernels [10,3,3,3,3,2,2] và strides [5,2,2,2,2,2,2]. Head CTC phoneme được khởi tạo mới trên backbone tự giám sát. Vì vậy BASE chưa train head không phải baseline MDD đã có chất lượng; baseline trước thích ứng Việt là model A sau source training."),
    note("**Qwen trong các lượt train thực tế là Qwen3-0.6B.** Qwen2-Audio-7B-Instruct chỉ có metadata và manifest ghi weights_downloaded=false; không có bằng chứng weights hoặc run train. Không gán các kết quả bên dưới cho Qwen2-Audio."),
    h("Corpus và nhãn"),
    p("L2-ARCTIC local có 3.599 câu được gán nhãn thủ công, thuộc 24 người nói; 600 câu của bốn người Việt HQTV, PNV, THV, TLV. Bốn người Việt có tổng 4.528 WAV trong phần đọc đầy đủ, nhưng chỉ 600 câu có nhãn lỗi MDD. Không xem toàn bộ WAV hoặc forced alignment là gold MDD. 3.599 câu annotated có tổng khoảng 3,663 giờ audio."),
    fig(13,"Phân bố ngôn ngữ, thời lượng và số lỗi trong corpus annotated của thí nghiệm local."),
    fig(14,"Waveform và phổ STFT của một audio source train; cho thấy tín hiệu đầu vào trước nhận dạng."),
    fig(15,"Waveform với khoảng thời gian lỗi từ annotation thủ công; đây là nhãn tham chiếu, không phải dự đoán model."),
    h("Hai protocol MDD cần phân biệt"),
    tab(["Thành phần","l2arctic-v1, fold 1","l2arctic-plus-cascade-v1"],[
        ["Source train / validation","2.249 / 750","1.949 / 300"],
        ["Vietnamese train","THV + TLV: 300","HQTV + THV: 300"],
        ["Vietnamese validation","PNV: 150","PNV: 150"],
        ["Vietnamese test","HQTV: 150","TLV: 150"],
        ["Joint train / validation / test","Vai trò theo source + fold Việt","2.249 / 450 / 900"],
        ["Joint test speakers","Chỉ HQTV trong test đã báo cáo","NJS, TLV, TNI, TXHC, YKWK, ZHAA; mỗi người 150"],
    ],[1.4,1.8,2.2],"Nguồn: hai protocol đã khóa, splits.json và data_readiness [S02], [S05]."),
    p("Trong protocol Plus, train/validation/test tách speaker và utterance ID; các câu đọc có thể trùng nội dung giữa các người. Đây là kiểm tra người nói mới trong miền câu đọc, chưa phải kiểm tra prompt hoàn toàn mới. Có 750/900 câu test Plus từng ở source train của protocol v1, nên A/B cũ không được dùng cho test Plus. A/B cascade được train lại theo split mới."),
    fig(2,"Số bản ghi train/test công bố và phân bố training speakers của L2-Arctic-plus theo notebook audit."),
    fig(5,"Số mẫu, phân bố thời lượng và độ dài prompt/target Qwen trên các split cascade."),
    fig(6,"Số mẫu theo speaker và loại lỗi phone thủ công trên các split cascade."),
    p("Feedback tiếng Anh của L2-Arctic-plus được giữ nguyên từ nguồn công bố: nhãn train do GPT-4o tạo, chưa được kiểm chứng từng mẫu bởi con người; feedback test được tác giả kiểm tra bằng annotator. Chúng khác với nhãn phone thủ công L2-ARCTIC. Không mô tả toàn bộ feedback train là nhãn chuyên gia."),
    fig(19,"Histogram thời lượng audio và phân bố speaker holdout của nhánh Projector, dùng cùng split Plus."),
)

old=w["old_mdd_experiment"]
cas=w["cascade_mdd_experiment"]
chapter("3. Cách huấn luyện wav2vec2 MDD",
    p("Mô hình học **chuỗi perceived phones** mà người gán nhãn nghe được bằng CTC, thay vì ép mọi audio thành canonical phones. Preprocessing đổi audio sang mono 16 kHz, chuẩn hóa phone bỏ silence và stress, giữ dấu biến dạng `*`; inventory có 40 ARPAbet phones, 40 biến thể có `*`, [PAD]/blank, [UNK] và err. Label padding dùng -100."),
    bullets("A: khởi tạo từ wav2vec2-base với phone head mới; train chỉ source, chọn checkpoint bằng F1 source validation.",
        "B: khởi tạo từ best A và optimizer mới; thích ứng bằng 300 câu Việt với source replay xác suất 0,5, chọn best bằng validation Việt.",
        "C: cùng best A, learning rate và ngân sách như B; chỉ cập nhật trên source, chọn best bằng cùng validation Việt. C là đối chứng cho việc train tiếp; vẫn dùng nhãn Việt để chọn checkpoint."),
    tab(["Thông số thực chạy","A","B","C"],[
        ["Max optimizer updates","2.000","400","400"],
        ["Batch × accumulation","1 × 4","1 × 4","1 × 4"],
        ["Backbone LR","3e-5","1e-5","1e-5"],
        ["Phoneme head LR","3e-4","1e-4","1e-4"],
        ["Validation mỗi","250 updates","100 updates","100 updates"],
        ["Lưu resume checkpoint mỗi","50 updates","50 updates","50 updates"],
        ["Source replay","Source 100%","Source xác suất 50%","Source 100%"],
        ["Optimizer / regularization","AdamW / decay 0,01","AdamW / decay 0,01","AdamW / decay 0,01"],
        ["Gradient clip / LR schedule","Norm 1,0 / LR hằng","Norm 1,0 / LR hằng","Norm 1,0 / LR hằng"],
        ["Seed / precision","42 / FP16","42 / FP16","42 / FP16"],
    ],[1.8,1,1,1],"Nguồn: config.json lưu trong run, training_log và runtime [S03], [S06]. C cascade mới chỉ có recipe; C hoàn tất là nhánh v1."),
    p("Batch hiệu dụng đầy đủ là 4 mẫu/update. Source/target được lấy mẫu có hoàn lại nên số seen_samples là số lượt lấy mẫu, không phải số câu duy nhất. B v1 hoàn tất với 781 lượt source và 819 lượt target; không ép hai tổng này bằng nhau. Không đổi 2.000 updates thành 2.000 epochs."),
    p("CNN feature encoder đóng băng; Transformer và phone head cập nhật. Dùng gradient checkpointing, eager attention và GradScaler FP16 để phù hợp RTX 3050 Laptop 4 GiB. Các run chạy tuần tự. Runtime xử lý overflow để chỉ tính optimizer update thành công; checkpoint resume lưu optimizer và RNG."),
    fig(16,"Learning curves từ log thật của A/B/C v1, gồm train loss và validation; các đường source/Vi validation có tập chấm khác nhau.","Đường technical smoke màu đỏ chỉ kiểm tra kỹ thuật, không phải scientific training."),
    h("Chọn checkpoint và suy luận"),
    p("Best checkpoint chọn theo F1 validation, không theo test; checkpoint cuối giữ để resume. Greedy CTC gộp token lặp và bỏ blank, không dùng beam search hay language model. Chuỗi dự đoán được căn chỉnh với canonical để tạo sự kiện substitution/deletion/addition và gán về từ."),
    tab(["Run","Updates hoàn tất","Best step","Best validation F1 (%)"],
        [["v1 "+k,str(r["completed"]["optimizer_steps"]),str(r["best_checkpoint"]["step"]),pct(r["completed"]["best_validation_value"])] for k,r in old["runs"].items()]
        +[["Plus "+k,str(r["completed"]["optimizer_steps"]),str(r["best_checkpoint"]["step"]),pct(r["completed"]["best_validation_value"])] for k,r in cas["runs"].items()],
        [1.1,1.4,1,1.8],"Nguồn: completed.json và best_checkpoint/metadata.json trong từng run [S03], [S06]."),
    note("Hai nhánh B/C có cùng ngân sách tối đa, nhưng best checkpoint được chọn riêng. Bảng test bên dưới chấm best-validation checkpoint; không tự gọi đó là so sánh weights cuối đúng 400 updates. A/B Plus chọn trên những tập validation khác nhau, vì vậy chỉ dùng test chung để kết luận trước/sau."),
)

chapter("4. Kết quả wav2vec2 v1: A/B/C",
    p("Cả ba model được đánh giá trên đúng 150 câu của HQTV, người Việt test fold 1, seed 42. A chỉ source; B thêm thích ứng Việt và source replay; C tiếp tục source với cùng ngân sách như B. Đây là một fold đã có kết quả, không phải trung bình bốn fold."),
    tab(["Chỉ số (%)","A","B","C"],[[label]+[pct(old["test_results"][k]["metrics"][metric]) for k in "ABC"] for label,metric in [
        ("Precision ↑","precision"),("Recall ↑","recall"),("Phone F1 ↑","f1"),("FAR ↓","far"),("FRR ↓","frr"),("DER ↓","der"),("PER ↓","per"),("Word F1 ↑","word_f1")]],
        [1.5,1,1,1],"Nguồn: runs/windows-notebook-seed42-*-test-fold1/metrics.json [S04]."),
    p("B tăng phone F1 **4,49 đpt so với A** và **5,66 đpt so với C**. PER giảm từ 21,39% xuống 19,25%; FAR giảm từ 34,98% xuống 28,57%. Kết quả nhất quán với lợi ích của việc thêm dữ liệu Việt trong fold đang xét, nhưng chưa cho biết mức cải thiện trên toàn bộ người Việt hoặc các seed khác."),
    fig(17,"Các chỉ số test A/B/C và chênh lệch B−A, B−C trên cùng HQTV fold 1."),
    fig(18,"Các lỗi substitution phổ biến từ alignment perceived-gold → predicted của model B; không phải confusion matrix phân loại frame độc lập.",maxh=105),
    h("Bộ đếm để kiểm tra kết quả"),
    tab(["Model","TR","FR","FA","PER S / D / I","PER N"],[[k]+[str(old["test_results"][k]["counts"][v]) for v in ["TR","FR","FA"]]+[" / ".join(str(old["test_results"][k]["counts"][v]) for v in ["PER_S","PER_D","PER_I"]),str(old["test_results"][k]["counts"]["PER_N"])] for k in "ABC"],
        [1,1,1,1,2,1.2]),
    p("Ví dụ B có F1 = 2×970 / (2×970 + 410 + 388) = 70,85%. PER = (617 + 164 + 111) / 4.633 = 19,25%. Số liệu được cộng từ các counter trước khi chia; không lấy trung bình F1 của từng câu."),
    note("Speaker/audio của test tách khỏi train, nhưng nội dung câu đọc đã xuất hiện ở các người train. Chưa có artifact đầy đủ cho fold 2–4 hay nhiều seed; không báo 4-fold mean hoặc ý nghĩa thống kê từ một fold."),
)

speaker_rows=[]
for sp in cas["test_results"]["A"]["by_speaker"]:
    a=cas["test_results"]["A"]["by_speaker"][sp]["metrics"]
    b=cas["test_results"]["B"]["by_speaker"][sp]["metrics"]
    speaker_rows.append([sp,pct(a["f1"]),pct(b["f1"]),f"{(b['f1']-a['f1'])*100:+.2f}",pct(a["per"]),pct(b["per"])])
chapter("5. Kết quả wav2vec2 trong cascade Plus",
    p("A và B của protocol Plus được train lại để tránh dùng 750 câu test Plus đã từng nằm trong source train v1. Test chung có 900 câu, sáu speaker; TLV là người Việt duy nhất trong nhóm test. A tốt nhất tại step 1.750, B tốt nhất tại step 200; cả hai đã chạy đủ ngân sách 2.000/400 updates."),
    fig(22,"Split đã khóa, tình trạng checkpoint và loss alignment thực tế được notebook validation đọc lại."),
    tab(["Chỉ số (%)","A","B","B−A (đpt)"],[[label,pct(cas["test_results"]["A"]["metrics"][m]),pct(cas["test_results"]["B"]["metrics"][m]),f"{100*(cas['test_results']['B']['metrics'][m]-cas['test_results']['A']['metrics'][m]):+.2f}"] for label,m in [
        ("Phone precision ↑","precision"),("Phone recall ↑","recall"),("Phone F1 ↑","f1"),("PER ↓","per"),("FRR ↓","frr"),("DER ↓","der"),("Word F1 ↑","word_f1")]],
        [1.7,1,1,1.4],"Nguồn: data/generated/cascade-l2arctic-plus-seed42/audio/{A,B}/metrics.json; đối chiếu validation summary [S08]."),
    fig(23,"Phone F1 của A/B theo từng speaker trên đủ 900 câu test protocol Plus; PER theo speaker được trình bày ở bảng dưới."),
    tab(["Speaker","F1 A (%)","F1 B (%)","Δ F1 (đpt)","PER A (%)","PER B (%)"],speaker_rows,[1,1,1,1.2,1,1]),
    p("Micro F1 toàn test tăng 0,93 đpt; trên TLV riêng tăng khoảng 4,97 đpt. F1 giảm ở NJS, TNI và ZHAA, nên không kết luận B cải thiện mọi accent. DER toàn test tăng từ 33,79% lên 35,81%, dù F1 detection và PER cải thiện nhẹ; phát hiện được lỗi không đồng nghĩa chẩn đoán chính xác loại lỗi."),
    note("Không so 70,85% của B v1 với 49,30% của B Plus để kết luận model mới kém hơn: hai số đo trên HQTV 150 câu và test hỗn hợp 900 câu khác nhau. C cascade mới chưa có run hoàn tất; các đối chứng A_Q0/A_Q1/B_Q0/B_Q1 đủ 900 feedback cũng chưa hoàn tất."),
)

qc=q["cascade_Qwen"]
qp=qc["provenance"]
feedback_rows=[]
for r in validation["feedback_generation_pilot"]:
    label=r["case"].removeprefix("feedback_")
    feedback_rows.append([label,num(r["json_valid_percent"]),num(r["grounded_json_percent"]),num(r["feedback_word_f1_percent"]),num(r["suggestion_rougeL_on_reference_error_rows"]),num(r["latency_median_sec"])])
chapter("6. Qwen cascade: huấn luyện và feedback",
    h("Dữ liệu, prompt và objective"),
    p("Qwen3-0.6B được fine-tune riêng để chuyển câu tham chiếu và nhãn phone thành feedback tiếng Anh dạng JSON, theo từng từ chứa issue/suggestion. Train 2.249 mẫu, validation 450, test 900 cùng protocol Plus. Lượt train sử dụng annotation được công bố; khi triển khai cascade, annotation_info phải được tạo từ chuỗi phone do wav2vec2 dự đoán, không đọc gold perceived phones."),
    p("Loss causal next-token chỉ tính trên assistant feedback tokens; prompt được mask bằng -100. Base Qwen đóng băng, chỉ LoRA ở q_proj/v_proj được cập nhật. Lượng tử NF4 4 bit, double quantization, BF16 compute; gradient checkpointing. Logit projection chia chunk chỉ trên vị trí được giám sát để giảm bộ nhớ, giữ nguyên objective và prompt đầy đủ."),
    tab(["Thông số thực chạy","Giá trị"],[
        ["Train / validation","2.249 / 450 mẫu"],["Epochs / optimizer steps","3 / 846"],
        ["Train/eval batch; accumulation","1 / 1; accumulation 8 → batch hiệu dụng 8"],
        ["Learning rate / scheduler","1e-4; linear decay; warmup ratio 0,05"],
        ["Optimizer / decay / clipping","AdamW torch; 0,01; grad norm 1,0"],
        ["LoRA rank / alpha / dropout","8 / 16 / 0,05; q_proj, v_proj"],
        ["Tham số cập nhật","1.146.880"],["Max length / loss chunk","3.072 / 128 tokens"],
        ["Eval và save","Mỗi 100 updates; chọn eval_loss thấp nhất"],
        ["Seed / precision","42 / BF16 compute; NF4 base"],
        ["Runtime / peak allocated","7.748,37 giây (~2,15 giờ) / ~1,83 GiB"],
        ["Best / last","checkpoint-800 / checkpoint-846"],
    ],[1.5,3],"Nguồn: provenance, training_args.bin và trainer_state thực tế [S07]; không chỉ đọc recipe."),
    tab(["Train loss trung bình","Best validation loss","Best / last checkpoint"],[[num(qp["metrics"]["train_loss"],6),num(qp["best_validation_loss"],6),"800 / 846"]],[1.3,1.3,1.5]),
    p("Adapter xuất ở thư mục Q/adapter có SHA256 khớp checkpoint-800 đã chọn. Loss train trung bình và validation loss không phải trước/sau trên cùng dữ liệu, cũng không phải độ chính xác phát âm. Cả Q0 và Q1 khi đánh giá được dùng cùng thiết lập precision/generation cho từng phép đo."),
    h("Kết quả hiện có: pilot 12 feedback"),
    p("Pilot cân bằng gồm 12 câu trong test, mỗi speaker hai câu; tất cả đều có lỗi trong feedback tham chiếu. Sinh greedy, num_beams=1, seed 42, max_new_tokens=512 cho feedback (pilot transcript ở chương 7 dùng 128). JSON sai vẫn nằm trong mẫu số, không được thay bằng template hợp lệ. B_Q0 là Qwen base; B_Q1 là Qwen đã fine-tune cùng đầu vào từ wav2vec2 B."),
    tab(["Cấu hình","JSON (%)","Grounded (%)","Word-key F1 (%)","ROUGE-L","Median (s)"],feedback_rows,[2.2,1,1.2,1.2,1,1],"Nguồn: projector-validation-v1/report/summary.json [S10]. Word-key F1 đo tập từ của feedback, khác phone F1."),
    fig(25,"Các cấu hình feedback trên cùng pilot 12 mẫu: Projector P0, cascade Q0/Q1, oracle Q1 và adapter transfer.",maxh=100),
    p("B_Q1 đạt JSON hợp lệ/grounded 100%, word-key F1 62,65 và ROUGE-L 22,66; B_Q0 bị 0% strict JSON do output có Markdown fences. Như vậy mức tăng bao gồm việc học đúng contract đầu ra. BLEU-2 của B_Q1 là 37,51 theo sacreBLEU. Oracle Q1 đạt word-key F1 85,00 nhưng dùng nhãn gold; không thể coi là chất lượng của app tự nhận audio. Word-key F1 chấm tập từ đã chuẩn hóa; các lần lặp cùng key bị gộp, không phải F1 của từng vị trí từ."),
    note("12 mẫu là pilot, không phải benchmark sinh đủ 900 test. Vì mọi mẫu đều có reference errors, pilot không đo false alarm trên câu hoàn toàn đúng. Grounded JSON là kiểm tra cấu trúc/gắn với từ trong câu, không bảo đảm lời khuyên đúng ngữ âm hoặc hữu ích cho người học."),
)

pr=q["projector"]["provenance"]
chapter("7. Projector: nối audio với Qwen",
    p("Nhánh multimodal đưa feature audio trực tiếp vào embedding của Qwen: **wav2vec2 B đóng băng → Linear(768,1024) → GELU → Linear(1024,1024) → Qwen3-0.6B**. Embedding được ghép prefix → audio frames → suffix; mask loss trên prompt/audio. Feature được cache để tách bước encoder khỏi huấn luyện decoder."),
    tab(["Thành phần","Alignment đã chạy","Feedback task tuning (recipe)"],[
        ["Supervision","Audio → transcript; prompt không chứa transcript","Audio + reference → feedback JSON"],
        ["Trainable","Projector 1.837.056 params","Projector + LoRA q/v Qwen"],
        ["Frozen","wav2vec2 B và toàn bộ Qwen base","wav2vec2 B và Qwen base"],
        ["Epoch / learning rate","1 / 1e-3","3 / 1e-4"],
        ["Batch × accumulation","1 × 8","1 × 8 trong recipe"],
        ["Context / loss chunk","3.072 / 128","3.072 / 128"],
        ["Trạng thái","Hoàn tất 282 updates; P0","Chưa train; chưa có P1"],
    ],[1.2,1.9,1.9],"Nguồn: projector_l2arctic_plus.json và alignment/training_provenance.json [S09]."),
    p("Alignment đạt best validation loss 3,464111; runtime 1.160,406 giây, peak allocated khoảng 1,22 GiB. P0 là Projector sau alignment với Qwen gốc. Adapter transfer lấy LoRA của cascade gắn sang nhánh Projector chỉ là kiểm tra chuyển adapter, không phải P1 đã task tuning."),
    tab(["900 câu test, teacher forced","NLL/token ↓","Perplexity ↓"],[[r["case"].replace("transcript_nll_",""),num(r["target_token_nll"],6),num(r["perplexity"],2)] for r in validation["transcript_full_nll"]],[2,1.4,1.4]),
    p("So với random projector khởi tạo tái lập, alignment giảm NLL 3,548258; bootstrap theo sáu speaker cho CI95% [−3,583479; −3,511515]. So aligned với muted audio, delta NLL là −2,781176, CI95% [−2,795881; −2,765003]. Teacher-forced NLL trên 900 câu đo xác suất target khi có prefix gold, không phải độ chính xác tự sinh transcript."),
    fig(20,"Waveform mẫu train ABA/arctic_a0501 dùng trong notebook Projector; transcript và feedback được in riêng trong output notebook, không hiển thị trên biểu đồ."),
    {"type":"pagebreak"},
    fig(24,"Trước/sau alignment và đối chứng tắt audio: NLL trên 900 câu và WER pilot sinh transcript 30 mẫu; perplexity được trình bày trong bảng."),
    tab(["Sinh transcript: 30 mẫu","WER (%) ↓","Câu khớp tuyệt đối (%)","EOS (%)"],[[r["case"].replace("transcript_generation_",""),num(r["wer_percent"]),num(r["exact_sentence_percent"]),num(r["eos_percent"])] for r in validation["transcript_generation_pilot"]],[2,1.1,1.6,1]),
    note("WER aligned vẫn 97,39% và exact sentence 0% trong pilot 30 mẫu. Alignment có tín hiệu học từ audio nhưng chưa đủ để khẳng định nhận dạng tốt. P0 feedback chỉ JSON 16,67%, grounded/word-key F1 0% trên 12 mẫu. Chưa có bằng chứng Projector vượt cascade; so hai nhánh cũng khác compute vì thêm alignment."),
    p("Dataset alignment chỉ 2.249 audio/transcript; không gán quy mô hay điểm của kiến trúc/bài báo gốc cho lượt train nhỏ này. Random baseline là khởi tạo mới theo seed, không phải file checkpoint chính xác của step 0 đã lưu."),
)

g=q["grammar_Qwen"]
gp=g["provenance"]
chapter("8. Qwen ngữ pháp và cách nói tự nhiên",
    p("Adapter grammar là lượt fine-tune riêng trên W&I/LOCNESS và JFLEG, không dùng audio và không ghi đè adapter phát âm. W&I cung cấp task `grammar_corrected`; JFLEG cung cấp task `natural`. Mỗi target là JSON một trường đúng theo task; không tự tạo explanation gold hoặc nhãn natural cho W&I."),
    tab(["Split","Records","Câu nguồn","Grammar","Natural","Target giữ nguyên"],[[s,str(v["rows"]),str(v["source_sentences"]),str(v["tasks"]["grammar_corrected"]),str(v["tasks"]["natural"]),str(v["unchanged_rows"])] for s,v in g["dataset_stats"].items()],[1.1,1,1.2,1,1,1.3],"Nguồn: grammar-wi-jfleg-v1/manifest.json [S11]."),
    p("W&I dev làm validation; 10% author train được giữ làm internal grammar test, đồng thời loại train authors trùng dev. JFLEG dev chia nhóm câu 80/20 train/validation; JFLEG test 747 câu giữ độc lập. Mỗi câu eval giữ đủ references, không nhân số câu test theo số người sửa. Internal grammar test không phải BEA official test."),
    p("Audit sau chuẩn hóa ghi 0 text overlap và 0 group overlap giữa các split. Đây là chống trùng exact, chưa phải semantic dedup hoặc kiểm chứng base Qwen chưa từng thấy corpus. JFLEG không có author ID nên không khẳng định author-disjoint cho nguồn này."),
    fig(33,"Phân bố split/task, target giữ nguyên và độ dài token của dataset Qwen grammar."),
    fig(34,"Loại sửa được thống kê từ nhãn W&I trong notebook chuẩn bị dữ liệu.",maxh=65),
    tab(["Thông số thực chạy","Giá trị"],[
        ["Train / validation; epochs / updates","28.501 / 4.292 records; 2 epochs / 3.564 updates"],
        ["Max length / token dài nhất train","768 / 607; không cắt target"],
        ["Batch × accumulation","1 × 16 → batch hiệu dụng 16"],
        ["LoRA","r=8; alpha=16; dropout=0,05; q_proj/v_proj"],
        ["Base / compute","NF4 4 bit double quant; BF16 compute"],
        ["LR / schedule / warmup","1e-4 / linear / ratio 0,05"],
        ["Optimizer / decay / clip","AdamW torch / 0,01 / norm 1,0"],
        ["Eval-save / loss chunk","250 updates / 64 tokens"],
        ["Trainable / seed; best / last","1.146.880 params / 42; checkpoint 3.500 / 3.564"],
        ["Runtime / peak allocated","38.422,22 giây (~10,67 giờ) / ~1,08 GiB"],
    ],[1.7,3],"Nguồn: training_args.bin, provenance và trainer_state trong run [S11]."),
    p("Eval loss giảm từ 0,217636 tại step 250 xuống best 0,184105 tại step 3.500; train loss trung bình 0,234158. Adapter xuất có SHA256 giống selected best, dù lượt train kết thúc ở step 3.564."),
    h("Probe trước/sau trên 16 câu validation"),
    tab(["Chỉ số","Grammar (8 câu)","Natural (8 câu)"],[
        ["Strict JSON trước → sau","0 → 100%","0 → 100%"],
        ["Exact reference sau train","50,00%","25,00%"],
        ["chrF sau train","86,95","93,78"],
        ["Giữ nguyên control sau train","100% (4 controls)","25% (4 controls)"],
    ],[1.7,1.5,1.5],"Nguồn: grammar-adapter-demo-v1/comparison.json; CPU FP32, greedy, max_new_tokens=512 [S12]."),
    note("Đây là development probe 16 câu, không phải full benchmark 3.746 test, grammar accuracy, ERRANT F0.5 hay JFLEG GLEU. Điểm base bằng 0 chịu ảnh hưởng strict JSON contract. Natural chỉ giữ nguyên 1/4 control là dấu hiệu cần xem lại mức sửa quá tay. Không suy ra chất lượng hội thoại từ dữ liệu văn viết."),
    p("Nguồn dữ liệu có giới hạn nghiên cứu/giáo dục phi thương mại và điều kiện phân phối riêng. Báo cáo không đính kèm corpus thô. Các artifact runtime/demo không được xem là bằng chứng adapter cải thiện tác vụ hội thoại tự do."),
)

asrp=w["asr_experiment"]["provenance"]
ar=asrp["results"]
chapter("9. Fine-tune wav2vec2 cho ASR tiếng Anh",
    p("Nhánh ASR dùng **wav2vec2-base-960h với vocabulary ký tự 32**, khác wav2vec2-base + phoneme head 83 tokens của MDD. Mục tiêu là nhận transcript tiếng Anh từ giọng Việt, giúp cấp câu đối chiếu/hội thoại; không trực tiếp chẩn đoán phone sai."),
    tab(["Dữ liệu / thông số","Giá trị"],[
        ["Train / validation / test","2.445 / 104 / 111 câu"],
        ["Speaker Việt","Train HQTV/THV; validation PNV; test TLV"],
        ["Split prompt","Bucket SHA256 cố định 80/10/10; normalized prompt-disjoint"],
        ["Source replay","611 records; fraction cấu hình 0,25"],
        ["Giới hạn train audio","8 giây; loại 6 câu dài, không crop giữ target cũ"],
        ["Epochs / optimizer steps","2 / 1.223"],
        ["Batch × accumulation","1 × 4"],
        ["LR backbone / character head","3e-6 / 1e-5"],
        ["Warmup / validation","50 steps / mỗi 150 steps"],
        ["Memory / precision","CNN frozen; gradient checkpointing; BF16; GPU fraction 0,65"],
        ["Best checkpoint / criterion","Step 1.050; validation WER thấp nhất"],
    ],[1.6,3],"Nguồn: asr-vietnamese-english-v1/training_provenance.json và notebook [S13]."),
    fig(3,"Train loss và validation WER trong 1.223 updates của nhánh ASR."),
    tab(["Tập / chỉ số","Base (%)","Fine-tuned (%)","Thay đổi (đpt)"],[
        ["Validation WER",pct(ar["baseline_validation"]["wer"]),pct(ar["best_validation"]["wer"]),f"{100*(ar['best_validation']['wer']-ar['baseline_validation']['wer']):+.2f}"],
        ["Validation CER",pct(ar["baseline_validation"]["cer"]),pct(ar["best_validation"]["cer"]),f"{100*(ar['best_validation']['cer']-ar['baseline_validation']['cer']):+.2f}"],
        ["Test WER",pct(ar["baseline_test"]["wer"]),pct(ar["trained_test"]["wer"]),f"{100*(ar['trained_test']['wer']-ar['baseline_test']['wer']):+.2f}"],
        ["Test CER",pct(ar["baseline_test"]["cer"]),pct(ar["trained_test"]["cer"]),f"{100*(ar['trained_test']['cer']-ar['baseline_test']['cer']):+.2f}"],
    ],[1.6,1.1,1.5,1.5]),
    fig(4,"So sánh WER/CER trước và sau fine-tune trên cùng validation/test ASR."),
    p("Test có 1.023 reference words: base sai 440 edits, model fine-tuned còn 388, giảm 52 word edits. WER giảm 5,08 đpt, tương đương khoảng 11,82% giảm lỗi tương đối. Nhãn là transcript chính tả của corpus, không phải bản ghi gold mọi từ người học thực sự phát ra."),
    p("Ba recording phát triển riêng chưa được cải thiện: hai mẫu giữ WER 66,67% và 114,29%, một mẫu tăng 100% → 125%. Các ví dụ này khác holdout 111 câu và cho thấy mức tăng trung bình không bảo đảm mọi audio đều tốt hơn."),
    note("Dù validation/test tách người và normalized prompt, đây là tập nhỏ về câu đọc của một speaker test. Kết quả không chứng minh chất lượng ASR hội thoại tự do hoặc độ chính xác phát âm; không cộng/gộp WER với phone F1."),
)

chapter("10. Kiểm chứng ngoài miền: SpeechOcean762",
    p("Notebook external validation kiểm tra A/B/C v1 trên SpeechOcean762 mà không fine-tune thêm. Corpus gồm người có tiếng mẹ đẻ Trung nói tiếng Anh, có cả trẻ em; không phải tập giọng bản ngữ/TTS. B được chạy trên đủ 2.500 câu test; cả ba model được so thêm trên cùng 179 câu mà cả năm chuyên gia đều chấm đúng. Rubric của SpeechOcean khác L2-ARCTIC và scoring canonical-position không hoàn toàn cùng MDD scorer, nên các F1 không so trực tiếp."),
    tab(["B, đủ 2.500 câu test","Phone detection (%)","Word detection (%)"],[
        ["Precision","11,94","12,97"],["Recall","88,46","97,37"],["F1","21,04","22,89"],
    ],[2,1.4,1.4],"Nguồn: data/generated/external/speechocean762/20261021T024600Z/B-full/metrics.json [S14]."),
    fig(10,"B v1 trên 2.500 test SpeechOcean: recall cao đi cùng precision thấp và nhiều cảnh báo nhầm."),
    tab(["Cùng 179 câu hoàn toàn đúng","A","B","C"],[
        ["Câu có bất kỳ cảnh báo (%)","93,30","96,65","92,74"],
        ["Từ đúng bị đánh dấu (%)","41,26","44,65","40,64"],
    ],[2.3,1,1,1]),
    fig(11,"A/B/C trên 179 câu được cả năm chuyên gia chấm đúng; cột cảnh báo càng cao càng bất lợi."),
    fig(12,"Các phone bị B báo nhầm nhiều nhất trong các vị trí cả năm chuyên gia chấm đúng, với ít nhất 100 vị trí/phone."),
    note("B đánh dấu 173/179 câu hoàn toàn đúng, cho thấy giới hạn lớn khi đưa decoder MDD v1 sang miền khác. Đây là bằng chứng cần kiểm chứng false alarms trước triển khai; không che khuất kết quả này bằng F1 tốt hơn trên HQTV."),
)

notes=read("docs/training_report/qa/figures_notes_vi.json")["notes"]
notes={x["figure_id"]:x for x in notes}
chapter("11. Hiệu chỉnh và kiểm tra hệ thống",
    p("Các thí nghiệm sau giữ nguyên trọng số mô hình, nhằm phân tích cách chấm, chuẩn hóa transcript hoặc parser. Chúng bổ sung góc nhìn triển khai nhưng không phải các lượt fine-tune mới."),
    h("Calibration điểm phát âm"),
    p("Notebook calibration thử margin posterior theo phone, CTC alignment và biến thể CMU trên cùng weights B. Có pilot 40 audio và full 450 validation: HJK/LXC 300 câu dùng chọn ngưỡng, PNV 150 dùng kiểm tra. PNV từng dùng chọn checkpoint B nên không phải test độc lập. Không dùng test 900 hoặc SpeechOcean để chọn ngưỡng."),
    fig(26,"Pilot 40 audio: 10 HJK + 10 LXC dùng fit, 20 PNV dùng audit; phân bố nhãn từ đúng, sai và thiếu."),
    p("Lưới đăng ký gồm 496 cặp ngưỡng, yêu cầu đồng thời FPR ≤10%, FNR ≤20%, coverage ≥50%. Không cặp nào đạt trên fit, artifact ghi **ineligible**. Policy an toàn có coverage 0; FPR ở ô trống là không xác định do không có quyết định được chấm, không phải 0%."),
    fig(27,"Pilot heldout PNV: strict FPR 23,71%, recall 72,63%, coverage 100%; candidate/safe abstain vì artifact ineligible.",maxh=85),
    fig(28,"Histogram posterior-margin error score của từ được chấm thủ công đúng/sai trong pilot; hai phân bố chồng lấn. Score chưa phải xác suất lỗi đã hiệu chuẩn.",maxh=85),
    fig(29,"Tìm cặp ngưỡng trên FIT pilot: trục FPR/FNR, màu coverage; đường đứt FPR=0,10 và FNR=0,20.",maxh=95),
    fig(30,"Full-validation heldout PNV: strict FPR 23,46%, recall 69,66%; candidate/safe coverage 0 vì artifact ineligible.",maxh=85),
    fig(31,"Tìm cặp ngưỡng trên FIT full-validation: đánh đổi FPR/FNR/coverage; không cặp nào đạt đồng thời policy.",maxh=95),
    note("Không xem abstention là phát âm đúng. Posterior margin đang thử chưa phải triển khai đầy đủ CTC-GOP của paper. Cách chấm mới và lựa chọn ngưỡng cần được đánh giá riêng, không dùng nó để gán điểm test mới cho weights đã train."),
    h("Ví dụ audio, parser và ứng viên ngữ âm"),
    fig(8,notes[8]["caption_vi"]),
    p("Bốn audio validation cho thấy cả TP, FP và FN: HJK/a0028 phát hiện được một từ lỗi; LXC/a0029 bỏ sót bốn từ; PNV/a0127 có một FP và một FN; HJK/b0047 là câu đúng vẫn có ba FP. Đây là ví dụ phát triển, không phải benchmark tổng hợp."),
    fig(9,notes[9]["caption_vi"],"Chỉ sửa parser trên raw Qwen outputs cũ; không train lại hoặc đo grammar accuracy."),
    fig(32,notes[32]["caption_vi"]),
    p("Các WER 66,67 / 114,29 / 100% trong notebook ứng viên phone là so transcript với hint ở ví dụ phát triển; không phải độ chính xác phát âm. Hints/candidate retrieval cần người dùng xác nhận nội dung thực nói trước khi dùng làm tham chiếu."),
    fig(35,notes[35]["caption_vi"]),
    p("Trong kiểm tra normalization, WER controls giảm 6,25% → 0%; trên audio, raw 16,67%, Qwen draft 13,16%, guarded output 16,67%, với 8/12 đề xuất được nhận. Guard chưa cải thiện word errors audio trong ví dụ này; không được báo thành benchmark tăng ASR chung."),
    src("Nguồn: các notebook mdd_pronunciation_calibration, mdd_chat_error_checks, phonetic_candidates_validation và transcript_normalization_validation; captions kiểm tra trong qa/figures_notes_vi.json [S15]."),
)

chapter("12. Môi trường và cách tái lập",
    h("Môi trường đã ghi trong artifact"),
    tab(["Thành phần","Phiên bản / thiết lập"],[
        ["OS / Python","Windows 11; Python 3.12.10"],
        ["GPU / CUDA","NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB; CUDA runtime 12.6"],
        ["PyTorch / Transformers","2.7.1+cu126 / 4.57.6"],
        ["PEFT / Accelerate / bitsandbytes","0.17.1 / 1.10.1 / 0.48.1"],
        ["NumPy / scipy / soundfile","2.2.6 / 1.15.3 / 0.13.1"],
        ["SacreBLEU / rouge-score","2.5.1 / 0.1.2"],
        ["Threads / seed","MDD: 4 threads; Projector/ASR: 2 threads; seed chính 42"],
    ],[1.7,3],"Nguồn: runs/*/environment.json và output môi trường notebook [S16]."),
    p("Các số peak allocated là bộ nhớ tensor CUDA được PyTorch ghi nhận, không phải tổng VRAM của hệ thống và không cộng GPU context/cache/process khác. FP16 của MDD và BF16/NF4 của Qwen là thiết lập riêng từng nhánh. Seed cố định không bảo đảm bitwise equality qua phần cứng khác."),
    h("Thứ tự train đã triển khai"),
    bullets("Chuẩn bị corpus/protocol, verify audio/hash, tạo vocabulary/normalization và audit speaker/utterance overlap.",
        "Train A → chọn best source validation → khởi tạo B và C độc lập từ cùng best A → chọn best validation Việt → chấm test đã khóa.",
        "Với Plus: tạo split mới → train lại A/B → train Qwen Q → chạy audio test và feedback comparison theo các cờ đã đặt.",
        "Projector: chờ B mới, cache feature → alignment P0 → nếu được triển khai thì mới task tuning P1. Ở snapshot này chỉ alignment hoàn tất.",
        "Grammar/ASR là các run riêng, với dữ liệu, head và metric riêng; giữ checkpoint/output tách biệt."),
    h("Lệnh tham chiếu — chạy từ project root MDD"),
    p("Các lệnh dưới đây mô tả cách dùng script đã có; không được thực thi để viết báo cáo. Các lệnh train sẽ tạo/cập nhật run khi chủ động chạy. Trên Windows dùng .venv/Scripts/python.exe; Linux thay bằng .venv/bin/python."),
    {"type":"code","text":"# wav2vec2 v1: chuỗi A/B/C\npwsh -File scripts/run_training_sequence.ps1 -Experiment windows-notebook-seed42 -Seed 42 -Fold 1\n# Resume chuỗi đã dừng: thêm -ResumeExisting"},
    {"type":"code","text":"# Cascade Plus: chuẩn bị / xem trạng thái / train\n.venv/Scripts/python.exe -X utf8 scripts/run_cascade_training.py --phase prepare\n.venv/Scripts/python.exe -X utf8 scripts/run_cascade_training.py --phase status\n.venv/Scripts/python.exe -X utf8 scripts/run_cascade_training.py --phase train"},
    {"type":"code","text":"# Qwen feedback: kiểm dữ liệu / resume adapter\n.venv/Scripts/python.exe -X utf8 scripts/train_feedback_model.py --config configs/cascade_feedback_qlora.json --check-data-only\n.venv/Scripts/python.exe -X utf8 scripts/train_feedback_model.py --config configs/cascade_feedback_qlora.json --resume"},
    {"type":"code","text":"# Projector: trạng thái / validation đã có runner riêng\n.venv/Scripts/python.exe -X utf8 scripts/run_projector_training.py --phase status\n.venv/Scripts/python.exe -X utf8 scripts/run_projector_validation.py --phase all"},
    {"type":"code","text":"# Qwen grammar: kiểm dữ liệu / chủ động train-resume\n.venv/Scripts/python.exe -X utf8 scripts/train_grammar_model.py --check-data-only\n.venv/Scripts/python.exe -X utf8 scripts/train_grammar_model.py --run --resume"},
    p("Notebook Colab là hướng dẫn, 0 code cell có execution count và không có saved outputs. Không có bằng chứng thực nghiệm Colab trong bộ hiện tại. Hai notebook cascade/Projector giữ RUN_TRAIN=False hoặc RUN_PROJECTOR_TRAIN=False mặc định; ảnh cũ không tự cập nhật khi run ngoài terminal."),
    h("Artifact cần giữ khi chuyển máy"),
    p("Giữ protocol/split và SHA256, preprocessing, config thực chạy, completed/provenance, best_checkpoint, final resume checkpoint, training/validation log, metrics và dự đoán từng câu. Với Qwen giữ base revision, tokenizer, adapter, prompt/evaluator hash. Dataset/weights/runs/data/generated thường bị Git ignore; Git không phải bản sao lưu của chúng."),
)

chapter("13. Ảnh lịch sử và phần còn thiếu",
    p("Hai hình sau được giữ để tài liệu chứa đủ ảnh notebook gốc. Chúng phản ánh trạng thái lúc notebook được lưu, không đại diện kết quả cuối. Bảng kết quả cập nhật ở các chương trước dùng log và metrics thực tế."),
    fig(7,"Snapshot learning curves cascade còn sớm: chỉ có audio loss A tới khoảng 41 step; hai panel validation F1 và Qwen còn ‘Awaiting real logs’.","Ảnh lịch sử, không dùng để chứng minh loss cuối hoặc model chưa train."),
    fig(21,"Snapshot Projector training với placeholder ‘Awaiting real training logs’ ở alignment/feedback.","Alignment hiện đã hoàn tất 282 updates; feedback P1 vẫn chưa train."),
    tab(["Mục","Trạng thái tại thời điểm tổng hợp"],[
        ["A/B/C v1","Có test fold 1; chưa có đủ fold 2–4/multi-seed"],
        ["A/B Plus","Đủ audio metrics 900 test; notebook/report cũ chưa phản ánh hết"],
        ["C Plus","Có config; chưa có scientific run hoàn tất"],
        ["Qwen cascade","Train xong; có feedback pilot 12, chưa full generation 900"],
        ["Projector","Alignment xong; P1 chưa train; pilot transcript 30/feedback 12"],
        ["Grammar","Train xong; development probe 16 validation; chưa full test benchmark"],
        ["Calibration","Không có ngưỡng đạt đồng thời tiêu chí; ineligible"],
        ["Qwen2-Audio","Chỉ metadata, chưa tải weights/train"],
    ],[1.3,3.5]),
    h("Các giới hạn quyết định cách diễn giải"),
    bullets("Không gộp F1 phone, word-key F1 feedback, WER, lexical overlap và validation loss thành một ‘độ chính xác chung’.",
        "Một seed, ít người Việt và câu đọc chung chưa chứng minh tổng quát hóa sang hội thoại, thiết bị/miền mới hoặc hiệu quả học tập.",
        "Mức tăng của pilot Qwen chịu ảnh hưởng học contract JSON; cần benchmark đủ test và đánh giá ngữ âm/lời khuyên bằng người để xác nhận chất lượng nội dung.",
        "Báo cả lỗi bỏ sót, cảnh báo nhầm và abstention; không suy ra phát âm đúng chỉ từ việc không có feedback."),
    h("Nhánh bản nháp tiếng Việt trước đây"),
    p("Notebook feedback review có 300 bản nháp template offline, 190 nhóm câu, 0 mẫu được duyệt; split 242/29/29 chỉ là dự kiến nếu toàn bộ được duyệt. Không có run khoa học chính của dataset feedback-v1. Các smoke trong local-checks chỉ kiểm tra kỹ thuật, khác với lượt train Qwen cascade trên feedback tiếng Anh công bố."),
    fig(1,"Phân bố số khác biệt model đánh dấu và độ dài 300 bản nháp review; đây là dữ liệu chưa duyệt, chưa phải bộ train đã dùng."),
)

sources=[
    ("S01","Kiến trúc và nhận dạng","models/pretrained/wav2vec2-base/config.json; models/pretrained/Qwen3-0.6B/config.json; src/mdd/runtime.py; runs/windows-notebook-seed42-A/parameter_counts.json"),
    ("S02","Protocol v1","data/protocol/l2arctic-v1/{protocol,splits}.json; docs/DATA_SPLIT.md"),
    ("S03","Train wav2vec2 v1","runs/windows-notebook-seed42-A/; runs/windows-notebook-seed42-{B,C}-fold1/ — config, completed, validation_log, best_checkpoint/metadata"),
    ("S04","Test v1","runs/windows-notebook-seed42-A-test-fold1/metrics.json; runs/windows-notebook-seed42-{B,C}-fold1-test-fold1/metrics.json"),
    ("S05","Protocol Plus","data/protocol/l2arctic-plus-cascade-v1/{protocol,splits}.json; data/generated/cascade-l2arctic-plus-seed42/data_readiness.json"),
    ("S06","Train audio Plus","runs/cascade-l2arctic-plus-seed42/{A,B}/ — config, completed, training_log, validation_log, best_checkpoint/metadata"),
    ("S07","Train Qwen cascade","runs/cascade-l2arctic-plus-seed42/Q/training_provenance.json; checkpoint-846/{training_args.bin,trainer_state.json}; checkpoint-800; adapter"),
    ("S08","Test audio Plus","data/generated/cascade-l2arctic-plus-seed42/audio/{A,B}/metrics.json"),
    ("S09","Train Projector","configs/projector_l2arctic_plus.json; runs/projector-l2arctic-plus-seed42/alignment/{training_provenance,parameter_counts}.json; training_history.jsonl; scripts/projector_model.py"),
    ("S10","Validation checkpoint hiện có","data/generated/projector-validation-v1/report/summary.json — audio 900, transcript NLL 900, transcript pilot 30 và feedback pilot 12"),
    ("S11","Train grammar","data/processed/grammar-wi-jfleg-v1/manifest.json; runs/grammar-qwen3-06b-wi-jfleg-v1/training_provenance.json; training_history.jsonl; checkpoint-3564/training_args.bin"),
    ("S12","Grammar probe","data/generated/grammar-adapter-demo-v1/comparison.json; runs/grammar-qwen3-06b-wi-jfleg-v1/demo_validation.json"),
    ("S13","Train/test ASR","configs/asr_vietnamese_english.json; runs/asr-vietnamese-english-v1/{training_provenance,history}.json; notebooks/mdd_asr_vi_finetuning.ipynb"),
    ("S14","SpeechOcean ngoài miền","notebooks/mdd_external_validation.ipynb; data/generated/external/speechocean762/20261021T024600Z/B-full/metrics.json; các output A/B/C perfect subset đã lưu trong notebook"),
    ("S15","Calibration và diagnostic","notebooks/mdd_pronunciation_calibration.ipynb; notebooks/mdd_chat_error_checks.ipynb; notebooks/phonetic_candidates_validation.ipynb; notebooks/transcript_normalization_validation.ipynb"),
    ("S16","Môi trường","runs/windows-notebook-seed42-A/environment.json; notebooks/mdd_cascade_training_comparison.ipynb cell 4"),
    ("S17","Scorer","src/mdd/metrics.py; scripts/published_feedback_metrics.py; scripts/evaluate_grammar_model.py"),
    ("S18","Các model/nhánh chưa train","models/pretrained/Qwen2-Audio-7B-Instruct/source_manifest.json; data/feedback/drafts-20261021-cef406b.manifest.json"),
]

metrics=[
    ["Precision","TR / (TR + FR)","Tỷ lệ cảnh báo lỗi đúng"],
    ["Recall","TR / (TR + FA)","Tỷ lệ lỗi được phát hiện"],
    ["Phone F1","2TR / (2TR + FR + FA)","Hài hòa precision/recall detection"],
    ["FAR","FA / (FA + TR)","Lỗi bị chấp nhận/bỏ sót"],
    ["FRR","FR / (TA + FR)","Âm đúng bị từ chối"],
    ["DER","DE / (CD + DE)","Sai chẩn đoán trong lỗi bắt được, đủ gold"],
    ["PER","(S + D + I) / số perceived-gold phones","Edit distance chuỗi phone"],
    ["WER / CER","Word / character edits chia số đơn vị gold","Edit distance transcript; có thể >100%"],
    ["Word-key F1","F1 trên tập từ chuẩn hóa có feedback; gộp key lặp","Khác word detection F1, từng vị trí từ và phone F1"],
    ["NLL / perplexity","−log p(target); perplexity = exp(NLL)","Causal target-token loss; không phải accuracy"],
    ["BLEU-2 / ROUGE-L / chrF","Overlap n-gram / LCS / ký tự","Gần reference về chữ, không đủ kết luận đúng ý/ngữ âm"],
]
manifest=read("docs/training_report/figures_manifest.json")
figure_order=[b["id"] for c in chapters for b in c["blocks"] if b["type"]=="figure"]
figure_by_id={f["figure_id"]:f for f in manifest["figures"]}
chapter("14. Chỉ số, nguồn và danh mục hình",
    h("Định nghĩa chỉ số"),
    p("Scorer MDD của dự án dùng canonical-phone-and-ordered-insertion-slots-v1: vị trí canonical cùng insertion slots theo thứ tự trong cùng gap. TA: âm đúng được chấp nhận; FR: âm đúng bị từ chối; FA: lỗi bị chấp nhận; TR: lỗi bị phát hiện; CD/DE: chẩn đoán đúng/sai trong lỗi đã bắt được."),
    tab(["Chỉ số","Công thức / phép đo","Cách hiểu"],metrics,[1.1,2.1,2.3],"Nguồn: [S17]. Micro metrics cộng counters trước khi chia; mẫu số 0 trả null."),
    p("Gold err/[UNK] không đủ cho exact diagnosis bị loại khỏi DER nhưng vẫn giữ trong detection/PER. PER giữ marker `*`. Khi suy luận, canonical dùng căn chỉnh dự đoán; không đưa perceived gold vào decoder hoặc prompt cascade. Perceived gold được dùng làm supervision khi train và để chấm. Alignment có thể không trùng từng vị trí chuyên gia, nên không tự xem đây là scorer tái hiện nguyên vẹn một paper."),
    h("Danh mục bằng chứng cục bộ"),
    p("Đường dẫn dưới đây tương đối với project root MDD. Các số liệu trình bày là tổng hợp lại từ artifact ở máy hiện tại; tham chiếu này giúp truy ngược và kiểm tra, không giả định tất cả tệp lớn đã được lưu trong Git."),
    *[{"type":"small","text":f"**[{sid}] {label}.** `{path}`"} for sid,label,path in sources],
    p("Kiểm kê 14 notebook ghi nhận 298 outputs và không có saved output_type=error; điều đó không bảo đảm mọi notebook đã chạy toàn bộ. Có 35/35 PNG khác nhau, được kiểm byte, SHA256 và PNG CRC; không tìm thấy ảnh liên kết Markdown/HTML bị bỏ sót. train_mdd_colab.ipynb có 0 hình/output thực thi nên không tạo hình thay thế."),
    h("Các tệp đi cùng báo cáo"),
    p("figures_manifest.json chứa nguồn, cell, output, kích thước và SHA256 của từng hình. evidence_wav2vec.json và evidence_qwen.json lưu chi tiết đối chiếu. report_content.json là bản nội dung có thể chỉnh sửa; author_report.py tạo bảng từ nguồn, build_report.py dựng PDF. report_build_manifest.json và qa/pdf_validation.json ghi thông tin xuất/kiểm tra tài liệu."),
    h("Danh mục đủ 35 hình notebook"),
    tab(["Hình","Notebook / cell / output","Chủ đề"],[[f"{n:02d}",f"{Path(f['notebook']).name}\ncell {f['cell_number_1']} · output {f['output_index_0']+1}",f["nearest_heading"]] for n,f in enumerate([figure_by_id[i] for i in figure_order],1)],[.55,2.5,2]),
)

(HERE/"report_content.json").write_text(json.dumps({"language":"vi","date":"2026-10-23","chapters":chapters},ensure_ascii=False,indent=2),encoding="utf-8")
print(f"Authored {len(chapters)} chapters; {sum(len(c['blocks']) for c in chapters)} blocks")
