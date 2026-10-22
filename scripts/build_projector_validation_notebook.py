"""Executable, read-only evidence notebook with explicit manual evaluation cells."""
from pathlib import Path
import textwrap
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []
    def md(source):
        cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip()))
    def code(source, manual=False):
        cell = nbformat.v4.new_code_cell(textwrap.dedent(source).strip())
        if manual:
            cell.metadata.tags = ["manual_run"]
        cells.append(cell)
    md('''
    # Kiểm thử Projector đã train: pipeline và bằng chứng

    Pipeline: **audio 16 kHz → wav2vec2 B đóng băng → Linear/GELU/Linear → Qwen3-0.6B → text**.
    Cascade đối chứng: **audio → B CTC → nhãn lỗi dự đoán → Qwen gốc / LoRA → JSON**.
    Hai nhánh giữ riêng trọng số. Notebook này không train, không gọi API ngoài.

    **Checkpoint hiện có mới hoàn tất alignment transcript, chưa có P1 học phản hồi lỗi.**
    NLL transcript được đo đủ 900 test; sinh transcript và phản hồi dùng pilot cố định cân bằng speaker.
    Run All đọc kết quả đã lưu. Hai cell chạy suy luận/đánh giá có cờ mặc định False.
    ''')
    code('''
    import os, sys, json, subprocess
    from pathlib import Path
    import pandas as pd
    from IPython.display import display, Image, Audio, HTML
    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'scripts/run_projector_validation.py').exists())
    os.chdir(ROOT)
    if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
    from scripts.projector_data import read_json, load
    from scripts.report_projector_validation import report
    settings = read_json(ROOT/'configs/projector_validation.json')
    config = read_json(ROOT/settings['projector_config'])
    parts, manifest = load(config)
    OUTPUT = ROOT/settings['output_dir']
    REPORT = OUTPUT/'report'
    PYTHON = ROOT/('.venv/Scripts/python.exe' if os.name=='nt' else '.venv/bin/python')
    summary = report(settings)
    print('Report:', REPORT/'report.html')
    print('Trạng thái:', summary['status'])
    ''')
    md('''
    ## 1. Dữ liệu đã khóa và tình trạng model

    Split 2.249 train / 450 validation / 900 test, không trùng speaker; test 6 speaker × 150.
    Câu đọc có thể trùng giữa split. Feedback train là nhãn GPT-4o của tác giả; test được tác giả kiểm tra bằng người.
    Không lấy nhãn lỗi gold làm đầu vào Projector hoặc cascade thực tế.
    ''')
    code('''
    display(pd.read_csv(REPORT/'dataset_counts.csv'))
    provenance = summary['projector_alignment']
    display(pd.DataFrame([{
        'stage': provenance['stage'], 'status': provenance['status'],
        'optimizer_steps': provenance['optimizer_steps'], 'validation_transcript_loss': provenance['best_validation_loss'],
        'trainable_projector': provenance['parameter_counts']['projector_trainable'],
        'trainable_qwen': provenance['parameter_counts']['llm_trainable'],
        'trainable_encoder': provenance['parameter_counts']['encoder_trainable'],
        'elapsed_min': provenance['elapsed_sec']/60
    }]))
    print('Feedback stage:', summary['feedback_projector_stage'])
    display(Image(filename=str(REPORT/'data_and_training.png')))
    ''')
    md('''
    ## 2. wav2vec2 A/B: đủ 900 test trên cùng split

    A là model sau source training; B tiếp tục thích nghi giọng Việt. Đây không phải so sánh wav2vec gốc chưa có CTC head.
    F1 tính trên lỗi âm vị theo rubric của project; PER là lỗi nhận dạng chuỗi âm. Không diễn giải F1 thành accuracy.
    ''')
    code('''
    display(pd.DataFrame(summary['audio_full_test']))
    display(Image(filename=str(REPORT/'audio_by_speaker.png')))
    ''')
    md('''
    ## 3. Projector: trước/sau alignment và đối chứng tắt audio

    Random là projector khởi tạo theo seed cố định, **không phải checkpoint step 0 đã lưu của lượt train**.
    Aligned là checkpoint tốt nhất được chọn bằng validation; Muted dùng đúng checkpoint nhưng đặt latent audio bằng 0.
    NLL đo teacher forcing đủ 900 test (nhỏ hơn tốt hơn); không phải phần trăm nhận dạng đúng.
    WER sinh tự do dùng pilot 5 mẫu/speaker = 30, không đưa câu chuẩn vào prompt transcript. WER có thể >100% khi sinh từ thừa.
    Chuẩn hóa case và dấu câu, giữ nguyên từ Unicode để không bỏ qua hallucination bằng ngôn ngữ khác.
    ''')
    code('''
    display(pd.DataFrame(summary['transcript_full_nll']))
    display(summary.get('alignment_delta', {}))
    display(summary.get('audio_ablation_delta', {}))
    display(pd.DataFrame(summary['transcript_generation_pilot']))
    if (REPORT/'projector_transcription.png').exists():
        display(Image(filename=str(REPORT/'projector_transcription.png')))
    ''')
    md('''
    ## 4. Phản hồi lỗi: cùng pilot cân bằng 12 mẫu

    P0 chỉ qua alignment, chưa được train schema/issue/suggestion. B_Q0/B_Q1 dùng lỗi từ B dự đoán.
    Projector + cascade_adapter_transfer ghép hai thành phần đã train, chưa học chung audio + feedback nên không gọi là P1.
    Oracle_Q1 nhận nhãn âm gold, chỉ là chẩn đoán khả năng Qwen; không đại diện pipeline triển khai.
    Cùng greedy decoding, tối đa 512 token; output thiếu EOS, JSON lỗi hoặc schema sai vẫn nằm trong mẫu số.
    Word-key F1 và ROUGE-L là độ khớp nhãn từ/lời khuyên, khác F1 âm vị; chưa đánh giá chất lượng sư phạm bằng người.
    ''')
    code('''
    display(pd.DataFrame(summary['feedback_generation_pilot']))
    display(summary.get('cascade_qwen_pilot_delta', {}))
    if (REPORT/'feedback_pilot.png').exists():
        display(Image(filename=str(REPORT/'feedback_pilot.png')))
    if (REPORT/'per_speaker.csv').exists():
        display(pd.read_csv(REPORT/'per_speaker.csv'))
    ''')
    md('''
    ## 5. Audio thật, raw output và lỗi quan sát được

    Kiểm tra một bản thu cố định/speaker: mã hóa từ WAV phải trùng chính xác cache đã khóa trước khi đưa vào Projector.
    Ví dụ dưới đây theo thứ tự test đã khóa, không chọn theo model thắng/thua. Có thể nghe audio và xem trực tiếp output.
    ''')
    code('''
    examples = read_json(REPORT/'examples.json')
    display(summary.get('raw_audio_parity', {'status':'pending'}))
    feedback_examples = [r for r in examples if r['case'].startswith('feedback_')]
    ids = list(dict.fromkeys(r['utterance_id'] for r in feedback_examples))
    if ids:
        uid = ids[0]
        source = next(r for r in parts['test'] if r['utterance_id']==uid)
        print(uid, source['reference_text'])
        display(Audio(filename=str(ROOT/source['audio_path'])))
        print('Published reference:')
        display(source['feedback'])
        for row in feedback_examples:
            if row['utterance_id']==uid:
                print(row['case'], 'JSON:', row['valid_json'], 'EOS:', row['finished_with_eos'])
                print(row.get('validation_error') or '')
                print(row['raw_text'])
    ''')
    md('''
    ## 6. Giới hạn và cách diễn giải

    Alignment cải thiện khả năng dùng âm thanh nếu NLL giảm và đối chứng muted kém hơn.
    Điều này chưa chứng minh model sửa phát âm chính xác: cần stage feedback và benchmark đủ 900 output trước/sau trên cùng protocol.
    Pilot là phép kiểm tra ban đầu để phát hiện lỗi pipeline; chưa dùng để chọn checkpoint hoặc chỉnh tham số.
    ''')
    code('''
    for item in summary['limitations']: print('•', item)
    print('Execution:', summary.get('execution', 'Chưa hoàn tất'))
    print('HTML:', REPORT/'report.html')
    print('Raw output:', OUTPUT)
    ''')
    md('''
    ## 7. Tự chạy lại pipeline kiểm thử

    Chỉ đánh giá các checkpoint hiện có; không có lệnh train. Resume chỉ chấp nhận cùng hash dữ liệu, trọng số và code.
    Để đo generation đủ 900 mẫu: tạo config mới, output_dir mới, đổi cả hai per_speaker thành 150 trước khi chạy.
    Không thay đổi config đã dùng cho lượt hiện tại. Lượt 900 generation sẽ lâu hơn pilot.
    ''')
    code('''
    RUN_EVALUATION = False
    if RUN_EVALUATION:
        subprocess.run([str(PYTHON), '-X', 'utf8', '-u', 'scripts/run_projector_validation.py', '--phase', 'all'], cwd=ROOT, check=True)
    else:
        print(str(PYTHON), '-X utf8 -u scripts/run_projector_validation.py --phase all')
        print('Mặc định chỉ in lệnh. Log: '+str(OUTPUT/'checkpoint_benchmark.log'))
    ''', manual=True)
    md('''
    ## 8. Thử một WAV riêng qua toàn bộ pipeline

    Mặc định dùng một audio test để minh họa transcription. Đổi AUDIO_FILE sang file của bạn.
    Với `--task feedback`, cần thêm `--reference` và câu chuẩn; checkpoint alignment sẽ ghi rõ chưa học feedback.
    ''')
    code('''
    RUN_AUDIO_INFERENCE = False
    USE_CASCADE_ADAPTER = False  # True: thử ghép Projector + LoRA cascade; không phải P1.
    AUDIO_FILE = ROOT/parts['test'][0]['audio_path']
    if RUN_AUDIO_INFERENCE:
        command = [str(PYTHON), '-X', 'utf8']
        if USE_CASCADE_ADAPTER:
            command += ['scripts/projector_composed_inference.py', '--audio', str(AUDIO_FILE),
                        '--reference', parts['test'][0]['reference_text'], '--output', str(OUTPUT/'my_audio_result.json')]
        else:
            command += ['scripts/projector_inference.py', '--audio', str(AUDIO_FILE), '--task', 'alignment',
                        '--max-new-tokens', '128', '--output', str(OUTPUT/'my_audio_result.json')]
        subprocess.run(command, cwd=ROOT, check=True)
        display(read_json(OUTPUT/'my_audio_result.json'))
    else:
        print('Đổi RUN_AUDIO_INFERENCE=True khi muốn tự chạy thử WAV.')
    ''', manual=True)
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "MDD (.venv)", "language": "python", "name": "mdd-local"},
                                                           "language_info": {"name": "python"}})
    path = ROOT / "notebooks/mdd_projector_validation.ipynb"
    NotebookClient(notebook, timeout=180, kernel_name="mdd-local", resources={"metadata": {"path": str(ROOT)}},
                   skip_cells_with_tag="manual_run").execute()
    nbformat.write(notebook, path)
    print(path)


if __name__ == "__main__":
    build()
