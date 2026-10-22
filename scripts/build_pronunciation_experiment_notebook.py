"""Build the calibration demo notebook; execute saved-result cells only on request."""
from __future__ import annotations

import argparse
from pathlib import Path
import textwrap
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/mdd_pronunciation_calibration.ipynb'


def build():
    cells = []

    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip()))

    def code(text, manual=False):
        cell = nbformat.v4.new_code_cell(textwrap.dedent(text).strip())
        if manual:
            cell.metadata.tags = ['manual_inference']
        cells.append(cell)

    md('''
    # Hiệu chỉnh đánh giá phát âm trên checkpoint B đã đóng băng

    **Không train thêm wav2vec2/Qwen.** Audio → posterior CTC của B → so sánh các cách phát âm CMU đã công bố → score theo từ → đúng / cần luyện / chưa đủ bằng chứng.
    Đây là score âm học tương đối, **không phải xác suất người học phát âm sai**. Không đưa audio, nhãn gold hoặc kết quả vào Qwen.

    Pilot được khóa **trước suy luận**: 10 HJK + 10 LXC để fit ngưỡng; 20 PNV để audit ngưỡng đã khóa.
    PNV khác speaker với tập fit nhưng cả 450 validation từng dùng chọn checkpoint B; chưa phải kết quả test độc lập.
    Run All đọc kết quả đã lưu. Các cell suy luận/hiệu chỉnh có cờ mặc định **False**. Test 900 và SpeechOcean không tự chạy.
    ''')
    code('''
    import os, sys, json, subprocess
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display, Audio, Image, HTML
    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
                if (p/'scripts/experiment_pronunciation_calibration.py').exists())
    os.chdir(ROOT)
    for path in (ROOT, ROOT/'src'):
        if str(path) not in sys.path: sys.path.insert(0, str(path))
    from scripts import experiment_pronunciation_calibration as experiment
    CONFIG = experiment.load_config()
    OUTPUT = ROOT/CONFIG['output_root']/'pilot'
    PYTHON = ROOT/('.venv/Scripts/python.exe' if os.name=='nt' else '.venv/bin/python')
    plan = experiment.prepare(CONFIG, OUTPUT, 'pilot')
    print('Python:', PYTHON)
    print('Checkpoint:', CONFIG['checkpoint'])
    print('Frozen selection:', plan['selection_sha256'])
    print('Mode: calibration only; model weights stay unchanged')
    ''')
    md('''
    ## 1. Audit dữ liệu và split

    Nhãn từ sai lấy từ **annotation âm vị thủ công của L2-ARCTIC**, gồm thay/bỏ/thêm âm. Từ không có lỗi là đối chứng âm tính.
    Không sử dụng feedback GPT của L2-Arctic-plus làm nhãn âm học. Word timing/gold/perceived phones chỉ nằm ở phía chấm điểm sau suy luận.
    Split gốc 2.249 train / 450 validation / 900 test giữ nguyên; pilot lấy mẫu bằng SHA256 ID, không lọc theo model thắng/thua.
    ''')
    code('''
    audit = pd.DataFrame(experiment.data_audit(CONFIG, plan))
    display(audit.groupby(['role','speaker_id']).agg(
        utterances=('utterance_id','count'), seconds=('duration_sec','sum'),
        words=('words','sum'), correct_words=('gold_correct','sum'),
        error_words=('gold_errors','sum'), unavailable_gold=('gold_unavailable','sum')))
    display(pd.DataFrame([plan['split_counts']]))
    print('No overlapping fit / heldout speakers:',
          not set(CONFIG['fit_speakers']) & set(CONFIG['heldout_speakers']))
    print('Test audio/gold used in this pilot:', plan['test_audio_or_gold_used'])
    print('Checkpoint SHA256:', plan['identity']['checkpoint_sha256'])
    ''')
    code('''
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    audit.groupby('speaker_id')['utterance_id'].count().plot.bar(ax=axes[0], color='#397b67')
    axes[0].set_title('Số audio đã khóa / speaker'); axes[0].set_ylabel('Utterances')
    audit.groupby('role')[['gold_correct','gold_errors','gold_unavailable']].sum().plot.bar(
        ax=axes[1], color=['#397b67','#bb7056','#b7beb8'])
    axes[1].set_title('Nhãn từ: đúng, sai, thiếu'); axes[1].set_ylabel('Words')
    fig.tight_layout(); fig.savefig(OUTPUT/'notebook_data_audit.png', dpi=150)
    plt.show()
    ''')
    md('''
    ## 2. Nghe audio và xem câu chuẩn

    Ví dụ đầu tiên theo selection đã khóa, không chọn theo chất lượng output. Câu đọc được cung cấp như intended reference;
    phép thử này chưa đo lỗi ASR trong hội thoại tự do. Metadata của nhãn không đi vào mạng audio hoặc scorer.
    ''')
    code('''
    first = plan['records'][0]
    annotations = experiment.selected_annotations(CONFIG, plan)
    example = annotations[first['utterance_id']]
    print(first['utterance_id'], first['role'])
    print('Reference:', example['prompt_text'])
    display(Audio(filename=str(ROOT/first['audio_path'])))
    print('Manual error events (evaluation labels only):')
    display(pd.DataFrame(example['gold_error_events'])[['kind','expected_phone','observed_phone','word_index']]
            if example['gold_error_events'] else pd.DataFrame())
    ''')
    md('''
    ## 3. Các cách phát âm được chấp nhận trước khi nhận audio

    CMU dictionary đóng băng cung cấp nhiều biến thể. Không đoán từ gần giống, không dùng model âm thanh để tự sửa câu chuẩn.
    Ví dụ có thể có các cách đọc khác nhau của `read`, `wind`, `the`. Chấp nhận biến thể từ điển không đồng nghĩa bao phủ mọi giọng hoặc mọi phát âm trong ngữ cảnh.
    ''')
    code('''
    from mdd.pronunciation_assessment import accepted_variants
    variants = accepted_variants(ROOT, 'the read wind')
    display(pd.DataFrame([{'word': word['text'], 'available': word['available'],
                          'accepted_phones': [' '.join(v['phones']) for v in word['variants']]}
                         for word in variants]))
    ''')
    md('''
    ## 4. Suy luận B trên CPU — thao tác riêng

    B chỉ nhận audio. Scorer sau mạng nhận text chuẩn + CMU + posterior; không nhận nhãn perceived/gold hoặc word timing thủ công.
    Posterior được cache riêng theo ID và kiểm tra hash audio/checkpoint/vocabulary. Chạy lại không cần suy luận lại audio đã cache đúng identity.
    ''')
    code('''
    RUN_INFERENCE = False  # Chủ động đổi True để suy luận pilot 40; không train.
    if RUN_INFERENCE:
        subprocess.run([str(PYTHON), '-u', 'scripts/experiment_pronunciation_calibration.py',
                        '--pilot', '--infer'], cwd=ROOT, check=True)
    else:
        print('Không chạy model trong cell này. Đọc cache/result có sẵn ở:', OUTPUT)
    print('Inference metadata:')
    display(experiment.read_json(OUTPUT/'inference.json') if (OUTPUT/'inference.json').exists()
            else {'status':'pending; run the inference cell manually'})
    ''', manual=True)
    md('''
    ## 5. Fit ngưỡng trên HJK/LXC, khóa rồi audit PNV

    `accept_threshold` thấp → nhận là khớp; `error_threshold` cao → nghi lỗi; ở giữa hoặc score không đủ tin cậy → **uncertain**.
    Ngưỡng fit chỉ nhìn 20 audio HJK/LXC. Điều kiện FPR/FNR/coverage/đủ số quyết định đã đăng ký trong config trước suy luận.
    Nếu không có ngưỡng đạt, hoặc PNV không đạt điều kiện, artifact **ineligible** và demo tiếp tục uncertain; không tự tuyên bố đã hiệu chỉnh thành công.
    ''')
    code('''
    display(CONFIG['threshold_policy'])
    RUN_CALIBRATION = False  # Fit ngưỡng, không cập nhật trọng số neural model.
    if RUN_CALIBRATION:
        subprocess.run([str(PYTHON), 'scripts/experiment_pronunciation_calibration.py',
                        '--pilot', '--calibrate'], cwd=ROOT, check=True)
        subprocess.run([str(PYTHON), 'scripts/experiment_pronunciation_calibration.py',
                        '--pilot', '--report'], cwd=ROOT, check=True)
    artifact = experiment.read_json(OUTPUT/'calibration.json') if (OUTPUT/'calibration.json').exists() else None
    if artifact:
        display({k: artifact[k] for k in ['status','eligible','experimental','accept_threshold','error_threshold']})
        print('Audit failures:', artifact['validation']['failures'])
    else:
        print('Pending: no thresholds have been fitted.')
    ''', manual=True)
    md('''
    ## 6. Đối chứng trước/sau cách chấm, cùng trọng số B

    **Strict:** CTC greedy + một cách đọc CMU đầu tiên + alignment, như baseline cũ.
    **Candidate:** score các biến thể + ngưỡng đã fit, vẫn hiển thị để phân tích nếu artifact không đủ điều kiện.
    **Safe:** chỉ ra quyết định khi artifact đủ điều kiện; nếu không thì abstain.

    Positive = từ có lỗi thủ công. FPR = FP/(FP+TN) trên từ đúng đã ra quyết định; FNR = FN/(TP+FN) trên từ sai đã ra quyết định.
    Với cách gọi vận hành, **false rejection FRR = FPR** (báo sai từ vốn đúng), **false acceptance FAR = FNR** (chấp nhận từ vốn sai).
    Xem thêm `recall_all_errors` và `coverage`: không che các từ bị abstain bằng cách chỉ báo cáo accuracy trên phần dễ.
    ''')
    code('''
    summary = experiment.read_json(OUTPUT/'report.json') if (OUTPUT/'report.json').exists() else None
    if summary:
        columns = ['role','system','tp','fp','tn','fn','correct_abstained','error_abstained',
                   'precision','recall_decided','recall_all_errors','false_positive_rate','false_negative_rate','coverage']
        display(pd.DataFrame(summary['metrics'])[columns])
        display(Image(filename=str(OUTPUT/'heldout_metrics.png')))
    else:
        print('Pending report. Run the manual inference/calibration cells first.')
    ''')
    md('''
    ## 7. Phân bố score và coverage theo speaker

    Score không phải xác suất; không diễn giải `0.8` thành “80% sai”. Uncertain bao gồm OOV, alignment không đủ bằng chứng, insert/delete chưa hỗ trợ và khoảng giữa hai ngưỡng.
    Một speaker PNV chưa đủ chứng minh khả năng tổng quát giữa các giọng; test gốc vẫn để riêng.
    ''')
    code('''
    if (OUTPUT/'score_distributions.png').exists():
        display(Image(filename=str(OUTPUT/'score_distributions.png')))
        display(pd.read_csv(OUTPUT/'per_speaker.csv'))
        words = pd.read_csv(OUTPUT/'words.csv')
        print('Ineligible / uncertain reasons:')
        display(words.loc[~words['score_eligible'].fillna(False),'reason'].value_counts(dropna=False))
    else:
        print('No score distribution yet.')
    if (OUTPUT/'threshold_tradeoff.png').exists():
        display(Image(filename=str(OUTPUT/'threshold_tradeoff.png')))
        print('Mỗi điểm là một cặp ngưỡng trên FIT; hai đường đứt là giới hạn đã đăng ký, không phải ngưỡng được chỉnh từ heldout.')
    ''')
    md('''
    ## 8. Một output đầy đủ theo selection cố định

    Score và lý do abstain giữ nguyên từ scorer. Nhãn đúng/sai chỉ được ghép vào bảng sau đó để chấm.
    Không thay các lỗi model bằng lời giải thích mẫu và không bỏ các từ model không chấm được.
    ''')
    code('''
    if (OUTPUT/'predictions.jsonl').exists():
        predictions = experiment.load_predictions(OUTPUT, plan)
        chosen = next(row for row in predictions if row['role']=='heldout')
        print(chosen['utterance_id'], chosen['reference_text'])
        print('Alignment:', chosen['scoring']['alignment'])
        table = pd.DataFrame(chosen['scoring']['words'])
        table['manual_word_error'] = chosen['gold_word_errors']
        display(table)
        print('Reference variants / target words:')
        display(chosen['scoring'].get('target_words', []))
    else:
        print('No model output saved yet.')
    ''')
    md('''
    ## 9. Pilot và đủ 450 validation, cùng phương pháp và policy đã khóa

    Full validation mở rộng số audio fit/audit theo cùng phân nhóm speaker, không thay công thức score hoặc nới ngưỡng FPR/FNR sau khi xem pilot.
    Đây vẫn là development audit; không gọi 450 validation là test độc lập. Nếu cả hai không có cặp ngưỡng đạt, kết luận là chưa đủ bằng chứng để bật chấm đúng/sai.
    ''')
    code('''
    comparison = []
    for mode in ['pilot','full-validation']:
        path = ROOT/CONFIG['output_root']/mode/'report.json'
        if not path.exists(): continue
        value = experiment.read_json(path)
        strict = next(row for row in value['metrics'] if row['role']=='heldout' and row['system']=='strict_greedy_single_CMU')
        safe = next(row for row in value['metrics'] if row['role']=='heldout' and row['system']=='safe_verified_policy')
        comparison.append({'mode': mode, 'utterances': value['utterances'], 'status': value['artifact_status'],
                           'heldout_gold_correct': strict['gold_correct'], 'heldout_gold_error': strict['gold_error'],
                           'strict_FP': strict['fp'], 'strict_FPR': strict['false_positive_rate'],
                           'strict_recall': strict['recall_all_errors'], 'safe_coverage': safe['coverage'],
                           'failures': value['validation']['failures']})
    display(pd.DataFrame(comparison))
    full = ROOT/CONFIG['output_root']/'full-validation'
    if (full/'heldout_metrics.png').exists():
        display(Image(filename=str(full/'heldout_metrics.png')))
        display(Image(filename=str(full/'threshold_tradeoff.png')) if (full/'threshold_tradeoff.png').exists() else 'Fit search plot pending')
    ''')
    md('''
    ## 10. Chạy đủ validation hoặc test sau khi khóa artifact

    Full validation = 300 HJK/LXC fit + 150 PNV audit. Test = 900 speaker khác, chỉ đánh giá bằng artifact eligible đã khóa, không refit.
    SpeechOcean test không dùng chọn ngưỡng và không được gọi từ notebook này. Các lệnh dưới chỉ được in ra, không tự chạy.
    ''')
    code('''
    print(f'"{PYTHON}" -u scripts/experiment_pronunciation_calibration.py --full-validation --run')
    print(f'"{PYTHON}" -u scripts/experiment_pronunciation_calibration.py --test --run '
          '--frozen-calibration data/generated/pronunciation-calibration-v1/full-validation/calibration.json')
    print('Test is blocked if the artifact is ineligible or checkpoint/CMU/vocabulary/method hashes differ.')
    ''')
    md('''
    ## 11. File kết quả và giới hạn

    `selection.json`, `predictions.jsonl`, `calibration.json`, `metrics.csv`, `words.csv`, `per_speaker.csv`, PNG và `report.html` được lưu riêng.
    Đây là thử nghiệm chấm điểm/hiệu chỉnh trên trọng số B hiện có, **không phải train model mới** và chưa chứng minh mức phát âm đạt chuẩn người bản ngữ.
    ''')
    code('''
    for item in CONFIG['limitations']: print('•', item)
    print('HTML report:', OUTPUT/'report.html')
    print('Calibration artifact:', OUTPUT/'calibration.json')
    print('Notebook HTML export:', OUTPUT/'mdd_pronunciation_calibration.html')
    ''')
    return nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'},
        'calibration_note': 'Manual inference flags default False; no training or API calls.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Execute saved-result cells; manual model flags remain False.')
    args = parser.parse_args()
    notebook = build()
    if args.execute:
        from nbclient import NotebookClient
        notebook = NotebookClient(notebook, timeout=180, kernel_name='python3',
                                  resources={'metadata': {'path': str(ROOT)}}).execute()
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, NOTEBOOK)
    if args.execute:
        from nbconvert import HTMLExporter
        html, _ = HTMLExporter().from_notebook_node(notebook)
        target = ROOT/'data/generated/pronunciation-calibration-v1/pilot/mdd_pronunciation_calibration.html'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding='utf-8')
        print(target)
    print(NOTEBOOK)


if __name__ == '__main__':
    main()
