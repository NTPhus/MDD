"""Build an executed casebook from saved audio/Qwen checks; no new inference."""
from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import nbformat

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'notebooks/mdd_chat_error_checks.ipynb'


def build():
    cells = []
    def md(source):
        cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip()))
    def code(source):
        cells.append(nbformat.v4.new_code_cell(textwrap.dedent(source).strip()))

    md('''
    # Kiểm thử audio có lỗi và khôi phục hội thoại

    Bốn bản thu validation đã chạy lại qua **ASR + checkpoint B trên CPU**; có cả từ sai, lỗi bị bỏ sót và một bản được gán nhãn hoàn toàn đúng.
    Nhãn thủ công chỉ dùng đối chiếu **sau inference**. Các ví dụ được chọn để kiểm tra lỗi cụ thể, không phải test set mới.

    Gợi ý màu vàng là khác biệt do model dự đoán, **chưa phải kết luận lỗi**. Hiệu chỉnh ngưỡng chưa đạt yêu cầu nên không gọi `uncertain` là phát âm đúng.
    Notebook đọc kết quả đã lưu, không train, không gọi model/API khi Run All. Kiểm thử Qwen thật được ghi rõ là nhận transcript fixture; kiểm thử web dùng audio thật được lưu riêng.
    ''')
    code('''
    import json, sys
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display, Audio, HTML
    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
                if (p/'scripts/check_pronunciation_error_examples.py').exists())
    OUT = ROOT/'data/generated/pronunciation-error-qa'
    load = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
    report = load(OUT/'report.json')
    actual = load(OUT/'real-audio.json')
    cases = actual['examples']
    assert len(cases) == 4 and actual['no_training'] and not actual['test_900_used']
    print('Fresh audio checks:', len(cases))
    print('Checkpoint SHA256:', actual['identity']['checkpoint_sha256'])
    print('Calibration:', report['calibration'])
    ''')
    md('''
    ## 1. Nghe bản thu và đối chiếu từng từ

    `manual_gold_error` là nhãn thủ công; `candidate_events` là dự đoán độc lập của model.
    TP = gợi ý trúng từ có lỗi, FP = báo nhầm từ đúng, FN = từ có lỗi bị bỏ sót.
    Mốc âm của model là xấp xỉ; không dùng timestamp thủ công để tạo gợi ý.
    ''')
    code('''
    summary = []
    for case in cases:
        print()
        print(case['utterance_id'] + ' · ' + case['case'])
        print('Reference:', case['reference_text'])
        print('ASR:', case['transcript'])
        print('Model suggests:', ', '.join(case['candidate_words']) or '(none; not a pronunciation pass)')
        display(Audio(filename=str(ROOT/case['audio_path'])))
        frame = pd.DataFrame(case['words'])
        display(frame[['text','manual_gold_error','manual_events','candidate_events',
                       'posterior_status','score_eligible','abstention_reason']])
        metrics = case['candidate_word_metrics']
        summary.append({'audio': case['utterance_id'], 'case': case['case'],
                        **{key: metrics[key] for key in ['tp','fp','tn','fn']}})
    checks = pd.DataFrame(summary)
    display(checks)
    ''')
    code('''
    ax = checks.set_index('audio')[['tp','fp','fn']].plot.bar(
        figsize=(10,3.8), color=['#4c8265','#c38c49','#947190'])
    ax.set_title('4 audio minh họa: gợi ý trúng / báo nhầm / bỏ sót')
    ax.set_ylabel('Số từ'); ax.set_xlabel('Audio validation')
    plt.xticks(rotation=15); plt.tight_layout(); plt.show()
    print('The fully correct control still has false alarms. No candidate is a claim of 100% correctness.')
    ''')
    md('''
    ## 2. Kiểm tra trên toàn bộ 450 validation

    Đây là thống kê khác biệt âm theo từ điển và decoder, không phải điểm train mới hay calibration đã đạt.
    Biến thể hợp lệ giảm một phần báo nhầm nhưng cũng bỏ sót thêm lỗi. PNV chỉ độc lập với bước chọn ngưỡng, từng được dùng chọn checkpoint B.
    ''')
    code('''
    display(pd.DataFrame(report['validation_candidate_metrics']).T)
    print('Exhaustive fit-only check:', report.get('exhaustive_fit_only'))
    print('Official 900 test was not used to choose thresholds or run these checks.')
    ''')
    md('''
    ### Gợi ý ngay theo ASR và sau khi sửa text

    Replay dưới đây dùng posterior đã lưu của cùng 4 bản thu, không chạy lại model.
    Cột ASR là gợi ý tạm tính xuất hiện ngay, không ép người dùng xác nhận trước.
    `frout` không có tham chiếu từ điển nên không thể đoán rằng nó phải là `fraud`; khi sửa đúng từ thì mới có gợi ý âm tương ứng.
    Bản thu đúng vẫn có báo nhầm: việc hiện gợi ý ngay không phải bằng chứng độ chính xác đã tăng.
    ''')
    code('''
    replay = load(ROOT/'data/generated/optional-reference-qa/diagnostics.json')
    display(pd.DataFrame([{'audio': item['utterance_id'],
                          'ASR': item['initial']['transcript'],
                          'initial_status': item['initial']['sound_candidates']['status'],
                          'initial_suggestions': ', '.join(w['text'] for w in item['initial']['sound_candidates']['words']),
                          'confirmed_text': item['confirmed']['reference_text'],
                          'confirmed_suggestions': ', '.join(w['text'] for w in item['confirmed']['sound_candidates']['words'])}
                         for item in replay['examples']]))
    ''')
    md('''
    ## 3. Lỗi bot chỉ nói Hi và các lượt Qwen thật

    Lượt được báo lỗi thực sự có output backend `Hi!`, không phải trình duyệt cắt mất câu.
    Các lượt Qwen dưới đây dùng transcript và history cố định, không giả làm kiểm thử nhận dạng audio.
    Giữ nguyên cả các câu hỏi lại thông tin đã biết để thấy giới hạn của Qwen3-0.6B.
    ''')
    code('''
    before = load(ROOT/'data/generated/chat-recovery-qa/reported-case.json')
    print('Reported ASR:', before['asr_transcript'])
    print('Reported reply:', before['text'])
    conversation = load(ROOT/'data/generated/chat-recovery-qa/live-fewshot.json')
    display(pd.DataFrame([{'case': case['label'], 'transcript_fixture': case['transcript_fixture'],
                          'actual_Qwen_reply': case['visible_output']} for case in conversation['cases']]))
    print('Qualitative review:', conversation.get('qualitative_assessment'))
    for limit in conversation.get('limitations', []): print('•', limit)
    ''')
    md('''
    ## 4. Đối chiếu lại: thay câu bot, giữ bản thu

    Kiểm thử web thật gửi audio local qua ASR + B + Qwen. Gợi ý âm xuất hiện ngay theo transcript ASR, chưa phải kết luận lỗi.
    **Đối chiếu** chỉ sinh lại khi các từ thay đổi. Giữ nguyên từ hoặc chỉ sửa chữ hoa/dấu câu không xóa câu trả lời và voice.
    Khi các từ thay đổi, bot xin lỗi một lần rồi trả lời tiếp theo text mới, cập nhật lịch sử cho lượt tiếp theo.
    Transcript ASR gốc, audio và phoneme vẫn giữ nguyên. Chỉ Qwen chạy lại; không gọi lại ASR/B, không yêu cầu ai33/LLM API.
    Gợi ý sửa ngữ pháp vẫn là tùy chọn, không tự thay câu tham chiếu.
    ''')
    code('''
    REGEN = ROOT/'data/generated/optional-reference-qa'
    pipeline = load(REGEN/'live-pipeline.json')
    assert pipeline['status'] == 'passed' and not pipeline['training'] and not pipeline['paid_api_calls']
    initial = pipeline['initial']
    assert initial['speech']['sound_candidates']['provisional'] is True
    rows = [{'revision': 0, 'conversation_text': initial['speech']['transcript'],
             'reply': initial['text'], 'candidate_words': '(awaiting confirmation)',
             'stages': ', '.join(pipeline['initial_stages'])}]
    for revision, item in enumerate(pipeline['revisions'], 1):
        accepted, final = item['accepted'], item['final']
        assert accepted['text'] == '' and accepted['status'] == 'generating'
        assert final['id'] == initial['id'] and final['metadata']['reference_revision'] == revision
        assert final['speech']['transcript'] == initial['speech']['transcript']
        assert final['speech']['phonemes'] == initial['speech']['phonemes']
        assert not {'transcribing','analyzing'} & set(item['stages'])
        rows.append({'revision': revision, 'conversation_text': final['speech']['conversation_text'],
                     'reply': final['text'],
                     'candidate_words': ', '.join(w['text'] for w in final['speech']['sound_candidates']['words']),
                     'stages': ', '.join(item['stages'])})
    display(pd.DataFrame(rows))
    unchanged = pipeline['unchanged_checks']
    for item in unchanged:
        assert item['response']['reference_update'] == {'changed': False, 'regenerated_reply': False}
    display(pd.DataFrame([{'check': item['phase'], 'submitted_text': item['text'],
                          'reply_unchanged': True, 'revision': item['response']['metadata']['reference_revision']}
                         for item in unchanged]))
    print('Initial provisional words:', [w['text'] for w in initial['speech']['sound_candidates']['words']])
    for limit in pipeline.get('limits', []): print('•', limit)
    ''')
    md('''
    ## 5. Lời xin lỗi, câu Qwen thật và giới hạn

    Lời xin lỗi cố định do ứng dụng thêm, không ghi nhận thành đóng góp do model sinh.
    Nội dung Qwen được kiểm tra lặp câu trước khi stream ra web/voice; câu lặp bị giữ lại, retry có giới hạn.
    Đây là kiểm thử chức năng, chưa chứng minh hội thoại tự nhiên hay bộ lọc nhận biết mọi cách diễn đạt lại cùng ý.
    Bảng lịch sử bên dưới giữ cả thất bại Qwen trong lượt kiểm tra trước khi thêm lời xin lỗi.
    ''')
    code('''
    for item in pipeline['revisions']:
        final = item['final']
        print('Application apology:', final['metadata'].get('reference_apology'))
        print('Visible reply:', final['text'])
        display(pd.DataFrame(final['metadata']['reply_attempts']))
    first_live = load(REGEN/'first-live-stock-praise.json')
    print('First live run (preserved, off-topic phrasing praise):', first_live['revisions'][0]['final']['text'])
    print('Passing the request/evidence checks does not certify a useful conversational answer.')
    qwen = load(ROOT/'data/generated/reference-regeneration-qa/real-qwen.json')
    display(pd.DataFrame([{'case': case['label'], 'visible_output': case['visible_output'],
                          'quality_review': case['qualitative_review']['status'],
                          'notes': case['qualitative_review']['notes']} for case in qwen['cases']]))
    print('Actual model generations:', len(qwen['calls']))
    print('Request/evidence checks:', qwen['review_summary']['request_evidence_checks_all_passed'])
    print('Natural response failures:', qwen['review_summary']['natural_response_failed_cases'])
    for limit in qwen['limitations']: print('•', limit)
    ''')
    md('''
    ## 6. Kiểm tra giao diện và lệnh chạy lại

    Browser test dùng job/voice giả lập để kiểm tra stream, đổi revision, lịch sử, hủy và chặn voice cũ.
    Kiểm tra thêm gợi ý âm xuất hiện ngay, text không đổi giữ câu cũ, và mở collapse/người dùng cuộn lên không bị kéo xuống khi stream/highlight.
    Đây là kiểm thử tích hợp giao diện, không phải đánh giá chất lượng model hay nghe loa thực tế.
    Từ chưa có trong từ điển vẫn dùng được cho hội thoại; phần âm báo rõ chưa có tham chiếu, không gọi là đúng.
    ''')
    code('''
    browser = load(REGEN/'browser-summary.json')
    assert browser['status'] == 'passed' and not browser['exceptions']
    display(pd.DataFrame({'browser_check': browser['checks']}))
    print('Physical audio limitation:', browser['speech'])
    print('Replay commands (not executed):')
    print(f'"{sys.executable}" scripts/check_pronunciation_error_examples.py --real-audio')
    print(f'"{sys.executable}" scripts/check_reference_regeneration_live.py')
    print('node scripts/check_reference_regeneration_browser.cjs')
    print('Local audio report:', OUT/'report.html')
    ''')
    md('''
    ## 7. Ngữ pháp: sửa wakes → wake nhưng bộ lọc từng loại mất kết quả

    Ba câu phát triển được thử bằng **Qwen3-0.6B base thật**, cùng prompt và greedy decoding trước/sau.
    Model đã sửa lỗi ở câu người dùng cung cấp; bộ lọc cũ lại loại vì lược đại từ lặp sau `then` hoặc đổi `am` thành `a.m.`.
    Thay đổi nằm ở bộ lọc và hiển thị từng đề xuất độc lập, không train hoặc đổi trọng số/prompt.
    Đây là regression có chủ đích, không phải benchmark ngữ pháp trên dữ liệu chưa thấy.
    ''')
    code('''
    GRAMMAR = ROOT/'data/generated/grammar-routine-qa'
    before_writing, after_writing = load(GRAMMAR/'before.json'), load(GRAMMAR/'after.json')
    comparison_writing = load(GRAMMAR/'comparison.json')
    old_by_label = {item['label']: item for item in before_writing['cases']}
    writing_rows = []
    for item in after_writing['cases']:
        previous = old_by_label[item['label']]
        assert item['request'] == previous['request']
        parsed = item['parsed']
        writing_rows.append({'case': item['label'], 'input': item['transcript'],
                             'before_parser': previous['parsed']['status'], 'after_parser': parsed['status'],
                             'grammar': parsed.get('grammar_corrected'), 'natural': parsed.get('natural'),
                             'edits': parsed.get('explanation')})
    writing_table = pd.DataFrame(writing_rows)
    display(writing_table)
    print('Causal check:', comparison_writing['method'])
    print('Fresh structured calls before / after:', before_writing['actual_generations'], after_writing['actual_generations'])
    for limit in after_writing['limitations']: print('•', limit)
    ''')
    code('''
    accepted = pd.Series({'Before parser fix': sum(row['before_parser'] == 'ready' for row in writing_rows),
                          'After parser fix': sum(row['after_parser'] == 'ready' for row in writing_rows)})
    ax = accepted.plot.bar(figsize=(6,3), color=['#bfa787','#6d8d64'], rot=0)
    ax.set_ylim(0, len(writing_rows) + .5)
    ax.set_ylabel('Output được phép hiển thị / 3 câu phát triển')
    ax.set_title('Chấp nhận output hợp lệ, không phải độ chính xác ngữ pháp')
    plt.tight_layout(); plt.show()
    ''')
    md('''
    ## 8. Câu nhiều lỗi và thứ tự collapse → bot trả lời

    Câu `Yesterday I wake up at 7, I no brush teeth...` được thử trước/sau với Qwen thật.
    Bộ lọc mới hỗ trợ các nhóm chỉnh sửa ngữ pháp có giới hạn: động từ bất quy tắc, phủ định của người học,
    khôi phục cùng chủ thể và từ chức năng tại một số cấu trúc. Chủ thể khác, sở hữu có sẵn, giờ và phủ định vẫn được bảo vệ.
    Đây là đề xuất viết lại, không thay bản thu, transcript, lịch sử hoặc tham chiếu phát âm.

    **Giới hạn thực tế:** prompt mới yêu cầu sửa câu nối bằng dấu phẩy, nhưng Qwen0.6B vẫn giữ comma splice ở ví dụ này.
    Output khả dụng không có nghĩa đã sửa ngữ pháp hoàn hảo. Không chọn lại lượt sinh để che lỗi.
    ''')
    code('''
    PAST = ROOT/'data/generated/grammar-past-qa'
    past_before, past_after = load(PAST/'before.json'), load(PAST/'after.json')
    old_past = {item['label']: item for item in past_before['cases']}
    display(pd.DataFrame([{'case': item['label'], 'input': item['transcript'],
                          'before': old_past[item['label']]['parsed']['status'],
                          'after': item.get('final_parsed', item['parsed'])['status'],
                          'grammar': item.get('final_parsed', item['parsed'])['grammar_corrected'],
                          'natural': item.get('final_parsed', item['parsed'])['natural']}
                         for item in past_after['cases']]))
    print('Actual calls before / after:', past_before['actual_generations'], past_after['actual_generations'])
    print('The model still leaves comma splices in the reported example; these are draft suggestions.')
    ''')
    code('''
    order = load(PAST/'live-order.json')
    assert order['status'] == 'passed'
    timing = order['feedback_first']
    assert timing['feedback_ready_sec'] <= timing['first_text_sec']
    display(pd.DataFrame([{'observed_event': 'Collapse feedback ready', 'seconds': timing['feedback_ready_sec']},
                          {'observed_event': 'First bot text', 'seconds': timing['first_text_sec']}]))
    print('Checks use one real local audio; polling timing is approximate. No TTS was requested.')
    print('Replay command (not executed):')
    print(f'"{sys.executable}" scripts/check_reference_regeneration_live.py --feedback-first-only '
          '--output data/generated/grammar-past-qa/live-order.json')
    ''')
    return nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'},
        'scope': 'Saved genuine inference checks. Run All performs no model inference, API calls or training.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    notebook = build()
    if args.execute:
        from nbclient import NotebookClient
        notebook = NotebookClient(notebook, timeout=180, kernel_name='python3',
                                  resources={'metadata': {'path': str(ROOT)}}).execute()
    nbformat.write(notebook, TARGET)
    print(TARGET)
    if args.execute:
        from nbconvert import HTMLExporter
        html, _ = HTMLExporter().from_notebook_node(notebook)
        path = ROOT/'data/generated/pronunciation-error-qa/mdd_chat_error_checks.html'
        path.write_text(html, encoding='utf-8')
        print(path)


if __name__ == '__main__':
    main()
