"""Read-only five-model pronunciation comparison and locked-policy report.

No neural inference, downloads, threshold fitting or production configuration.
Rejects incomplete arms, changed cached controls and mismatched audio identities.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from mdd.metrics import aggregate_scores

DEFAULT = ROOT / 'data/generated/pronunciation-improvement-20261024'
BASELINE = ROOT / 'data/generated/model-comparison-20261024/pronunciation'
MODELS = ('cascade-B', 'timit-xlsr-300m', 'crottc-if', 'charsiu-ctc', 'charsiu-fc10ms')
PANELS = ('research_original_inventory', 'research_full39', 'research_common37')
NAMES = {'cascade-B': 'wav2vec2 B · local', 'timit-xlsr-300m': 'XLS-R 300M · TIMIT',
         'crottc-if': 'CROTTC-IF', 'charsiu-ctc': 'Charsiu · CTC', 'charsiu-fc10ms': 'Charsiu · FC 10 ms'}
SHORT = {'cascade-B': 'B', 'timit-xlsr-300m': 'XLS-R TIMIT', 'crottc-if': 'CROTTC-IF',
         'charsiu-ctc': 'Charsiu CTC', 'charsiu-fc10ms': 'Charsiu FC'}
COLORS = ('#7650ad', '#a2a5b4', '#dbb0c8', '#3e9a98', '#bb8a36')
PANEL_NAMES = {'research_original_inventory': 'Nhãn chuyên gia gốc · chính', 'research_full39': 'Full39 · chuẩn hóa nhãn âm',
               'research_common37': 'Common37 · tác vụ giảm inventory'}
POLICY_NAMES = {'dictionary-variant-review': 'Danh sách xem lại · baseline',
                'error-cutoff-f05': 'Chỉ gắn cờ lỗi', 'selective-two-threshold': 'Hai ngưỡng · có abstain'}
GROUP_NAMES = {'fit': 'Fit · HJK + LXC', 'heldout': 'Audit · PNV', 'test': 'Test · 6 speaker'}
SOURCES = (
    ('Charsiu · ICASSP 2022', 'https://arxiv.org/abs/2110.03876'),
    ('Charsiu · code, model và vocab của tác giả', 'https://github.com/lingjzhu/charsiu'),
    ('Charsiu FC 10 ms · checkpoint', 'https://huggingface.co/charsiu/en_w2v2_fc_10ms'),
    ('XLS-R TIMIT · checkpoint tác giả', 'https://huggingface.co/vitouphy/wav2vec2-xls-r-300m-timit-phoneme'),
    ('CROTTC-IF · bài báo', 'https://arxiv.org/abs/2604.22133'),
    ('CROTTC-IF · checkpoint', 'https://huggingface.co/Haopeng/CROTTC-IF-l2-arctic'),
    ('L2-ARCTIC · corpus gốc', 'https://psi.engr.tamu.edu/l2-arctic-corpus/'),
    ('CTC-GOP · Interspeech 2024', 'https://www.isca-archive.org/interspeech_2024/cao24b_interspeech.html'),
    ('GOPT · bài báo và model của tác giả', 'https://github.com/YuanGongND/gopt'),
    ('SSL-FT-PRON · APSIPA ASC 2024, code tác giả', 'https://github.com/hy310/ssl_finetuning'),
    ('SSL-FT-PRON · WavLM/HuBERT checkpoint', 'https://huggingface.co/haeylee/ssl_ft_pron'),
    ('HMamba · NAACL 2025', 'https://aclanthology.org/2025.naacl-long.98/'),
    ('Phoneme Similarity Modeling · Interspeech 2025', 'https://www.isca-archive.org/interspeech_2025/zhou25h_interspeech.html'),
    ('PSM · project và tài nguyên tác giả', 'https://github.com/Berkeley-Speech-Group/Phonetic-Error-Detection'),
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def policy_label(bundle, name):
    selected = bundle['policy'].get('selected', {})
    no_threshold = name in selected and selected[name].get('thresholds') is None
    return POLICY_NAMES.get(name, name) + (' · không tìm được ngưỡng' if no_threshold else '')


def review_totals(rows, key):
    counts = Counter()
    for row in rows:
        counts.update({k: v for k, v in row[key]['counts'].items() if type(v) is int})
    n, errors = counts['reference_words'], counts['gold_error_words']
    correct = n - errors
    if not (counts['hint_tp'] + counts['hint_fn'] + counts['unavailable_error_words'] == errors and
            counts['hint_fp'] + counts['hint_tn'] + counts['unavailable_correct_words'] == correct and
            counts['hint_tp'] + counts['hint_fn'] + counts['hint_fp'] + counts['hint_tn'] == counts['available_words']):
        raise ValueError('Review denominators do not partition all reference words')
    return {**dict(counts), 'gold_correct_words': correct,
            'precision': ratio(counts['hint_tp'], counts['hint_tp'] + counts['hint_fp']),
            'recall_all_errors': ratio(counts['hint_tp'], errors),
            'recall_available_only': ratio(counts['hint_tp'], counts['hint_tp'] + counts['hint_fn']),
            'false_positive_all_correct': ratio(counts['hint_fp'], correct),
            'false_positive_rate_decided': ratio(counts['hint_fp'], counts['hint_fp'] + counts['hint_tn']),
            'coverage': ratio(counts['available_words'], n)}


def load_bundle(directory=DEFAULT, baseline=BASELINE):
    directory, baseline = Path(directory).resolve(), Path(baseline).resolve()
    cm_path, cr_path = directory / 'charsiu/manifest.json', directory / 'charsiu/results.json'
    cm, cr = read(cm_path), read(cr_path)
    bm, br = read(baseline / 'manifest.json'), read(baseline / 'results.json')
    if cr.get('status') != 'completed' or br.get('status') != 'completed':
        raise ValueError('Wait for completed five-model and cached baseline evaluations')
    if cr.get('manifest_sha256') != canonical_hash(cm) or br.get('manifest_sha256') != canonical_hash(bm):
        raise ValueError('Evaluation manifest identity differs')
    if cm['cases'] != bm['cases']:
        raise ValueError('New models do not use exactly the same cases/audio/annotations as cached controls')
    cases = cm['cases']
    ids = [c['id'] for c in cases]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('Empty or duplicated cases')
    expert = [c for c in cases if c['group'] == 'expert_test']
    if len(expert) != 120 or Counter(c['speaker_id'] for c in expert) != {s: 20 for s in {c['speaker_id'] for c in expert}} or len({c['speaker_id'] for c in expert}) != 6:
        raise ValueError('Expected the registered 120 expert cases, 20 per six speakers')
    for model in MODELS:
        rows = cr['predictions'].get(model, {})
        if set(rows) != set(ids):
            raise ValueError('Incomplete matched model: ' + model)
        if model in MODELS[:3] and rows != br['predictions'][model]:
            raise ValueError('Cached control predictions changed: ' + model)
        for case in cases:
            row = rows[case['id']]
            if row.get('status') != 'completed' or row.get('audio_sha256') != case['audio_sha256']:
                raise ValueError('Incomplete or changed audio/model pair: ' + model + ':' + case['id'])
            if row.get('reference_supplied_to_model') is not False:
                raise ValueError('Unexpected reference-conditioned neural decoder')
            if case['group'] != 'expert_test' and any(row.get(panel) is not None for panel in PANELS):
                raise ValueError('Development audio must not have invented expert metrics')
        for panel in PANELS:
            actual = aggregate_scores(rows[c['id']][panel] for c in expert)
            if actual != cr['summary']['models'][model][panel]:
                raise ValueError('Saved aggregate differs from per-case counters: ' + model + ':' + panel)
    policy_path = directory / 'policies/results.json'
    policy = read(policy_path)
    if policy.get('status') != 'completed':
        raise ValueError('Wait for completed locked-policy evaluation')
    plan_path, locked_path = directory / 'policies/plan.json', directory / 'policies/locked_policy.json'
    plan, locked = read(plan_path), read(locked_path)
    if policy.get('plan_sha256') not in {digest(plan_path), canonical_hash(plan)}:
        raise ValueError('Policy plan identity differs')
    if policy.get('locked_policy_sha256') not in {digest(locked_path), canonical_hash(locked)}:
        raise ValueError('Locked thresholds identity differs')
    if (locked.get('status') != 'locked' or locked.get('test_features_read_before_lock') is not False
            or locked.get('test_threshold_tuning') is not False or locked.get('heldout_threshold_tuning') is not False):
        raise ValueError('Expected a fit-only threshold lock before test evidence')
    if locked.get('validation_features_sha256') != digest(directory / 'policies/validation_features.jsonl'):
        raise ValueError('Validation features changed after thresholds were locked')
    if policy.get('selected') != locked.get('selected'):
        raise ValueError('Reported thresholds differ from the locked thresholds')
    for name, expected in policy.get('feature_file_sha256', {}).items():
        if Path(name).name != name or digest(directory / 'policies' / name) != expected:
            raise ValueError('Policy feature identity differs: ' + name)
    test_identity_path = directory / 'policies/test_cache_identity.json'
    test_identity = read(test_identity_path)
    if test_identity.get('lock_sha256_before_test') != digest(locked_path):
        raise ValueError('Test posterior scoring did not use the unchanged threshold lock')
    for name, eligibility in policy.get('eligibility', {}).items():
        if eligibility.get('test_used_to_decide_eligibility') is not False:
            raise ValueError('Test must not decide policy eligibility: ' + name)
    for group in ('fit', 'heldout', 'test'):
        if group not in policy['groups']:
            raise ValueError('Missing fit/heldout/test policy group: ' + group)
        if policy['groups'][group]['cases'] != {'fit':300,'heldout':150,'test':120}[group]:
            raise ValueError('Policy case cohort differs: ' + group)
        for name, row in policy['groups'][group]['policies'].items():
            if row['tp'] + row['fp'] + row['tn'] + row['fn'] != row['decided_words']:
                raise ValueError('Policy decision matrix includes abstentions: ' + group + ':' + name)
            if row['decided_words'] > row['total_words']:
                raise ValueError('Policy coverage denominator differs')
    paths = [cm_path, cr_path, baseline / 'manifest.json', baseline / 'results.json', policy_path,
             plan_path, locked_path, test_identity_path]
    paths.extend(p for p in (directory / 'policies/validation_features.jsonl', directory / 'policies/test_features.jsonl') if p.exists())
    failure_audit_path = ROOT / 'docs/pronunciation_improvement_audit_20261024.json'
    failure_audit = read(failure_audit_path) if failure_audit_path.exists() else None
    if failure_audit:
        audit_counts = failure_audit['saved_120_case_B_word_review']
        measured = review_totals([cr['predictions']['cascade-B'][c['id']] for c in expert], 'variant_review_full39')
        for audit_key, measured_key in (('reference_words','reference_words'),('expert_error_words','gold_error_words'),('fp','hint_fp'),('tp','hint_tp')):
            if audit_counts[audit_key] != measured[measured_key]:
                raise ValueError('Failure audit refers to a different B word cohort')
        paths.append(failure_audit_path)
    hashes = {os.path.relpath(p, directory).replace('\\', '/'): digest(p) for p in paths}
    return {'directory': directory, 'manifest': cm, 'results': cr, 'policy': policy,
            'plan': plan, 'locked': locked, 'input_sha256': hashes, 'failure_audit':failure_audit}


def model_rows(bundle, *, by_speaker=False):
    expert = [c for c in bundle['manifest']['cases'] if c['group'] == 'expert_test']
    groups = sorted({c['speaker_id'] for c in expert}) if by_speaker else ['all']
    output = []
    for speaker in groups:
        selected = [c for c in expert if speaker == 'all' or c['speaker_id'] == speaker]
        for model in MODELS:
            rows = [bundle['results']['predictions'][model][c['id']] for c in selected]
            for panel in PANELS:
                scores = aggregate_scores(row[panel] for row in rows)
                review = review_totals(rows, 'variant_review_common37' if panel == 'research_common37' else 'variant_review_full39')
                output.append({'speaker': speaker, 'model': model, 'inventory': panel, 'cases': len(rows),
                    **scores['counts'], **scores['metrics'], **{'review_' + k: v for k, v in review.items()},
                    'removed_expert_events': sum(len(r['research_common37']['removed_indistinguishable_expert_events']) for r in rows) if panel == 'research_common37' else 0,
                    'mean_inference_sec': sum(row['inference_sec'] for row in rows) / len(rows),
                    'devices': ', '.join(sorted({str(r.get('runtime', {}).get('device', 'unknown')) for r in rows})),
                    'unknown_prediction_phones': sum(len(r.get('unknown_tokens', [])) for r in rows)})
    return output


def policy_rows(bundle, *, by_speaker=False):
    output = []
    source = bundle['policy']['per_speaker'] if by_speaker else bundle['policy']['groups']
    for label, group in source.items():
        for name, row in group['policies'].items():
            eligibility = bundle['policy'].get('eligibility', {}).get(name, {})
            output.append({'group': group.get('group', label), 'speaker': label if by_speaker else 'all',
                           'policy': name, 'cases': group['cases'], **row,
                           'risk_on_decided': ratio(row['fp'] + row['fn'], row['decided_words']),
                           'eligible': eligibility.get('eligible'),
                           'eligibility_failures': json.dumps(eligibility.get('failures', []), ensure_ascii=False)})
    return output


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def figures(bundle, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'axes.titleweight': 'bold'})
    all_rows = model_rows(bundle)
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    for i, panel in enumerate(PANELS):
        chosen = [r for r in all_rows if r['inventory'] == panel]
        for j, key in enumerate(('f1', 'per')):
            values = [100*r[key] if r[key] is not None else float('nan') for r in chosen]
            bars = axes[i, j].bar([SHORT[r['model']] for r in chosen], values, color=COLORS)
            axes[i, j].bar_label(bars, fmt='%.1f', padding=3)
            axes[i, j].set(title=PANEL_NAMES[panel] + (' · F1 ↑' if key == 'f1' else ' · PER ↓'),
                           ylabel='%' if key == 'f1' else 'Lỗi âm / âm tham chiếu (%)',
                           ylim=(0, max(100 if key == 'f1' else 10, max(values)*1.17)))
    fig.suptitle('Năm checkpoint · cùng 120 audio có nhãn chuyên gia\nGiữ riêng nhãn gốc, Full39 và tác vụ giảm Common37; không trộn các điểm', fontsize=13)
    fig.savefig(output / 'model_matrix.png', dpi=160); plt.close(fig)
    chosen = [r for r in all_rows if r['inventory'] == 'research_full39']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, key, title in zip(axes, ('review_false_positive_all_correct', 'review_recall_all_errors', 'review_coverage'),
                             ('Báo nhầm / mọi từ đúng ↓', 'Bắt lỗi / mọi lỗi từ ↑', 'Có gợi ý / mọi từ ↑')):
        bars = ax.bar([SHORT[r['model']] for r in chosen], [100*r[key] if r[key] is not None else float('nan') for r in chosen], color=COLORS)
        ax.bar_label(bars, fmt='%.1f', padding=3)
        ax.set(title=title, ylabel='%', ylim=(0, 112))
        ax.tick_params(axis='x', labelrotation=18)
    fig.suptitle('Danh sách từ cần xem lại · cùng dictionary-variant reviewer\nKhông bỏ các lỗi bị abstain khỏi mẫu số recall', fontsize=13)
    fig.savefig(output / 'word_review.png', dpi=160); plt.close(fig)
    subgroup = model_rows(bundle, by_speaker=True)
    speakers = sorted({r['speaker'] for r in subgroup})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for index, model in enumerate(MODELS):
        selected = [next(r for r in subgroup if r['model'] == model and r['speaker'] == speaker and r['inventory'] == 'research_full39') for speaker in speakers]
        for ax, key in zip(axes, ('f1', 'review_false_positive_all_correct')):
            ax.bar(np.arange(len(speakers)) + (index-2)*.15,
                   [100*r[key] if r[key] is not None else float('nan') for r in selected], width=.14,
                   label=SHORT[model], color=COLORS[index])
    for ax, title in zip(axes, ('Phone F1 ↑', 'Gợi ý nhầm / mọi từ đúng ↓')):
        ax.set_xticks(range(len(speakers)), [s + ('\nViệt Nam' if s == 'TLV' else '') for s in speakers])
        ax.set(title=title, ylabel='%', ylim=(0, 100))
    axes[0].legend(loc='upper center', ncol=3, fontsize=8, frameon=False)
    fig.suptitle('Mỗi speaker có 20 audio · TLV là một speaker Việt, không đại diện dân số', fontsize=13)
    fig.savefig(output / 'by_speaker.png', dpi=160); plt.close(fig)
    policy = policy_rows(bundle)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    groups = ('fit', 'heldout', 'test')
    names = list(bundle['policy']['groups']['fit']['policies'])
    for i, name in enumerate(names):
        selected = [next(r for r in policy if r['group'] == group and r['policy'] == name) for group in groups]
        for ax, key in zip(axes, ('false_positive_all_correct', 'recall_all_errors', 'coverage')):
            ax.bar(np.arange(3) + (i-(len(names)-1)/2)*.23,
                   [100*r[key] if r.get(key) is not None else float('nan') for r in selected], width=.22,
                   color=COLORS[i], label=policy_label(bundle,name))
    for ax, title in zip(axes, ('Báo nhầm / mọi từ đúng ↓', 'Bắt lỗi / mọi lỗi từ ↑', 'Quyết định / mọi từ')):
        ax.set_xticks(range(3), ['Fit HJK/LXC', 'PNV audit', 'Test 6 speaker'])
        ax.set(title=title, ylabel='%', ylim=(0, 108))
    axes[0].legend(fontsize=8, loc='upper left', frameon=False)
    fig.suptitle('B · cùng trọng số, thay chính sách quyết định\nNgưỡng khóa trước khi đọc posterior và nhãn test; kết quả fail vẫn giữ nguyên', fontsize=13)
    fig.savefig(output / 'locked_policy.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, len(names), figsize=(5*len(names), 4.5), constrained_layout=True, squeeze=False)
    for ax, name in zip(axes.flat, names):
        for i, group in enumerate(groups):
            row = next(r for r in policy if r['group'] == group and r['policy'] == name)
            if row.get('coverage') is not None and row['risk_on_decided'] is not None:
                ax.scatter(100*row['coverage'], 100*row['risk_on_decided'], s=85, color=COLORS[i], label=GROUP_NAMES[group])
        ax.set(title=policy_label(bundle,name), xlabel='Coverage quyết định (%)', ylabel='Lỗi / quyết định đã đưa ra (%)', xlim=(0, 105), ylim=(0, 105))
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, frameon=False)
        else:
            ax.text(.5,.5,'Không có quyết định\nRisk không xác định',ha='center',va='center',transform=ax.transAxes,color='#706978')
    fig.suptitle('Risk–coverage tại các ngưỡng đã khóa\nChỉ gắn cờ lỗi: risk là tỷ lệ cờ sai; hai ngưỡng: gồm cờ sai và chấp nhận sai', fontsize=12)
    fig.savefig(output / 'risk_coverage.png', dpi=160); plt.close(fig)


def esc(value):
    return html.escape(str(value if value is not None else '—'), quote=True)


def number(value, percent=False):
    return '—' if value is None else f'{100*value:.2f}%' if percent else f'{value:.2f}'


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+esc(h)+'</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def picture(output, name):
    return '<img class="chart" alt="Biểu đồ từ kết quả đã lưu" src="data:image/png;base64,' + base64.b64encode((output/name).read_bytes()).decode() + '">'


def findings(bundle):
    rows = [r for r in model_rows(bundle) if r['inventory'] == 'research_original_inventory']
    b = next(r for r in rows if r['model'] == 'cascade-B')
    output = []
    for row in rows[3:]:
        output.append(f'{NAMES[row["model"]]} trên nhãn chuyên gia gốc: phone F1 {number(row["f1"], True)} '
                      f'(B {number(b["f1"], True)}), PER {number(row["per"], True)} '
                      f'(B {number(b["per"], True)}); ở reviewer Full39, gợi ý nhầm / mọi từ đúng '
                      f'{number(row["review_false_positive_all_correct"], True)}, '
                      f'recall mọi lỗi từ {number(row["review_recall_all_errors"], True)} '
                      f'với coverage {number(row["review_coverage"], True)}.')
    for name, eligibility in bundle['policy'].get('eligibility', {}).items():
        selected = bundle['policy'].get('selected',{}).get(name,{})
        if selected.get('thresholds') is None:
            output.append(policy_label(bundle,name)+': không có ngưỡng đạt điều kiện trên fit; tất cả từ ở trạng thái chưa quyết định. FPR = 0 cùng recall = 0 và coverage = 0 không phải cải thiện phát âm.')
            continue
        audit = bundle['policy']['groups']['heldout']['policies'][name]
        output.append(policy_label(bundle,name) + ': ' +
                      ('đạt điều kiện đăng ký trên fit và PNV audit.' if eligibility.get('eligible') is True
                       else 'chưa đạt điều kiện đăng ký; không tự đưa vào demo.') +
                      f' Trên PNV, recall mọi lỗi {number(audit.get("recall_all_errors"),True)}, '
                      f'coverage {number(audit.get("coverage"),True)}, '
                      f'FNR phần quyết định {number(audit.get("false_negative_rate_decided"),True)}. '+
                      ' Chi tiết: ' + json.dumps(eligibility.get('failures', []), ensure_ascii=False))
    test_policies = bundle['policy']['groups']['test']['policies']
    baseline = test_policies['dictionary-variant-review']
    selective = test_policies['selective-two-threshold']
    output.append(
        f'Tradeoff của hai ngưỡng trên test: báo nhầm {baseline["fp"]}/{baseline["gold_correct"]} '
        f'({number(baseline["false_positive_all_correct"],True)}) → '
        f'{selective["fp"]}/{selective["gold_correct"]} '
        f'({number(selective["false_positive_all_correct"],True)}), nhưng recall mọi lỗi giảm '
        f'{baseline["tp"]}/{baseline["gold_error"]} ({number(baseline["recall_all_errors"],True)}) → '
        f'{selective["tp"]}/{selective["gold_error"]} ({number(selective["recall_all_errors"],True)}); '
        f'coverage {number(baseline["coverage"],True)} → '
        f'{selective["decided_words"]}/{selective["total_words"]} ({number(selective["coverage"],True)}). '
        'Ít cờ lỗi hơn đi cùng ít lỗi được bắt hơn; không coi đây là cải thiện tổng thể.')
    return output


def case_cards(bundle):
    parts = []
    for case in bundle['manifest']['cases']:
        annotation = case.get('annotation')
        words = ({w['index']: w['text'] for w in annotation['word_intervals']} if annotation
                 else dict(enumerate(case['reference_text'].split())))
        gold = sorted({event['word_index'] for event in annotation['gold_error_events'] if type(event.get('word_index')) is int and event['word_index'] in words}) if annotation else []
        parts.append('<details class="case" data-search="'+esc(case['id']+' '+case['reference_text'])+'"><summary><b>'+esc(case.get('speaker_id', 'User'))+'</b> · '+esc(case['reference_text'])+'</summary><div class="casebody">')
        if annotation:
            parts.append('<p><b>Từ có lỗi theo chuyên gia:</b> '+esc(', '.join(words[i] for i in gold) or 'Không có từ lỗi được ánh xạ rõ')+'</p>')
        else:
            parts.append('<p class="notice">Audio phát triển của người dùng; không có nhãn phoneme chuyên gia, không tính F1/PER.</p>')
        audio = (ROOT / case['audio']).resolve()
        if audio.is_relative_to(ROOT):
            source = os.path.relpath(audio, bundle['directory']).replace('\\', '/')
            parts.append('<audio controls preload="none" src="'+esc(source)+'"></audio>')
        model_table = []
        for model in MODELS:
            row = bundle['results']['predictions'][model][case['id']]
            review = row['variant_review_full39']
            hints = review.get('hint_word_indices', [])
            model_table.append([NAMES[model], number(row['research_full39']['metrics']['f1'], True) if annotation else 'Không chấm',
                                number(row['research_full39']['metrics']['per'], True) if annotation else 'Không chấm',
                                ', '.join(words.get(i, str(i)) for i in hints) if review['status'] == 'available' else review.get('reason', 'Không có gợi ý'),
                                number(row['inference_sec'])])
        parts.append(table(['Model', 'F1 full39', 'PER full39', 'Từ model gợi ý xem lại', 'Giây'], model_table))
        for model in MODELS:
            row = bundle['results']['predictions'][model][case['id']]
            parts.append('<details><summary>'+esc(NAMES[model])+' · phone và dữ liệu gốc</summary><pre>'+esc(json.dumps({k: row.get(k) for k in ('phonemes','native_decoded_labels','unknown_tokens','runtime','variant_review_full39','variant_review_common37')},ensure_ascii=False,indent=2))+'</pre></details>')
        parts.append('</div></details>')
    return ''.join(parts)


CSS = 'body{margin:0;background:#f8f6fc;color:#262032;font:16px/1.55 Segoe UI,Arial,sans-serif}main{max-width:1240px;margin:auto;padding:32px 22px}header,section{background:white;border:1px solid #e7e0f1;border-radius:16px;padding:24px;margin:0 0 20px}h1{font-size:33px;line-height:1.2;margin:8px 0 14px}h2{font-size:24px;color:#60408e}h3{font-size:19px}nav{display:flex;gap:18px;flex-wrap:wrap;margin:20px 0}a{color:#7047a3}.eyebrow{font-size:12px;letter-spacing:1.3px;color:#7c61a0}.notice{padding:14px 18px;border-left:4px solid #a889cd;background:#f5f1fb}.muted,small{color:#706978}.chart{display:block;width:100%;height:auto;margin:16px 0}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:11px 12px;text-align:left;border-bottom:1px solid #eee7f5;vertical-align:top}th{background:#f4effa;color:#604483;white-space:nowrap}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#f8f6fb;padding:14px;border-radius:9px}.case{border:1px solid #e5deee;border-radius:9px;margin:8px 0}.case>summary{cursor:pointer;padding:13px 16px}.casebody{padding:0 16px 16px}.casebody details{padding:8px 0}audio{max-width:100%;margin:8px 0}input{box-sizing:border-box;width:100%;padding:13px;border:1px solid #d5c7e6;border-radius:9px;font:inherit}.pill{display:inline-block;background:#f0e8fa;color:#674190;border-radius:20px;padding:5px 12px;margin:0 8px 8px 0;font-size:13px}@media(max-width:650px){main{padding:15px 10px}header,section{padding:16px}h1{font-size:27px}table{font-size:12px}}'


def render_html(bundle, output):
    rows = model_rows(bundle)
    policies = policy_rows(bundle)
    chunks = ['<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cải thiện đánh giá phát âm · 24/10/2026</title><style>'+CSS+'</style><body><main><header><span class="eyebrow">PRONUNCIATION · LOCAL EVIDENCE · 24/10/2026</span><h1>Có cải thiện được việc bắt lỗi phát âm?</h1><p>So sánh năm checkpoint trên cùng audio và thử riêng cách ra quyết định của B. Không train thêm, không dùng LLM/TTS để gán nhãn.</p><span class="pill">120 audio chuyên gia × 5 model</span><span class="pill">6 speaker · 20 audio/người</span><span class="pill">3 audio người dùng · không chấm</span></header><nav><a href="#models">Model</a><a href="#policies">Chính sách của B</a><a href="#cases">Nghe và xem từng câu</a><a href="#sources">Nguồn và tái lập</a></nav>']
    chunks.append('<section><h2>Kết quả thực nghiệm</h2>'+''.join('<p>'+esc(s)+'</p>' for s in findings(bundle))+'<p class="notice">Điểm trên subset không chứng minh model tốt hơn với mọi người. Đổi checkpoint và giảm báo nhầm là hai thí nghiệm riêng; policy không đạt điều kiện vẫn được báo cáo.</p></section>')
    chunks.append('<section id="models"><h2>1. Năm checkpoint, cùng 120 audio</h2><p>Ba model B, XLS-R TIMIT và CROTTC-IF tái sử dụng nguyên output đã đóng băng. Hai checkpoint Charsiu chạy mới trên chính các file, checksum và nhãn đó. Charsiu CTC và FC đều dùng backbone wav2vec2-base, không phải hai họ backbone mới. FC dùng frame classification 10 ms; CTC dùng mục tiêu CTC và cách decode khác, nên đây không phải ablation chỉ thay một yếu tố.</p>')
    chunks.append(picture(output, 'model_matrix.png'))
    chunks.append('<p class="notice">Nhãn chuyên gia gốc là bảng chính, giữ ký hiệu distortion. Full39 bỏ stress/ký hiệu distortion và chuẩn hóa AX/AH cho mọi model và reference; các error events chuyên gia vẫn được giữ, nhưng đây là cách chuẩn hóa nhãn khác với task gốc. Common37 còn gộp AO/AA và ZH/SH cho mọi model, reference và nhãn lỗi; một số lỗi trở nên không phân biệt được và được đếm riêng. Không dùng điểm tác vụ giảm inventory để tuyên bố đạt cùng chất lượng ở đầy đủ âm. PER đo nhận dạng âm; F1 đo phát hiện lỗi theo protocol của dự án, không phải độ chính xác ASR hay điểm phần trăm phát âm đúng.</p>')
    chunks.append(table(['Inventory','Model','Audio','Phone F1 ↑','PER ↓','Precision','Recall','Lỗi chuyên gia bị gộp','Giây/audio','Device'], [[PANEL_NAMES[r['inventory']],NAMES[r['model']],r['cases'],number(r['f1'],True),number(r['per'],True),number(r['precision'],True),number(r['recall'],True),r['removed_expert_events'],number(r['mean_inference_sec']),r['devices']] for r in rows]))
    chunks.append('<p class="muted">Thời gian là inference đã lưu, không gồm nạp model. Các lượt đo không đồng thời; decoder, độ phân giải và lượng công việc khác nhau, nên không phải phép đo nhân quả về tốc độ backbone. CROTTC có protocol scoring và split huấn luyện của tác giả khác; không coi đây là tái lập điểm bài báo.</p>')
    chunks.append('<h3>Gợi ý từ: bao nhiêu đúng, bao nhiêu bị báo nhầm?</h3>'+picture(output,'word_review.png'))
    chunks.append(table(['Model','Từ có reference','Gợi ý khả dụng','FPR mọi từ đúng','FPR phần khả dụng','Recall mọi lỗi từ','Recall phần khả dụng','Precision cờ lỗi','Coverage'], [[NAMES[r['model']],r['review_reference_words'],r['review_available_words'],number(r['review_false_positive_all_correct'],True),number(r['review_false_positive_rate_decided'],True),number(r['review_recall_all_errors'],True),number(r['review_recall_available_only'],True),number(r['review_precision'],True),number(r['review_coverage'],True)] for r in rows if r['inventory']=='research_full39']))
    chunks.append('<p class="muted">Cùng dictionary-variant reviewer cho cả năm model, khác với policy posterior đang dùng trong demo. Không có reference hoặc gặp nhãn âm chưa hỗ trợ thì abstain; các lỗi bị abstain vẫn nằm trong mẫu số recall toàn bộ. Nhãn lỗi từ ở panel này chỉ dùng word_index chuyên gia hợp lệ; các insertion không có chỉ số từ vẫn giữ trong phone metrics. Không trộn mẫu số này với word metrics suy từ canonical phone positions.</p>')
    chunks.append(picture(output,'by_speaker.png')+'<p class="notice">TLV là duy nhất một speaker Việt trong sáu speaker test của lượt này, với 20 audio. Không suy ra chất lượng chung cho giọng Việt từ một speaker.</p></section>')
    chunks.append('<section id="policies"><h2>2. Giữ model B, thử thay cách đưa ra kết quả</h2><p>Fit dùng HJK/LXC; PNV là speaker audit riêng. Mọi ngưỡng được khóa trước khi đọc posterior/nhãn test 120 audio. Đây là hai policy đăng ký trước và một baseline danh sách xem lại; không fine-tune trọng số và không chọn ngưỡng trên test.</p>'+picture(output,'locked_policy.png'))
    chunks.append(table(['Nhóm','Policy','Từ','Quyết định','Cờ lỗi','FPR mọi từ đúng','Recall mọi lỗi','Coverage','Precision cờ','Risk trên quyết định'], [[GROUP_NAMES.get(r['group'],r['group']),policy_label(bundle,r['policy']),r['total_words'],r['decided_words'],r['flagged_words'],number(r.get('false_positive_all_correct'),True),number(r.get('recall_all_errors'),True),number(r.get('coverage'),True),number(r.get('precision'),True),number(r['risk_on_decided'],True)] for r in policies]))
    chunks.append(picture(output,'risk_coverage.png')+'<p class="notice">Policy chỉ gắn cờ lỗi không gọi các từ còn lại là đúng: chúng là chưa quyết định. Vì vậy risk của policy này là tỷ lệ cờ sai; risk của hai ngưỡng còn gồm cả chấp nhận sai. Xem đồng thời FPR, recall toàn bộ và coverage, không chọn model/policy chỉ vì ít cờ lỗi.</p>')
    chunks.append('<h3>Điều kiện đạt và ngưỡng đã khóa</h3><pre>'+esc(json.dumps({'selected':bundle['policy'].get('selected'), 'eligibility':bundle['policy'].get('eligibility'), 'definitions':bundle['policy'].get('definitions')},ensure_ascii=False,indent=2))+'</pre><details><summary>Giới hạn và đăng ký thí nghiệm</summary><pre>'+esc(json.dumps({'limitations':bundle['policy'].get('limitations'), 'plan':bundle['plan'], 'locked_policy':bundle['locked']},ensure_ascii=False,indent=2))+'</pre></details><p class="muted">HJK/LXC/PNV là validation của model B trước đây; PNV là heldout cho bước chọn ngưỡng, không hoàn toàn độc lập với việc chọn checkpoint B. Subset test đã được quan sát ở lượt benchmark trước, nên đây là kiểm tra tiếp theo trên dữ liệu cố định, không phải test mù mới.</p></section>')
    chunks.append('<section><h2>Cải thiện tiếp theo</h2><p>Không thể chỉ hạ ngưỡng để vừa bắt nhiều lỗi vừa ít báo nhầm với feature hiện tại: một policy không tìm được cutoff đạt điều kiện, policy còn lại giảm báo nhầm bằng cách bỏ qua phần lớn lỗi. Bước tiếp theo là sửa việc ánh xạ từ/reference rồi đánh giá feature âm học hỗ trợ cả chèn và thiếu âm; margin hiện tại chưa chấm được các cấu trúc đó.</p>')
    reasons=[]
    for group in ('fit','heldout','test'):
        value=bundle['policy']['groups'][group]['policies']['dictionary-variant-review']
        for reason,count in value.get('score_ineligible_reasons',{}).items():
            reasons.append([GROUP_NAMES[group],reason,count])
    chunks.append(table(['Nhóm','Vì sao chưa có score','Số từ'],reasons))
    chunks.append('<p class="muted">Các lý do không có score không đồng nghĩa toàn bộ là lỗi encoder. Kiểm tra validation cho thấy 33 từ thuộc ba bản đọc “God bless \'em…” bị mất score khi prompt giữ dấu nháy đầu nhưng annotation ghi “em”; 19 từ khác bị lệch số từ. Cần sửa bước mapping và đăng ký một phiên bản mới, không refit các ngưỡng vừa báo cáo trên test.</p>')
    audit=bundle.get('failure_audit')
    if audit:
        details=audit['saved_120_case_B_word_review']
        dictionary=details['false_positive_words_with_manual_canonical_sequence_absent_from_all_CMU_variants']['count']
        fp_dh=next(row['count'] for row in details['frequent_events_inside_false_positive_words'] if row.get('expected')=='DH' and row.get('predicted')=='D')
        missed_dh=next(row['count'] for row in details['frequent_expert_events_inside_false_negative_words'] if row.get('expected')=='DH' and row.get('observed')=='D')
        link=os.path.relpath(ROOT/'docs/pronunciation_improvement_audit_20261024.json',output).replace('\\','/')
        chunks.append('<p>Trong audit B cùng 120 audio, chỉ '+str(dictionary)+'/'+str(details['fp'])+' từ bị báo nhầm có chuỗi canonical chuyên gia nằm ngoài mọi CMU variant đang có. Đây không chứng minh các lỗi còn lại chỉ do encoder, nhưng cho thấy thêm dictionary đơn thuần chưa xử lý hết vấn đề. DH→D xuất hiện ở '+str(fp_dh)+' event trong từ báo nhầm và '+str(missed_dh)+' event trong từ có lỗi bị bỏ sót: không nên gộp mặc định hai âm thành luôn đúng. <a href="'+esc(link)+'">Xem audit các lỗi</a>.</p>')
    chunks.append('<p>Nên benchmark công thức posterior/CTC-GOP đúng mô tả của tác giả và kiểm thử toán trên ví dụ nhỏ trước khi calibration; đây là hướng tiếp theo, chưa được triển khai hay bảo đảm tốt hơn trong lượt này. Nếu cần fine-tune tiếp, dùng audio người Việt nói tiếng Anh có nhãn chuyên gia công bố, kèm mẫu phát âm đúng và lỗi để đo cả báo nhầm lẫn bỏ sót; tách speaker train/validation/test và chỉ chọn ngưỡng trên validation. Bạn chủ động chạy training sau khi protocol và dữ liệu đã chuẩn bị.</p></section>')
    chunks.append('<section id="cases"><h2>3. Nghe audio và đối chiếu từng model</h2><input id="search" type="search" placeholder="Tìm speaker, ID hoặc câu tiếng Anh" aria-label="Tìm câu">'+case_cards(bundle)+'</section>')
    chunks.append('<section id="sources"><h2>4. Nguồn nghiên cứu và tái lập</h2><p>CTC-GOP là hướng nghiên cứu dựa trên posterior, không bảo đảm chỉ thêm alignment là phát hiện đúng. Policy hiện tại là công thức nội bộ đã đăng ký, không được gọi là bản tái lập CTC-GOP của tác giả.</p><p>WavLM/HuBERT trong SSL-FT-PRON có checkpoint chấm bốn điểm toàn câu và báo cáo PCC; chúng không trả trực tiếp lỗi âm từng từ nên chưa đưa vào bảng MDD này. GOPT có head nhỏ nhưng cần frontend GOP/Kaldi phù hợp. HMamba có head nhỏ nhưng pipeline còn cần nhiều SSL extractor và phụ thuộc CUDA/Mamba; số tham số head không đại diện tài nguyên toàn hệ thống. PSM (Phoneme Similarity Modeling, Interspeech 2025) hướng tới verbatim phoneme recognition; tài nguyên đã tra có project/demo, mô phỏng dữ liệu và metric AER, nhưng chưa xác minh được một checkpoint suy luận đầy đủ để chạy cùng protocol. Vì vậy chưa đưa PSM vào bảng số đo local. Các hướng thiếu checkpoint đầy đủ hoặc khác tác vụ được giữ ở phần nghiên cứu, không gán điểm đo local.</p>')
    chunks.append('<ul>'+''.join('<li><a target="_blank" rel="noreferrer" href="'+esc(url)+'">'+esc(name)+'</a></li>' for name,url in SOURCES)+'</ul>')
    chunks.append('<p><a href="model_metrics.csv">CSV model</a> · <a href="by_speaker.csv">CSV theo speaker</a> · <a href="policy_metrics.csv">CSV policy</a> · <a href="per_case.csv">CSV từng câu</a> · <a href="report_manifest.json">Manifest báo cáo</a></p><details><summary>Checksum và cấu hình model</summary><pre>'+esc(json.dumps({'input_sha256':bundle['input_sha256'],'models':bundle['manifest']['models'],'protocol':bundle['manifest'].get('protocol')},ensure_ascii=False,indent=2))+'</pre></details></section><script>document.getElementById("search").addEventListener("input",e=>{const q=e.target.value.toLowerCase();document.querySelectorAll(".case").forEach(c=>c.hidden=!c.dataset.search.toLowerCase().includes(q));});</script></main></body></html>')
    (output/'report.html').write_text(''.join(chunks),encoding='utf-8')


def case_rows(bundle):
    output=[]
    for case in bundle['manifest']['cases']:
        for model in MODELS:
            row=bundle['results']['predictions'][model][case['id']]
            output.append({'id':case['id'],'cohort':case['group'],'speaker':case.get('speaker_id'),
                           'model':model,'reference_text':case['reference_text'],'audio_sha256':case['audio_sha256'],
                           'phonemes':' '.join(row['phonemes']),'native_labels':json.dumps(row.get('native_decoded_labels'),ensure_ascii=False),
                           'unknown_tokens':json.dumps(row.get('unknown_tokens'),ensure_ascii=False),
                           'review':json.dumps(row['variant_review_full39'],ensure_ascii=False),
                           'inference_sec':row['inference_sec'],'runtime':json.dumps(row.get('runtime'),ensure_ascii=False)})
    return output


def build(directory=DEFAULT,baseline=BASELINE):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    bundle=load_bundle(directory,baseline)
    models,subgroups,policies=model_rows(bundle),model_rows(bundle,by_speaker=True),policy_rows(bundle)
    write_csv(directory/'model_metrics.csv',models)
    write_csv(directory/'by_speaker.csv',subgroups)
    write_csv(directory/'policy_metrics.csv',policies)
    write_csv(directory/'policy_by_speaker.csv',policy_rows(bundle,by_speaker=True))
    write_csv(directory/'per_case.csv',case_rows(bundle))
    figures(bundle,directory)
    render_html(bundle,directory)
    save(directory/'report_manifest.json',{'status':'completed','report_source_sha256':digest(__file__),
         'input_sha256':bundle['input_sha256'],'findings':findings(bundle),
         'expert_cases':120,'development_cases':len(bundle['manifest']['cases'])-120,'models':list(MODELS),
         'no_neural_inference':True,'no_training':True,'no_threshold_fitting':True,'no_promotion':True,
         'outputs':{p.name:digest(p) for p in directory.iterdir() if p.is_file() and p.suffix in {'.html','.png','.csv'}}})
    print('Report:',directory/'report.html')
    return bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=DEFAULT)
    parser.add_argument('--baseline',type=Path,default=BASELINE)
    args=parser.parse_args();build(args.input,args.baseline)


if __name__=='__main__':main()
