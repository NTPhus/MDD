"""Read-only report for registered SpeechOcean/FCE/JFLEG component tests."""
from __future__ import annotations
import argparse
import base64
from collections import Counter, defaultdict
import csv
import html
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'.tools/model-comparison/python')]
from scripts.evaluate_extended_validation import OUT, read, sha, canonical_sha, lines, verify, write
from scripts.evaluate_external_speechocean import derive_metrics


def divide(a,b): return a/b if b else None


def word_review(rows):
    counts=Counter()
    for row in rows: counts.update(row['demo_review_counts'])
    n=counts['reference_words']; errors=counts['gold_error_words']; correct=n-errors
    available=sum(counts[k] for k in ('tp','fp','tn','fn'))
    if counts['tp']+counts['fn']+counts['unavailable_error_words']!=errors or counts['fp']+counts['tn']+counts['unavailable_correct_words']!=correct:
        raise ValueError('Review denominators do not partition all expert words')
    return {'counts':dict(counts),'gold_correct_words':correct,'precision':divide(counts['tp'],counts['tp']+counts['fp']),
        'recall_all_errors':divide(counts['tp'],errors),'false_positive_all_correct':divide(counts['fp'],correct),
        'recall_available':divide(counts['tp'],counts['tp']+counts['fn']),
        'false_positive_available':divide(counts['fp'],counts['fp']+counts['tn']),
        'coverage':divide(available,n),'f1_all':divide(2*counts['tp'],2*counts['tp']+counts['fp']+errors-counts['tp'])}


def clustered_intervals(rows,iterations=2000):
    import numpy as np
    speakers=defaultdict(list)
    for row in rows: speakers[row['speaker_id']].append(row)
    counts=[word_review(group)['counts'] for group in speakers.values()]
    matrix=np.asarray([[c.get(k,0) for k in ('tp','fp','reference_words','gold_error_words')] for c in counts],dtype=np.int64)
    sampled=matrix[np.random.default_rng(42).integers(0,len(matrix),size=(iterations,len(matrix)))].sum(axis=1)
    fp=sampled[:,1]/(sampled[:,2]-sampled[:,3]); recall=sampled[:,0]/sampled[:,3]
    return {'method':f'speaker cluster bootstrap, {len(matrix)} speakers, {iterations} iterations, seed42; uncertainty within this corpus only',
            'fpr_all_correct_95':np.quantile(fp,[.025,.975]).tolist(),
            'recall_all_errors_95':np.quantile(recall,[.025,.975]).tolist()}


def load():
    manifest=read(OUT/'manifest.json'); verify(manifest); manifest_sha=sha(OUT/'manifest.json')
    groups={}
    for name,key in (('pronunciation','utterance_id'),('grammar','id')):
        folder=OUT/name; meta=read(folder/'metadata.json')
        if meta['status']!='completed' or meta['predictions_sha256']!=sha(folder/'predictions.jsonl') or meta['manifest_sha256']!=manifest_sha:
            raise ValueError('Wait for complete unchanged '+name)
        rows=lines(folder/'predictions.jsonl')
        if [r[key] for r in rows]!=[r[key] for r in manifest[name]]:
            raise ValueError('Incomplete or reordered '+name)
        for original,row in zip(manifest[name],rows):
            if row['source_row_sha256']!=canonical_sha(original) or row['manifest_sha256']!=manifest_sha:
                raise ValueError('Changed case '+row[key])
            if row['reference_supplied_to_model'] is not False or not row['runtime']['device'].startswith('cuda'):
                raise ValueError('Unexpected conditioned/CPU inference')
        groups[name]=rows
    errant=read(OUT/'grammar/errant.json')
    if errant['status']!='completed' or errant['predictions_sha256']!=sha(OUT/'grammar/predictions.jsonl'):
        raise ValueError('ERRANT score identity differs')
    return manifest,groups,errant


def text_metrics(rows,hypotheses,destination):
    from sacrebleu.metrics import CHRF
    from scripts import compare_grammar_models as original
    original_gleu=ROOT/'data/generated/model-comparison-20261024/grammar/source-audit/official_jfleg_gleu.py'
    original_manifest=read(original_gleu.parents[1]/'manifest.json')
    if sha(original_gleu)!=original_manifest['gleu_sha256']:
        raise ValueError('Official GLEU source differs')
    target=OUT/'source-audit/official_jfleg_gleu.py'; target.parent.mkdir(exist_ok=True)
    shutil.copyfile(original_gleu,target)
    original.OUT=OUT
    references=[[r['references'][min(i,len(r['references'])-1)] for r in rows]
                for i in range(max(len(r['references']) for r in rows))]
    metric=CHRF(); value=metric.corpus_score(hypotheses,references)
    norm=lambda x:' '.join(x.split())
    return {'chrf':value.score,'exact_reference_rate':sum(norm(h) in {norm(r) for r in row['references']}
            for row,h in zip(rows,hypotheses))/len(rows),
            'gleu':original.gleu(rows,hypotheses,destination),'rows':len(rows)}


def summarize(manifest,data,errant):
    from scripts.grammar_data import canonical
    pron=data['pronunciation']; counts=Counter()
    for row in pron: counts.update(row['canonical_counts'])
    approved=Counter(word['status'] for row in pron for word in row['assessment'].get('words',[]))
    groups={}
    for group in ('child','adult'):
        rows=[r for r in pron if r['age_group']==group]
        groups[group]={'audio':len(rows),'speakers':len({r['speaker_id'] for r in rows}),'demo_review':word_review(rows)}
    summary={'pronunciation':{'audio':len(pron),'speakers':len({r['speaker_id'] for r in pron}),
        'canonical_counts':dict(counts),'canonical':derive_metrics(counts),'demo_review':word_review(pron),
        'cluster_bootstrap':clustered_intervals(pron),'by_age':groups,'categorical_word_status':dict(approved),
        'calibrated_decisions':sum(approved[k] for k in ('match','difference')),
        'unavailable_recordings':sum(r['assessment']['status']!='ready' for r in pron),
        'mean_inference_seconds':sum(r['inference_seconds'] for r in pron)/len(pron)},'grammar':{}}
    for corpus in ('FCE','JFLEG'):
        rows=[r for r in data['grammar'] if r['corpus']==corpus]
        controls=[r for r in rows if canonical(r['source']) in {canonical(x) for x in r['references']}]
        group={'rows':len(rows),'native_available':sum(r['prediction'] is not None for r in rows),
            'delivered':sum(r['delivered'] for r in rows),'changed_delivered':sum(r['delivered'] and canonical(r['prediction'])!=canonical(r['source']) for r in rows),
            'reference_matching_native':sum(r['prediction'] in r['references'] for r in rows),
            'reference_matching_withheld':sum(r['prediction'] in r['references'] and not r['delivered'] for r in rows),
            'lexical_unchanged_controls':len(controls),
            'unchanged_controls_preserved':sum(r['prediction'] is not None and canonical(r['prediction'])==canonical(r['source']) for r in controls),
            'mean_seconds':sum(r['seconds'] for r in rows)/len(rows),'text_similarity':{}}
        for arm in ('copy','raw','delivered'):
            hypotheses=[r['source'] if arm=='copy' else (r['prediction'] or r['source']) if arm=='raw' else (r['delivered_prediction'] or r['source']) for r in rows]
            group['text_similarity'][arm]=text_metrics(rows,hypotheses,OUT/'metric-inputs'/corpus/arm)
        if corpus=='FCE': group['errant']={arm:{k:v for k,v in scores.items() if k not in {'per_case','by_error_type'}} for arm,scores in errant['models'].items()}
        summary['grammar'][corpus]=group
    summary['limits']=manifest['protocol']['limits']+[
        'Canonical-position phone F1 here has a different target/rubric from L2-ARCTIC perceived-phone error F1; never rank across these values.',
        'Current demo-review flags are unvalidated hints, distinct from categorical decisions. Unflagged/uncertain is not confirmed correct.',
        'All-row grammar similarity and ERRANT use explicit original-source fallback when no suggestion is delivered; report availability alongside scores.',
        'FCE ERRANT v3/spaCy3 tokenization on a selected subset does not reproduce historical leaderboard tokenization.',
        'JFLEG GLEU/chrF/exact reference match are text similarity, not human naturalness or meaning-preservation scores.',
        'Canonical Wilson intervals treat phone/word trials independently; speaker-cluster bootstrap is supplied for demo word-review rates.']
    return summary


def csv_file(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def figures(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    p=summary['pronunciation']; canonical=p['canonical']; review=p['demo_review']
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
    values=((canonical['word_detection']['precision'],review['precision']),
            (canonical['word_detection']['recall'],review['recall_all_errors']),
            (canonical['majority_accepted_word_false_positive_rate']['rate'],review['false_positive_all_correct']))
    for ax,nums,title in zip(axes,values,('Precision cờ lỗi ↑','Recall mọi từ có lỗi ↑','Báo nhầm / mọi từ chấp nhận được ↓')):
        bars=ax.bar(['Canonical strict','Demo CMU review'],[100*x for x in nums],color=['#b7a7d1','#7850b2'])
        ax.bar_label(bars,fmt='%.1f',padding=3); ax.set(title=title,ylabel='%',ylim=(0,110))
    fig.suptitle('SpeechOcean762 · 500 audio / 125 speaker · nhãn đa số 5 chuyên gia\nHai cách tạo gợi ý; không phải kết luận phát âm đã calibration',fontsize=12)
    fig.savefig(OUT/'pronunciation_metrics.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for ax,metric,title in zip(axes,('false_positive_all_correct','recall_all_errors'),('Báo nhầm từ chấp nhận được ↓','Bắt được mọi từ có lỗi ↑')):
        groups=p['by_age']; bars=ax.bar(list(groups),[100*r['demo_review'][metric] for r in groups.values()],color=['#aa8cc9','#7650ad'])
        ax.bar_label(bars,fmt='%.1f',padding=3); ax.set(title=title,ylabel='%',ylim=(0,110))
    fig.suptitle('CMU review theo nhóm tuổi · tiếng Anh giọng Trung Quốc, không đại diện giọng Việt')
    fig.savefig(OUT/'pronunciation_age.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    fce=summary['grammar']['FCE']; jfleg=summary['grammar']['JFLEG']; arms=('copy','raw','delivered')
    panels=([100*fce['errant'][a]['f05'] for a in arms],
            [jfleg['text_similarity'][a]['gleu']['score'] for a in arms],
            [100*summary['grammar'][c]['delivered']/100 for c in ('FCE','JFLEG')])
    for ax,values,labels,title in zip(axes,panels,(['Copy','Qwen raw','Qua bộ lọc'],['Copy','Qwen raw','Qua bộ lọc'],['FCE','JFLEG']),
                                    ('FCE · ERRANT F0.5 ↑','JFLEG · GLEU ↑','Gợi ý thực sự hiển thị')):
        bars=ax.bar(labels,values,color=['#b7a7d1','#7850b2','#b18bbf'][:len(values)]); ax.bar_label(bars,fmt='%.1f',padding=3)
        ax.set(title=title,ylim=(0,110))
    fig.suptitle('Qwen grammar adapter hiện tại · 100 FCE + 100 JFLEG mới\nRaw/qua bộ lọc: fallback câu gốc khi không có đầu ra; độ sẵn sàng được báo riêng',fontsize=12)
    fig.savefig(OUT/'grammar_metrics.png',dpi=160); plt.close(fig)


def esc(value): return html.escape(str(value),quote=True)
def percent(value): return '—' if value is None else f'{value*100:.2f}%'
def table(headers,rows):
    return '<div class="scroll"><table><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'
def picture(name): return '<img alt="Biểu đồ kết quả" src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'">'


def html_report(manifest,data,summary):
    p=summary['pronunciation']; d=p['demo_review']; c=d['counts']
    parts=['<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kiểm thử mở rộng phát âm và ngữ pháp</title><style>body{margin:0;background:#f8f6fc;color:#292134;font:16px/1.55 Segoe UI,Arial}main{max-width:1200px;margin:auto;padding:28px 20px}section,header{padding:24px;background:white;border:1px solid #e5dcee;border-radius:14px;margin-bottom:18px}h1,h2{color:#66418f}h1{font-size:32px}h2{font-size:24px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #e9e0f1;text-align:left}th{background:#f2edf9}a{color:#70439e}.notice{background:#f5f0fa;border-left:4px solid #9c78be;padding:14px}details{border:1px solid #e4daef;border-radius:8px;margin:9px 0;padding:12px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}input{width:100%;padding:12px;box-sizing:border-box;margin:12px 0;font:inherit}audio{max-width:100%}.muted{color:#74697d}</style><body><main><header><h1>Kiểm thử mở rộng hệ thống local</h1><p>500 audio SpeechOcean762 từ 125 người nói và 200 câu mới FCE/JFLEG. Chọn mẫu và khóa model, prompt, bộ lọc trước inference; không train hoặc chỉnh ngưỡng bằng test.</p><p class="notice">Đây là đánh giá từng thành phần với reference đọc có sẵn và văn bản người học. Không phải đánh giá hội thoại/ASR đầu-cuối. F1 SpeechOcean có rubric khác với F1 L2-ARCTIC.</p></header>']
    parts.append('<section><h2>1. Phát âm · SpeechOcean762</h2><p>Corpus công bố tại Interspeech 2021; 5 chuyên gia chấm mỗi mẫu. Score 0 là sai/thiếu, score 1 là đúng âm nhưng accent mạnh và được chấp nhận trong protocol này. Gold dùng ít nhất 3/5 chuyên gia đánh dấu lỗi.</p>')
    parts.append('<p><b>CMU review hiện tại:</b> precision '+percent(d['precision'])+', recall toàn bộ lỗi '+percent(d['recall_all_errors'])+', báo nhầm '+percent(d['false_positive_all_correct'])+' ('+str(c['fp'])+'/'+str(d['gold_correct_words'])+' từ được chuyên gia chấp nhận). Coverage '+percent(d['coverage'])+'.</p>')
    parts.append('<p class="notice">Có '+str(c['gold_error_words'])+'/'+str(c['reference_words'])+' từ lỗi theo nhãn đa số. Hệ thống gợi ý quá nhiều từ cần xem lại trên corpus ngoài miền; không diễn giải recall cao là chấm chuẩn. Calibration chưa đạt nên số quyết định đúng/sai được duyệt vẫn là '+str(p['calibrated_decisions'])+'.</p>'+picture('pronunciation_metrics.png')+picture('pronunciation_age.png'))
    parts.append('<details><summary>Đếm chi tiết, controls đúng và khoảng tin cậy</summary><pre>'+esc(json.dumps(p,ensure_ascii=False,indent=2))+'</pre></details></section>')
    parts.append('<section><h2>2. Sửa ngữ pháp và cách nói tự nhiên</h2><p>FCE: 100 câu official test từ nguồn Cambridge, có M2 của người chấm. JFLEG: 100 câu official test mới, mỗi câu có các cách sửa tham chiếu của con người. Đã loại source trùng train/validation và lượt benchmark trước bằng chuẩn hóa lexical chính xác.</p>'+picture('grammar_metrics.png'))
    scores=[]
    for corpus,group in summary['grammar'].items():
        for arm in ('copy','raw','delivered'):
            metric=group['text_similarity'][arm]
            scores.append([corpus,arm,group['rows'],f"{metric['gleu']['score']:.2f}",f"{metric['chrf']:.2f}",percent(metric['exact_reference_rate']),percent(group['errant'][arm]['f05']) if corpus=='FCE' else 'Không áp dụng',group['delivered'] if arm=='delivered' else '—'])
    parts.append(table(['Corpus','Đầu ra','Câu','GLEU','chrF','Trùng reference','ERRANT F0.5','Gợi ý hiển thị'],scores))
    parts.append('<p class="notice">Raw/qua bộ lọc dùng câu gốc làm fallback để tính metric khi không có gợi ý; UI không giả vờ đó là một lần sửa thành công. GLEU/chrF là tương đồng với reference, không phải phần trăm câu đúng hay giữ nguyên nghĩa. ERRANT FCE là metric sửa edit trên subset, không phải điểm leaderboard toàn bộ.</p><details><summary>Chất lượng, availability và controls từng tác vụ</summary><pre>'+esc(json.dumps(summary['grammar'],ensure_ascii=False,indent=2))+'</pre></details></section>')
    parts.append('<section><h2>3. Xem từng câu ngữ pháp</h2><input type="search" id="grammar-search" placeholder="Tìm câu, corpus hoặc ID">')
    for row in data['grammar']:
        parts.append('<details data-grammar="'+esc(row['id']+' '+row['source'])+'"><summary>'+esc(row['corpus']+' · '+row['source'])+'</summary><p><b>Model:</b> '+esc(row['prediction'] or '[Không có đầu ra hợp lệ]')+'</p><p><b>Qua bộ lọc:</b> '+esc(row['delivered_prediction'] or '[Không hiển thị gợi ý]')+'</p><p><b>Reference:</b> '+esc(' | '.join(row['references']))+'</p><pre>'+esc(json.dumps({'generation':row['generation'],'guard':row['guard_validation']},ensure_ascii=False,indent=2))+'</pre></details>')
    parts.append('</section><section><h2>4. Nghe và xem từng audio</h2><input type="search" id="audio-search" placeholder="Tìm speaker, câu hoặc ID">')
    for row in data['pronunciation']:
        reference=row['human']['reference_text']; gold=[w for w,error in zip(row['human']['words'],row['human']['majority_word_error']) if error]
        hints=[w['text'] for w in row['sound_candidates']['words']]
        source=os.path.relpath(row['audio_path'],OUT).replace('\\','/')
        parts.append('<details data-audio="'+esc(row['speaker_id']+' '+row['utterance_id']+' '+reference)+'"><summary>'+esc(row['utterance_id']+' · '+reference)+'</summary><audio controls preload="none" src="'+esc(source)+'"></audio><p><b>Từ lỗi theo đa số chuyên gia:</b> '+esc(', '.join(gold) or 'Không có')+'</p><p><b>Demo gợi ý xem lại:</b> '+esc(', '.join(hints) or row['sound_candidates']['reason'])+'</p><pre>'+esc(json.dumps({'phones':row['phonemes'],'hints':row['sound_candidates'],'canonical_events':row['canonical_events']},ensure_ascii=False,indent=2))+'</pre></details>')
    parts.append('</section><section><h2>Nguồn, protocol và giới hạn</h2><ul><li><a href="https://www.isca-archive.org/interspeech_2021/zhang21x_interspeech.html">SpeechOcean762 · Interspeech 2021</a></li><li><a href="https://www.cl.cam.ac.uk/research/nl/bea2019st/">FCE chuẩn BEA-2019 · nguồn Cambridge</a></li><li><a href="https://aclanthology.org/P11-1019/">FCE · ACL 2011</a></li><li><a href="https://aclanthology.org/E17-2037/">JFLEG · EACL 2017</a></li><li><a href="https://github.com/chrisjbryant/errant">ERRANT · công cụ tác giả</a></li></ul>')
    parts.append('<ul>'+''.join('<li>'+esc(x)+'</li>' for x in summary['limits'])+'</ul><p><a href="summary.json">Summary JSON</a> · <a href="grammar_cases.csv">CSV ngữ pháp</a> · <a href="pronunciation_cases.csv">CSV phát âm</a> · <a href="manifest.json">Protocol đã khóa</a></p></section><script>for(const [id,key] of [["grammar-search","grammar"],["audio-search","audio"]])document.getElementById(id).addEventListener("input",e=>{const q=e.target.value.toLowerCase();document.querySelectorAll("[data-"+key+"]").forEach(x=>x.hidden=!x.dataset[key].toLowerCase().includes(q))});</script></main></body></html>')
    (OUT/'report.html').write_text(''.join(parts),encoding='utf-8')


def build():
    manifest,data,errant=load(); summary=summarize(manifest,data,errant)
    write(OUT/'summary.json',summary); figures(summary); html_report(manifest,data,summary)
    csv_file(OUT/'grammar_cases.csv',[{k:row[k] for k in ('id','corpus','task','source','prediction','delivered','delivered_prediction','seconds')} for row in data['grammar']])
    csv_file(OUT/'pronunciation_cases.csv',[{'id':r['utterance_id'],'speaker':r['speaker_id'],'age_group':r['age_group'],
        'reference':r['human']['reference_text'],**r['demo_review_counts'],'inference_seconds':r['inference_seconds']} for r in data['pronunciation']])
    inputs=[OUT/'manifest.json',OUT/'grammar/errant.json',OUT/'grammar/predictions.jsonl',OUT/'grammar/metadata.json',
            OUT/'pronunciation/predictions.jsonl',OUT/'pronunciation/metadata.json',Path(__file__),ROOT/'scripts/score_extended_grammar_errant.py']
    write(OUT/'report_manifest.json',{'status':'completed','input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
        'output_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix in {'.png','.html','.csv'} or p==OUT/'summary.json'},
        'no_training':True,'no_model_calls_in_report':True})
    print(json.dumps({'report':str(OUT/'report.html'),'summary':summary},ensure_ascii=False))


if __name__=='__main__': build()
