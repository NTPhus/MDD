"""Create an executed data/dashboard notebook; long training stays opt-in by cell."""
from __future__ import annotations

from pathlib import Path
import textwrap
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells=[]
    def md(text): cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip()))
    def code(text, *, manual=False):
        source=textwrap.dedent(text).strip()
        if manual:
            source="RUN_TRAIN = False\nif not RUN_TRAIN:\n    print('Chỉ xem dữ liệu/trạng thái. Khi muốn train, tự đổi RUN_TRAIN=True rồi chạy cell này.')\nelse:\n"+textwrap.indent(source,"    ")
        cell=nbformat.v4.new_code_cell(source)
        if manual: cell.metadata.tags=["manual_run"]
        cells.append(cell)

    md('''
    # wav2vec2 + Qwen3-0.6B: train và so sánh có kiểm soát

    Audio → wav2vec2 → đối chiếu âm chuẩn → Qwen → hướng dẫn tiếng Anh.
    Model âm thanh và Qwen được train riêng, chạy lần lượt trên GPU. Không dùng Qwen2-Audio.
    Nhãn hướng dẫn gốc từ [L2-Arctic-plus / Findings of EACL 2026](https://aclanthology.org/2026.findings-eacl.190/).
    Train feedback do tác giả tạo bằng GPT-4o; 900 test reference được tác giả kiểm tra bằng người.
    Không yêu cầu bạn duyệt 300 template tự tạo trước đó.

    **Kernel: MDD (.venv).** Các cell xem dữ liệu/tiến trình có thể chạy ngay.
    Cell **CHẠY TRAIN** mặc định có `RUN_TRAIN=False`; Run All chỉ xem dữ liệu/trạng thái.
    Khi chủ động muốn train, tự đổi cờ thành `True` rồi chạy cell đó hoặc dùng lệnh terminal.
    Nếu pipeline đã chạy nền, cell chỉ hiển thị tiến trình, không tạo lượt train thứ hai.
    ''')
    code('''
    from pathlib import Path
    import json, sys, os, subprocess, math
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display, Audio, Markdown, Image, HTML
    ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'scripts/run_cascade_training.py').is_file())
    os.chdir(ROOT)
    if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
    from scripts.prepare_cascade_data import check_common_split
    CONFIG=ROOT/'configs/cascade_l2arctic_plus.json'
    config=json.loads(CONFIG.read_text(encoding='utf-8'))
    PYTHON=ROOT/('.venv/Scripts/python.exe' if os.name=='nt' else '.venv/bin/python')
    if not PYTHON.is_file(): PYTHON=Path(sys.executable)
    REPORTS=ROOT/config['report_root']; RUNS=ROOT/config['run_root']
    def run_script(*args):
        command=[str(PYTHON),'-X','utf8','-u',*[str(value) for value in args]]
        print(subprocess.list2cmdline(command))
        subprocess.run(command,cwd=ROOT,check=True)
    print('Project:',ROOT)
    print('Reports:',REPORTS)
    ''')
    md('''
    ## 1. Môi trường và lệnh cài đặt

    Giữ bản Torch CUDA đang dùng. Cài phần thiếu bằng Python của kernel, không thay Torch bằng bản CPU.
    Các lệnh dưới được in ra để dùng lại khi chuyển máy.
    ''')
    code('''
    from importlib.metadata import version
    import torch
    display(pd.DataFrame([{'package':name,'version':version(name)} for name in ['torch','transformers','peft','accelerate','bitsandbytes','sacrebleu','rouge-score']]))
    print('CUDA:',torch.cuda.is_available())
    if torch.cuda.is_available(): print('GPU:',torch.cuda.get_device_name(),'· GiB:',round(torch.cuda.get_device_properties(0).total_memory/2**30,2))
    for requirement in ['requirements/training.txt','requirements/feedback.txt','requirements/feedback-evaluation.txt','requirements/notebook.txt']:
        print(subprocess.list2cmdline([str(PYTHON),'-m','pip','install','-r',requirement]))
    ''')
    md('''
    ## 2. Split chung và kiểm tra rò rỉ

    Qwen: 2.249 train / 450 validation / 900 test. Validation: HJK, LXC, PNV.
    wav2vec2 A: 1.949 source train / 300 source validation. B: 300 target train,
    150 target validation, source replay 50%. Người Việt test là TLV (150 câu);
    toàn bộ test chung còn có 750 câu của năm người nói ngoài Việt Nam.

    Không dùng checkpoint A/B cũ: 750/900 test chung từng xuất hiện trong source train cũ.
    A mới khởi tạo từ wav2vec2-base, với phone head mới. Vocabulary âm vị được khai báo trước,
    không fit từ test. Câu đọc có thể chung giữa các người nói: đây là speaker holdout.
    ''')
    code('''
    if not (ROOT/config['protocol_dir']/'protocol.json').is_file():
        run_script('scripts/prepare_cascade_data.py')
    protocol,splits,qwen_manifest=check_common_split(config)
    readiness=json.loads((REPORTS/'data_readiness.json').read_text(encoding='utf-8'))
    display(pd.DataFrame([{'split':name,'utterances':value} for name,value in protocol['joint_counts'].items()]))
    display(pd.Series(readiness,name='readiness'))
    print('Qwen speakers:',qwen_manifest['speakers'])
    assert readiness['test_train_overlap']==readiness['test_validation_overlap']==0
    ''')
    md('## 3. Visualization dữ liệu thật')
    code('''
    annotations={r['utterance_id']:r for r in (json.loads(line) for line in (ROOT/config['processed_dir']/'annotations.jsonl').read_text(encoding='utf-8').splitlines() if line.strip())}
    frames=[]
    qwen_config=json.loads((ROOT/config['feedback_config']).read_text(encoding='utf-8'))
    for split in ['train','validation','test']:
        rows=[json.loads(line) for line in (ROOT/qwen_config['dataset_dir']/f'{split}.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        for row in rows:
            a=annotations[row['utterance_id']]
            frames.append({'id':row['utterance_id'],'split':split,'speaker':row['speaker_id'],'language':a['native_language'],'seconds':a['duration_sec'],
                'canonical_phones':len(a['canonical_phones']),'perceived_phones':len(a['perceived_phones']),
                'gold_phone_errors':len(a['gold_error_events']),'tokens':row['total_tokens'],'target_tokens':row['target_tokens']})
    frame=pd.DataFrame(frames)
    display(frame.groupby('split').agg(samples=('id','size'),speakers=('speaker','nunique'),hours=('seconds',lambda x:x.sum()/3600),max_tokens=('tokens','max')))
    fig,axes=plt.subplots(1,3,figsize=(14,3.5))
    frame.groupby(['language','split']).size().unstack(fill_value=0).plot.bar(ax=axes[0],rot=25)
    for name,part in frame.groupby('split'):
        axes[1].hist(part['seconds'],bins=20,alpha=.45,label=name)
        axes[2].hist(part['tokens'],bins=20,alpha=.45,label=name)
    axes[0].set(title='Speakers / native-language distribution',ylabel='Utterances')
    axes[1].set(title='Audio duration',xlabel='Seconds',ylabel='Utterances');axes[1].legend()
    axes[2].set(title='Full Qwen prompt + target',xlabel='Tokens',ylabel='Utterances');axes[2].legend()
    fig.tight_layout();display(fig);plt.close(fig)
    ''')
    code('''
    fig,axes=plt.subplots(1,2,figsize=(12,3.5))
    frame.groupby('speaker')['id'].count().plot.bar(ax=axes[0],color='#278391')
    error_counts={split:{} for split in ['train','validation','test']}
    for row in frames:
        for event in annotations[row['id']]['gold_error_events']:
            key=event['kind']; group=error_counts[row['split']]
            group[key]=group.get(key,0)+1
    pd.DataFrame(error_counts).fillna(0).plot.bar(ax=axes[1],rot=0)
    axes[0].set(title='Samples per speaker',ylabel='Utterances')
    axes[1].set(title='Manual phone errors by split',ylabel='Events')
    fig.tight_layout();display(fig);plt.close(fig)
    ''')
    md('''
    ## 4. Một ví dụ audio, nhãn và hướng dẫn gốc

    Đây là một mẫu train được chọn cố định. Audio là đầu vào wav2vec2; Qwen chỉ nhận
    câu tham chiếu và nhãn âm vị/sai khác. Ở lượt oracle, dùng nhãn công bố; ở lượt
    cascade, thay bằng âm dự đoán thật. Nhãn perceived/gold chỉ được dùng để train/chấm,
    không đưa vào nhánh nhận diện audio khi suy luận.
    ''')
    code('''
    samples=[json.loads(line) for line in (ROOT/qwen_config['dataset_dir']/'train.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    sample=samples[0]; annotation=annotations[sample['utterance_id']]
    print(sample['utterance_id'],sample['reference_text'])
    display(Audio(filename=str(ROOT/annotation['audio_path']),embed=True))
    display(pd.DataFrame(annotation['gold_error_events']))
    display(pd.DataFrame([{'word':word,**pair} for word,pairs in sample['feedback'].items() for pair in pairs]))
    ''')
    md('''
    ## 5. Bốn cấu hình và chỉ số

    A là source-trained CTC baseline, B là sau thích ứng Việt. Q0 là Qwen gốc,
    Q1 là sau LoRA. Cả bốn cùng 900 bản thu, cùng câu tham chiếu và cấu hình sinh greedy.
    Báo cáo oracle riêng để thấy mức suy giảm khi dùng lỗi dự đoán thực.

    Audio: PER, precision/recall/F1, FRR và word F1 theo scorer đã khóa.
    Feedback: JSON hợp lệ, từ được tham chiếu đúng, BLEU-2 và ROUGE-L của suggestion,
    NLL/perplexity, latency. F1 của feedback là **word-key coverage**, không phải phone F1.
    JSON sai, model hết token, hoặc input không gán được từ đều giữ trong mẫu số.
    Delta ROUGE-L có paired bootstrap 95% theo speaker; một seed không đo biến thiên giữa seed train.
    Các chỉ số này chưa tự chứng minh người học tiến bộ hay mọi lời hướng dẫn đều đúng ngữ âm.
    ''')
    code('''
    display(pd.DataFrame([{'case':a+'_'+q,'audio':a,'Qwen':q,'input':'Predicted phones'} for a in ['A','B'] for q in ['Q0','Q1']]))
    display(pd.Series(json.loads((ROOT/config['feedback_eval_config']).read_text(encoding='utf-8')),name='locked_evaluation'))
    for name,path in {**config['audio_configs'],'Q':config['feedback_config']}.items():
        print(name,json.loads((ROOT/path).read_text(encoding='utf-8')))
    ''')
    md('''
    ## 6. Kiểm tra kỹ thuật trước train chính

    Qwen dùng QLoRA NF4, full prompt, assistant-only loss. Projection từ vựng tính theo
    chunk và checkpoint để giữ loss/gradient tương đương, giảm peak VRAM cho target dài.
    Smoke lưu riêng trong `runs/local-checks/`; không lấy adapter này làm kết quả sau train.
    Các report kỹ thuật dưới đây không được dùng để kết luận cải thiện chất lượng.
    ''')
    code('''
    technical=[]
    for path in sorted((ROOT/'runs/local-checks').glob('l2plus-*/training_provenance.json')):
        r=json.loads(path.read_text(encoding='utf-8'))
        if r.get('purpose')=='technical_smoke' and r.get('effective_config',{}).get('warmup_ratio')==0:
            technical.append({'run':path.parent.name,'status':r['status'],'optimizer_steps':r.get('global_step'),'samples':r.get('samples_used'),
                'peak_GiB':r.get('peak_allocated_bytes',0)/2**30,'validation_loss':r.get('best_validation_loss')})
    display(pd.DataFrame(technical))
    run_script('scripts/train_feedback_model.py','--config',config['feedback_config'],'--check-data-only')
    ''')
    md('''
    ## 7. CHẠY TRAIN / RESUME toàn bộ pipeline

    Cell này chạy A (2.000 updates) → B (400 updates) → Qwen (3 epochs) → hai lượt
    audio test → hai lượt Qwen oracle → bốn lượt cascade → report. Đây là tác vụ dài;
    GPU được dùng lần lượt. Chọn checkpoint audio bằng validation F1, Qwen bằng validation loss.
    Test không chọn checkpoint hay thay recipe. Log/checkpoint/optimizer được giữ để resume.
    Nếu muốn tách giai đoạn, CLI có `--phase train` và `--phase evaluate`.
    ''')
    code('''
    import psutil
    status_path=REPORTS/'pipeline_status.json'
    status=json.loads(status_path.read_text(encoding='utf-8')) if status_path.is_file() else {}
    if status.get('status')=='running' and psutil.pid_exists(status.get('pid',-1)):
        print('Pipeline đang chạy nền; xem cell tiến trình bên dưới.')
        display(pd.Series(status))
    elif status.get('status')=='completed' and status.get('phase')=='all':
        print('Pipeline đã hoàn thành; mở report hoặc chạy lại cell báo cáo.')
    else:
        run_script('scripts/run_cascade_training.py','--phase','all')
    ''',manual=True)
    md('## 8. Tiến trình hiện tại — chạy lại cell này để cập nhật')
    code('''
    status_path=REPORTS/'pipeline_status.json'
    status=json.loads(status_path.read_text(encoding='utf-8')) if status_path.is_file() else {'status':'not_started'}
    display(pd.Series(status,name='pipeline'))
    if status.get('log') and Path(status['log']).is_file():
        print(chr(10).join(Path(status['log']).read_text(encoding='utf-8',errors='replace').splitlines()[-12:]))
    ''')
    md('''
    ## 9. Learning curves và kết quả trước/sau

    Chạy lại cell để đọc log thật. Khi các lượt test chưa xong, bảng sẽ ghi đang chờ;
    không có điểm sau train giả định. Report xuất JSON, CSV, PNG, SVG và HTML tự chứa.
    Ví dụ phản hồi được chọn cố định theo speaker, không chọn riêng những mẫu cải thiện.
    ''')
    code('''
    run_script('scripts/report_cascade.py')
    summary=json.loads((REPORTS/'report/comparison.json').read_text(encoding='utf-8'))
    print('Trạng thái:',summary['status'])
    print('Còn chờ:',summary['missing_audio'],summary['missing_feedback'])
    for filename in ['audio_comparison.csv','feedback_comparison.csv','feedback_by_speaker.csv']:
        path=REPORTS/'report'/filename
        if path.is_file(): display(pd.read_csv(path))
    for filename in ['training_curves.png','before_after.png']:
        path=REPORTS/'report'/filename
        if path.is_file(): display(Image(filename=str(path)))
    if summary.get('paired_intervals'): display(pd.DataFrame(summary['paired_intervals']).T)
    print('HTML:',REPORTS/'report/report.html')
    ''')
    notebook=nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec={'name':'mdd-local','language':'python','display_name':'MDD (.venv)'}
    return notebook


def main():
    notebook=build()
    NotebookClient(notebook,timeout=180,kernel_name='mdd-local',skip_cells_with_tag='manual_run',resources={'metadata':{'path':str(ROOT)}}).execute()
    path=ROOT/'notebooks/mdd_cascade_training_comparison.ipynb'
    nbformat.write(notebook,path)
    print('Saved:',path)


if __name__=='__main__':main()
