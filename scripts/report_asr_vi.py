"""Build an executed results notebook and standalone ASR before/after report."""
import json
from pathlib import Path
import sys
import html

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    import nbformat
    from nbclient import NotebookClient
    from scripts.train_asr_vi import write
    out=ROOT/'data/generated/asr-vietnamese-english-v1'
    result=json.loads((out/'results.json').read_text())
    notebook=nbformat.v4.new_notebook()
    markdown=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
    notebook.cells=[
      markdown('# Vietnamese-accented English ASR: fine-tuning and validation\n'
        'This trains English character CTC, separately from phoneme-B. All references are corpus orthographic '
        'transcripts. Speaker and normalized prompt holdouts are disjoint. Test is excluded from checkpoint selection. '
        'Results on one Vietnamese test speaker do not establish general conversational or pronunciation accuracy.'),
      code("from pathlib import Path\nimport json\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
           "ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'configs/asr_vietnamese_english.json').is_file())\n"
           "OUT = ROOT/'data/generated/asr-vietnamese-english-v1'\n"
           "config = json.loads((ROOT/'configs/asr_vietnamese_english.json').read_text())\n"
           "manifest = json.loads((ROOT/config['dataset_dir']/'manifest.json').read_text())\n"
           "results = json.loads((OUT/'results.json').read_text())\n"
           "pd.DataFrame(manifest['parts']).T[['count','hours','max_seconds']]"),
      markdown('## Run training manually\nThese commands never execute when the notebook is opened. '
               'The existing run is preserved; use `--resume` only for an interrupted matching run.'),
      code("RUN_TRAINING = False\nif RUN_TRAINING:\n    import subprocess, sys\n"
           "    subprocess.run([sys.executable, '-X','utf8','scripts/train_asr_vi.py','--prepare'],cwd=ROOT,check=True)\n"
           "    subprocess.run([sys.executable, '-X','utf8','scripts/train_asr_vi.py','--run'],cwd=ROOT,check=True)\n"
           "else:\n    print('Training disabled in this cell. Recorded results are visualized below.')"),
      markdown('## Learning curves\nTraining loss is diagnostic. Lowest validation WER selects the checkpoint; '
               'test is evaluated only after selection.'),
      code("history = pd.read_json(ROOT/config['run_dir']/'training_history.jsonl', lines=True)\n"
           "validation = pd.DataFrame(json.loads((ROOT/config['run_dir']/'history.json').read_text()))\n"
           "fig, axes = plt.subplots(1,2,figsize=(11,3.4))\n"
           "axes[0].plot(history.step,history.loss.rolling(10,min_periods=1).mean(),color='#7c3aed')\n"
           "axes[0].set(xlabel='Optimizer step',ylabel='Smoothed train CTC loss')\n"
           "axes[1].plot(validation.step,validation.validation.map(lambda r:r['wer']*100),marker='o',color='#7c3aed')\n"
           "axes[1].axhline(results['baseline_validation']['wer']*100,color='#888888',linestyle='--',label='Pretrained baseline')\n"
           "axes[1].set(xlabel='Optimizer step',ylabel='Validation WER (%)'); axes[1].legend()\n"
           "fig.tight_layout();fig.savefig(OUT/'learning-curves.png',dpi=160);plt.show()"),
      markdown('## Matched before/after results\nWER and CER measure transcription errors, not pronunciation correctness. '
               'These are a small read-speech benchmark; errors in corpus references may affect results.'),
      code("rows=[]\nfor split,before,after in [('validation','baseline_validation','best_validation'),('test','baseline_test','trained_test')]:\n"
           "    for label,key in [('Before',before),('After',after)]:\n"
           "        r=results[key];rows.append({'split':split,'model':label,'utterances':r['utterances'],'WER (%)':r['wer']*100,'CER (%)':r['cer']*100})\n"
           "table=pd.DataFrame(rows);display(table.round(3));table.to_csv(OUT/'metrics.csv',index=False)\n"
           "fig,axes=plt.subplots(1,2,figsize=(10,3.4))\n"
           "for ax,metric in zip(axes,['WER (%)','CER (%)']):\n"
           "    table.pivot(index='split',columns='model',values=metric)[['Before','After']].plot.bar(ax=ax,color=['#aaa2b4','#7c3aed'],rot=0)\n"
           "    ax.set_ylabel(metric);ax.set_xlabel('');ax.set_title('Lower is better')\n"
           "fig.tight_layout();fig.savefig(OUT/'before-after.png',dpi=160);plt.show()"),
      markdown('## Previously supplied development recordings\nThese recordings were inspected during planning. '
               'They are not the held-out benchmark. Reference text was not supplied to the decoder. '
               'This comparison uses GPU BF16 greedy decoding, while the demo runs inference on CPU float32.'),
      code("examples=OUT/'development_examples.json'\nif examples.is_file():\n"
           "    examples=json.loads(examples.read_text())['cases']\n"
           "    display(pd.DataFrame([{'reference':r['reference'],'before':r['before']['transcript'],'after':r['after']['transcript'],"
           "'WER before (%)':r['before']['wer_vs_intended']*100,'WER after (%)':r['after']['wer_vs_intended']*100} for r in examples]))\n"
           "else:\n    print('Development comparison pending.')"),
      markdown('## Weight changes\nCNN stays frozen; attention and character-CTC weights are updated.'),
      code("weights=OUT/'weight_changes.json'\nif weights.is_file():\n    display(pd.DataFrame(json.loads(weights.read_text())).T)"),
      markdown('## Qwen grammar adapter\nMatched text-only development probe; format compliance and chrF are not '
               'human grammar quality, semantic faithfulness, official ERRANT F0.5 or JFLEG GLEU.'),
      code("comparison=ROOT/'data/generated/grammar-adapter-demo-v1/comparison.json'\n"
           "if comparison.is_file():\n    q=json.loads(comparison.read_text()); qr=[]\n"
           "    for model in ['before','after']:\n        for task,r in q[model]['tasks'].items():\n"
           "            qr.append({'model':model,'task':task,'rows':r['rows'],'valid JSON (%)':r['valid_json_rate']*100,'reference chrF':r['chrf']})\n"
           "    display(pd.DataFrame(qr).round(3))\nelse:\n    print('Grammar comparison pending.')"),
      markdown('## Demo model provenance\nIncomplete ASR runs and runs without improved validation WER use the base model. '
               'The grammar adapter requires a completed checksum-matched validation record. The conversation model remains base Qwen.'),
      code("import sys\nsys.path.insert(0,str(ROOT/'src'))\nfrom mdd.demo_model_registry import asr_selection, grammar_selection\n"
           "for task,selection in [('ASR',asr_selection(ROOT)),('Grammar',grammar_selection(ROOT))]:\n    print(task,selection['id'])\n"
           "print('Conversation: Qwen3-0.6B base; pronunciation: phoneme-B unchanged.')")]
    notebook.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    path=ROOT/'notebooks/mdd_asr_vi_finetuning.ipynb'
    nbformat.write(notebook,path)
    NotebookClient(notebook,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(notebook,path)
    rows=[]
    for name in ['baseline_validation','best_validation','baseline_test','trained_test']:
        r=result[name];rows.append('<tr><td>'+name+'</td><td>'+str(r['utterances'])+'</td><td>'+f"{100*r['wer']:.2f}%"+'</td><td>'+f"{100*r['cer']:.2f}%"+'</td></tr>')
    page='<!doctype html><meta charset="utf-8"><title>Vietnamese English ASR</title><style>body{max-width:1050px;margin:35px auto;padding:0 20px;font:17px system-ui;color:#30233f}h1{color:#7c3aed}td,th{padding:10px;border-bottom:1px solid #ddd}img{max-width:100%}</style>'
    page+='<h1>Vietnamese-accented English ASR</h1><p>Speaker and normalized prompt disjoint. Test excluded from checkpoint selection. Small read-speech benchmark; not pronunciation accuracy.</p><table><tr><th>Split/model</th><th>Utterances</th><th>WER</th><th>CER</th></tr>'+''.join(rows)+'</table><img src="before-after.png"><img src="learning-curves.png">'
    page+='<p>Selected optimizer step: '+str(result['best_validation']['step'])+'. Training steps: '+str(result['steps'])+'.</p>'
    examples_path=out/'development_examples.json'
    if examples_path.is_file():
        page+='<h2>Previously supplied recordings (development only)</h2><p>These difficult recordings remain incorrectly transcribed. No intended text was given to the decoder. GPU BF16 inference.</p><table><tr><th>Intended sentence</th><th>Before</th><th>After</th></tr>'
        for r in json.loads(examples_path.read_text())['cases']:
            page+='<tr>'+''.join('<td>'+html.escape(text)+'</td>' for text in [r['reference'],r['before']['transcript'],r['after']['transcript']])+'</tr>'
        page+='</table>'
    page+='<details><summary>Provenance</summary><pre>'+html.escape(json.dumps(result,indent=2))+'</pre></details>'
    (out/'report.html').write_text(page,encoding='utf-8')
    print('Executed notebook:',path);print('Report:',out/'report.html')

if __name__=='__main__':main()
