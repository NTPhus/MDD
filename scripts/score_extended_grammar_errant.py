"""Score frozen FCE predictions using isolated, unmodified ERRANT 3.0.0.

Run with Python -S: the evaluation dependencies never replace demo libraries.
Only FCE has original M2 edit references; JFLEG is scored separately with GLEU.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / '.tools/errant-eval/python'
sys.path.insert(0,str(DEPS))
OUT = ROOT / 'data/generated/extended-validation-20261024'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def score():
    import errant
    from errant.commands import compare_m2 as official
    import spacy
    folder = OUT / 'grammar'
    metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    if metadata['status']!='completed' or metadata['predictions_sha256']!=sha(folder/'predictions.jsonl'):
        raise ValueError('Wait for unchanged complete model outputs')
    rows=[json.loads(x) for x in (folder/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=[row for row in rows if row['corpus']=='FCE']
    if len(rows)!=100: raise ValueError('Expected the registered 100 FCE test sentences')
    annotator=errant.load('en')
    args=SimpleNamespace(beta=.5,verbose=False,dt=False,ds=False,cs=True,cse=False,single=False,multi=False,filt=[],cat=3)
    output={}
    inputs={}
    gold=''.join(row['gold_m2'].strip()+'\n\n' for row in rows)
    (folder/'fce_gold.m2').write_text(gold,encoding='utf-8')
    for arm in ('copy','raw','delivered'):
        best=Counter(tp=0,fp=0,fn=0); categories={}; blocks=[]; observations=[]
        for index,row in enumerate(rows):
            source=annotator.parse(row['source_tokenized'],tokenise=False)
            if arm=='copy': hypothesis=row['source']
            elif arm=='raw': hypothesis=row['prediction'] or row['source']
            else: hypothesis=row['delivered_prediction'] or row['source']
            # Preserve FCE source offsets, tokenize all generated strings using
            # the same English tokenizer. Copy/lexically unchanged sentences
            # preserve original source tokens, avoiding retokenization edits.
            if hypothesis==row['source']:
                corrected=source
            else:
                corrected=annotator.parse(hypothesis,tokenise=True)
            edits=annotator.annotate(source,corrected)
            block='S '+row['source_tokenized']
            if edits: block+='\n'+'\n'.join(e.to_m2() for e in edits)
            blocks.append(block+'\n\n')
            hyp=official.process_edits(official.simplify_edits(block),args)
            ref=official.process_edits(official.simplify_edits(row['gold_m2'].strip()),args)
            counts,cats=official.evaluate_edits(hyp,ref,best,index,row['source_tokenized'],args)
            best.update(counts); categories=official.merge_dict(categories,cats)
            observations.append({'id':row['id'],**counts,'edits':[e.to_m2() for e in edits]})
        p,r,f=official.computeFScore(best['tp'],best['fp'],best['fn'],.5)
        output[arm]={'counts':dict(best),'precision':p,'recall':r,'f05':f,'by_error_type':categories,'per_case':observations}
        name=folder/f'fce_{arm}.m2'; name.write_text(''.join(blocks),encoding='utf-8'); inputs[name.name]=sha(name)
    package_files=[p for p in (DEPS/'errant').rglob('*.py') if p.is_file()]
    model_files=[p for p in (DEPS/'en_core_web_sm').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    result={'status':'completed','dataset':'FCE official test, registered 100-case subset','models':output,
        'metric':'ERRANT 3.0.0 default span correction, F0.5; original author compare_m2 functions unchanged',
        'official_fce_gold':True,'tokenization':'FCE source fixed gold tokens; generated strings spaCy en_core_web_sm 3.8.0 tokenizer',
        'explicit_source_fallback':True,'predictions_sha256':sha(folder/'predictions.jsonl'),
        'source_sha256':sha(__file__),'packages':{n:version(n) for n in ('errant','spacy','en-core-web-sm','rapidfuzz','numpy')},
        'package_files_sha256':{str(p.relative_to(DEPS)):sha(p) for p in package_files},
        'english_model_sha256':{str(p.relative_to(DEPS)):sha(p) for p in model_files},
        'm2_sha256':inputs,'limits':['Subset score, not published full FCE leaderboard.',
            'ERRANT overlap is not human semantic correctness; unmatched alternate corrections may be valid.',
            'Span metrics penalize spelling/punctuation as well as grammar; inspect type breakdown.',
            'Raw/delivered metrics use source fallback on unavailable outputs; availability shown separately.'],
        'finished_utc':datetime.now(timezone.utc).isoformat()}
    (folder/'errant.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({arm:{k:v for k,v in data.items() if k not in {'per_case','by_error_type'}} for arm,data in output.items()}))


def self_test():
    import errant
    from errant.commands import compare_m2 as official
    annotator=errant.load('en')
    source=annotator.parse('He go home .')
    args=SimpleNamespace(beta=.5,verbose=False,dt=False,ds=False,cs=True,cse=False,single=False,multi=False,filt=[],cat=3)
    reference='S He go home .\nA 1 2|||R:VERB:SVA|||goes|||REQUIRED|||-NONE-|||0'
    gold=official.process_edits(official.simplify_edits(reference),args)
    for text,expected in (('He go home .',{'tp':0,'fp':0,'fn':1}),
                          ('He goes home .',{'tp':1,'fp':0,'fn':0}),
                          ('He went home .',{'tp':0,'fp':1,'fn':1})):
        edits=annotator.annotate(source,annotator.parse(text))
        block='S He go home .'+('' if not edits else '\n'+'\n'.join(e.to_m2() for e in edits))
        hypothesis=official.process_edits(official.simplify_edits(block),args)
        actual,_=official.evaluate_edits(hypothesis,gold,Counter(tp=0,fp=0,fn=0),0,'He go home .',args)
        assert actual==expected,(text,actual,expected)
    print(json.dumps({'errant_self_test':'passed','cases':3}))


if __name__=='__main__':
    self_test() if '--self-test' in sys.argv else score()
