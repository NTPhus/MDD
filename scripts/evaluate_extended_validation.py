"""Frozen external pronunciation and editing validation; no training or APIs.

Select cases before inference. SpeechOcean labels are used only after the
audio-only B forward; the grammar model never receives a reference correction.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / '.tools/model-comparison/python')]
OUT = ROOT / 'data/generated/extended-validation-20261024'
FCE = ROOT / 'data/external/fce-bea2019/fce_v2.1.bea19.tar.gz'
FCE_SHA = 'c574c1cdba6d3ab5a87280f180133cdcf0609848f5dc87cfa2c3f4b0c07ec67e'
SALT = 'external-validation-after-guard-v2-20261024'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def rank(identifier):
    return hashlib.sha256((SALT + ':' + identifier).encode()).hexdigest()


def choose_grammar():
    from nltk.tokenize.treebank import TreebankWordDetokenizer
    from scripts.grammar_data import canonical, parse_m2, load
    detok = TreebankWordDetokenizer().detokenize
    paths = [ROOT / f'data/processed/grammar-wi-jfleg-v1/{name}.jsonl' for name in ('train', 'validation')]
    load(ROOT / 'data/processed/grammar-wi-jfleg-v1')
    prior = ROOT / 'data/generated/model-comparison-20261024/grammar/manifest.json'
    seen = set()
    for row in [row for path in paths for row in lines(path)]:
        seen.update(canonical(text) for text in (row['source'], *row['references']))
    prior_seen = {canonical(text) for row in read(prior)['rows'] for text in (row['source'], *row['references'])}
    if sha(FCE) != FCE_SHA:
        raise ValueError('Official FCE archive hash changed')
    with tarfile.open(FCE) as archive:
        content = archive.extractfile('fce/m2/fce.test.gold.bea19.m2').read()
    source = OUT / 'sources/fce.test.gold.bea19.m2'
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(content)
    fce = []
    for index, row in enumerate(parse_m2(content.decode('utf-8'))):
        if any(v['unknown'] for v in row['variants']):
            continue
        refs = [detok(v['corrected_tokens']) for v in row['variants']]
        fce.append({'id': f'fce:test:{index}', 'corpus': 'FCE', 'task': 'grammar_corrected',
                    'official_split': 'test', 'source': detok(row['tokens']), 'source_tokenized': ' '.join(row['tokens']),
                    'references': refs, 'gold_m2': row['m2'], 'target': refs[0], 'cohort': 'new_external'})
    jfleg = [dict(row, corpus='JFLEG', cohort='new_external') for row in
             lines(ROOT / 'data/processed/grammar-wi-jfleg-v1/test.jsonl') if row['task'] == 'natural']
    selected, excluded = [], {}
    for name, values in (('FCE', fce), ('JFLEG', jfleg)):
        eligible, counts, duplicates = [], Counter(), set()
        for row in values:
            key = canonical(row['source'])
            if key in seen: counts['train_or_validation_exact_overlap'] += 1; continue
            if key in prior_seen: counts['previous_benchmark_source_or_reference_overlap'] += 1; continue
            if key in duplicates: counts['duplicate_source'] += 1; continue
            if not 1 <= len(row['source']) <= 1000 or not 1 <= len(row['source'].split()) <= 150:
                counts['outside_demo_text_bounds'] += 1; continue
            duplicates.add(key); eligible.append(row)
        if len(eligible) < 100:
            raise ValueError('Not enough independent cases: ' + name)
        chosen = sorted(eligible, key=lambda r: rank(r['id']))[:100]
        selected.extend(chosen)
        excluded[name] = {'input_rows': len(values), 'eligible': len(eligible), 'excluded': dict(counts)}
    return selected, {'selection': '100 FCE official-test + 100 unused JFLEG official-test; fixed SHA256 sort, no output/label-density selection',
        'fce_rows_with_unknown_edits_excluded': 195, 'corpora': excluded,
        'zero_selected_source_overlap_with_training_or_validation': True,
        'zero_selected_source_overlap_with_previous_benchmark_sources_or_references': True,
        'exact_overlap_only': True}, paths + [prior, FCE, source,
            ROOT / 'data/processed/grammar-wi-jfleg-v1/test.jsonl', ROOT / 'configs/grammar_sources.lock.json']


def prepare():
    from scripts.evaluate_external_speechocean import load_dataset, COMMIT
    grammar, audit, paths = choose_grammar()
    records, source_audit = load_dataset(ROOT / 'data/external/speechocean762' / COMMIT, 'full-test')
    speakers = defaultdict(list)
    for row in records:
        speakers[row['speaker_id']].append(row)
    pronunciation = [row for speaker in sorted(speakers)
                     for row in sorted(speakers[speaker], key=lambda r: rank(r['utterance_id']))[:4]]
    if len(pronunciation) != 500 or len(speakers) != 125:
        raise ValueError('Expected four recordings per 125 official test speakers')
    baseline = read(ROOT / 'data/generated/model-comparison-20261024/grammar/manifest.json')
    checkpoints = {'B': ROOT / 'runs/cascade-l2arctic-plus-seed42/B/best_checkpoint/model.safetensors',
                   'B_vocab': ROOT / 'runs/cascade-l2arctic-plus-seed42/B/best_checkpoint/vocab.json',
                   'qwen_base': ROOT / baseline['models']['qwen-base']['path'] / 'model.safetensors',
                   'qwen_grammar_adapter': ROOT / baseline['models']['qwen-grammar']['path'] / 'adapter_model.safetensors'}
    source_names = ('scripts/evaluate_extended_validation.py', 'scripts/evaluate_external_speechocean.py',
        'scripts/local_chat_worker.py', 'scripts/grammar_inference.py', 'scripts/grammar_data.py',
        'scripts/evaluate_grammar_model.py', 'scripts/sentence_rewrite.py', 'src/mdd/demo_inference.py',
        'src/mdd/chat_assessment.py', 'src/mdd/chat_diagnostics.py', 'src/mdd/pronunciation_assessment.py',
        'src/mdd/demo_model_registry.py', 'src/mdd/chat_runtime.py')
    paths.extend(ROOT / name for name in source_names)
    paths.extend((ROOT / 'configs/pronunciation_calibration.json',
                  ROOT / 'data/generated/pronunciation-calibration-v1/full-validation/calibration.json'))
    paths.extend(ROOT / 'models/pretrained/Qwen3-0.6B' / name for name in
                 ('config.json','generation_config.json','tokenizer.json','tokenizer_config.json'))
    paths.extend(ROOT / 'runs/cascade-l2arctic-plus-seed42/B/best_checkpoint' / name for name in ('config.json','metadata.json'))
    paths.extend(ROOT / 'runs/grammar-qwen3-06b-wi-jfleg-v1' / name for name in
                 ('adapter/adapter_config.json','training_provenance.json','demo_validation.json'))
    manifest = {'schema': 'extended-component-validation-v1', 'date': '2026-10-24', 'salt': SALT,
        'grammar': grammar, 'grammar_selection': audit, 'pronunciation': pronunciation,
        'pronunciation_selection': {'rule': 'four audio per official test speaker; SHA256-ranked ID, independent of scores',
            'audio': 500, 'speakers':125, 'groups':dict(Counter(r['age_group'] for r in pronunciation)),
            'dataset': source_audit, 'reference_given_to_neural_decoder':False,
            'reference': 'Official reading text; not ASR or a Qwen grammar correction',
            'scope': 'SpeechOcean is Chinese-accented English, not a Vietnamese-accent benchmark'},
        'input_sha256': {str(p.relative_to(ROOT)):sha(p) for p in paths},
        'model_sha256': {name:{'path':str(path.relative_to(ROOT)), 'sha256':sha(path)} for name,path in checkpoints.items()},
        'protocol': {'training':False,'paid_api':False,'demo_config_changes':False,
            'pronunciation': 'B CUDA FP32 audio-only evidence, original SpeechOcean canonical-position majority0 rubric and separate actual CMU-review hints; accents score1 accepted. Phone insertions reported separately. No perceived-phone PER or comparison with L2 phone F1.',
            'grammar': 'Actual local Qwen grammar adapter, greedy production prompts, 320 tokens <=40 words else512, 45sec, original JSON parser and current delivery guard. One task per sampled sentence; references withheld from model.',
            'precision': 'Qwen uses production CUDA runtime; record actual dtype and adapters. B FP32. No fallback silently.',
            'limits': ['Component evaluation, not an end-to-end ASR/conversation test.',
                'FCE and JFLEG are written learner text; free speech/microphone errors differ.',
                'Unknown upstream foundation pretraining exposure remains possible.',
                'New cases excluded from prior 200 examples and Qwen train/validation by normalized exact source only; not semantic dedup.',
                'No thresholds, prompts or guard are changed based on the new test outputs.',
                'Pronunciation review hints are not calibrated diagnoses; failed calibration means zero approved categorical decisions.']}}
    path = OUT / 'manifest.json'
    if path.exists() and read(path) != manifest:
        raise ValueError('Existing registered study differs; cannot overwrite')
    write(path, manifest)
    print(json.dumps({'status':'frozen','grammar':len(grammar),'audio':len(pronunciation),'selection':audit},ensure_ascii=False),flush=True)


def verify(manifest):
    for name, expected in manifest['input_sha256'].items():
        if sha(ROOT / name) != expected: raise ValueError('Frozen source changed: ' + name)
    for info in manifest['model_sha256'].values():
        if sha(ROOT / info['path']) != info['sha256']: raise ValueError('Frozen weights changed')


def gpu_gate():
    import torch
    from scripts.crottc_worker import physical_cuda_memory
    if not torch.cuda.is_available(): raise RuntimeError('This study requires CUDA')
    state = physical_cuda_memory()
    if state and state['free_mib'] < 2100: raise RuntimeError('GPU busy; wait rather than evict demo')
    return state


def prior_rows(path, planned, manifest_sha):
    previous = lines(path) if path.exists() else []
    key = 'id' if planned and 'id' in planned[0] else 'utterance_id'
    if [r[key] for r in previous] != [r[key] for r in planned[:len(previous)]]:
        raise ValueError('Resume must be the exact planned prefix')
    for original, observed in zip(planned, previous):
        if observed['source_row_sha256'] != canonical_sha(original) or observed['manifest_sha256'] != manifest_sha:
            raise ValueError('Resume row changed')
    return previous


def append(path, value):
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False) + '\n')


def pronunciation():
    import torch
    from mdd.demo_inference import LocalInference
    from mdd.chat_assessment import assessment_fields
    from scripts.evaluate_external_speechocean import read_verified_audio, score_prediction
    from scripts.experiment_pronunciation_calibration import save_evidence
    manifest = read(OUT / 'manifest.json'); verify(manifest); gpu_gate()
    manifest_sha = sha(OUT / 'manifest.json')
    folder = OUT / 'pronunciation'; folder.mkdir(exist_ok=True)
    path = folder / 'predictions.jsonl'
    previous = prior_rows(path,manifest['pronunciation'],manifest_sha)
    if len(previous) == 500: return
    engine = LocalInference(ROOT,threads=2,device='cuda',compute_type='float32')
    model = {'id':'cascade-B','kind':'phoneme','path':str(ROOT / 'runs/cascade-l2arctic-plus-seed42/B/best_checkpoint'),
             'label':'Current trained B','experimental':False}
    write(folder / 'metadata.json',{'status':'running','manifest_sha256':manifest_sha,'started_utc':datetime.now(timezone.utc).isoformat()})
    try:
        for index,row in enumerate(manifest['pronunciation'][len(previous):],len(previous)+1):
            audio,digest = read_verified_audio(row)
            started = time.perf_counter()
            evidence = engine.posterior_evidence(model,audio)
            seconds = time.perf_counter()-started
            if not evidence['runtime']['device'].startswith('cuda') or evidence['runtime']['compute_type'] != 'float32':
                raise RuntimeError('Unexpected B device/precision')
            phones = [p['phone'] for p in evidence['phonemes']]
            canonical = score_prediction(row['human'],phones)
            fields = assessment_fields(ROOT,row['human']['reference_text'],evidence,confirmed=True)
            hints = fields['sound_candidates']
            predicted = {word['index'] for word in hints['words']}
            available = hints['status'] == 'available'
            counts = Counter(tp=0,fp=0,tn=0,fn=0,unavailable_error_words=0,unavailable_correct_words=0,
                             reference_words=len(row['human']['words']),gold_error_words=sum(row['human']['majority_word_error']))
            for i,gold in enumerate(row['human']['majority_word_error']):
                if not available: counts['unavailable_error_words' if gold else 'unavailable_correct_words'] += 1
                else: counts[('tp' if gold else 'fp') if i in predicted else ('fn' if gold else 'tn')] += 1
            cache = folder / 'evidence' / (row['utterance_id']+'.npz'); cache.parent.mkdir(exist_ok=True)
            evidence['source_audio_sha256'] = digest
            save_evidence(cache,evidence)
            result = {**row,'status':'completed','manifest_sha256':manifest_sha,'source_row_sha256':canonical_sha(row),
                'audio_sha256':digest,'phonemes':phones,'runtime':evidence['runtime'],'inference_seconds':seconds,
                'canonical_counts':canonical['counts'],'canonical_events':canonical['events'],
                'demo_review_counts':dict(counts),'assessment':fields['assessment'],'sound_candidates':hints,
                'cache':{'path':str(cache.relative_to(OUT)),'sha256':sha(cache)},'reference_supplied_to_model':False}
            append(path,result)
            if index % 25 == 0: print(json.dumps({'stage':'pronunciation','done':index,'total':500}),flush=True)
            del evidence,fields
    finally:
        del engine; gc.collect(); torch.cuda.empty_cache()
    verify(manifest)
    write(folder / 'metadata.json',{'status':'completed','manifest_sha256':manifest_sha,'rows':500,
          'predictions_sha256':sha(path),'finished_utc':datetime.now(timezone.utc).isoformat()})


def grammar():
    import torch
    from scripts.local_chat_worker import QwenChatEngine
    from scripts.grammar_data import messages
    from scripts.evaluate_grammar_model import parse_output
    from scripts.sentence_rewrite import parse_rewrite
    manifest = read(OUT / 'manifest.json'); verify(manifest); gpu_gate()
    manifest_sha = sha(OUT / 'manifest.json')
    folder = OUT / 'grammar'; folder.mkdir(exist_ok=True)
    path = folder / 'predictions.jsonl'
    previous = prior_rows(path,manifest['grammar'],manifest_sha)
    if len(previous) == 200: return
    engine = QwenChatEngine(device='cuda'); engine.load('qwen-grammar')
    if not engine.runtime['device'].startswith('cuda'): raise RuntimeError('Unexpected Qwen CPU fallback')
    write(folder / 'metadata.json',{'status':'running','manifest_sha256':manifest_sha,'runtime':engine.runtime})
    try:
        for index,row in enumerate(manifest['grammar'][len(previous):],len(previous)+1):
            chunks=[]; task=row['task']; cancelled=threading.Event()
            request={'model_id':'qwen-grammar','messages':messages(row),
                     'max_tokens':512 if len(row['source'].split())>40 else 320,'timeout':45.}
            started=time.perf_counter()
            info=engine.generate(request,cancelled,lambda event:chunks.append(event['text']) if event['type']=='chunk' else None)
            seconds=time.perf_counter()-started; raw=''.join(chunks)
            prediction,error=parse_output(raw,task,info.get('finish_reason')=='stop')
            payload={'grammar_corrected':row['source'],'natural':row['source'],'explanation':'Independent saved-output validation.'}
            if prediction: payload[task]=prediction
            guarded=parse_rewrite(json.dumps(payload),True,row['source']) if prediction else None
            prefix='grammar' if task=='grammar_corrected' else 'natural'
            delivered=bool(guarded and guarded.get(prefix+'_status')=='ready')
            append(path,{**row,'status':'completed','manifest_sha256':manifest_sha,'source_row_sha256':canonical_sha(row),
                'raw':raw,'prediction':prediction,'error':error,'generation':info,'seconds':seconds,
                'delivered':delivered,'delivered_prediction':prediction if delivered else None,
                'guard_validation':guarded.get('guard_validation') if guarded else None,'runtime':dict(engine.runtime),
                'request_max_tokens':request['max_tokens'],'reference_supplied_to_model':False})
            if index%10==0: print(json.dumps({'stage':'grammar','done':index,'total':200,'seconds':round(seconds,2)}),flush=True)
    finally:
        del engine; gc.collect(); torch.cuda.empty_cache()
    verify(manifest)
    write(folder / 'metadata.json',{'status':'completed','manifest_sha256':manifest_sha,'rows':200,
          'predictions_sha256':sha(path),'finished_utc':datetime.now(timezone.utc).isoformat()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','pronunciation','grammar'))
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    {'prepare':prepare,'pronunciation':pronunciation,'grammar':grammar}[args.action]()


if __name__=='__main__': main()
