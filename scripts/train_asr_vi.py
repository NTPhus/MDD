"""Train English character-CTC on Vietnamese-accented L2-ARCTIC, local CUDA.

Separate from phoneme-B. Requires --run; held-out test is evaluated only after
the checkpoint has been selected on validation. No LLM/TTS/API calls.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.asr_vi_data import prepare,load,rooted,sha

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)

def now(): return datetime.now(timezone.utc).isoformat()

def train(config, resume=False):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from transformers import Wav2Vec2Config,Wav2Vec2ForCTC,Wav2Vec2Processor
    from scripts.evaluate_transcript_normalization import distance,words
    torch.set_num_threads(config['cpu_threads']);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise ValueError('CUDA BF16 required')
    torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'])
    torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    parts,manifest=load(config);run=rooted(config['run_dir']);reports=rooted(config['report_dir'])
    provenance_path=run/'training_provenance.json'
    if provenance_path.exists():
        previous=json.loads(provenance_path.read_text())
        if previous['config']!=config or previous['dataset']!=manifest: raise ValueError('Run provenance changed')
        if previous['status']=='completed': print('Run already completed:',run,flush=True);return
        if not resume: raise ValueError('Run exists; use --resume explicitly')
    run.mkdir(parents=True,exist_ok=True);reports.mkdir(parents=True,exist_ok=True)
    base=rooted(config['base_model'])
    download=json.loads((base/'download_manifest.json').read_text())
    base_sha=next(row['sha256'] for row in download['files'] if row['filename']=='model.safetensors')
    if sha(base/'model.safetensors')!=base_sha: raise ValueError('Base weights digest mismatch')
    processor=Wav2Vec2Processor.from_pretrained(base,local_files_only=True)
    def build(path):
        cfg=Wav2Vec2Config.from_pretrained(path,local_files_only=True)
        cfg.ctc_loss_reduction='mean';cfg.apply_spec_augment=False
        cfg.mask_time_prob=cfg.mask_feature_prob=0.
        m=Wav2Vec2ForCTC.from_pretrained(path,config=cfg,local_files_only=True,attn_implementation='eager')
        if config['freeze_cnn']:m.freeze_feature_encoder()
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        return m.cuda()
    model=build(run/'checkpoint' if resume else base)
    groups=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('lm_head.')],
             'lr':config['backbone_learning_rate'],'base_lr':config['backbone_learning_rate']},
            {'params':list(model.lm_head.parameters()),'lr':config['head_learning_rate'],'base_lr':config['head_learning_rate']}]
    optimizer=torch.optim.AdamW(groups,weight_decay=.01,foreach=False)
    trainable=[p for p in model.parameters() if p.requires_grad]
    checked=set()
    def inputs(row,labels=False):
        audio=rooted(row['audio'])
        if row['utterance_id'] not in checked:
            if sha(audio)!=row['audio_sha256']:raise ValueError('Audio changed: '+row['utterance_id'])
            checked.add(row['utterance_id'])
        wave,rate=sf.read(audio,dtype='float32')
        if wave.ndim!=1 or not np.isfinite(wave).all():raise ValueError('Invalid audio')
        if rate!=16000:
            divisor=math.gcd(rate,16000);wave=resample_poly(wave,16000//divisor,rate//divisor).astype(np.float32)
        data=dict(processor(wave,sampling_rate=16000,return_tensors='pt'))
        if labels:
            target=processor.tokenizer(row['text'],return_tensors='pt').input_ids
            sequence=target[0].tolist()
            frames=int(model._get_feat_extract_output_lengths(len(wave)))
            if frames<len(sequence)+sum(a==b for a,b in zip(sequence,sequence[1:])):
                raise ValueError('CTC-infeasible training utterance: '+row['utterance_id'])
            data['labels']=target
        return {key:value.cuda() for key,value in data.items()}
    def evaluate(rows,label):
        model.eval();errors=count=char_errors=chars=0;predictions=[]
        began=time.monotonic()
        with torch.inference_mode():
            for row in rows:
                with torch.autocast('cuda',dtype=torch.bfloat16): result=model(**inputs(row))
                if not torch.isfinite(result.logits).all():raise ValueError('Nonfinite evaluation logits')
                text=processor.batch_decode(result.logits.argmax(-1))[0]
                ref,hyp=words(row['text']),words(text)
                e=distance(ref,hyp);errors+=e;count+=len(ref)
                c=distance(' '.join(ref),' '.join(hyp));char_errors+=c;chars+=len(' '.join(ref))
                predictions.append({'utterance_id':row['utterance_id'],'reference':row['text'],'prediction':text,'word_errors':e})
                del result
        metrics={'wer':errors/max(count,1),'cer':char_errors/max(chars,1),'utterances':len(rows),
                 'word_errors':errors,'reference_words':count,'seconds':round(time.monotonic()-began,2)}
        print(label,json.dumps(metrics),flush=True)
        return metrics,predictions
    provenance={'status':'started','started_at_utc':now(),'config':config,'dataset':manifest,
                'base_sha256':base_sha,'task':'English orthographic ASR, not phoneme recognition',
                'test_used_for_training_or_selection':False,'trainable_parameters':sum(p.numel() for p in trainable),
                'gpu':torch.cuda.get_device_name(0),'precision':'bf16','model_type':'wav2vec2-base-960h'}
    write(provenance_path,provenance)
    cursor=step=0;best=float('inf');history=[]
    if resume:
        state=torch.load(run/'optimizer.pt',map_location='cpu',weights_only=True)
        optimizer.load_state_dict(state['optimizer'])
        cursor,step,best=state['cursor'],state['step'],state['best']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        history=json.loads((run/'history.json').read_text())
    try:
        if not (run/'baseline_validation.json').exists():
            metrics,predictions=evaluate(parts['validation'],'baseline_validation')
            write(run/'baseline_validation.json',metrics);write(reports/'baseline_validation_predictions.json',predictions)
        baseline=json.loads((run/'baseline_validation.json').read_text())
        ordered=[]
        for epoch in range(config['epochs']):
            indices=list(range(len(parts['train'])));random.Random(config['seed']+epoch).shuffle(indices)
            ordered.extend(indices)
        total=math.ceil(len(ordered)/config['gradient_accumulation'])
        started=time.monotonic()
        while cursor<len(ordered):
            model.train();optimizer.zero_grad(set_to_none=True);losses=[]
            batch=ordered[cursor:cursor+config['gradient_accumulation']]
            factor=min(1.,(step+1)/config['warmup_steps']) if step<config['warmup_steps'] else max(0.,(total-step)/(total-config['warmup_steps']))
            for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
            for index in batch:
                with torch.autocast('cuda',dtype=torch.bfloat16): output=model(**inputs(parts['train'][index],labels=True))
                loss=output.loss
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                losses.append(float(loss.detach()));(loss/len(batch)).backward()
                del output,loss
            norm=torch.nn.utils.clip_grad_norm_(trainable,1.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite training gradient')
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            cursor+=len(batch);step+=1
            record={'step':step,'total_steps':total,'loss':sum(losses)/len(losses),'gradient_norm':float(norm),
                    'elapsed_sec':round(time.monotonic()-started,2),'at_utc':now(),
                    'peak_allocated_mib':round(torch.cuda.max_memory_allocated()/2**20,1)}
            if step%10==0:
                print(json.dumps(record),flush=True)
                with (run/'training_history.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
            if step%config['eval_steps']==0 or cursor==len(ordered):
                metrics,predictions=evaluate(parts['validation'],'validation_step_'+str(step))
                history.append({**record,'validation':metrics})
                if metrics['wer']<best:
                    best=metrics['wer'];model.save_pretrained(run/'best_checkpoint');processor.save_pretrained(run/'best_checkpoint')
                    write(run/'best_validation.json',{'step':step,**metrics})
                    write(reports/'best_validation_predictions.json',predictions)
                model.save_pretrained(run/'checkpoint');processor.save_pretrained(run/'checkpoint')
                torch.save({'optimizer':optimizer.state_dict(),'cursor':cursor,'step':step,'best':best,
                            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},run/'optimizer.pt')
                write(run/'history.json',history)
        del model,optimizer,trainable,groups
        torch.cuda.empty_cache()
        best_meta=json.loads((run/'best_validation.json').read_text())
        model=build(run/'best_checkpoint')
        candidate_test,predictions=evaluate(parts['test'],'trained_test')
        write(reports/'trained_test_predictions.json',predictions)
        del model;torch.cuda.empty_cache();model=build(base)
        baseline_test,predictions=evaluate(parts['test'],'baseline_test')
        write(reports/'baseline_test_predictions.json',predictions)
        result={'baseline_validation':baseline,'best_validation':best_meta,'baseline_test':baseline_test,
                'trained_test':candidate_test,'selection':'lowest validation WER; test excluded from selection',
                'validation_improved':best_meta['wer']<baseline['wer'],
                'trained_sha256':sha(run/'best_checkpoint/model.safetensors'),
                'base_sha256':base_sha,'steps':step,'epochs':config['epochs'],
                'scope':'Small, speaker-and-prompt-disjoint Vietnamese read-speech benchmark; not conversational or pronunciation accuracy.'}
        write(reports/'results.json',result)
        provenance.update(status='completed',finished_at_utc=now(),global_step=step,results=result)
        write(provenance_path,provenance)
        print('COMPLETED',json.dumps(result),flush=True)
    except BaseException as error:
        provenance.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                          error_type=type(error).__name__,error=str(error)[:500],global_step=step)
        write(provenance_path,provenance)
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/asr_vietnamese_english.json')
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    config=json.loads(rooted(args.config).read_text())
    if args.prepare:print(json.dumps(prepare(config),indent=2),flush=True)
    if args.run:train(config,args.resume)
    elif not args.prepare:parser.error('Use --prepare or --run explicitly')

if __name__=='__main__':main()
