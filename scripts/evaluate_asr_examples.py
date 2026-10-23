"""Paired ASR on user-confirmed development recordings, never used for training."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.asr_vi_data import rooted,sha
from scripts.train_asr_vi import write

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    parser.add_argument('--cases',type=Path,required=True);args=parser.parse_args()
    if not args.run:parser.error('--run is required for inference')
    import torch
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from transformers import Wav2Vec2ForCTC,Wav2Vec2Processor,Wav2Vec2Config
    from scripts.evaluate_transcript_normalization import distance,words
    torch.set_num_threads(2)
    cases=json.loads(args.cases.read_text(encoding='utf-8'))
    config=json.loads(rooted('configs/asr_vietnamese_english.json').read_text())
    completed=json.loads((rooted(config['run_dir'])/'training_provenance.json').read_text())
    if completed['status']!='completed':raise ValueError('Training must be completed first')
    records=[{'file':Path(row['audio']).name,'sha256':sha(row['audio']),'reference':row['hint'],
              'reference_source':row['hint_source']} for row in cases]
    for label,path in [('before',rooted(config['base_model'])),('after',rooted(config['run_dir'])/'best_checkpoint')]:
        cfg=Wav2Vec2Config.from_pretrained(path,local_files_only=True)
        cfg.apply_spec_augment=False
        model=Wav2Vec2ForCTC.from_pretrained(path,config=cfg,local_files_only=True,attn_implementation='eager').cuda().eval().requires_grad_(False)
        processor=Wav2Vec2Processor.from_pretrained(path,local_files_only=True)
        for row,result in zip(cases,records):
            wave,rate=sf.read(row['audio'],dtype='float32')
            if wave.ndim==2:wave=wave.mean(axis=1)
            if rate!=16000:
                divisor=math.gcd(rate,16000);wave=resample_poly(wave,16000//divisor,rate//divisor).astype(np.float32)
            inputs={key:value.cuda() for key,value in processor(wave,sampling_rate=16000,return_tensors='pt').items()}
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(**inputs)
            prediction=processor.batch_decode(output.logits.argmax(-1))[0]
            ref,hyp=words(result['reference']),words(prediction)
            result[label]={'transcript':prediction,'wer_vs_intended':distance(ref,hyp)/max(1,len(ref))}
        del model,processor;torch.cuda.empty_cache()
    result={'purpose':'Development cases already inspected before this experiment; not a held-out benchmark.',
            'references_used_for_decoding':False,'training_on_these_files':False,
            'reference_limit':'Intended sentence, not human annotation of realized pronunciation/omitted words.',
            'trained_sha256':completed['results']['trained_sha256'],'cases':records}
    write(rooted(config['report_dir'])/'development_examples.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
