"""Prepare local English ASR data; speaker and normalized prompt disjoint.

Read transcripts are corpus orthographic references, not verified annotations of
every omitted/spoken word. No pronunciation labels enter ASR targets.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()

def rooted(path):
    path = (ROOT / path).resolve()
    if not path.is_relative_to(ROOT): raise ValueError('Path outside workspace')
    return path

def normalized(text):
    text = text.replace('’', "'").replace('‘', "'").upper()
    if re.search(r'\d|[^A-Z\s\W]', text): raise ValueError('Unsupported ASR characters')
    text = ' '.join(re.sub("[^A-Z' ]", ' ', text).split())
    if not text: raise ValueError('Empty transcript')
    return text

def prompt_split(text, seed=42):
    bucket = int(hashlib.sha256((str(seed)+'\n'+normalized(text)).encode()).hexdigest()[:16],16)%100
    return 'train' if bucket < 80 else 'validation' if bucket < 90 else 'test'

def audit(parts):
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        for key in ['speaker_id','text','audio_sha256']:
            if {r[key] for r in parts[a]} & {r[key] for r in parts[b]}:
                raise ValueError('Split leakage: '+a+'/'+b+'/'+key)
    if any(not rows for rows in parts.values()): raise ValueError('Empty split')

def prepare(config):
    import soundfile as sf
    corpus = rooted(config['corpus_root']); dest = rooted(config['dataset_dir'])
    source = rooted(config['source_protocol'])
    protocol = json.loads((source/'splits.json').read_text())
    with (source/'manifest.jsonl').open(encoding='utf-8') as stream:
        old = {r['utterance_id']:r for r in (json.loads(line) for line in stream)}
    candidates = []
    for speaker in [*config['train_speakers'],*config['validation_speakers'],*config['test_speakers']]:
        for transcript in sorted((corpus/speaker/'transcript').glob('*.txt')):
            candidates.append((speaker,transcript,False))
    for identifier in protocol['source_train_ids']:
        row=old[identifier]
        candidates.append((row['speaker_id'],rooted(row['transcript']['path']),True))
    parts={k:[] for k in ['train','validation','test']}; replay=[]; excluded=Counter()
    indexed=Counter()
    for speaker, transcript, is_replay in candidates:
        indexed['source' if is_replay else speaker]+=1
        identifier=speaker+'/'+transcript.stem
        audio=transcript.parent.parent/'wav'/(transcript.stem+'.wav')
        if not audio.is_file(): excluded['missing_audio']+=1; continue
        raw=transcript.read_text(encoding='utf-8').strip()
        try: text=normalized(raw)
        except ValueError: excluded['unsupported_transcript']+=1; continue
        split=prompt_split(text,config['seed'])
        expected=('train' if is_replay or speaker in config['train_speakers'] else
                  'validation' if speaker in config['validation_speakers'] else 'test')
        if split!=expected: excluded['speaker_prompt_holdout']+=1; continue
        info=sf.info(audio); duration=info.frames/info.samplerate
        if info.channels!=1 or duration<=0: excluded['invalid_audio']+=1; continue
        if split=='train' and duration>config['max_train_seconds']:
            excluded['long_train_utterance']+=1; continue
        audio_hash=sha(audio); text_hash=sha(transcript)
        if identifier in old:
            if audio_hash!=old[identifier]['audio']['sha256'] or text_hash!=old[identifier]['transcript']['sha256']:
                raise ValueError('Existing corpus provenance mismatch: '+identifier)
        row={'utterance_id':identifier,'speaker_id':speaker,'split':split,'text':text,'raw_transcript':raw,
             'audio':audio.relative_to(ROOT).as_posix(),'transcript':transcript.relative_to(ROOT).as_posix(),
             'audio_sha256':audio_hash,'transcript_sha256':text_hash,'duration_sec':duration,
             'sample_rate':info.samplerate,'source_replay':is_replay}
        (replay if is_replay else parts[split]).append(row)
    count=round(len(parts['train'])*config['source_replay_fraction']/(1-config['source_replay_fraction']))
    random.Random(config['seed']).shuffle(replay)
    parts['train'].extend(replay[:count]); audit(parts)
    manifest={'dataset':'L2-ARCTIC v5.0, full scripted Vietnamese transcripts + source replay',
              'indexed_counts':dict(indexed),'excluded_counts':dict(excluded),'seed':config['seed'],
              'split_policy':'Speaker-disjoint and normalized-prompt-disjoint; fixed SHA256 prompt buckets 80/10/10.',
              'train_speakers':config['train_speakers'],'validation_speakers':config['validation_speakers'],
              'test_speakers':config['test_speakers'],'source_replay_count':min(count,len(replay)),
              'source_replay_candidates':len(replay),'labels':'Corpus orthographic English transcripts; not gold realized-word annotations.',
              'license':'CC BY-NC 4.0; research use', 'parts':{}}
    contents={}
    for name,rows in parts.items():
        content=''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in rows)
        contents[name+'.jsonl']=content
        manifest['parts'][name]={'count':len(rows),'hours':sum(r['duration_sec'] for r in rows)/3600,
            'max_seconds':max(r['duration_sec'] for r in rows),'sha256':hashlib.sha256(content.encode()).hexdigest()}
    contents['manifest.json']=json.dumps(manifest,ensure_ascii=False,indent=2)+'\n'
    dest.mkdir(parents=True,exist_ok=True)
    for name,content in contents.items():
        path=dest/name
        if path.exists() and path.read_text(encoding='utf-8')!=content: raise ValueError('Prepared data changed: '+str(path))
        path.write_text(content,encoding='utf-8',newline='\n')
    return manifest

def load(config):
    dest=rooted(config['dataset_dir']); manifest=json.loads((dest/'manifest.json').read_text(encoding='utf-8'))
    parts={}
    for name,meta in manifest['parts'].items():
        path=dest/(name+'.jsonl')
        if sha(path)!=meta['sha256']: raise ValueError('Dataset digest mismatch')
        with path.open(encoding='utf-8') as stream:
            parts[name]=[json.loads(line) for line in stream]
    audit(parts)
    return parts,manifest
