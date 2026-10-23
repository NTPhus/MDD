"""Freeze public Vietnamese-accent English examples before blind local inference.

Audio must already be downloaded from the recorded official sources. No network,
training, inference or API key access. Does not crop, select by ASR output, or
feed published references into recognition.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/external/speech-accent-archive-vietnamese'
OUT = ROOT / 'data/generated/vi-voice-pipeline-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    import soundfile as sf
    metadata = json.loads((SOURCE / 'source_metadata.json').read_text(encoding='utf-8'))
    downloads = json.loads((SOURCE / 'downloads.json').read_text(encoding='utf-8'))
    train = [json.loads(line) for line in (ROOT / 'data/processed/asr-vietnamese-english-v1/train.jsonl').read_text(encoding='utf-8').splitlines()]
    test = [json.loads(line) for line in (ROOT / 'data/processed/asr-vietnamese-english-v1/test.jsonl').read_text(encoding='utf-8').splitlines()]
    cascade = json.loads((ROOT / 'data/protocol/l2arctic-plus-cascade-v1/splits.json').read_text(encoding='utf-8'))
    train_hashes = {row['audio_sha256'] for row in train}
    train_prompts = {row['text'] for row in train}
    cases = []
    for identifier in ('vietnamese12', 'vietnamese13', 'vietnamese22', 'vietnamese32'):
        source = next(row for row in metadata['records'] if row['speaker_id'] == identifier)
        audio = next(row for row in downloads if row['id'] == identifier)
        path = ROOT / audio['audio']
        info = sf.info(str(path))
        assert .25 <= info.duration <= 30
        assert sha(path) == audio['sha256'] and audio['sha256'] not in train_hashes
        assert source['speaker_metadata_as_published']['native language'] == 'vietnamese'
        cases.append({'id': 'saa-' + identifier, 'corpus': 'Speech Accent Archive', 'speaker_id': identifier,
            'audio': audio['audio'], 'audio_sha256': audio['sha256'], 'duration_sec': info.duration,
            'reference_text': source['reading_paragraph'], 'reference_kind': 'published_reading_prompt',
            'source_url': source['page']['source_url'], 'audio_url': source['official_audio_url'],
            'speaker_metadata': source['speaker_metadata_as_published'], 'license': source['license'],
            'split_notes': 'External source corpus, absent from our ASR-v1 training manifest by original audio SHA. Cross-corpus speaker identity not independently verified. Same paragraph across these four speakers.'})
    selected = []
    for low, high in ((0, 3), (3, 5), (5, 30)):
        group = [row for row in test if low < row['duration_sec'] <= high]
        group.sort(key=lambda row: hashlib.sha256(('vi-voice-pipeline-v1\n' + row['utterance_id']).encode()).hexdigest())
        selected.extend(group[:2])
    assert len(selected) == 6
    for row in selected:
        assert row['audio_sha256'] not in train_hashes and row['text'] not in train_prompts
        assert row['utterance_id'] not in cascade['joint_train_ids']
        assert 'TLV' not in {item.split('/')[0] for item in cascade['joint_train_ids']}
        cases.append({'id': 'l2-' + row['utterance_id'].replace('/', '-'), 'corpus': 'L2-ARCTIC',
            'speaker_id': row['speaker_id'], 'audio': row['audio'], 'audio_sha256': row['audio_sha256'],
            'duration_sec': row['duration_sec'], 'reference_text': row['raw_transcript'],
            'reference_kind': 'published_reading_prompt', 'source_url': 'https://psi.engr.tamu.edu/l2-arctic-corpus/',
            'license': {'id': 'CC-BY-NC-4.0', 'url': 'https://creativecommons.org/licenses/by-nc/4.0/'},
            'split_notes': 'ASR-v1 test: TLV speaker and normalized prompt held out from train. TLV excluded from cascade joint train. Existing test corpus, not a new independent dataset.'})
    manifest = {'schema': 'vi-voice-pipeline-examples-v1', 'cases': cases,
        'selection': 'Fixed before inference: four full SAA clips within 30s; six ASR test TLV clips, two per duration bucket (0,3],(3,5],(5,30], SHA256 order seed vi-voice-pipeline-v1.',
        'excluded_downloads': [{**row, 'reason': 'Exceeds the current demo 30-second input limit; retained full file, not cropped.'} for row in downloads if row['duration_sec'] > 30],
        'reference_caveat': 'Corpus reading prompts, not expert verification of every realized word or omission. WER/CER measure differences against the intended prompt.',
        'protocol': 'Every clip is submitted once, whole, with empty history and no reference_text. Source/reference is used for scoring only. No training or TTS.',
        'source_metadata_sha256': sha(SOURCE / 'source_metadata.json'),
        'asr_training_manifest_sha256': sha(ROOT / 'data/processed/asr-vietnamese-english-v1/manifest.json')}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'manifest.json'
    content = json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
    if path.exists() and path.read_text(encoding='utf-8') != content:
        raise ValueError('Existing frozen manifest differs; do not overwrite an evaluated selection')
    path.write_text(content, encoding='utf-8', newline='\n')
    print(json.dumps({'manifest': str(path), 'sha256': sha(path), 'cases': len(cases),
        'speakers': len({row['speaker_id'] for row in cases}), 'reference_words': sum(len(row['reference_text'].split()) for row in cases),
        'excluded_over_30_seconds': len(manifest['excluded_downloads'])}, indent=2))


if __name__ == '__main__':
    main()
