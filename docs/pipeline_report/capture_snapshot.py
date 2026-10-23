"""Read-only source/config snapshot for the current-pipeline document."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import urllib.request

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FILES=[
    'src/mdd/demo.py','src/mdd/demo_chat.py','src/mdd/demo_model_registry.py',
    'src/mdd/chat_runtime.py','src/mdd/demo_inference.py','src/mdd/demo_reference.py',
    'src/mdd/demo_realtime.py','src/mdd/demo_feedback.py','src/mdd/chat_assessment.py',
    'src/mdd/pronunciation_assessment.py','src/mdd/chat_pronunciation.py',
    'src/mdd/chat_reference.py','src/mdd/chat_diagnostics.py','src/mdd/phonetic_candidates.py',
    'src/mdd/acoustic_evidence.py','src/mdd/speech_topics.py','src/mdd/chat_voices.py',
    'scripts/serve_demo.py','scripts/local_chat_process.py','scripts/local_chat_worker.py',
    'scripts/local_speech_engine.py','scripts/grammar_inference.py','scripts/transcript_normalization.py',
    'scripts/conversation_context.py','scripts/conversation_reply.py','scripts/conversation_policy.py',
    'scripts/conversation_state.py','scripts/sentence_rewrite.py','scripts/sentence_correction.py',
    'scripts/cascade_bridge.py','scripts/ai33_tts.py','scripts/llm_client.py',
    'demo/index.html','demo/chat.js','demo/app.js','demo/recorder-worklet.js',
    'runs/asr-vietnamese-english-v1/training_provenance.json',
    'runs/cascade-l2arctic-plus-seed42/B/completed.json',
    'runs/cascade-l2arctic-plus-seed42/B/best_checkpoint/metadata.json',
    'runs/cascade-l2arctic-plus-seed42/Q/training_provenance.json',
    'runs/grammar-qwen3-06b-wi-jfleg-v1/training_provenance.json',
    'runs/grammar-qwen3-06b-wi-jfleg-v1/demo_validation.json',
    'data/generated/pronunciation-calibration-v1/full-validation/calibration.json',
]
fields=['default_model_id','models','device','acoustic_device','asr_model_id','grammar_model_id',
    'reply_model_id','sentence_mode','rewrite_mode','dialogue_policy','pronunciation_method',
    'pronunciation_uncertainty','voice_content','reference_confirmation','reference_regenerates_reply',
    'reference_regenerates_only_on_change','pronunciation_review','max_audio_seconds','speech_ready',
    'ai33_available','default_voice_provider']
snapshot={'captured_at_bangkok':datetime.now(timezone(timedelta(hours=7))).isoformat(),
    'method':'Read local code/metadata and GET config only; no audio, inference, training, TTS or state mutation.',
    'sources':[]}
for name in FILES:
    p=ROOT/name
    if p.is_file():
        b=p.read_bytes()
        snapshot['sources'].append({'path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
try:
    value=json.load(urllib.request.urlopen('http://127.0.0.1:8765/api/chat/config',timeout=4))
    snapshot['runtime_config']={k:value.get(k) for k in fields}
    snapshot['runtime_config_status']='GET completed'
except Exception as e:
    snapshot['runtime_config_status']=type(e).__name__
(HERE/'source_snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
print(snapshot['captured_at_bangkok'],len(snapshot['sources']),snapshot['runtime_config_status'])
