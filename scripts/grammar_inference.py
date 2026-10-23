"""Use the grammar adapter's original two single-field tasks, then lexical guard."""
import json
from scripts.grammar_data import messages
from scripts.evaluate_grammar_model import parse_output
from scripts.sentence_rewrite import _edit_summary, parse_rewrite

def rewrite(qwen,transcript,cancelled,history=()):
    predictions={};failures={}
    # Longer utterances need room for a complete JSON string and an EOS. Keep
    # short conversational turns at the existing budget and retain the timeout.
    max_tokens=512 if len(transcript.split())>40 else 320
    for task in ['grammar_corrected','natural']:
        if cancelled.is_set():return None
        chunks=[]
        request={'model_id':'qwen-grammar','messages':messages({'task':task,'source':transcript}),
                 'max_tokens':max_tokens,'timeout':45.}
        try:
            metadata=qwen.generate(request,cancelled,lambda value:chunks.append(value['text']) if value['type']=='chunk' else None)
            prediction,error=parse_output(''.join(chunks),task,metadata.get('finish_reason')=='stop')
        except Exception:
            if cancelled.is_set():return None
            prediction,error=None,'generation_failed'
        predictions[task]=prediction or transcript
        if error:failures[task]=error
    result=parse_rewrite(json.dumps({**predictions,'explanation':'Adapter rewrite; inspect actual edits.'}),True,transcript,history=history)
    # A failed model task is not an unchanged successful correction.
    fields={'grammar_corrected':'grammar','natural':'natural'}
    available=[];field_reasons={}
    for task,prefix in fields.items():
        usable=(task not in failures and result.get(prefix+'_status')=='ready'
                and isinstance(result.get(task),str) and bool(result[task].strip()))
        if usable:
            available.append(task)
            continue
        field_reasons[task]=failures.get(task,'unsupported_edit')
        result[task]='';result[prefix+'_status']='unavailable'
        result[prefix+'_changed']=False
        result[prefix+'_format_changed']=False

    # Recompute the aggregate after dropping failed tasks. A guard rejection in
    # one field and a generation failure in the other can leave no usable text.
    lexical_changed=any(result.get(prefix+'_changed',False) for prefix in fields.values())
    format_changed=any(result.get(prefix+'_format_changed',False) for prefix in fields.values())
    result['status']='ready' if available else 'unavailable'
    result['changed']=bool(available and (lexical_changed or format_changed))
    if not available:
        result['explanation']='No writing suggestion is available for this sentence.'
    elif result.get('grammar_changed'):
        result['explanation']=_edit_summary(transcript,result['grammar_corrected'])
    elif result.get('natural_changed'):
        result['explanation']=_edit_summary(transcript,result['natural'])
    elif format_changed:
        result['explanation']='Capitalization or punctuation adjusted.'
    else:
        result['explanation']='No wording change was suggested.'
    # A copied sentence only means that no change was proposed. It is not a
    # grammar, meaning, or naturalness verdict, including for an unclear ASR.
    feedback_state=('unavailable' if not available else 'partial' if len(available)==1
                    else 'suggestions_available' if lexical_changed else 'no_change_suggested')
    result.update(model_id='qwen-grammar',source='trained_adapter',task_failures=failures,
                  field_reasons=field_reasons,partial=len(available)==1,feedback_state=feedback_state)
    return result
