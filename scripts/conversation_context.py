"""Prepare a dialogue-only view without changing stored ASR/audio evidence.

The service owns validation/binding of a confirmed edit and supplies the history
from BEFORE the turn being regenerated. This helper never guesses corrections,
checks pronunciations, or removes history by matching text strings.
"""
from __future__ import annotations

from copy import deepcopy

SOURCES = {'asr', 'reading_prompt', 'corrected_sentence', 'user_confirmed'}
MAX_TEXT_CHARS = 1000


def _validate_text(value, name):
    if (not isinstance(value, str) or not value.strip() or len(value) > MAX_TEXT_CHARS
            or any(ord(character) < 32 and character not in '\t\r\n' for character in value)):
        raise ValueError(f'{name} must be nonempty text of at most {MAX_TEXT_CHARS} characters')
    return value.strip()


def prepare_dialogue_context(request, raw_transcript, *, dialogue_text=None, source='asr', missing_words=(), normalized_text=None):
    """Select confirmed text or a guarded ASR formatting view and its clarity.

    ``user_confirmed`` is an internal service provenance, not a client flag.
    The caller must bind it to the recording/reference revision before calling.
    A confirmed name missing from CMUdict is not an ASR misunderstanding.
    """
    if not isinstance(request, dict) or not isinstance(source, str) or source not in SOURCES:
        raise ValueError('Invalid dialogue request or conversation source')
    _validate_text(raw_transcript, 'raw_transcript')
    view = deepcopy(request)
    if source == 'user_confirmed':
        effective = _validate_text(dialogue_text, 'dialogue_text')
        # This is a chat edit, not a reading-practice instruction. Only the
        # model-request view changes; pronunciation reference/audio stay owned
        # by the caller's independent speech snapshot.
        view['reference_text'] = None
        needs_clarification = False
    else:
        if dialogue_text is not None and dialogue_text != raw_transcript:
            raise ValueError('A changed dialogue_text requires user_confirmed provenance')
        effective = raw_transcript
        if normalized_text is not None:
            from scripts.transcript_normalization import preservation_reason
            if source != 'asr' or preservation_reason(raw_transcript, normalized_text):
                raise ValueError('Normalization must preserve the ASR lexical content')
            effective = _validate_text(normalized_text, 'normalized_text')
            source = 'asr_normalized'
        missing_count = len(missing_words)
        needs_clarification = source in {'asr', 'asr_normalized'} and missing_count >= 2 and missing_count / max(len(effective.split()), 1) >= .25
    view['conversation_source'] = source
    view['needs_clarification'] = needs_clarification
    return {'text': effective, 'request': view,
            'clarity': {'needs_clarification': needs_clarification,
                        'reason': 'multiple_unresolved_words' if needs_clarification else None}}
