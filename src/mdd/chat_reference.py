"""Chat-only reference drafts and reassessment of existing acoustic evidence.

Spelling suggestions are editable drafts, never inferred ground truth.  An
explicitly confirmed target can be aligned with the original model phones;
neither operation runs ASR, changes the recording, or trains a model.
"""
from __future__ import annotations

from copy import deepcopy
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
import re
import unicodedata

from . import demo_reference
from .demo_inference import canonical_comparison

MAX_TEXT_CHARS = 1000
MAX_WORDS = 150
MAX_SUGGESTION_WORDS = 12
MAX_CANDIDATES = 4
_EDGE_PUNCTUATION = '.,!?;:()[]{}"“”'


def _text(value):
    if (not isinstance(value, str) or not value.strip() or len(value) > MAX_TEXT_CHARS
            or len(value.split()) > MAX_WORDS
            or any(ord(character) < 32 and character not in '\t\r\n' for character in value)):
        raise ValueError('Use English reference text of 1–1,000 characters and at most 150 words.')
    return value.strip()


def _word(value):
    # Match the strict CMUdict lookup, including its treatment of contractions.
    key = unicodedata.normalize('NFKC', value).replace('’', "'").lower().strip(_EDGE_PUNCTUATION)
    return key[1:-1] if key.startswith("'") and key.endswith("'") and len(key) > 2 else key


def _dictionary(root):
    root = Path(root).resolve()
    path = root / 'data/external/cmudict' / demo_reference.COMMIT / 'cmudict.dict'
    try:
        if not path.resolve().is_relative_to(root):
            return {}, None
        info = path.stat()
        identity = (str(path), info.st_mtime_ns, info.st_size)
        # The shared loader checks the pinned dictionary's SHA-256 before use.
        return demo_reference._load(*identity), identity
    except (OSError, ValueError):
        return {}, None


@lru_cache(maxsize=4)
def _buckets(path, modified, size):
    result = {}
    for word in demo_reference._load(path, modified, size):
        if re.fullmatch('[a-z]{3,24}', word):
            result.setdefault((word[0], len(word)), []).append(word)
    return {key: tuple(sorted(values)) for key, values in result.items()}


def _distance(left, right, limit):
    """Bounded Levenshtein distance, with an early exit outside the edit band."""
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for index, character in enumerate(left, 1):
        current = [limit + 1] * (len(right) + 1)
        current[0] = index
        for position in range(max(1, index - limit), min(len(right), index + limit) + 1):
            current[position] = min(current[position - 1] + 1, previous[position] + 1,
                                    previous[position - 1] + (character != right[position - 1]))
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


@lru_cache(maxsize=512)
def _candidates(path, modified, size, word):
    # Short tokens, numbers, names containing punctuation and very long tokens
    # have no useful safe lexical draft.  Missing words remain visible for edits.
    if not re.fullmatch('[a-z]{3,24}', word):
        return ()
    limit = 1 if len(word) <= 4 else 2
    buckets = _buckets(path, modified, size)
    ranked = []
    for length in range(max(3, len(word) - limit), min(24, len(word) + limit) + 1):
        for candidate in buckets.get((word[0], length), ()):
            distance = _distance(word, candidate, limit)
            if distance <= limit:
                similarity = SequenceMatcher(None, word, candidate, autojunk=False).ratio()
                ranked.append((distance, -similarity, candidate))
    return tuple(candidate for _, _, candidate in sorted(ranked)[:MAX_CANDIDATES])


def _draft_token(original, replacement):
    # Keep punctuation and case in the user's editable draft.  Known tokens are
    # never passed through this function and stay byte-for-byte unchanged.
    match = re.fullmatch(r'([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)', original)
    if not match:
        return original
    prefix, word, suffix = match.groups()
    if word.isupper():
        replacement = replacement.upper()
    elif word.istitle():
        replacement = replacement.capitalize()
    return prefix + replacement + suffix


def reference_resolution(root, text):
    """Return a spelling-only draft; OOV replacements always need confirmation.

    ``target_text`` is deliberately null when any word is missing.  Consumers
    must not use ``suggested_text`` to assign sound errors before confirmation.
    Candidate ordering expresses spelling similarity, not intent confidence.
    """
    raw_text = _text(text)
    dictionary, identity = _dictionary(root)
    tokens = raw_text.split()
    missing, suggestions, replacements, checks = [], [], {}, []
    searched = set()
    for index, token in enumerate(tokens):
        word = _word(token)
        checks.append({'word': token, 'index': index,
                       'in_dictionary': word in dictionary if dictionary else None})
        if word in dictionary:
            continue
        missing.append(token)
        candidates = []
        if identity is not None and dictionary:
            if word in searched or len(searched) < MAX_SUGGESTION_WORDS:
                searched.add(word)
                candidates = list(_candidates(*identity, word))
        suggestions.append({'word': token, 'index': index, 'candidates': candidates})
        if candidates:
            replacements[index] = _draft_token(token, candidates[0])
    # Preserve original spacing, punctuation, tense, negation and numbers.  Only
    # words missing from the verified dictionary can change in this draft.
    count = iter(range(len(tokens)))
    suggested = re.sub(r'\S+', lambda match: replacements.get(next(count), match.group()), raw_text)
    return {'status': 'needs_confirmation' if missing else 'asr', 'raw_text': raw_text,
            'target_text': None if missing else raw_text, 'suggested_text': suggested,
            'missing_words': missing, 'suggestions': suggestions, 'word_checks': checks,
            'dictionary_source': 'cmudict' if dictionary else None,
            'dictionary_sha256': demo_reference.DICTIONARY_SHA256 if dictionary else None,
            'suggestion_method': 'bounded_levenshtein_then_character_similarity',
            'suggestions_require_confirmation': True}


def reassess_reference(root, speech, target):
    """Align a confirmed target with unchanged, server-owned original phones.

    Returns a new speech snapshot; transcript, recording metadata and acoustic
    evidence are retained.  Earlier coaching is cleared because its reference
    may have changed.  This performs no neural inference or voice generation.
    """
    from .chat_pronunciation import target_words
    target = _text(target)
    if (not isinstance(speech, dict) or not isinstance(speech.get('transcript'), str)
            or not speech['transcript'].strip()):
        raise ValueError('The original speech transcript is unavailable.')
    phonemes = speech.get('phonemes')
    if (not isinstance(phonemes, list) or not 0 < len(phonemes) <= 1500
            or any(not isinstance(item, dict) or not isinstance(item.get('phone'), str)
                   or not re.fullmatch(r'(?:[A-Z]{1,4}\*?|err|\[UNK\])', item['phone']) for item in phonemes)):
        raise ValueError('The original recording has no usable acoustic phone evidence.')
    reference, missing = demo_reference.dictionary_reference(Path(root), target)
    display_reference = reference
    if missing or reference is None:
        # A learner may confirm a real name or place outside CMUdict. That
        # text is usable by the conversation model even though this recording
        # cannot be acoustically compared as a whole. Known-word dictionary
        # phones below are display metadata only, never a partial alignment.
        dictionary, _ = _dictionary(root)
        canonical, indices = [], []
        for index, token in enumerate(target.split()):
            phones = dictionary.get(_word(token), [])
            canonical.extend(phones)
            indices.extend([index] * len(phones))
        display_reference = {'canonical_phones': canonical, 'canonical_word_indices': indices}
        comparison = {'words': [{'text': token, 'status': 'unavailable'} for token in target.split()],
                      'events': [], 'counts': dict.fromkeys(('match', 'substitution', 'deletion', 'insertion'), 0)}
    else:
        canonical = reference.get('canonical_phones')
        if not isinstance(canonical, list) or not 0 < len(canonical) < 1000:
            raise ValueError('The confirmed target has too many reference sounds.')
        comparison = canonical_comparison(target, reference, [item['phone'] for item in phonemes])
        if comparison is None:
            raise ValueError('The confirmed target could not be aligned with its dictionary reference.')
    result = deepcopy(speech)
    result.update(reference_text=target, conversation_text=target, reference_source='user_confirmed', comparison=comparison,
                  missing_words=list(missing),
                  target_words=target_words(root, target, display_reference),
                  reference_resolution={'status': 'confirmed', 'raw_text': speech['transcript'],
                      'target_text': target, 'suggested_text': target, 'missing_words': list(missing), 'suggestions': []},
                  coaching={'status': 'unavailable', 'feedback': {},
                      'message': 'Sound estimates were updated for your confirmed text. Previous coaching was cleared.'},
                  notice='Model sound estimates compare the original recording with the English text you confirmed.')
    if '_acoustic_evidence' in speech or 'assessment' in speech:
        from .acoustic_evidence import unpack_evidence
        from .chat_assessment import assessment_fields
        try:
            evidence = unpack_evidence(speech.get('_acoustic_evidence'))
        except ValueError:
            # New pipeline snapshots may never degrade to the raw categorical
            # edit baseline merely because retained evidence is unavailable.
            raise ValueError('The stored sound evidence is unavailable; record the sentence again.') from None
        fields = assessment_fields(root, target, evidence, confirmed=True,
                                   raw_comparison=comparison if reference is not None and not missing else None)
        if not fields.get('target_words'):
            fields['target_words'] = result['target_words']
        result.update(fields)
    elif missing or reference is None:
        from .chat_diagnostics import sound_candidates
        unavailable = {'status': 'unavailable', 'reason': 'reference_contains_unknown_words',
                       'missing_words': list(missing)}
        result.update(sound_candidates=sound_candidates(unavailable, target, reference_confirmed=True))
    else:
        # A legacy snapshot has no posterior assessment. Do not retain an OOV
        # diagnostic from an earlier target after returning to known words.
        result.pop('sound_candidates', None)
    if missing:
        result['notice'] = ('Your confirmed words can be used for the reply. Some words have no dictionary pronunciation; '
                            'this recording received no pronunciation judgement.')
    return result
