"""Remove complete learner-utterance restatements before text reaches the UI/TTS."""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable


_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_QUOTES = '\"\'“”‘’«»‹›'
_END = '.!?…\r\n'
_MARKERS = (
    ('you', 'said'), ('you', 'mentioned'), ('you', 'wrote'),
    ('your', 'sentence', 'is'), ('your', 'sentence'), ('your', 'words'),
    ('you', 'meant'), ('you', 'mean'), ('you', 'meant', 'that'), ('you', 'said', 'that'), ('you', 'mentioned', 'that'),
    ('your', 'corrected', 'sentence', 'is'), ('the', 'corrected', 'sentence', 'is'),
    ('i', 'understand', 'that'), ('i', 'understand', 'you', 'said'),
)
_ACKNOWLEDGMENTS = (('so',), ('okay',), ('ok',), ('i', 'see'), ('well',),
                   ('right',), ('alright',), ('got', 'it'), ('sure',))
_SOCIAL = {('hi',), ('hello',), ('hey',), ('thanks',), ('thank', 'you')}
_PERSON = {'i': 'you', 'my': 'your', 'me': 'you', 'mine': 'yours', 'myself': 'yourself',
           "i'm": "you're", "i've": "you've", "i'll": "you'll", "i'd": "you'd"}
_CONTRACTIONS = {
    ('i', 'am'): "i'm", ('you', 'are'): "you're", ('we', 'are'): "we're", ('they', 'are'): "they're",
    ('i', 'have'): "i've", ('you', 'have'): "you've", ('we', 'have'): "we've", ('they', 'have'): "they've",
    ('i', 'will'): "i'll", ('you', 'will'): "you'll", ('we', 'will'): "we'll", ('they', 'will'): "they'll",
    ('do', 'not'): "don't", ('does', 'not'): "doesn't", ('did', 'not'): "didn't",
    ('can', 'not'): "can't", ('will', 'not'): "won't", ('would', 'not'): "wouldn't",
    ('should', 'not'): "shouldn't", ('could', 'not'): "couldn't", ('is', 'not'): "isn't",
    ('are', 'not'): "aren't", ('was', 'not'): "wasn't", ('were', 'not'): "weren't",
}
_EXPANSIONS = {contracted: expanded for expanded, contracted in _CONTRACTIONS.items()}
_WAIT = object()
REFERENCE_APOLOGY = 'Sorry for the misunderstanding.'
MODEL_APOLOGIES = (REFERENCE_APOLOGY, 'I am sorry for the misunderstanding.',
                   'Sorry about the misunderstanding.')


def _key(word: str) -> str:
    normalized = word.replace('’', "'").casefold()
    # A spelling variant must not make an already asked question look new.
    return 'favourite' if normalized == 'favorite' else normalized


def _spellings(word: str) -> tuple[str, ...]:
    return ('favourite', 'favorite') if word == 'favourite' else (word,)


def _leading(text: str, start: int = 0) -> int:
    while start < len(text) and (text[start].isspace() or text[start] in _QUOTES):
        start += 1
    return start


def _prefix(text: str, start: int, words: tuple[str, ...], separators: str, *, allow_plain: bool = False):
    """Find an exact wrapper with required punctuation, keeping partials private."""
    position = start
    for index, word in enumerate(words):
        if index:
            if position == len(text):
                return _WAIT
            if not text[position].isspace():
                return None
            while position < len(text) and text[position].isspace():
                position += 1
        for char in word:
            if position == len(text):
                return _WAIT
            if text[position].casefold() != char:
                return None
            position += 1
    word_end = position
    while position < len(text) and text[position].isspace():
        position += 1
    if position == len(text):
        return _WAIT
    if text[position] not in separators:
        return _leading(text, position) if allow_plain and position > word_end else None
    return _leading(text, position + 1)


def _offsets(text: str, extra_wrappers=()):
    start = _leading(text)
    offsets = [(start, False)]
    seen = set(offsets)
    waiting = start == len(text)
    for offset, labelled in offsets:
        for phrases, separators, explicit, allow_plain in ((_MARKERS, ',:' + _QUOTES, True, True),
                                               (_ACKNOWLEDGMENTS, ',:', False, False),
                                               (extra_wrappers, ',:;—–-' + _END + _QUOTES, False, False)):
            for phrase in phrases:
                result = _prefix(text, offset, phrase, separators, allow_plain=allow_plain)
                if result is _WAIT:
                    waiting = True
                elif result is not None and (item := (result, labelled or explicit)) not in seen:
                    offsets.append(item)
                    seen.add(item)
    return offsets, waiting


def _sentence(text: str, start: int, words: tuple[str, ...], *, final: bool):
    """Match a whole sentence, never a shared prefix with additional content."""
    matches = list(_WORD.finditer(text, start))
    if not matches:
        return _WAIT if start == len(text) else None
    if matches[0].start() != start:
        return None
    previous = start
    for index, expected in enumerate(words):
        if index >= len(matches):
            return None if any(char in _END for char in text[previous:]) else _WAIT
        match = matches[index]
        # Do not combine words from different reply sentences into an echo.
        if any(char in _END for char in text[previous:match.start()]):
            return None
        actual = _key(match.group())
        if actual != expected:
            suffix = text[match.end():]
            partial = actual + ("'" if suffix in ("'", '’') else '')
            return _WAIT if (not suffix or suffix in ("'", '’')) and any(spelling.startswith(partial) for spelling in _spellings(expected)) else None
        previous = match.end()
    last = matches[len(words) - 1]
    next_word = matches[len(words)].start() if len(matches) > len(words) else len(text)
    position = last.end()
    while position < next_word and (text[position] in _QUOTES or text[position] in ' \t'):
        position += 1
    if position == next_word:
        if final and next_word == len(text):
            return position, False
        return _WAIT if next_word == len(text) else None
    if text[position] not in _END:
        return None
    question = text[position] == '?'
    position += 1
    while position < len(text) and (text[position].isspace() or text[position] in _QUOTES + _END):
        position += 1
    return position, question


def _contraction_variants(words):
    expanded = tuple(part for word in words for part in _EXPANSIONS.get(word, (word,)))
    variants = [expanded]
    seen = {expanded}
    # Keep combinations bounded; no fuzzy word substitutions or negation loss.
    for variant in variants:
        for index in range(len(variant) - 1):
            contracted = _CONTRACTIONS.get(variant[index:index + 2])
            if contracted:
                candidate = (*variant[:index], contracted, *variant[index + 2:])
                if candidate not in seen:
                    variants.append(candidate)
                    seen.add(candidate)
                    if len(variants) >= 16:
                        return variants
    return variants


def _candidates(candidates: Iterable[str]):
    result = []
    for text in candidates:
        if not isinstance(text, str):
            continue
        words = tuple(_key(match.group()) for match in _WORD.finditer(text))
        if not words:
            continue
        result.extend(_contraction_variants(words))
        # Exact grammatical person substitutions, never semantic paraphrases.
        # Negation and every content word stay unchanged.
        shifted = tuple('are' if word == 'am' and index and words[index - 1] == 'i'
                        else _PERSON.get(word, word) for index, word in enumerate(words))
        if shifted != words:
            result.extend(_contraction_variants(shifted))
    return list(dict.fromkeys(result))


class ReplyWithoutEcho:
    """Filter complete echoes at every sentence boundary before emitting them.

    Ordinary text streams as soon as its prefix cannot be an utterance or an
    explicit echo wrapper. A possible echo stays buffered until its sentence
    ends or becomes substantive new content. This includes one-word statements
    and exact first-to-second-person restatements; it does not guess paraphrases.
    Bare short questions ("Japan?") and social greetings remain conversational.
    Regeneration can disable those short-echo exceptions and omit a supplied
    apology before model-contribution validation. This is a lexical guard, not
    a guarantee against all semantic paraphrases of a learner's meaning.
    raw_text and removed retain an audit of the model's original output.
    """

    def __init__(self, candidates: Iterable[str], emit: Callable[[str], None], *, omit_phrases: Iterable[str] = (),
                 preserve_short_exchanges: bool = True, embedded_echoes: bool = False):
        # Application-supplied text (such as the fixed regeneration apology)
        # must not be counted as a useful model contribution. Buffer and remove
        # a model copy just like an echo, including a comma-separated wrapper.
        self._omit = _candidates(omit_phrases)
        self._candidates = list(dict.fromkeys([*_candidates(candidates), *self._omit]))
        self._preserve_short_exchanges = preserve_short_exchanges
        self._embedded_candidates = [words for words in self._candidates if len(words) >= 4] if embedded_echoes else []
        self._embedded_buffer = ''
        self._embedded_after_echo = False
        self._emit = emit
        self._buffer = ''
        self._raw: list[str] = []
        self._visible: list[str] = []
        self._checking = bool(self._candidates)
        self._after_echo = False
        self._pending_removed = ''
        self._join_after_echo = False
        self._finished = False
        self.removed: list[str] = []

    @property
    def raw_text(self) -> str:
        return ''.join(self._raw)

    @property
    def text(self) -> str:
        return ''.join(self._visible)

    def _forward(self, value: str) -> None:
        if value:
            if self._join_after_echo and self._visible and not self._visible[-1][-1].isspace() and not value[0].isspace():
                value = ' ' + value
            self._join_after_echo = False
            self._visible.append(value)
            self._emit(value)

    def push(self, chunk: str) -> None:
        if self._finished:
            raise RuntimeError('The reply stream is already finished')
        if not isinstance(chunk, str):
            raise TypeError('Reply chunks must be text')
        self._raw.append(chunk)
        if self._embedded_candidates:
            # A new prefix does not prove that a later part of the same
            # sentence is safe. Regenerated replies wait for a sentence so a
            # verbatim copy after an unfamiliar wrapper never reaches a callback.
            self._embedded_buffer += chunk
            self._drain_embedded(final=False)
            return
        if not self._candidates:
            self._forward(chunk)
            return
        self._buffer += chunk
        self._drain(final=False)

    def _embedded_sentence(self, text: str):
        matches = list(_WORD.finditer(text))
        words = tuple(_key(match.group()) for match in matches)
        spans = []
        for candidate in self._embedded_candidates:
            width = len(candidate)
            for start in range(len(words) - width + 1):
                if words[start:start + width] == candidate:
                    spans.append((matches[start].start(), matches[start + width - 1].end()))
        if not spans:
            return text, None
        # Replaying the complete learner statement inside an unfamiliar
        # preamble is still an echo. Drop that sentence conservatively. An
        # explicit new question after the copied clause can stand on its own;
        # preserve it without authoring replacement model content.
        end = max(item[1] for item in spans)
        remainder = text[end:]
        tail = remainder.lstrip(_QUOTES + ' \t,:;—–-')
        tail = re.sub(r'^(?:and|so)\s+', '', tail, flags=re.IGNORECASE)
        question = re.match(r'(?:what|where|when|why|which|who|how|is|are|was|were|do|does|did|'
                            r'would|could|can|will|should|have|has)\b', tail, flags=re.IGNORECASE)
        if question and tail.rstrip().endswith('?'):
            return tail, text[:len(text) - len(tail)]
        return '', text

    def _drain_embedded(self, *, final: bool):
        while self._embedded_buffer:
            if self._embedded_after_echo:
                position = 0
                while position < len(self._embedded_buffer) and (self._embedded_buffer[position].isspace()
                        or self._embedded_buffer[position] in _QUOTES + _END):
                    position += 1
                self._embedded_buffer = self._embedded_buffer[position:]
                if not self._embedded_buffer:
                    return
                self._embedded_after_echo = False
            boundary = next((index for index, char in enumerate(self._embedded_buffer) if char in _END), None)
            if boundary is None and not final:
                return
            end = len(self._embedded_buffer) if boundary is None else boundary + 1
            sentence, self._embedded_buffer = self._embedded_buffer[:end], self._embedded_buffer[end:]
            safe, removed = self._embedded_sentence(sentence)
            if removed:
                self.removed.append(removed.strip())
                self._join_after_echo = True
                self._embedded_after_echo = not safe
            if safe:
                self._buffer += safe
                self._drain(final=final and not self._embedded_buffer)

    def _drain(self, *, final: bool) -> None:
        while self._buffer:
            if self._after_echo:
                position = 0
                while position < len(self._buffer) and (self._buffer[position].isspace() or self._buffer[position] in _QUOTES + _END):
                    position += 1
                self._pending_removed += self._buffer[:position]
                self._buffer = self._buffer[position:]
                if not self._buffer:
                    return
                self.removed.append(self._pending_removed.strip())
                self._pending_removed = ''
                self._after_echo = False
            if self._checking:
                offsets, waiting = _offsets(self._buffer, self._omit)
                removed_end = None
                for offset, labelled in offsets:
                    for candidate in self._candidates:
                        result = _sentence(self._buffer, offset, candidate, final=final)
                        if result is _WAIT:
                            waiting = True
                        elif result is not None:
                            end, question = result
                            # A short bare clarification or intentional greeting
                            # can have meaning despite sharing the learner's words.
                            if self._preserve_short_exchanges and not labelled and (question and len(candidate) <= 3 or candidate in _SOCIAL):
                                continue
                            removed_end = end
                            break
                    if removed_end is not None:
                        break
                if removed_end is not None:
                    self._pending_removed = self._buffer[:removed_end]
                    self._buffer = self._buffer[removed_end:]
                    self._after_echo = self._join_after_echo = True
                    continue
                if waiting and not final:
                    return
                self._checking = False
            boundary = next((index for index, char in enumerate(self._buffer) if char in _END), None)
            if boundary is None:
                self._forward(self._buffer)
                self._buffer = ''
            else:
                self._forward(self._buffer[:boundary + 1])
                self._buffer = self._buffer[boundary + 1:]
                self._checking = True

    def finish(self) -> str:
        if not self._finished:
            if self._embedded_candidates:
                self._drain_embedded(final=True)
            self._drain(final=True)
            if self._pending_removed:
                self.removed.append(self._pending_removed.strip())
                self._pending_removed = ''
            self._finished = True
        return self.text
