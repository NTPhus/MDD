"""Keep greeting-only failures and unsupported cafe clauses off text/voice.

This validates local model output; it does not supply authored reply text.
A substantive learner turn requires a new contribution, not only a greeting.
Short social exchanges stay valid without forcing an extra question.
"""
from __future__ import annotations

import re


def _plain(text):
    return ' '.join(re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower().replace('’', "'")))


def _social_turn(text):
    """Short social exchanges do not require an extra topic or a question."""
    return bool(re.fullmatch(
        r"(?:hi(?: there)?|hello(?: there)?|hey|good (?:morning|afternoon|evening)|"
        r"thanks(?: a lot| so much)?|thank you(?: very much)?|thanks? (?:that helps|for your help)|"
        r"bye|goodbye|see you(?: later| soon)?|talk to you later|good night)", _plain(text)))


def _empty_contribution(text):
    """Recognize narrow greetings/stock acknowledgements, not all short replies."""
    normalized = _plain(text)
    return not normalized or bool(re.fullmatch(
        r"(?:hi(?: [a-z]+)?|hello(?: [a-z]+)?|hey(?: there)?|welcome|greetings|"
        r"good (?:morning|afternoon|evening)|okay|ok|sure|certainly|great|nice|cool|"
        r"i see|got it|alright|sounds (?:good|nice|great|fun|interesting)|"
        r"that sounds (?:good|nice|great|fun|interesting)|"
        r"thanks|thank you|you're welcome|you are welcome)", normalized))


def _asks_about_language(text):
    """Allow direct wording/grammar requests without treating all chat as class."""
    normalized = _plain(text)
    request = '?' in text or bool(re.match(
        r'^(?:please|can|could|would|will|help|check|correct|rewrite|rephrase|tell me|explain|how|what|is|does|do|are|should)\b',
        normalized))
    language = re.search(r'\b(?:grammar|grammatical|grammatically|pronunciation|pronounce|wording|phrase|phrasing|'
                         r'sentence|spelling|spell|meaning)\b|\bsound(?:s)? natural\b|\bway to say\b|'
                         r'\b(?:say|express) (?:this|that|it)\b', normalized)
    return bool(request and language)


def _stock_language_praise(text):
    """A bounded lexical check for meta-language praise, not a relevance judge."""
    return bool(re.fullmatch(
        r"(?:(?:that|this|it)(?:'s| is) (?:a )?(?:(?:very|really|quite) )?"
        r"(?:good|great|nice|correct|natural|clear|better|perfect) way to (?:say|phrase|put|express) (?:it|that|this)|"
        r"(?:your|that|this|the) (?:sentence|phrasing|wording|pronunciation) (?:is|sounds) "
        r"(?:(?:very|really|quite|grammatically) )?(?:good|great|nice|correct|natural|clear|better|perfect)|"
        r"(?:you (?:said|phrased|put|expressed) (?:it|that|this) (?:very )?well)|"
        r"well said|nicely put)", _plain(text)))


class GroundedReply:
    def __init__(self, state, emit, *, transcript=None, history=(), needs_clarification=False):
        self.state = state
        self.emit = emit
        self.buffer = ''
        self.blocked = []
        self.accepted_question = False
        self.finished = False
        self.pending = ''
        self.contextual_contribution = state is None and isinstance(transcript, str) and not _social_turn(transcript)
        self.accepted_contribution = False
        self.language_request = isinstance(transcript, str) and _asks_about_language(transcript)
        self.pending_language_praise = False
        self.history = history
        self.transcript = transcript if isinstance(transcript, str) else ''
        self.needs_clarification = needs_clarification
        self.accepted_clarification = False

    def _clause(self, text):
        if self.needs_clarification:
            # An unrelated fluent answer must not masquerade as understanding
            # a transcript whose words remain unresolved. This narrow gate is
            # not a semantic judge; rejected output uses the existing fallback.
            wording = _plain(text)
            clarification = ('?' in text and bool(re.search(
                r'\b(?:clarify|clarification|repeat|rephrase|mishear|misheard)\b|'
                r'\b(?:what|which).*(?:means?|refers?|word|part|said|say)\b|'
                r'\b(?:say|read|type|write|explain).*(?:again|sentence|said|mean|last part)\b|'
                r'\bdid you mean\b|\bnot (?:understand|catch|hear)\b', wording)))
            if not clarification:
                self.blocked.append({'text': text.strip(), 'reason': 'unresolved_transcript_requires_clarification'})
                return
            # Small local models can copy a fictional unclear word from an
            # instruction example. A question about such a word is fluent but
            # still invents evidence. Only quote spans from this actual turn.
            quotes = re.findall(r'''"([^"\n]{1,120})"|“([^”\n]{1,120})”|(?<!\w)'([^'\n]{1,120})'(?!\w)|‘([^’\n]{1,120})’''', text)
            original = ' ' + _plain(self.transcript) + ' '
            if any(' ' + _plain(next(part for part in groups if part)) + ' ' not in original
                   for groups in quotes):
                self.blocked.append({'text': text.strip(), 'reason': 'clarification_invented_quote'})
                return
            if len(wording.split()) > 30:
                self.blocked.append({'text': text.strip(), 'reason': 'clarification_too_long'})
                return
            if self.accepted_clarification:
                self.blocked.append({'text': text.strip(), 'reason': 'clarification_already_asked'})
                return
            prior_questions = {_plain(row['content']) for row in self.history
                               if isinstance(row, dict) and row.get('role') == 'assistant'
                               and isinstance(row.get('content'), str) and '?' in row['content']}
            if wording in prior_questions:
                self.blocked.append({'text': text.strip(), 'reason': 'clarification_repeated_question'})
                return
            self.accepted_clarification = True
            self.emit(text)
            return
        if self.contextual_contribution:
            if (not self.accepted_contribution and not self.language_request and _stock_language_praise(text)):
                self.pending += text
                self.pending_language_praise = True
                return
            if not self.accepted_contribution and _empty_contribution(text):
                self.pending += text
                return
            self.accepted_contribution = True
            if self.pending:
                self.emit(self.pending)
                self.pending = ''
            self.emit(text)
            return
        from scripts.conversation_state import question_violation
        reason = question_violation(text, self.state)
        if reason:
            self.blocked.append({'text': text.strip(), 'reason': reason})
        else:
            question = '?' in text
            if question:
                self.accepted_question = True
                if self.pending:
                    self.emit(self.pending)
                    self.pending = ''
            if self.requires_question and not self.accepted_question:
                self.pending += text
            else:
                self.emit(text)

    def push(self, text):
        if self.finished:
            raise RuntimeError('The reply policy stream is finished')
        if self.state is None and not self.contextual_contribution and not self.needs_clarification:
            self.emit(text)
            return
        self.buffer += text
        while match := re.search(r'[.!?\n]+', self.buffer):
            self._clause(self.buffer[:match.end()])
            self.buffer = self.buffer[match.end():]

    def finish(self):
        if not self.finished:
            if self.buffer:
                self._clause(self.buffer)
                self.buffer = ''
            if self.pending.strip():
                self.blocked.append({'text': self.pending.strip(), 'reason':
                                     'off_topic_language_praise' if self.pending_language_praise else
                                     'no_conversation_contribution' if self.contextual_contribution else 'missing_next_detail'})
                self.pending = ''
            self.finished = True

    @property
    def requires_question(self):
        return bool(self.state and str(self.state.get('next_action', '')).startswith('ASK_'))

    @property
    def needs_retry(self):
        if self.needs_clarification:
            return not self.accepted_clarification
        if self.contextual_contribution:
            return not self.accepted_contribution
        if self.state is None or self.accepted_question:
            return False
        return bool(self.blocked) or self.requires_question
