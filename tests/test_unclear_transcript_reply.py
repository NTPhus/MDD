"""Unresolved recovery evidence asks for another utterance, without guessing."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.conversation_context import prepare_dialogue_context
from scripts.conversation_policy import GroundedReply


def recovered(raw, text, *, unresolved=(), evidence=(), status='accepted'):
    return {'provider': 'bert', 'original': raw, 'text': text, 'status': status,
            'reference_confirmed': False, 'requires_confirmation': text != raw,
            'unresolved_words': list(unresolved), 'evidence': list(evidence)}


def rejected(word):
    return {'original': word, 'reason': 'unknown_word', 'accepted': False,
            'decision': 'competing_candidates'}


class UnclearRecoveryContextTests(unittest.TestCase):
    def test_partial_repair_cannot_hide_a_remaining_ambiguous_word_in_a_long_turn(self):
        raw = 'My fravorit place is the small coffee shop near the blerm station'
        text = 'My favorite place is the small coffee shop near the blerm station'
        interpretation = recovered(raw, text, unresolved=['blerm'], evidence=[
            {'original': 'fravorit', 'reason': 'unknown_word', 'accepted': True}, rejected('blerm')])
        result = prepare_dialogue_context({}, raw, missing_words=['fravorit', 'blerm'],
                                          interpretation=interpretation)
        self.assertEqual(result['text'], text)
        self.assertTrue(result['clarity']['needs_clarification'])
        self.assertEqual(result['clarity']['reason'], 'unclear_interpretation')

    def test_rejected_evidence_must_still_match_a_missing_word_in_selected_text(self):
        raw, text = 'I took the blerm train to work this morning', 'I took the blue train to work this morning'
        for unresolved in ([], ['another'], ['blerm']):
            with self.subTest(unresolved=unresolved):
                result = prepare_dialogue_context({}, raw, interpretation=recovered(
                    raw, text, unresolved=unresolved, evidence=[rejected('blerm')]))
                self.assertFalse(result['clarity']['needs_clarification'])

    def test_one_dictionary_missing_name_does_not_require_a_full_sentence(self):
        raw = 'I work with Quorvax on the new project'
        result = prepare_dialogue_context({}, raw, missing_words=['Quorvax'],
            interpretation=recovered(raw, raw, unresolved=['Quorvax'], status='unchanged'))
        self.assertFalse(result['clarity']['needs_clarification'])

    def test_short_answers_and_grammar_mistakes_are_not_treated_as_unintelligible(self):
        for text in ('Iced, please', 'At seven', 'Yesterday I go to work', 'Both', 'No, thanks'):
            with self.subTest(text=text):
                result = prepare_dialogue_context({}, text, interpretation=recovered(text, text, status='unchanged'))
                self.assertFalse(result['clarity']['needs_clarification'])

    def test_multiple_unknown_words_without_candidates_keep_existing_clarification(self):
        raw = 'And he dishurt my blerm'
        result = prepare_dialogue_context({}, raw, missing_words=['dishurt', 'blerm'],
            interpretation=recovered(raw, raw, unresolved=['dishurt', 'blerm'], status='unchanged'))
        self.assertTrue(result['clarity']['needs_clarification'])

    def test_malformed_or_known_word_evidence_is_not_an_unknown_span(self):
        raw = 'I work near Quorvax every weekday'
        for evidence in (None, {}, [None], [{'reason': 'unknown_word', 'accepted': False}],
                         [{'original': 'Quorvax', 'reason': 'phonetic_word', 'accepted': False}]):
            with self.subTest(evidence=evidence):
                value = recovered(raw, raw, unresolved=['Quorvax'])
                value['evidence'] = evidence
                result = prepare_dialogue_context({}, raw, interpretation=value)
                self.assertFalse(result['clarity']['needs_clarification'])

    def test_confirmed_text_and_reading_prompts_keep_priority(self):
        raw, text = 'My blerm train', 'My blue train'
        interpretation = recovered(raw, raw, unresolved=['blerm'], evidence=[rejected('blerm')])
        result = prepare_dialogue_context({}, raw, dialogue_text=text, source='user_confirmed',
                                          interpretation=interpretation)
        self.assertEqual(result['text'], text)
        self.assertFalse(result['clarity']['needs_clarification'])
        result = prepare_dialogue_context({}, raw, source='reading_prompt', missing_words=['My', 'blerm'])
        self.assertFalse(result['clarity']['needs_clarification'])


class ConciseClarificationPolicyTests(unittest.TestCase):
    raw = 'I went near the blerm station'

    def run_policy(self, reply, *, transcript=None, history=(), split=None):
        emitted = []
        policy = GroundedReply(None, emitted.append, transcript=transcript or self.raw,
                               history=history, needs_clarification=True)
        chunks = list(reply) if split is None else [reply[:split], reply[split:]]
        for chunk in chunks:
            policy.push(chunk)
        policy.finish()
        return ''.join(emitted), policy

    def test_invented_quoted_words_never_reach_text_or_voice_at_any_boundary(self):
        for quote in ('"green flerm"', '“green flerm”', "'green flerm'", '‘green flerm’'):
            reply = f'Could you explain what {quote} means?'
            for split in range(len(reply) + 1):
                with self.subTest(quote=quote, split=split):
                    output, policy = self.run_policy(reply, split=split)
                    self.assertEqual(output, '')
                    self.assertTrue(policy.needs_retry)
                    self.assertEqual(policy.blocked[0]['reason'], 'clarification_invented_quote')

    def test_actual_transcript_word_can_be_quoted_in_a_clarification(self):
        reply = 'Could you explain what "blerm" means?'
        output, policy = self.run_policy(reply)
        self.assertEqual(output, reply)
        self.assertFalse(policy.needs_retry)

    def test_only_one_short_clarification_is_published(self):
        output, policy = self.run_policy('Could you say that again, please? What did you mean by the last part?')
        self.assertEqual(output, 'Could you say that again, please?')
        self.assertFalse(policy.needs_retry)
        self.assertEqual(policy.blocked[0]['reason'], 'clarification_already_asked')

    def test_previous_model_question_is_not_replayed_as_a_new_clarification(self):
        reply = 'What did you mean by the last part?'
        output, policy = self.run_policy(reply, history=[{'role': 'assistant', 'content': reply}])
        self.assertEqual(output, '')
        self.assertTrue(policy.needs_retry)
        self.assertEqual(policy.blocked[0]['reason'], 'clarification_repeated_question')

    def test_explicit_uncertainty_cannot_bypass_policy_with_a_social_transcript(self):
        output, policy = self.run_policy('Hi!', transcript='Hi')
        self.assertEqual(output, '')
        self.assertTrue(policy.needs_retry)
        self.assertEqual(policy.blocked[0]['reason'], 'unresolved_transcript_requires_clarification')

    def test_long_abstract_questions_are_kept_private(self):
        reply = 'Could you repeat ' + 'the unclear part of your previous sentence ' * 5 + 'again?'
        output, policy = self.run_policy(reply)
        self.assertEqual(output, '')
        self.assertTrue(policy.needs_retry)
        self.assertEqual(policy.blocked[0]['reason'], 'clarification_too_long')


if __name__ == '__main__':
    unittest.main()
