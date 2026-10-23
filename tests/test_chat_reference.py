"""A reference draft must not silently become pronunciation ground truth."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.mdd import demo_reference
from src.mdd.chat_reference import (_buckets, _candidates, _distance,
                                    reference_resolution, reassess_reference)


LEXICON = b'''i AY1
my M AY1
favorite F EY1 V ER0 IH0 T
favourite F EY1 V ER0 IH0 T
flavorist F L EY1 V ER0 IH0 S T
food F UW1 D
is IH1 Z
rice R AY1 S
yesterday Y EH1 S T ER0 D EY2
go G OW1
went W EH1 N T
not N AA1 T
to T UW1
the DH AH0
park P AA1 R K
hello HH AH0 L OW1
friend F R EH1 N D
fraud F R AO1 D
forces F AO1 R S IH0 Z
'''


class ChatReferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / 'data/external/cmudict' / demo_reference.COMMIT / 'cmudict.dict'
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(LEXICON)
        self.hash = patch.object(demo_reference, 'DICTIONARY_SHA256', hashlib.sha256(LEXICON).hexdigest())
        self.hash.start()
        self.addCleanup(self.hash.stop)
        self.addCleanup(demo_reference._load.cache_clear)
        self.addCleanup(_buckets.cache_clear)
        self.addCleanup(_candidates.cache_clear)

    def speech(self):
        return {'transcript': 'My fravorit food is rice', 'reference_text': 'My fravorit food is rice',
                'reference_source': 'asr', 'audio': {'duration_sec': 2.5},
                'phonemes': [{'phone': phone, 'confidence': .8, 'start_sec': index * .1}
                             for index, phone in enumerate('M AY F EY V ER IH T F UW D IH Z R AY S'.split())],
                'coaching': {'status': 'ready', 'feedback': {'fravorit': 'obsolete tip'}},
                'comparison': None, 'missing_words': ['fravorit']}

    def test_known_words_keep_original_grammar_negation_and_spacing(self):
        text = 'Yesterday  I go not to the park!'
        result = reference_resolution(self.root, text)
        expected = {'status': 'asr', 'raw_text': text, 'target_text': text,
            'suggested_text': text, 'missing_words': [], 'suggestions': []}
        self.assertEqual({key: result[key] for key in expected}, expected)
        self.assertTrue(all(item['in_dictionary'] is True for item in result['word_checks']))
        self.assertTrue(result['suggestions_require_confirmation'])
        self.assertEqual(result['dictionary_sha256'], demo_reference.DICTIONARY_SHA256)

    def test_ambiguous_spelling_is_only_an_editable_draft(self):
        text = 'My fravorit food is rice'
        result = reference_resolution(self.root, text)
        self.assertEqual(result['status'], 'needs_confirmation')
        self.assertIsNone(result['target_text'])
        self.assertEqual(result['raw_text'], text)
        self.assertEqual(result['missing_words'], ['fravorit'])
        self.assertEqual(result['suggested_text'], 'My favorite food is rice')
        self.assertEqual(result['suggestions'][0]['word'], 'fravorit')
        self.assertEqual(result['suggestions'][0]['index'], 1)
        self.assertIn('favorite', result['suggestions'][0]['candidates'])
        self.assertIn('flavorist', result['suggestions'][0]['candidates'])
        self.assertEqual(_distance('fravorit', 'favorite', 2), 2)
        self.assertEqual(_distance('fravorit', 'flavorist', 2), 2)
        self.assertEqual([item['in_dictionary'] for item in result['word_checks']], [True, False, True, True, True])
        self.assertEqual(result['suggestion_method'], 'bounded_levenshtein_then_character_similarity')

    def test_draft_preserves_case_punctuation_known_tokens_and_numbers(self):
        text = 'Yesterday I go not 42, (FRAVORIT)!'
        result = reference_resolution(self.root, text)
        self.assertEqual(result['suggested_text'], 'Yesterday I go not 42, (FAVORITE)!')
        self.assertEqual(result['missing_words'], ['42,', '(FRAVORIT)!'])
        self.assertEqual(result['suggestions'][0]['candidates'], [])

    def test_names_and_unrelated_words_are_not_silently_replaced(self):
        result = reference_resolution(self.root, 'Qzx Phu zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
        self.assertEqual(result['suggested_text'], result['raw_text'])
        self.assertIsNone(result['target_text'])
        self.assertTrue(all(item['candidates'] == [] for item in result['suggestions']))

    def test_missing_or_corrupt_dictionary_produces_no_candidates(self):
        self.path.write_bytes(b'favorite MADEUP\n')
        result = reference_resolution(self.root, 'fravorit')
        self.assertIsNone(result['word_checks'][0]['in_dictionary'])
        self.assertEqual(result['status'], 'needs_confirmation')
        self.assertEqual(result['suggestions'][0]['candidates'], [])
        self.assertEqual(result['suggested_text'], 'fravorit')
        self.path.unlink()
        self.assertEqual(reference_resolution(self.root, 'favorite')['missing_words'], ['favorite'])

    def test_suggestion_search_is_bounded_per_request(self):
        words = ['f' + chr(97 + index) * 5 for index in range(20)]
        with patch('src.mdd.chat_reference._candidates', return_value=()) as candidates:
            result = reference_resolution(self.root, ' '.join(words))
        self.assertEqual(len(result['missing_words']), 20)
        self.assertEqual(candidates.call_count, 12)

    def test_confirmed_target_reuses_original_phones_and_clears_stale_coaching(self):
        speech = self.speech()
        before = deepcopy(speech)
        result = reassess_reference(self.root, speech, 'My favorite food is rice')
        self.assertEqual(speech, before)
        self.assertEqual(result['transcript'], 'My fravorit food is rice')
        self.assertEqual(result['conversation_text'], 'My favorite food is rice')
        self.assertEqual(result['phonemes'], before['phonemes'])
        self.assertEqual(result['audio'], before['audio'])
        self.assertEqual(result['reference_source'], 'user_confirmed')
        self.assertEqual(result['reference_text'], 'My favorite food is rice')
        self.assertEqual(result['reference_resolution']['status'], 'confirmed')
        self.assertEqual(result['reference_resolution']['raw_text'], before['transcript'])
        self.assertEqual(result['missing_words'], [])
        self.assertEqual(result['coaching']['status'], 'unavailable')
        self.assertEqual(result['coaching']['feedback'], {})
        self.assertEqual(result['target_words'][1], {'text': 'favorite', 'phones': 'F EY V ER IH T'.split(),
                                                   'pronunciation_phones': 'F EY1 V ER0 IH0 T'.split()})
        self.assertTrue(all(word['status'] == 'match' for word in result['comparison']['words']))
        result['phonemes'][0]['phone'] = 'Z'
        self.assertEqual(speech, before)

    def test_changed_target_can_reveal_an_actual_original_audio_difference(self):
        speech = self.speech()
        speech['phonemes'][2]['phone'] = 'P'
        result = reassess_reference(self.root, speech, 'My favorite food is rice')
        self.assertEqual(result['comparison']['words'][1]['status'], 'difference')
        self.assertIn({'kind': 'substitution', 'expected_phone': 'F', 'observed_phone': 'P', 'word_index': 1},
                      result['comparison']['events'])

    def test_posterior_reassessment_abstains_without_calibration_and_preserves_writing(self):
        import numpy as np
        from src.mdd.acoustic_evidence import pack_evidence
        speech = self.speech()
        phones = [item['phone'] for item in speech['phonemes']]
        vocabulary = {phone: index for index, phone in enumerate(['[PAD]', *sorted(set(phones))])}
        matrix = np.full((len(phones) * 2, len(vocabulary)), -9., dtype=np.float32)
        for index, phone in enumerate(phones):
            matrix[index * 2, vocabulary[phone]] = 0.
            matrix[index * 2 + 1, 0] = 0.
        matrix -= np.log(np.exp(matrix).sum(axis=1, keepdims=True))
        speech['_acoustic_evidence'] = pack_evidence({'log_probs': matrix,
            'id_to_phone': {index: phone for phone, index in vocabulary.items()},
            'blank_id': 0, 'duration_sec': 2.5, 'checkpoint_sha256': 'a' * 64, 'vocabulary_sha256': 'b' * 64})
        speech['assessment'] = {'method': 'ctc_variant_posterior_margin_v1'}
        speech['rewrite'] = {'status': 'ready', 'grammar_corrected': 'An unrelated grammar suggestion.'}
        result = reassess_reference(self.root, speech, 'My favorite food is rice')
        self.assertEqual(result['phonemes'], speech['phonemes'])
        self.assertEqual(result['_acoustic_evidence'], speech['_acoustic_evidence'])
        self.assertEqual(result['rewrite'], speech['rewrite'])
        self.assertEqual(result['reference_text'], 'My favorite food is rice')
        self.assertEqual(result['assessment']['status'], 'ready')
        self.assertFalse(result['assessment']['calibration']['applied'])
        self.assertTrue(all(word['status'] == 'uncertain' for word in result['comparison']['words']))
        self.assertTrue(all(word['status'] == 'match' for word in result['raw_comparison']['words']))

    def test_new_assessment_never_falls_back_to_raw_edits_when_cache_is_missing(self):
        speech = {**self.speech(), 'assessment': {'method': 'ctc_variant_posterior_margin_v1'}}
        with self.assertRaisesRegex(ValueError, 'stored sound evidence'):
            reassess_reference(self.root, speech, 'My favorite food is rice')

    def test_confirming_target_exposes_substitution_and_deletion_candidates_without_calibration(self):
        import numpy as np
        from src.mdd.acoustic_evidence import pack_evidence
        phones = 'F R AW T F AO S IH Z'.split()
        target_phones = 'F R AO D F AO R S IH Z'.split()
        vocabulary = ['[PAD]', *sorted(set(phones + target_phones))]
        matrix = np.full((len(phones) * 3, len(vocabulary)), -12., dtype=np.float32)
        for index, phone in enumerate(phones):
            matrix[index * 3:index * 3 + 2, vocabulary.index(phone)] = 0.
            matrix[index * 3 + 2, 0] = 0.
        matrix -= np.log(np.exp(matrix).sum(axis=1, keepdims=True))
        speech = {'transcript': 'Frout forces', 'reference_text': 'Frout forces', 'reference_source': 'asr',
            'phonemes': [{'phone': phone} for phone in phones],
            'assessment': {'method': 'ctc_variant_posterior_margin_v1'},
            '_acoustic_evidence': pack_evidence({'log_probs': matrix,
                'id_to_phone': dict(enumerate(vocabulary)), 'blank_id': 0, 'duration_sec': 2.,
                'checkpoint_sha256': 'a' * 64, 'vocabulary_sha256': 'b' * 64})}
        from src.mdd.acoustic_evidence import unpack_evidence
        from src.mdd.chat_assessment import assessment_fields
        initial = assessment_fields(self.root, 'fraud forces', unpack_evidence(speech['_acoustic_evidence']))
        self.assertEqual(initial['sound_candidates']['status'], 'available')
        self.assertTrue(initial['sound_candidates']['provisional'])
        self.assertEqual(initial['sound_candidates']['count'], 2)
        self.assertTrue(all(word['status'] == 'uncertain' for word in initial['comparison']['words']))
        self.assertFalse(initial['assessment']['calibration']['applied'])
        self.assertNotIn('Confirm what you actually said before', initial['notice'])
        before = deepcopy(speech)
        result = reassess_reference(self.root, speech, 'fraud forces')
        self.assertEqual(speech, before)
        self.assertEqual(result['transcript'], 'Frout forces')
        self.assertEqual(result['conversation_text'], 'fraud forces')
        self.assertEqual(result['reference_source'], 'user_confirmed')
        self.assertEqual(result['_acoustic_evidence'], before['_acoustic_evidence'])
        self.assertFalse(result['assessment']['calibration']['applied'])
        self.assertTrue(all(word['status'] == 'uncertain' for word in result['comparison']['words']))
        candidates = result['sound_candidates']
        self.assertTrue(candidates['reference_confirmed'])
        self.assertFalse(candidates['provisional'])
        self.assertEqual([word['text'] for word in candidates['words']], ['fraud', 'forces'])
        self.assertIn({'kind': 'deletion', 'expected_phone': 'R', 'observed_phone': None,
            'word_index': 1, 'tentative': True, 'source': 'unvalidated_greedy_edit'}, candidates['words'][1]['events'])

    def test_reassess_rejects_unknown_invalid_or_absent_audio_evidence(self):
        for target in ['', None, 'x' * 1001, 'I\x00 go', 'I ' * 151]:
            with self.subTest(target=str(target)[:30]), self.assertRaises(ValueError):
                reassess_reference(self.root, self.speech(), target)
        for phones in [None, [], ['F'], [{'phone': ''}], [{'phone': '[PAD]'}], [{'phone': 'GOLDERROR'}],
                       [{'phone': 'F'}] * 1501]:
            with self.subTest(phones=str(phones)[:30]), self.assertRaises(ValueError):
                reassess_reference(self.root, {**self.speech(), 'phonemes': phones}, 'favorite')
        with self.assertRaises(ValueError):
            reassess_reference(self.root, {'phonemes': [{'phone': 'F'}]}, 'favorite')

    def test_confirmed_name_outside_dictionary_is_conversation_text_not_an_acoustic_error(self):
        speech = self.speech()
        before = deepcopy(speech)
        result = reassess_reference(self.root, speech, 'My friend Nguyenphu')
        self.assertEqual(speech, before)
        self.assertEqual(result['transcript'], speech['transcript'])
        self.assertEqual(result['conversation_text'], 'My friend Nguyenphu')
        self.assertEqual(result['reference_source'], 'user_confirmed')
        self.assertEqual(result['reference_resolution']['status'], 'confirmed')
        self.assertEqual(result['reference_resolution']['target_text'], 'My friend Nguyenphu')
        self.assertEqual(result['reference_resolution']['missing_words'], ['Nguyenphu'])
        self.assertEqual(result['missing_words'], ['Nguyenphu'])
        self.assertEqual(result['comparison']['events'], [])
        self.assertTrue(all(word['status'] == 'unavailable' for word in result['comparison']['words']))
        self.assertEqual(result['sound_candidates']['status'], 'unavailable')
        self.assertEqual(result['sound_candidates']['reason'], 'reference_contains_unknown_words')
        self.assertEqual(result['sound_candidates']['count'], 0)
        self.assertEqual(result['target_words'][1]['phones'], ['F', 'R', 'EH', 'N', 'D'])
        self.assertEqual(result['target_words'][2], {'text': 'Nguyenphu', 'phones': []})
        restored = reassess_reference(self.root, result, 'My favorite food is rice')
        self.assertEqual(restored['missing_words'], [])
        self.assertTrue(all(word['status'] == 'match' for word in restored['comparison']['words']))
        self.assertNotIn('sound_candidates', restored)

    def test_confirmed_oov_with_posterior_evidence_keeps_missing_reason_and_never_scores_known_words(self):
        import numpy as np
        from src.mdd.acoustic_evidence import pack_evidence
        speech = self.speech()
        matrix = np.log(np.array([[.1, .9], [.9, .1]], dtype=np.float32))
        speech['_acoustic_evidence'] = pack_evidence({'log_probs': matrix,
            'id_to_phone': {0: '[PAD]', 1: 'F'}, 'blank_id': 0, 'duration_sec': 2.,
            'checkpoint_sha256': 'a' * 64, 'vocabulary_sha256': 'b' * 64})
        speech['assessment'] = {'method': 'ctc_variant_posterior_margin_v1'}
        result = reassess_reference(self.root, speech, 'My friend Nguyenphu')
        self.assertEqual(result['conversation_text'], 'My friend Nguyenphu')
        self.assertEqual(result['assessment']['status'], 'unavailable')
        self.assertEqual(result['assessment']['reason'], 'reference_contains_unknown_words')
        self.assertEqual(result['sound_candidates']['reason'], 'reference_contains_unknown_words')
        self.assertEqual(result['_acoustic_evidence'], speech['_acoustic_evidence'])
        self.assertEqual(result['comparison']['events'], [])
        self.assertTrue(all(word['status'] == 'unavailable' for word in result['comparison']['words']))
        self.assertTrue(all(word['error_score'] is None for word in result['assessment']['words']))

    def test_reassess_rejects_excessively_large_canonical_sequence(self):
        reference = {'prompt_text': 'favorite', 'canonical_phones': ['F'] * 1000,
                     'canonical_word_indices': [0] * 1000}
        with patch.object(demo_reference, 'dictionary_reference', return_value=(reference, [])):
            with self.assertRaises(ValueError):
                reassess_reference(self.root, self.speech(), 'favorite')

    def test_shared_reference_still_abstains_for_entire_unknown_sentence(self):
        reference, missing = demo_reference.dictionary_reference(self.root, 'My fravorit food')
        self.assertIsNone(reference)
        self.assertEqual(missing, ['fravorit'])


if __name__ == '__main__':
    unittest.main()
