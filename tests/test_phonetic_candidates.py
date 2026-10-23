"""Sound-neighbour retrieval must use audio phones, never the intended hint."""
from pathlib import Path
import hashlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import mdd.phonetic_candidates as retrieval
from mdd.demo_reference import COMMIT
from mdd.pronunciation_assessment import _read_dictionary

LEXICON = b'''difficult D IH1 F AH0 K AH0 L T
it IH1 T
itch IH1 CH
dovecote D AH1 V K OW0 T
true T R UW1
to T UW1
get G EH1 T
jet JH EH1 T
read R IY1 D
read(2) R EH1 D
red R EH1 D
'''


class PhoneticCandidateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        path = self.root / 'data/external/cmudict' / COMMIT / 'cmudict.dict'
        path.parent.mkdir(parents=True); path.write_bytes(LEXICON)
        identity = patch.object(retrieval, 'DICTIONARY_SHA256', hashlib.sha256(LEXICON).hexdigest())
        identity.start(); self.addCleanup(identity.stop)
        self.addCleanup(retrieval._phone_buckets.cache_clear)
        self.addCleanup(retrieval._nearest.cache_clear)
        self.addCleanup(_read_dictionary.cache_clear)

    def retrieve(self, word, phones):
        phonemes = [{'phone': phone, 'start_sec': i / len(phones), 'end_sec': (i + 1) / len(phones)} for i, phone in enumerate(phones)]
        return retrieval.sound_candidates(self.root, [{'text': word, 'start_sec': 0., 'end_sec': 1.}], phonemes, word)

    def test_rank_by_phonemes_instead_of_the_nearest_spelling(self):
        result = self.retrieve('divicote', 'D IH F AH K AH L T'.split())
        self.assertEqual(result['words'][0]['candidates'][0]['word'], 'difficult')
        self.assertEqual(result['words'][0]['candidates'][0]['edit_distance'], 0)
        self.assertTrue(result['diagnostic_only'])
        self.assertTrue(result['requires_confirmation'])
        self.assertNotIn('reference_text', result)

    def test_known_asr_word_can_have_a_different_sound_neighbour(self):
        result = self.retrieve('true', ['T', 'UW'])
        self.assertEqual(result['words'][0]['candidates'][0]['word'], 'to')
        self.assertEqual(result['words'][0]['asr_word'], 'true')

    def test_exact_sound_neighbour_can_coexist_with_the_intended_asr_word(self):
        result = self.retrieve('it', ['IH', 'CH'])
        candidates = result['words'][0]['candidates']
        self.assertEqual(candidates[0]['word'], 'itch')
        self.assertTrue(any(row['word'] == 'it' and row['edit_distance'] == 1 for row in candidates))

    def test_all_dictionary_variants_are_considered_without_duplicate_words(self):
        result = self.retrieve('read', ['R', 'EH', 'D'])
        candidates = result['words'][0]['candidates']
        self.assertEqual(candidates[0]['word'], 'read')
        self.assertEqual(candidates[0]['edit_distance'], 0)
        self.assertEqual(sum(row['word'] == 'read' for row in candidates), 1)

    def test_word_timing_mismatch_refuses_to_associate_another_word(self):
        result = retrieval.sound_candidates(self.root, [{'text': 'other', 'start_sec': 0., 'end_sec': 1.}], [], 'true')
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['words'], [])

    def test_ctc_repetition_blanks_and_edges_keep_all_word_time(self):
        result = retrieval.word_intervals([1, 1, 0, 2, 2, 3, 4, 5, 2, 0],
            {0: '<pad>', 1: 'I', 2: 'T', 3: '|', 4: 'G', 5: 'E'}, 0, '|', 1.)
        self.assertEqual([row['text'] for row in result], ['IT', 'GET'])
        self.assertEqual(result[0]['start_sec'], 0.)
        self.assertEqual(result[-1]['end_sec'], 1.)
        self.assertEqual(result[0]['end_sec'], result[1]['start_sec'])

    def test_missing_evidence_produces_no_fake_word_or_probability(self):
        result = self.retrieve('unknown', ['Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z'])
        self.assertEqual(result['words'][0]['candidates'], [])
        self.assertNotIn('confidence', result['words'][0])


if __name__ == '__main__': unittest.main()
