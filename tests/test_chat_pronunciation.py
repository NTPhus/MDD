"""Display stress must come from the same verified canonical pronunciation."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.mdd import demo_reference
from src.mdd.chat_pronunciation import _load, target_words


LEXICON = b'''added AE1 D AH0 D
added(2) AE1 D IH0 D
with W IH1 DH
with(2) W IH0 TH
genuine JH EH1 N Y UW2 AH0 N
genuine(2) JH EH1 N Y UW0 IH0 N
i'm AY1 M
hello HH AH0 L OW1
'''


class ChatPronunciationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / 'data/external/cmudict' / demo_reference.COMMIT / 'cmudict.dict'
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(LEXICON)
        checksum = patch.object(demo_reference, 'DICTIONARY_SHA256', hashlib.sha256(LEXICON).hexdigest())
        checksum.start()
        self.addCleanup(checksum.stop)
        self.addCleanup(_load.cache_clear)
        self.addCleanup(demo_reference._load.cache_clear)

    def reference(self, text):
        reference, missing = demo_reference.dictionary_reference(self.root, text)
        self.assertEqual(missing, [])
        self.assertIsNotNone(reference)
        return reference

    def test_preserves_primary_secondary_and_unstressed_vowels_for_display_only(self):
        reference = self.reference('added with genuine')
        result = target_words(self.root, 'added with genuine', reference)
        self.assertEqual(result, [
            {'text': 'added', 'phones': 'AE D AH D'.split(), 'pronunciation_phones': 'AE1 D AH0 D'.split()},
            {'text': 'with', 'phones': 'W IH DH'.split(), 'pronunciation_phones': 'W IH1 DH'.split()},
            {'text': 'genuine', 'phones': 'JH EH N Y UW AH N'.split(), 'pronunciation_phones': 'JH EH1 N Y UW2 AH0 N'.split()},
        ])
        self.assertTrue(all(not phone[-1:].isdigit() for phone in reference['canonical_phones']))

    def test_lookup_matches_case_punctuation_unicode_and_apostrophe_normalization(self):
        text = '(ADDED)! “ＷＩＴＨ,” \'Genuine\' I’m'
        result = target_words(self.root, text, self.reference(text))
        self.assertEqual([item['text'] for item in result], text.split())
        self.assertEqual([item['pronunciation_phones'] for item in result], [
            'AE1 D AH0 D'.split(), 'W IH1 DH'.split(), 'JH EH1 N Y UW2 AH0 N'.split(), 'AY1 M'.split(),
        ])

    def test_first_listed_variant_is_deterministic_and_never_selected_by_audio(self):
        reference = self.reference('with genuine')
        first = target_words(self.root, 'with genuine', reference)
        second = target_words(self.root, 'with genuine', deepcopy(reference))
        self.assertEqual(first, second)
        self.assertEqual(first[0]['pronunciation_phones'], 'W IH1 DH'.split())
        self.assertEqual(first[1]['pronunciation_phones'], 'JH EH1 N Y UW2 AH0 N'.split())

    def test_different_selected_reference_does_not_receive_wrong_variant_metadata(self):
        reference = {'canonical_phones': 'W IH TH'.split(), 'canonical_word_indices': [0, 0, 0]}
        self.assertEqual(target_words(self.root, 'with', reference), [{'text': 'with', 'phones': 'W IH TH'.split()}])

    def test_corrupt_or_missing_dictionary_keeps_existing_reference_phones(self):
        reference = self.reference('added with')
        expected = [{'text': 'added', 'phones': 'AE D AH D'.split()}, {'text': 'with', 'phones': 'W IH DH'.split()}]
        self.path.write_bytes(b'added MADEUP1\nwith MADEUP2\n')
        self.assertEqual(target_words(self.root, 'added with', reference), expected)
        self.path.unlink()
        self.assertEqual(target_words(self.root, 'added with', reference), expected)

    def test_word_absent_from_dictionary_keeps_its_selected_canonical_phones(self):
        reference = {'canonical_phones': ['F', 'UW'], 'canonical_word_indices': [0, 0]}
        self.assertEqual(target_words(self.root, 'unknown', reference), [{'text': 'unknown', 'phones': ['F', 'UW']}])

    def test_no_reference_means_no_target_words(self):
        self.assertEqual(target_words(self.root, 'added', None), [])
        self.assertEqual(target_words(self.root, 'added', {}), [])

    def test_inputs_and_cached_dictionary_cannot_be_mutated_through_output(self):
        text = 'added genuine'
        reference = self.reference(text)
        before = deepcopy(reference)
        result = target_words(self.root, text, reference)
        self.assertEqual(reference, before)
        result[0]['phones'][0] = 'Z'
        result[0]['pronunciation_phones'][0] = 'Z1'
        self.assertEqual(reference, before)
        fresh = target_words(self.root, text, reference)
        self.assertEqual(fresh[0]['phones'][0], 'AE')
        self.assertEqual(fresh[0]['pronunciation_phones'][0], 'AE1')

    def test_dictionary_symlink_outside_root_is_not_used(self):
        reference = self.reference('added')
        expected = [{'text': 'added', 'phones': 'AE D AH D'.split()}]
        with tempfile.TemporaryDirectory() as outside:
            external = Path(outside) / 'cmudict.dict'
            external.write_bytes(LEXICON)
            self.path.unlink()
            try:
                self.path.symlink_to(external)
            except OSError:
                # Creating symlinks needs a Windows privilege on some machines.
                self.skipTest('File symlinks are not available on this platform.')
            self.assertEqual(target_words(self.root, 'added', reference), expected)


if __name__ == '__main__':
    unittest.main()
