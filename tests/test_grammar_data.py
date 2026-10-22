"""Published annotation conversion, isolation and exact assistant supervision."""
from copy import deepcopy
import unittest

from scripts.grammar_data import canonical, document_map, encode, leakage_audit, parse_m2, reserve_splits


class GrammarDataTests(unittest.TestCase):
    def group(self, identifier, source, refs, split, task='grammar_corrected', **kwargs):
        return {'id': identifier, 'source': source, 'references': refs, 'split': split,
                'task': task, 'group_id': identifier, 'variants': [{'unknown': False}], **kwargs}

    def test_annotations_are_separate_not_concatenated(self):
        rows = parse_m2('S This are a sentence .\n'
                        'A 1 2|||R:VERB:SVA|||is|||REQUIRED|||-NONE-|||0\n'
                        'A 3 3|||M:ADJ|||good|||REQUIRED|||-NONE-|||0\n'
                        'A 1 2|||R:VERB:SVA|||is|||REQUIRED|||-NONE-|||1\n'
                        'A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||2\n')
        self.assertEqual([' '.join(v['corrected_tokens']) for v in rows[0]['variants']],
                         ['This is a good sentence .', 'This is a sentence .', 'This are a sentence .'])

    def test_insert_delete_adjacent_and_empty_edits(self):
        row = parse_m2('S I go school now .\n'
                       'A 1 2|||R:VERB:TENSE|||went|||REQUIRED|||-NONE-|||0\n'
                       'A 2 2|||M:PREP|||to|||REQUIRED|||-NONE-|||0\n'
                       'A 3 4|||U:ADV||||||REQUIRED|||-NONE-|||0\n')[0]
        self.assertEqual(row['variants'][0]['corrected_tokens'], ['I', 'went', 'to', 'school', '.'])
        self.assertEqual(parse_m2('S Hello .\n')[0]['variants'][0]['corrected_tokens'], ['Hello', '.'])

    def test_overlaps_invalid_offsets_and_malformed_data_fail_closed(self):
        for edits in [
            'A 0 2|||R:OTHER|||hi|||REQUIRED|||-NONE-|||0\nA 1 3|||R:OTHER|||bad|||REQUIRED|||-NONE-|||0',
            'A -1 2|||R:OTHER|||hi|||REQUIRED|||-NONE-|||0',
            'A 0 9|||R:OTHER|||hi|||REQUIRED|||-NONE-|||0',
            'A malformed',
        ]:
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                parse_m2('S These are words .\n' + edits)

    def test_unknown_is_not_a_correct_control(self):
        row = parse_m2('S Unknownword .\nA 0 1|||UNK|||Unknownword|||REQUIRED|||-NONE-|||0')[0]
        group = self.group('a', 'Unknownword.', ['Unknownword.'], 'train')
        group['variants'] = row['variants']
        parts, audit = reserve_splits([group])
        self.assertEqual(parts['train'], [])
        self.assertEqual(audit['train:unknown_reference'], 1)

    def test_essay_mapping_proves_boundaries_and_rejects_guessing(self):
        records = parse_m2('S It is here .\n\nS Do n\'t go .\n\nS Fine !\n')
        docs = [{'id': 'a', 'text': "It is here. Don't go."}, {'id': 'b', 'text': 'Fine!'}]
        self.assertEqual([d['id'] for d in document_map(docs, records)], ['a', 'a', 'b'])
        docs[0]['text'] = 'It is wrong.'
        with self.assertRaises(ValueError):
            document_map(docs, records)

    def test_all_references_and_hidden_test_are_reserved_across_tasks(self):
        groups = [self.group('test', 'I goes there.', ['I go there.'], 'test'),
                  self.group('val', 'I GO there!', ['I go there.'], 'validation', task='natural'),
                  self.group('train', 'I go there', ['I go there.'], 'train'),
                  self.group('hidden', 'Secret sentence.', ['Secret sentence.'], 'train'),
                  self.group('ok', 'A separate text.', ['A separate text.'], 'train')]
        parts, audit = reserve_splits(groups, ['Secret sentence .'])
        self.assertEqual([r['id'] for r in parts['train']], ['ok'])
        self.assertEqual(parts['validation'], [])
        self.assertEqual(audit['train:cross_split_overlap'], 2)
        leakage_audit(parts)

    def test_leakage_check_rejects_same_author_or_reference(self):
        a = self.group('a', 'First source.', ['First reference.'], 'train')
        b = self.group('b', 'Second source.', ['Second reference.'], 'test')
        parts = {'train': [a], 'test': [b], 'validation': []}
        leakage_audit(parts)
        b['group_id'] = a['group_id']
        with self.assertRaisesRegex(ValueError, 'leakage'):
            leakage_audit(parts)
        b['group_id'] = 'b'
        b['references'] = ['First Reference!']
        with self.assertRaisesRegex(ValueError, 'leakage'):
            leakage_audit(parts)

    def test_task_target_only_and_prompt_mask_no_truncation(self):
        class Tokenizer:
            eos_token = '!'
            def apply_chat_template(self, messages, **kwargs):
                assert kwargs['enable_thinking'] is False
                assert 'task' in messages[-1]['content']
                return 'PROMPT'
            def __call__(self, text, **kwargs):
                return {'input_ids': list(text.encode())}
        row = {'id': 'a', 'task': 'natural', 'source': 'I likes this.', 'target': 'I like this.'}
        encoded = encode(Tokenizer(), row, 100)
        self.assertEqual(encoded['labels'][:6], [-100] * 6)
        supervised = bytes(x for x in encoded['labels'] if x != -100).decode()
        self.assertEqual(supervised, '{"natural":"I like this."}!')
        self.assertNotIn('grammar_corrected', supervised)
        with self.assertRaisesRegex(ValueError, 'never truncate'):
            encode(Tokenizer(), row, 7)


if __name__ == '__main__':
    unittest.main()
