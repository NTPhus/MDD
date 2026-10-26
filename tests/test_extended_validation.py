"""External tests must preserve abstention denominators and frozen case identity."""
from unittest import TestCase
from unittest.mock import Mock, patch
from scripts import evaluate_extended_validation as evaluation
from scripts import report_extended_validation as report


class ExternalValidationTests(TestCase):
    def row(self):
        return {'speaker_id':'speaker','demo_review_counts':{
            'reference_words':10,'gold_error_words':4,'tp':1,'fp':1,'tn':2,'fn':1,
            'unavailable_error_words':2,'unavailable_correct_words':3}}

    def test_all_error_recall_keeps_abstained_errors_in_denominator(self):
        result=report.word_review([self.row()])
        self.assertEqual(result['recall_all_errors'],.25)
        self.assertEqual(result['recall_available'],.5)
        self.assertAlmostEqual(result['false_positive_all_correct'],1/6)
        self.assertEqual(result['coverage'],.5)

    def test_all_abstention_has_no_precision_or_successful_decisions(self):
        row={'demo_review_counts':{'reference_words':10,'gold_error_words':4,
            'unavailable_error_words':4,'unavailable_correct_words':6}}
        result=report.word_review([row])
        self.assertEqual(result['recall_all_errors'],0)
        self.assertEqual(result['coverage'],0)
        self.assertIsNone(result['precision'])

    def test_inconsistent_reference_counts_fail_visibly(self):
        row=self.row(); row['demo_review_counts']['gold_error_words']=5
        with self.assertRaisesRegex(ValueError,'denominators'):
            report.word_review([row])

    def test_speaker_bootstrap_is_reproducible_without_changing_point_estimates(self):
        rows=[self.row(),dict(self.row(),speaker_id='other')]
        result=report.clustered_intervals(rows,100)
        self.assertEqual(result,report.clustered_intervals(rows,100))
        self.assertEqual(result['recall_all_errors_95'],[.25,.25])

    def test_resume_accepts_only_registered_prefix_and_model_identity(self):
        original={'id':'first','source':'A learner sentence'}
        row={**original,'source_row_sha256':evaluation.canonical_sha(original),'manifest_sha256':'locked'}
        path=Mock(); path.exists.return_value=True
        with patch.object(evaluation,'lines',return_value=[row]):
            self.assertEqual(evaluation.prior_rows(path,[original],'locked'),[row])
            with self.assertRaisesRegex(ValueError,'changed'):
                evaluation.prior_rows(path,[dict(original,source='Different text')],'locked')
            with self.assertRaisesRegex(ValueError,'changed'):
                evaluation.prior_rows(path,[original],'another-manifest')

    def test_resume_cannot_silently_drop_reordered_or_duplicate_records(self):
        original={'id':'first'}; path=Mock(); path.exists.return_value=True
        with patch.object(evaluation,'lines',return_value=[{'id':'another'}]):
            with self.assertRaisesRegex(ValueError,'prefix'):
                evaluation.prior_rows(path,[original],'locked')
