"""Protect reference-wide denominators and policy abstention semantics."""
import unittest
from scripts import report_pronunciation_improvement as report


class PronunciationReportTests(unittest.TestCase):
    def test_abstention_cannot_inflate_all_error_recall(self):
        row={'variant_review_full39':{'counts':{
            'reference_words':10,'gold_error_words':4,'available_words':4,
            'hint_tp':1,'hint_fn':1,'hint_fp':0,'hint_tn':2,
            'unavailable_error_words':2,'unavailable_correct_words':4,
            'calibrated_decided_words':0}}}
        actual=report.review_totals([row],'variant_review_full39')
        self.assertEqual(actual['recall_all_errors'],.25)
        self.assertEqual(actual['recall_available_only'],.5)
        self.assertEqual(actual['coverage'],.4)

    def test_all_abstain_is_not_perfect_score(self):
        row={'review':{'counts':{'reference_words':4,'gold_error_words':2,'available_words':0,
                                'unavailable_error_words':2,'unavailable_correct_words':2,
                                'hint_tp':0,'hint_fn':0,'hint_fp':0,'hint_tn':0}}}
        value=report.review_totals([row],'review')
        self.assertEqual(value['recall_all_errors'],0)
        self.assertEqual(value['coverage'],0)
        self.assertIsNone(value['false_positive_rate_decided'])
        self.assertIsNone(value['precision'])

    def test_inconsistent_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'denominators'):
            report.review_totals([{'review':{'counts':{'reference_words':10,'gold_error_words':4,
                'available_words':10,'hint_tp':1,'hint_fn':1,'hint_fp':1,'hint_tn':7,
                'unavailable_error_words':0,'unavailable_correct_words':0}}}],'review')

    def test_policy_risk_keeps_abstention_outside_decided(self):
        row={'tp':1,'fp':1,'tn':1,'fn':1,'decided_words':4,'total_words':10}
        value=report.policy_rows({'policy':{'groups':{'test':{'cases':1,'policies':{'selective-two-threshold':row}}},
            'eligibility':{'selective-two-threshold':{'eligible':False,'failures':['heldout_coverage']}}}})[0]
        self.assertEqual(value['risk_on_decided'],.5)
        self.assertFalse(value['eligible'])

    def test_human_text_is_escaped(self):
        self.assertEqual(report.esc('<script>'), '&lt;script&gt;')
        self.assertEqual(report.number(None,True),'—')

    def test_missing_threshold_is_explicit_not_zero_error_success(self):
        value={'policy':{'selected':{'error-cutoff-f05':{'thresholds':None}}}}
        self.assertIn('không tìm được ngưỡng',report.policy_label(value,'error-cutoff-f05'))
        self.assertNotIn('không tìm được ngưỡng',report.policy_label(value,'dictionary-variant-review'))


if __name__=='__main__':unittest.main()
