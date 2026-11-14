"""Fix 5: the Isolation Forest model ranks anomalies above normal flows.

The bug
    pyod's IForest already turns scikit-learn's scores round, so that outliers get the higher
    scores (pyod's convention for every model). netml's IF wrapper turned them round a second time
    with pyod's invert_order(), so normal flows got the highest anomaly scores and the area under
    the ROC curve (AUC) fell below 0.5, i.e. worse than chance.

Affected
    IF.decision_function(), and so the AUC that ``netml learn --algorithm if`` reports.
"""
import unittest

import numpy as np
from pyod.models.iforest import IForest
from sklearn.metrics import roc_auc_score

from netml.ndm.iforest import IF
from netml.pparser.parser import PCAP

from ._helpers import DEMO_LABELS, DEMO_PCAP, RealCaptureTestCase, quiet

RANDOM_STATE = 42


def _labeled_demo_features():
    """IAT features and labels (0 normal, 1 anomaly) of the labeled flows of demo.pcap.

    demo.pcap and its label file demo.csv (in tests/data) are the Assignment 3 demo data.
    """
    pcap = PCAP(str(DEMO_PCAP))
    with quiet():
        pcap.pcap2flows(q_interval=0.9)
        pcap.label_flows(label_file=str(DEMO_LABELS))
        pcap.flow2features('IAT', fft=False, header=False)
    return pcap.features, pcap.labels


class IsolationForestTests(RealCaptureTestCase):
    """IF on the labeled flows of the Assignment 3 demo (102 flows, 16 of them anomalies)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        (cls.X, cls.y) = _labeled_demo_features()
        with quiet():
            cls.model = IF(random_state=RANDOM_STATE, verbose=0).fit(cls.X)

    def test_test_data(self):
        """Check of the test data itself: both classes are present."""
        self.assertEqual(sorted(set(self.y.tolist())), [0, 1])

    def test_anomalies_ranked_above_normal_flows(self):
        """The anomaly scores rank anomalies above normal flows more often than not.

        Expected: AUC above 0.5 (0.5 is chance; about 0.59 here).
        Before the fix: about 0.41, the mirror image (1 - AUC), because the scores were inverted.
        """
        auc = roc_auc_score(self.y, self.model.decision_function(self.X))
        self.assertGreater(auc, 0.5)

    def test_scores_match_pyod(self):
        """IF's scores are those of pyod's IForest trained with the same settings.

        Expected: identical scores. Before the fix: the scores were turned round
        (pyod's invert_order: the scores multiplied by -1).
        """
        with quiet():
            reference = IForest(n_estimators=100, max_samples='auto', contamination=0.1,
                                max_features=1., bootstrap=False, n_jobs=-1,
                                random_state=RANDOM_STATE, verbose=0).fit(self.X)
        np.testing.assert_allclose(self.model.decision_function(self.X),
                                   reference.decision_function(self.X))


if __name__ == '__main__':
    unittest.main()
