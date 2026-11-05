"""Detection model
    train and test a model on a given data
"""
# Authors: kun.bj@outlook.com
#
# License: xxx

import numpy as np
from pyod.models.base import BaseDetector
from sklearn import metrics
from sklearn.base import OutlierMixin
from sklearn.metrics import roc_curve

from netml.utils.tool import timing


def fit_threshold(detector, X_train):
    """Set the decision threshold of a pyod-based detector, after it is fitted.

    pyod's predict() needs the scores of the training data ("decision_scores_") and a threshold
    ("threshold_"): the flows scoring above the threshold are anomalies. The threshold is set so
    that a fraction "contamination" (e.g. 0.1) of the training flows score above it.

    Also records the number of features, so predictions on data of another shape fail clearly.
    """
    X_train = np.asarray(X_train)
    detector.decision_scores_ = detector.decision_function(X_train)
    detector._process_decision_scores()

    # (some pyod detectors define n_features_in_ as a read-only property themselves)
    if not isinstance(getattr(type(detector), 'n_features_in_', None), property):
        detector.n_features_in_ = X_train.shape[1]


def predict_anomalies(model, X):
    """Predict which flows are anomalies, whichever convention the model follows.

    The OCSVM model (scikit-learn) predicts 1 for normal and -1 for anomalies; the other models
    (pyod) predict 0 for normal and 1 for anomalies.

    Parameters
    ----------
    model: a fitted netml model (see netml.ndm)

    X: array of shape (n_flows, n_features)

    Returns
    -------
        is_anomaly: boolean array of shape (n_flows,)
    """
    X = np.asarray(X)

    n_features = getattr(model, 'n_features_in_', None)
    if n_features is not None and X.ndim == 2 and X.shape[1] != n_features:
        raise ValueError(f'the model expects {n_features} features per flow, but got '
                         f'{X.shape[1]}: extract features with the settings the model was '
                         'trained with')

    if isinstance(model, BaseDetector):
        if not hasattr(model, 'threshold_'):
            raise ValueError(f'this {type(model).__name__} model has no decision threshold: it '
                             'was trained by an older version of netml, whose IF, KDE, GMM, PCA '
                             'and AE models could not classify; retrain it (netml learn)')
        return model.predict(X) == 1

    if isinstance(model, OutlierMixin):
        return model.predict(X) == -1

    raise TypeError(f'cannot predict with a {type(model).__name__} model')


class MODEL:

    def __init__(self, model=None, *, score_metric='auc', verbose=1, random_state=42):
        """Train and test a model on a given data.

        Parameters
        ----------
        model: instance
            a detection model instance.

        score_metric: str (default is 'auc')
            a score we used to evaluate the model

        verbose: int (default is 1)
            a print level is to control what information should be printed according to the given
            value.
            The higher the value is, the more info is printed.

        random_state: int
            a value is to make your experiments more reproducible.

        Returns
        -------
            a MODEL instance
        """

        self.model = model
        self.model_name = model.name
        self.score_metric = score_metric
        self.verbose = verbose
        self.random_state = random_state
        # store all data generated during training and testing the model.
        self.history = {}

    @timing
    def _train(self, X_train, y_train=None):
        """fit the model on the train set

        Parameters
        ----------
        X_trian: array

        y_train: array (default is None)
            in unsupervised learning setting, there is no requirement of ground truth to
            fit the model

        Returns
        -------
            self
        """

        self.model.fit(X_train, y_train)

    def train(self, X_train, y_train=None):
        """fit the model on the train set

        Parameters
        ----------
        X_trian: array

        y_train: array (default is None)
            in unsupervised learning setting, there is no requirement of ground truth to
            fit the model

        Returns
        -------
            self
        """
        _, tot_time = self._train(X_train, y_train)
        self.train.__dict__['tot_time'] = tot_time

    @timing
    def _test(self, X_test, y_test):
        """Evaluate the model on the test set

        Parameters
        ----------
        X_test: array

        y_test: array
            ground true

        Returns
        -------
            self
        """
        y_score = self.model.decision_function(X_test)
        if self.score_metric == 'auc':
            # For binary  y_true, y_score is supposed to be the score of the class with greater
            # label.
            # pos_label = 1, so y_score should be the corresponding score (i.e., novel score)
            fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
            self.score = metrics.auc(fpr, tpr)

            self.history['score'] = self.score

    def test(self, X_test, y_test):
        """Evaluate the model on the test set

        Parameters
        ----------
        X_test: array

        y_test: array
            ground true

        Returns
        -------
            self
        """
        _, tot_time = self._test(X_test, y_test)
        self.test.__dict__['tot_time'] = tot_time
