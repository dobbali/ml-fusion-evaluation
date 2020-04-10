from typing import List
from dataclasses import dataclass, field
from sklearn.metrics import accuracy_score, \
    roc_curve, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, precision_recall_curve
import pandas as pd
import numpy as np


@dataclass
class Roc:
    """Creates a class for Roc metrics

     Attributes:
         fpr: Increasing False Positive rates
         tpr: Increasing True Positive rates
         thr_roc: thresholds
    """
    fpr: List[float]
    tpr: List[float]
    thr_roc: List[float]


@dataclass
class Prc:
    """Creates a class for Precision curve metrics

    Attributes:
        precision_thr: Increasing False Positive rates
        recall_thr: Increasing True Positive rates
        thr_prc: thresholds
    """
    precision_thr: List[float]
    recall_thr: List[float]
    thr_prc: List[float]


@dataclass
class Fvr:
    """Creates a class for Recall Curve test_metrics

    Attributes:
        fraction:
        recall_fvr:
        thr_fvr:
    """
    fraction: List[float]
    recall_fvr: List[float]
    thr_fvr: List[float]


@dataclass
class ClassificationEval:
    """Creates ClassificationEval class"""
    y_pred: List
    y_true: List
    threshold: float = 0.5
    pos_label: int = 1
    neg_label: int = field(init=False)

    def __post_init__(self):
        self.neg_label = int(not bool(self.pos_label))

    def get_binary_y_pred(self) -> List:
        """Updates attribute y_pred_binary using y_pred"""
        y_pred_binary = np.array(self.y_pred)
        y_pred_binary[y_pred_binary >= self.threshold] = self.pos_label
        y_pred_binary[y_pred_binary < self.threshold] = self.neg_label
        return list(y_pred_binary)


@dataclass
class ClassificationMetrics:
    # pylint: disable=too-many-instance-attributes
    """Creates a class for Classification Metrics

    Attributes:
        accuracy (float): Accuracy of the model
        auc_roc (float) :  Area Under the ROC curve for prediction scores
        auc_prc (float) :  Average precision (AP) from prediction scores
        precision (float): Precision of the model at certain threshold
        recall (float): Recall of the model at certain threshold
        f1score (float): f1score of the model at certain threshold
        roc (Roc) : Metrics to build ROC curve
        prc (Prc) : Metrics to build Precision Recall curve
        fvr (Fvr) : Metrics to build Recall curve
    """

    accuracy: float
    auc_roc: float
    auc_prc: float
    precision: float
    recall: float
    f1score: float
    roc: Roc
    prc: Prc
    fvr: Fvr


def _get_fvr(y_true: List, y_pred: List):
    """Provide metrics fraction, recall, thr to plot test population above decision threshold
    vs associated recall(TPR)"""
    df_aud = pd.DataFrame({'y': y_true, 'prob_a': y_pred}).sort_values('prob_a', ascending=True)
    fraction = np.arange(df_aud.shape[0]) / float(df_aud.shape[0])
    recall_fvr = df_aud['y'].cumsum() / df_aud['y'].sum()
    thr_fvr = df_aud['prob_a']

    return fraction, recall_fvr, thr_fvr


def metrics(classification_eval: ClassificationEval) -> ClassificationMetrics:
    # pylint: disable=too-many-locals
    """Returns Metrics for classification model"""
    y_pred = classification_eval.y_pred
    y_true = classification_eval.y_true
    y_pred_binary = classification_eval.get_binary_y_pred()
    pos_label = classification_eval.pos_label

    accuracy = accuracy_score(y_true, y_pred_binary)
    auc_roc = roc_auc_score(y_true, y_pred)
    auc_prc = average_precision_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1score = f1_score(y_true, y_pred_binary)

    fpr, tpr, thr_roc = roc_curve(y_true, y_pred, pos_label)
    precision_thr, recall_thr, thr_prc = precision_recall_curve(y_true, y_pred, pos_label)
    fraction, recall_fvr, thr_fvr = _get_fvr(y_true, y_pred)

    roc = Roc(fpr.tolist(), tpr.tolist(), thr_roc.tolist())
    prc = Prc(precision_thr.tolist(), recall_thr.tolist, thr_prc.tolist())
    fvr = Fvr(fraction.tolist(), recall_fvr.tolist(), thr_fvr.tolist())

    return ClassificationMetrics(accuracy=accuracy, auc_roc=auc_roc, auc_prc=auc_prc,
                                 precision=precision, recall=recall, f1score=f1score,
                                 roc=roc, prc=prc, fvr=fvr)
