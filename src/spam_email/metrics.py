from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics


@dataclass(slots=True)
class MetricsReport:
    """
    封裝模型評估指標與圖表資料。
    """

    labels: List[str]
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1: Dict[str, float]
    support: Dict[str, float]
    roc_auc: float
    confusion_matrix: pd.DataFrame
    roc_curve: Tuple[np.ndarray, np.ndarray, np.ndarray]
    pr_curve: Tuple[np.ndarray, np.ndarray, np.ndarray]

    @classmethod
    def from_predictions(
        cls,
        y_true,
        y_pred,
        y_prob,
        target_names: List[str],
    ) -> "MetricsReport":
        report = metrics.classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        confusion = metrics.confusion_matrix(y_true, y_pred, labels=target_names)
        confusion_df = pd.DataFrame(
            confusion,
            index=[f"實際-{label}" for label in target_names],
            columns=[f"預測-{label}" for label in target_names],
        )

        fpr, tpr, roc_thresholds = metrics.roc_curve(
            (y_true == target_names[1]).astype(int),
            y_prob[:, 1],
        )
        precision_curve, recall_curve, pr_thresholds = metrics.precision_recall_curve(
            (y_true == target_names[1]).astype(int),
            y_prob[:, 1],
        )

        return cls(
            labels=target_names,
            precision={label: report[label]["precision"] for label in target_names},
            recall={label: report[label]["recall"] for label in target_names},
            f1={label: report[label]["f1-score"] for label in target_names},
            support={label: report[label]["support"] for label in target_names},
            roc_auc=metrics.roc_auc_score((y_true == target_names[1]).astype(int), y_prob[:, 1]),
            confusion_matrix=confusion_df,
            roc_curve=(fpr, tpr, roc_thresholds),
            pr_curve=(recall_curve, precision_curve, pr_thresholds),
        )
