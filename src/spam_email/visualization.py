from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .metrics import MetricsReport


plt.rcParams['font.family'] = 'Times New Roman'


@dataclass(slots=True)
class VisualizationBuilder:
    """Generate evaluation visuals for the Streamlit dashboard."""

    report: MetricsReport

    def confusion_matrix_fig(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            self.report.confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
        )
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        return fig

    def roc_curve_fig(self) -> plt.Figure:
        fpr, tpr, _ = self.report.roc_curve
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, label=f"ROC AUC = {self.report.roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        return fig

    def pr_curve_fig(self) -> plt.Figure:
        recall, precision, _ = self.report.pr_curve
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(recall, precision)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        return fig

    def metrics_table(self) -> pd.DataFrame:
        data = {
            "Label": self.report.labels,
            "Precision": [self.report.precision[label] for label in self.report.labels],
            "Recall": [self.report.recall[label] for label in self.report.labels],
            "F1": [self.report.f1[label] for label in self.report.labels],
            "Support": [self.report.support[label] for label in self.report.labels],
        }
        return pd.DataFrame(data)
