from __future__ import annotations

import joblib
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import AppConfig
from .metrics import MetricsReport


@dataclass(slots=True)
class SpamClassifier:
    """
    Logistic Regression + TF-IDF 垃圾郵件分類器。
    """

    config: AppConfig
    pipeline: Optional[Pipeline] = None

    def build_pipeline(self) -> Pipeline:
        """
        建立完整的特徵工程與模型管線。
        """
        pipeline = Pipeline(
            steps=[
                (
                    "vectorize",
                    TfidfVectorizer(
                        max_features=self.config.max_features,
                        ngram_range=(1, 2),
                        stop_words="english",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=200,
                        random_state=self.config.random_state,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        self.pipeline = pipeline
        return pipeline

    @property
    def is_trained(self) -> bool:
        return self.pipeline is not None

    def train(
        self,
        frame: pd.DataFrame,
    ) -> Tuple[Pipeline, MetricsReport]:
        """
        訓練模型並回傳指標。
        """
        pipeline = self.build_pipeline()
        x_train, x_test, y_train, y_test = train_test_split(
            frame[[self.config.text_column]],
            frame[self.config.label_column],
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=frame[self.config.label_column],
        )

        pipeline.fit(x_train[self.config.text_column], y_train)
        x_eval = x_test[self.config.text_column]
        y_pred = pipeline.predict(x_eval)
        y_prob = pipeline.predict_proba(x_eval)

        report = MetricsReport.from_predictions(
            y_true=y_test.reset_index(drop=True),
            y_pred=y_pred,
            y_prob=y_prob,
            target_names=sorted(frame[self.config.label_column].unique()),
        )

        self.pipeline = pipeline
        return pipeline, report

    def predict(
        self,
        text_inputs: list[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.pipeline:
            raise RuntimeError("模型尚未訓練或載入。")
        labels = self.pipeline.predict(text_inputs)
        probabilities = self.pipeline.predict_proba(text_inputs)
        return labels, probabilities

    def save(self, path: Optional[Path | str] = None) -> Path:
        if not self.pipeline:
            raise RuntimeError("無法儲存：模型尚未訓練。")
        target = Path(path) if path else self.config.model_path
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, target)
        return target

    def load(self, path: Optional[Path | str] = None) -> Pipeline:
        target = Path(path) if path else self.config.model_path
        self.pipeline = joblib.load(target)
        return self.pipeline

    def evaluate(self, frame: pd.DataFrame) -> MetricsReport:
        """
        利用既有模型對完整資料集進行評估。
        """
        if not self.pipeline:
            raise RuntimeError("模型尚未載入。")

        y_true = frame[self.config.label_column]
        x_text = frame[self.config.text_column]
        y_pred = self.pipeline.predict(x_text)
        y_prob = self.pipeline.predict_proba(x_text)
        target_names = sorted(y_true.unique())

        return MetricsReport.from_predictions(
            y_true=y_true.reset_index(drop=True),
            y_pred=y_pred,
            y_prob=y_prob,
            target_names=target_names,
        )
