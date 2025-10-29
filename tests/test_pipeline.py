from __future__ import annotations

import pandas as pd
import pytest
from src.spam_email.config import AppConfig
from src.spam_email.model import SpamClassifier


@pytest.fixture()
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"label": "ham", "text": "Hi there, let's catch up tomorrow."},
            {"label": "spam", "text": "WIN a brand new car by clicking here!!!"},
            {"label": "ham", "text": "Reminder: project meeting at 2pm."},
            {"label": "spam", "text": "Exclusive offer!!! Limited time deal."},
            {"label": "ham", "text": "Lunch plans for next week?"},
            {"label": "spam", "text": "Get rich quick by subscribing now!!!"},
            {"label": "ham", "text": "Team standup moved to 10am."},
            {"label": "spam", "text": "Mega discount newsletter just for you."},
        ]
    )


def test_training_and_prediction(sample_frame: pd.DataFrame, tmp_path):
    config = AppConfig.from_env(
        model_path=str(tmp_path / "model.joblib"),
        local_data_path=str(tmp_path / "data.csv"),
    )
    sample_frame.to_csv(config.local_data_path, index=False)

    classifier = SpamClassifier(config=config)
    pipeline, report = classifier.train(sample_frame)

    assert pipeline is not None
    assert report.roc_auc >= 0

    classifier.save()
    classifier.load()
    labels, probs = classifier.predict(
        ["Limited time offer, claim your reward now!"]
    )

    assert labels.shape == (1,)
    assert probs.shape == (1, 2)
