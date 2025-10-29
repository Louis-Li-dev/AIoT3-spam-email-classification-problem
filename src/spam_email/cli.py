from __future__ import annotations

from pathlib import Path

import typer

from .config import AppConfig
from .data import DatasetLoader
from .model import SpamClassifier

app = typer.Typer(help="垃圾郵件分類 CLI 工具")


@app.command()
def train(
    data_path: Path = typer.Option(
        None,
        "--data",
        "-d",
        help="自訂資料集路徑，預設自動下載。",
    ),
    model_path: Path = typer.Option(
        None,
        "--model",
        "-m",
        help="儲存模型的目標檔案。",
    ),
) -> None:
    """
    訓練模型並輸出指標。
    """
    config = AppConfig.from_env(
        local_data_path=str(data_path) if data_path else None,
        model_path=str(model_path) if model_path else None,
    )
    loader = DatasetLoader(config=config)
    frame = loader.load()
    classifier = SpamClassifier(config=config)
    _, report = classifier.train(frame)
    saved_path = classifier.save()
    typer.echo(f"模型已儲存於: {saved_path}")
    typer.echo("=== 評估指標 ===")
    for label in report.labels:
        typer.echo(
            f"{label}: Precision={report.precision[label]:.3f}, "
            f"Recall={report.recall[label]:.3f}, F1={report.f1[label]:.3f}"
        )
    typer.echo(f"ROC AUC: {report.roc_auc:.3f}")


@app.command()
def predict(text: str, model_path: Path = typer.Option(None, "--model", "-m")) -> None:
    """
    對單一郵件文字進行分類。
    """
    config = AppConfig.from_env(model_path=str(model_path) if model_path else None)
    classifier = SpamClassifier(config=config)
    classifier.load()
    labels, probs = classifier.predict([text])
    typer.echo(f"預測結果: {labels[0]}, 垃圾郵件機率: {probs[0][1]:.3f}")


if __name__ == "__main__":
    app()
