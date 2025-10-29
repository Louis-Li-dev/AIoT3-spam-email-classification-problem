from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class AppConfig:
    """
    應用程式組態設定，集中管理資料、模型與視覺化選項。
    """

    data_url: str = (
        "https://raw.githubusercontent.com/PacktPublishing/"
        "Hands-On-Artificial-Intelligence-for-Cybersecurity/"
        "master/chapter3/datasets/spam.csv"
    )
    local_data_path: Path = Path("datasets/sms_spam_no_header.csv")
    model_path: Path = Path("models/spam_classifier.joblib")
    label_column: str = "label"
    text_column: str = "text"
    random_state: int = 42
    test_size: float = 0.2
    max_features: int = 10000
    cache_dir: Path = Path(".cache")
    streamlit_cache_ttl: int = 3600

    def ensure_directories(self) -> None:
        """
        建立必要的資料夾。
        """
        for path in {
            self.local_data_path.parent,
            self.model_path.parent,
            self.cache_dir,
        }:
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(
        cls,
        data_url: Optional[str] = None,
        local_data_path: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> "AppConfig":
        """
        建立覆寫特定欄位的設定。
        """
        base = cls()
        return cls(
            data_url=data_url or base.data_url,
            local_data_path=Path(local_data_path) if local_data_path else base.local_data_path,
            model_path=Path(model_path) if model_path else base.model_path,
        )
