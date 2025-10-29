from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from requests import Response

from .config import AppConfig


class DatasetDownloadError(RuntimeError):
    """
    資料下載失敗時拋出。
    """


@dataclass(slots=True)
class DatasetLoader:
    """
    管理垃圾郵件資料集的載入與快取。
    """

    config: AppConfig

    def _download(self, dest: Path) -> None:
        self.config.ensure_directories()
        try:
            response: Response = requests.get(self.config.data_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise DatasetDownloadError(f"無法下載資料集：{exc}") from exc

        dest.write_bytes(response.content)

    def ensure_local_copy(self) -> Path:
        """
        確保資料集存在於本機，若不存在則自動下載。
        """
        target = self.config.local_data_path
        if not target.exists():
            self._download(target)
        return target

    def load(self, source: Optional[Path | str] = None) -> pd.DataFrame:
        """
        載入資料集為 DataFrame。
        """
        if source:
            path = Path(source)
        else:
            path = self.ensure_local_copy()

        frame = pd.read_csv(path, encoding="latin1")
        expected_columns = {self.config.label_column, self.config.text_column}

        if not expected_columns.issubset(set(frame.columns)) or len(frame.columns) < 2:
            frame = pd.read_csv(
                path,
                encoding="latin1",
                names=[self.config.label_column, self.config.text_column],
                header=None,
            )
        else:
            frame = frame[[self.config.label_column, self.config.text_column]]

        frame[self.config.label_column] = frame[self.config.label_column].astype(str)
        frame[self.config.text_column] = frame[self.config.text_column].astype(str)

        return frame.dropna()
