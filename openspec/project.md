# 專案脈絡

## 目的
本專案聚焦於延伸 Packt《Hands-On Artificial Intelligence for Cybersecurity》第三章的垃圾郵件偵測例題，建立一套可視覺化、可操作的垃圾郵件分類系統，涵蓋資料前處理、模型訓練、效果評估，以及 CLI 與 Streamlit 互動介面。

## 技術棧
- Python 3.10+
- pandas、NumPy、scikit-learn、NLTK
- Streamlit、typer CLI
- Matplotlib、Seaborn
- Poetry 或 pip + requirements.txt 套件管理

## 專案慣例

### 程式風格
- 遵循 PEP 8 與 Google-style docstring。
- 變數與函式使用 snake_case，類別使用 PascalCase。
- 模組化資料處理、特徵工程、模型訓練與評估流程，並於關鍵步驟撰寫簡潔註解。

### 架構模式
- 採分層式結構：`data/`（資料）、`features/`（特徵工程）、`models/`（模型）、`services/`（服務層，含 CLI 與 Web）。
- 以配置檔集中管理資料路徑、模型參數與視覺化選項。
- Streamlit 頁面著重易用性與視覺一致性，包含側邊欄設定與主要內容區塊。

### 測試策略
- 以 pytest 撰寫單元測試，重點驗證資料清理函式、特徵轉換器與分類器流程。
- 為 CLI 指令與關鍵服務類別提供最小黏附度的整合測試。
- 使用 pytest-cov 追蹤涵蓋率，目標達 80% 以上核心模組涵蓋率。

### Git 流程
- 採 GitHub Flow：從 `main` 切出功能分支，完成後送出 PR 審查。
- Commit 訊息使用動詞開頭的英文敘述（如 `Add`, `Update`, `Fix`），並在 PR 描述連結對應的 OpenSpec 變更。

## 領域脈絡
- 垃圾郵件資料集含常見電子郵件內容與標籤，需處理英文文字、HTML 內容與可能的特殊符號。
- 模型需提供關鍵評估指標（精確率、召回率、F1、ROC AUC）與可解釋性輸出（重要特徵、關鍵詞頻率）。
- 實作須同時考量 CLI 報告與 Streamlit 即時互動式展示。

## 重要限制
- 介面文字與文件需以繁體中文呈現；Matplotlib 圖表字型統一使用 Times New Roman 以避免字型缺字。
- Demo 網站部署於 Streamlit Cloud，需確保依賴與快取設定相容雲端環境。
- 資料存取需支援本機 CSV 與可選用的雲端來源（如 GitHub Raw URL）。

## 外部依賴
- 參考資料與啟發：`https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity`
- 線上 Demo：`https://2025spamemail.streamlit.app/`
- 專案原始碼遠端：`https://github.com/huanchen1107/2025ML-spamEmail`
