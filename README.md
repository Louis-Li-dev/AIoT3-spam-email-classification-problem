# 垃圾郵件智慧分析平台

本專案延伸 Packt《Hands-On Artificial Intelligence for Cybersecurity》第三章垃圾郵件案例，提供自動化資料處理、模型訓練，以及 Streamlit 互動式 Demo。介面全面採用繁體中文，提供教學與展示使用。

## 快速開始

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

### 訓練模型（CLI）

```bash
python -m src.spam_email.cli train
```

### 啟動 Streamlit

```bash
streamlit run streamlit_app.py
```

## 功能特色

- 自動下載垃圾郵件資料集並完成預處理
- 共用 TF-IDF + Logistic Regression 管線，CLI 與 Streamlit 能共用同一模型
- Streamlit 介面提供資料上傳、單筆推論與批次推論
- 以 Times New Roman 顯示 ROC、PR、混淆矩陣等 Matplotlib 圖表
- pytest 單元測試驗證基本管線運作

## 專案結構

- `src/spam_email/`：核心模組（設定、資料載入、模型、視覺化、CLI）
- `streamlit_app.py`：Streamlit 主程式
- `tests/`：pytest 測試案例
- `openspec/`：OpenSpec 變更與規格

## 部署建議

- Streamlit Cloud：設定 `requirements.txt`、`streamlit_app.py`
- 環境需允許外部 HTTP 以便下載 Packt 垃圾郵件資料集
- 若有自訂資料集，可於啟動前將 CSV 置於 `data/raw/spam.csv`
