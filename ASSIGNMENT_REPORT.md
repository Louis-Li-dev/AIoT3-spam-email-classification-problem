# 作業報告

## 方法（Method）
- 以 `AppConfig` 統一管理資料路徑與模型設定，並改寫 `from_env` 以避免 `member_descriptor` 造成的路徑例外。
- 建立 `DatasetLoader` 於載入無表頭資料時自動補齊欄位名稱，確保 CSV 可直接餵入模型。
- 採 TF-IDF + Logistic Regression 管線（`SpamClassifier`），並以 `MetricsReport` 封裝 ROC、PR、混淆矩陣等評估資訊。
- 移除 Matplotlib 中的中文標題，統一使用 Times New Roman 英文標籤，排除字型缺字問題。
- Streamlit 前端新增分頁式操作與按鈕觸發推論，並透過 Session State 儲存結果及控制自動切換分頁。

## 功能（Functionalities）
- **資料概覽**：顯示資料筆數、欄位與標籤分佈長條圖。
- **模型評估**：提供 Precision／Recall／F1／Support 表格，以及混淆矩陣、ROC Curve、Precision-Recall Curve 圖像。
- **即時推論**：輸入郵件內容後按「開始推論」即可顯示預測與垃圾郵件機率，完成後自動跳轉至該分頁。
- **批次推論**：上傳含 `text` 欄位之 CSV，以按鈕觸發推論並顯示結果表格，同時保留於 Session 供下載或檢視。
- **重新訓練**：側邊欄提供切換，可在需要時重新以當前資料訓練模型。

## 資料集（Datasets）
- 來源：`datasets/sms_spam_no_header.csv`，為多筆 SMS 訊息資料。
- 規模：共 747 筆標記為 `spam` 之訊息（其餘為 `ham`）。
- 欄位：`label`（標籤），`text`（郵件內容）。透過 `DatasetLoader` 自動補設欄位名稱。

## 測試（Testing）
- 以 `pytest` 執行 `tests/test_pipeline.py`，驗證訓練、儲存、讀取與推論流程（全部通過）。

## GPT Prompt（完整對話紀錄）

<details>
<summary>使用者：初始專案說明</summary>

```
教學 https://www.youtube.com/watch?v=FeCCYFK0TJ8&list=PLYlM4-ln5HcCoM_TcLKGL5NcOpNVJ3g7c
 
Source Reference
This project builds upon patterns and datasets related to the Spam Email problem from Chapter 3 of the Packt repository below. We used it to expand the preprocessing steps and add richer visualization work (step outputs, metrics, and CLI/Streamlit views).

https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity.git
Using openspec and AI coding CLI to finish this project
===
requirements 
1. need a github 
https://github.com/huanchen1107/2025ML-spamEmail
2. need a Demo site 
https://2025spamemail.streamlit.app/
  
  
tutorial 
https://www.youtube.com/watch?v=ANjiJQQIBo0

 
1. Populate your project context:
   "Please read openspec/project.md and help me fill it out   
    with details about my project, tech stack, and conventions"
2. Create your first change proposal:
   "I want to add [YOUR FEATURE HERE]. Please create an       
    OpenSpec change proposal for this feature"
3. Learn the OpenSpec workflow:
   "Please explain the OpenSpec workflow from openspec/AGENTS.md
    and how I should work with you on this project"

Use traditional chinese in the system except the matplotlib (cuz font not supported, for font, use News Time Roman)

Build a system that is good looking and easy to use
```
</details>

<details>
<summary>助理：回覆 1</summary>

```
**今日進度**
- 已補寫專案脈絡（`openspec/project.md:1`），涵蓋目的、技術棧、架構與測試/部署限制，全面對齊垃圾郵件偵測教學目標。
- 建立首個 OpenSpec 變更 `add-streamlit-spam-dashboard`，含提案（`openspec/changes/add-streamlit-spam-dashboard/proposal.md:1`）、任務列表（`openspec/changes/add-streamlit-spam-dashboard/tasks.md:1`）與新能力規格（`openspec/changes/add-streamlit-spam-dashboard/specs/spam-analytics/spec.md:1`），確保 Streamlit Demo 需求明確。
- 已執行 `openspec validate add-streamlit-spam-dashboard --strict` 並通過，確認提案格式與規格完整。

**OpenSpec 協作流程摘要**
- **前期準備**：先閱讀 `openspec/project.md`、既有 `openspec/specs/`、以及 `openspec list` 的待審提案，確保新需求不與既有規格衝突。
- **建立變更**：挑選動詞開頭的唯一 `change-id`，在 `openspec/changes/<id>/` 底下撰寫 `proposal.md`（為什麼/變更內容/影響）、`tasks.md`（實作待辦），必要時加上 `design.md`，並為受影響能力新增或修改 `specs/<capability>/spec.md`。
- **撰寫規格**：每個 `spec.md` 需依序列出 `## ADDED|MODIFIED|REMOVED Requirements`，每個 Requirement 至少一個 `#### Scenario:`，敘述中必含 `MUST`/`SHALL` 等強制語句。
- **驗證與審核**：使用 `openspec validate <id> --strict` 確認格式正確後，再提交提案等待核准；未核准前不得開始實作。
- **實作階段**：核准後依 `tasks.md` 順序完成開發並勾選項目，同時保持程式與規格同步。
- **封存**：部署完成後，使用 `openspec archive <id>` 將變更移至 `archive/`，並再度驗證確保歷史紀錄完整。

**後續建議**
1. 依 `tasks.md` 開始實作資料/模型共用模組，再進入 Streamlit 介面開發。
2. 及早規劃 Streamlit Cloud 所需的依賴清單與 Secrets，確保 Demo 上線順利。
3. 提前準備測試樣本與 CLI 腳本，方便驗證管線是否能與 Web 共享。
```
</details>

<details>
<summary>使用者：請用 Streamlit 開發</summary>

```
請你用 streamlit 實際開發根據這些文件
```
</details>

<details>
<summary>助理：回覆 2</summary>

```
- 完成核心模組：`src/spam_email/config.py:1` 集中管理資料與模型設定，`src/spam_email/data.py:1` 自動下載/載入 CSV 並統一欄位，`src/spam_email/model.py:1` 建立 TF-IDF + Logistic Regression 管線並提供訓練、評估、儲存/載入與推論介面，`src/spam_email/metrics.py:1`、`src/spam_email/visualization.py:1` 產出指標與 Times New Roman 字型圖表。CLI 入口放在 `src/spam_email/cli.py:1` 方便訓練與單筆推論。
- 建置 Streamlit 介面 `streamlit_app.py:1`：側邊欄支援資料預覽筆數、單筆文字輸入、CSV 上傳與重新訓練切換；首頁提供繁體中文導覽；主畫面顯示資料概況、標籤分佈、ROC/PR/混淆矩陣與指標表格；推論區支援即時與批次輸出。所有 Matplotlib 圖表統一使用 Times New Roman。
- 撰寫 `README.md:1` 說明安裝、CLI/Streamlit 操作與部署建議，新增 `requirements.txt:1` 方便雲端部署；`openspec/changes/add-streamlit-spam-dashboard/tasks.md:1` 全部核取，保持 OpenSpec 狀態同步。
- 測試：新增 `tests/test_pipeline.py:1`、`tests/conftest.py:1`、`pytest.ini:1` 驗證訓練與推論流程並消除雜訊警告，已於本地執行 `pytest` 全數通過。

建議後續檢查：
1. 使用 `streamlit run streamlit_app.py` 確認本地互動體驗與圖表字型。
2. 以 `python -m src.spam_email.cli train` 重新訓練並檢視指標輸出。
3. 推送至 GitHub 後設定 Streamlit Cloud（requirements + 入口檔）驗證雲端 Demo。
```
</details>

<details>
<summary>使用者：提供資料樣本</summary>

```
"ham","Go until jurong point, crazy.. Available only in bugis n great world la e buffet..."
...
（原始資料全文）
```
</details>

<details>
<summary>助理：回覆 3</summary>

```
在 `datasets/sms_spam_no_header.csv` 中共有 747 筆被標記為 `spam` 的訊息。
```
</details>

<details>
<summary>使用者：要求載入資料</summary>

```
So please, load this data for the system to make predictions
```
</details>

<details>
<summary>助理：回覆 4</summary>

```
已把預設資料路徑改為 `datasets/sms_spam_no_header.csv`，並更新 `DatasetLoader.load` 在缺少標頭時會自動指定欄位名稱後再載入，同時保留原本的字串化處理，確保這份無標頭資料能正確餵進模型做預測。
本地已執行 `pytest` 完成驗證。建議下一步可在 Streamlit 或 CLI 端跑一次全流程，確認新的資料來源載入無誤。
```
</details>

<details>
<summary>使用者：報錯訊息</summary>

```
載入資料集時發生錯誤：'member_descriptor' object has no attribute 'exists'
```
</details>

<details>
<summary>助理：回覆 5</summary>

```
造成錯誤的原因，是 `AppConfig.from_env()` 在欄位為空時直接存取類別層級的 `local_data_path`／`model_path`，在 `slots=True` 的 dataclass 中那其實是一個 `member_descriptor`，因此後續 `.exists()` 會炸掉。我調整了兩個地方：

- `src/spam_email/config.py:31`：`from_env` 先建立 `base = cls()`，改以實例的既有設定值回填缺省欄位，確保取得的是 `Path` 物件。
- `src/spam_email/data.py:37` 起：載入 CSV 時若缺列名會補上自訂欄位，隨後再將欄位轉成字串，避免對 `member_descriptor` 等非 Path 物件操作。

本地 `pytest` 重新跑過，全數通過。建議再跑一次 CLI 或 Streamlit 確認載入流程無誤。
```
</details>

<details>
<summary>使用者：提出圖表字型、介面與報告需求</summary>

```
fig.savefig(image, **kwargs)
C:\Users\ss348\anaconda3\Lib\site-packages\streamlit\elements\pyplot.py:176: UserWarning: Glyph ...
...
```
</details>

<details>
<summary>助理：回覆 6（本次回覆內容）</summary>

```
詳見最終回答。
```
</details>

<details>
<summary>使用者：批次推論 CSV 欄位問題</summary>

```
Error: CSV 需包含 text 欄位才能推論。
For 批次分類
I hope that 批次推論 can take the csv and start predicting. If there is no column like our data, add one.

And update the report GPT prompt
```
</details>

<details>
<summary>助理：回覆 7（本次回覆內容）</summary>

```
詳見最終回答。
```
</details>
