## ADDED Requirements
### Requirement: Streamlit 首頁導覽
Streamlit 應用 MUST 提供清楚的專案說明、資料處理流程摘要與操作指引。

#### Scenario: 訪客開啟頁面
- **GIVEN** 使用者開啟 `https://2025spamemail.streamlit.app/`
- **WHEN** 首頁完成載入
- **THEN** 首屏需顯示專案用途、資料來源、模型版本與基礎操作說明
- **AND** 側邊欄需提供資料上傳、篩選和模型切換控件

### Requirement: 推論與批次分析
系統 MUST 支援單筆文字輸入與批次 CSV 上傳，並使用共用模型流程提供分類結果與信心分數。

#### Scenario: 使用者上傳郵件資料
- **GIVEN** 使用者透過側邊欄選擇 CSV 檔案並指定標題欄位
- **WHEN** 使用者按下推論或分析按鈕
- **THEN** 系統需呼叫共用前處理與模型管線產生分類結果
- **AND** 顯示每筆郵件的分類標籤、機率、以及錯誤訊息（若格式不正確）

### Requirement: 指標視覺化與字型設定
應用 MUST 顯示核心指標（Precision、Recall、F1、ROC AUC）與前處理統計圖表，並設定圖表字型為 Times New Roman。

#### Scenario: 顯示模型性能圖表
- **GIVEN** 模型已完成訓練或載入
- **WHEN** 使用者展開評估區塊
- **THEN** 系統需顯示至少一張 ROC 或 PR 曲線及指標表格
- **AND** 所有 Matplotlib 圖表需以 Times New Roman 為字型
- **AND** 若缺少必要資料則顯示友善錯誤訊息並提供重新計算選項
