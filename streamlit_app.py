from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.spam_email.config import AppConfig
from src.spam_email.data import DatasetLoader
from src.spam_email.metrics import MetricsReport
from src.spam_email.model import SpamClassifier
from src.spam_email.visualization import VisualizationBuilder

st.set_page_config(
    page_title="垃圾郵件智慧分析平台",
    layout="wide",
    page_icon="📧",
)

BASE_CONFIG = AppConfig()
TAB_TITLES = ["資料概覽", "模型評估", "即時推論", "批次推論"]


@st.cache_data(show_spinner=True, ttl=BASE_CONFIG.streamlit_cache_ttl)
def load_dataset(source: Optional[Path | str]) -> pd.DataFrame:
    config = AppConfig.from_env(local_data_path=str(source) if source else None)
    loader = DatasetLoader(config=config)
    return loader.load()


@st.cache_resource(show_spinner=True)
def prepare_classifier(frame: pd.DataFrame) -> tuple[SpamClassifier, MetricsReport]:
    classifier = SpamClassifier(config=BASE_CONFIG)
    _, report = classifier.train(frame)
    classifier.save()
    return classifier, report


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """
    Read an uploaded CSV and ensure a `text` column exists for inference.
    If missing, columns will be renamed with the last column treated as text.
    """

    content = uploaded_file.read()
    uploaded_file.seek(0)

    # Initial read to inspect structure
    dataframe = pd.read_csv(io.BytesIO(content), encoding="latin1")

    if BASE_CONFIG.text_column in dataframe.columns and len(dataframe.columns) >= 1:
        dataframe[BASE_CONFIG.text_column] = dataframe[BASE_CONFIG.text_column].astype(str)
        return dataframe

    column_count = dataframe.shape[1]
    if column_count == 0:
        raise ValueError("CSV 檔案內沒有任何欄位。")

    # Re-read treating the file as having no header so we can assign names explicitly
    header_names = []
    for idx in range(column_count):
        if idx == column_count - 1:
            header_names.append(BASE_CONFIG.text_column)
        elif idx == 0:
            header_names.append("label")
        else:
            header_names.append(f"col_{idx}")

    dataframe = pd.read_csv(
        io.BytesIO(content),
        encoding="latin1",
        names=header_names,
        header=None,
    )

    dataframe[BASE_CONFIG.text_column] = dataframe[BASE_CONFIG.text_column].astype(str)
    return dataframe


def run_batch_prediction(classifier: SpamClassifier, dataframe: pd.DataFrame) -> pd.DataFrame:
    labels, probs = classifier.predict(dataframe[BASE_CONFIG.text_column].astype(str).tolist())
    return pd.DataFrame(
        {
            "郵件內容": dataframe[BASE_CONFIG.text_column],
            "預測標籤": labels,
            "垃圾機率": probs[:, 1],
        }
    )


def render_intro() -> None:
    st.title("垃圾郵件智慧分析平台")
    st.markdown(
        """
        本系統延伸 Packt《Hands-On Artificial Intelligence for Cybersecurity》第三章案例，
        以 **繁體中文** 介面整合資料探索、模型訓練與推論功能，並提供 CLI 與 Streamlit 雙介面支援。

        - 快速掌握資料內容與標籤分佈
        - 查看模型評估指標與多種視覺化圖表
        - 執行單筆與批次推論，下載結果以便後續分析
        """
    )


def main() -> None:
    render_intro()

    try:
        frame = load_dataset(None)
    except Exception as exc:  # noqa: BLE001
        st.error(f"載入資料集時發生錯誤：{exc}")
        return

    retrain = st.toggle("重新訓練模型", value=False, key="retrain_toggle")

    if retrain:
        st.toast("開始重新訓練模型，請稍候……", icon="⏳")
        classifier, report = prepare_classifier(frame)
    else:
        try:
            classifier = SpamClassifier(BASE_CONFIG)
            classifier.load()
            report = classifier.evaluate(frame)
        except Exception:
            classifier, report = prepare_classifier(frame)

    tabs = st.tabs(TAB_TITLES)

    if "target_tab" in st.session_state:
        target = st.session_state.pop("target_tab")
        components.html(
            f"<script>var tabs = parent.document.querySelectorAll('.stTabs button');if (tabs[{target}]) {{ tabs[{target}].click(); }}</script>",
            height=0,
        )

    with tabs[0]:
        st.subheader("資料概覽")
        preview_rows = st.slider("預覽筆數", min_value=5, max_value=50, value=10, step=1, key="preview_rows")
        st.write(f"資料筆數：{len(frame)}，欄位：{list(frame.columns)}")
        st.dataframe(frame.head(preview_rows))
        st.markdown("**標籤分佈**")
        st.bar_chart(frame[BASE_CONFIG.label_column].value_counts())

    with tabs[1]:
        st.subheader("模型評估")
        builder = VisualizationBuilder(report=report)
        st.dataframe(builder.metrics_table())
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(builder.confusion_matrix_fig())
        with col2:
            st.pyplot(builder.roc_curve_fig())
        st.pyplot(builder.pr_curve_fig())

    with tabs[2]:
        st.subheader("即時推論")
        single_text = st.text_area("輸入郵件內容", height=160, key="single_text")
        if st.button("開始推論", key="single_infer_button"):
            if not single_text.strip():
                st.warning("請先輸入郵件內容。")
            else:
                labels, probs = classifier.predict([single_text.strip()])
                st.session_state["single_result"] = {
                    "label": labels[0],
                    "prob": float(probs[0][1]),
                }
                st.session_state["target_tab"] = TAB_TITLES.index("即時推論")
                st.rerun()

        if "single_result" in st.session_state:
            result = st.session_state["single_result"]
            st.success(
                f"預測標籤：{result['label']}，垃圾郵件機率 **{result['prob']:.2%}**"
            )

    with tabs[3]:
        st.subheader("批次推論")
        uploaded_file = st.file_uploader("上傳含 text 欄位的 CSV", type=["csv"], key="batch_uploader")
        if st.button("開始批次推論", key="batch_infer_button"):
            if not uploaded_file:
                st.warning("請先上傳 CSV 檔案。")
            else:
                try:
                    dataframe = read_uploaded_csv(uploaded_file)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"檔案讀取失敗，請確認為有效的 CSV：{exc}")
                else:
                    if BASE_CONFIG.text_column not in dataframe.columns:
                        st.error(f"CSV 需包含 {BASE_CONFIG.text_column} 欄位才能推論。")
                    else:
                        batch_result = run_batch_prediction(classifier, dataframe)
                        st.session_state["batch_result"] = batch_result
                        st.session_state["target_tab"] = TAB_TITLES.index("批次推論")
                        st.rerun()

        if "batch_result" in st.session_state:
            st.dataframe(st.session_state["batch_result"])

    st.caption("模型採用 TF-IDF + Logistic Regression，僅供教學示範。")


if __name__ == "__main__":
    main()
