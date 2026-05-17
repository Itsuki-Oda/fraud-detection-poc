import requests
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

azure_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
)

AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

FEATURE_LABELS = {

    # PCA系
    "num__V1": "潜在異常パターン01",
    "num__V2": "潜在異常パターン02",
    "num__V3": "潜在異常パターン03",
    "num__V4": "潜在異常パターン04",
    "num__V5": "潜在異常パターン05",
    "num__V6": "潜在異常パターン06",
    "num__V7": "潜在異常パターン07",
    "num__V8": "潜在異常パターン08",
    "num__V9": "潜在異常パターン09",
    "num__V10": "潜在異常パターン10",
    "num__V11": "潜在異常パターン11",
    "num__V12": "潜在異常パターン12",
    "num__V13": "潜在異常パターン13",
    "num__V14": "潜在異常パターン14",
    "num__V15": "潜在異常パターン15",
    "num__V16": "潜在異常パターン16",
    "num__V17": "潜在異常パターン17",
    "num__V18": "潜在異常パターン18",
    "num__V19": "潜在異常パターン19",
    "num__V20": "潜在異常パターン20",
    "num__V21": "潜在異常パターン21",
    "num__V22": "潜在異常パターン22",
    "num__V23": "潜在異常パターン23",
    "num__V24": "潜在異常パターン24",
    "num__V25": "潜在異常パターン25",
    "num__V26": "潜在異常パターン26",
    "num__V27": "潜在異常パターン27",
    "num__V28": "潜在異常パターン28",

    # 通常特徴量
    "num__customer_tenure_months": "利用継続期間",
    "num__Amount": "取引金額",
    "num__Time": "取引時刻",
    "num__is_foreign": "海外取引フラグ",
    "num__is_new_device": "新端末利用",
    "num__is_ec": "EC利用",

    "cat__entry_mode_chip": "ICチップ利用",
    "cat__entry_mode_manual": "手入力取引",
    "cat__country_JP": "国内取引",
}

def generate_azure_review_comment(row, top_shap_text):

    prompt = f"""
あなたはカード会社の不正検知アナリストです。

以下の取引について、審査担当者向けのコメントを日本語で作成してください。

条件:
- 200字以内
- 箇条書き不要
- 実務的に
- 不正の可能性、確認ポイント、推奨アクションを含める
- 匿名特徴量V1〜V28は「潜在的な異常パターン」と表現する
- 顧客向けではなく、社内審査担当者向けにする

取引情報:
fraud_score: {row.get("fraud_score")}
risk_band: {row.get("risk_band")}
actual_label: {row.get("actual_label")}
predicted_label: {row.get("predicted_label")}
rule_reason: {row.get("rule_reason")}
shap_reason: {row.get("shap_reason")}
top_shap_features: {row.get("top_shap_features")}

SHAP上位寄与:
{top_shap_text}

Amount: {row.get("Amount")}
country: {row.get("country")}
channel: {row.get("channel")}
entry_mode: {row.get("entry_mode")}
merchant_category: {row.get("merchant_category")}
is_foreign: {row.get("is_foreign")}
is_ec: {row.get("is_ec")}
is_new_device: {row.get("is_new_device")}
customer_tenure_months: {row.get("customer_tenure_months")}
authentication_result: {row.get("authentication_result")}
"""

    response = azure_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": "あなたはカード不正検知の審査支援AIです。簡潔で実務的に説明してください。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=300
    )

    return response.choices[0].message.content

FASTAPI_URL = "http://api:8000/rag-review-comment"

def generate_review_comment_via_api(row, top_shap_text):
    payload = {
        "fraud_score": float(row.get("fraud_score")),
        "risk_band": str(row.get("risk_band")),
        "amount": float(row.get("Amount")) if row.get("Amount") is not None else None,
        "country": row.get("country"),
        "channel": row.get("channel"),
        "entry_mode": row.get("entry_mode"),
        "merchant_category": row.get("merchant_category"),
        "is_foreign": int(row.get("is_foreign")),
        "is_ec": int(row.get("is_ec")),
        "is_new_device": int(row.get("is_new_device")),
        "customer_tenure_months": int(row.get("customer_tenure_months")),
        "authentication_result": row.get("authentication_result"),
        "top_shap_text": top_shap_text,
    }

    response = requests.post(FASTAPI_URL, json=payload, timeout=30)
    response.raise_for_status()

    result = response.json()

    return {
        "review_comment": result["review_comment"],
        "retrieved_context": result["retrieved_context"]
    }

def plot_shap_contribution(shap_row, top_n=10):
    top_features = shap_row.abs().sort_values(ascending=False).head(top_n).index

    plot_df = pd.DataFrame({
        "feature_raw": top_features,
        "shap_value": shap_row[top_features].values
    })

    plot_df["feature"] = plot_df["feature_raw"].apply(
        lambda f: FEATURE_LABELS.get(f, f)
    )

    feature_values = []

    for raw_feature in plot_df["feature_raw"]:
        original_col = raw_feature.replace("num__", "").replace("cat__", "")

        if original_col in row.index:
            val = row[original_col]

            if "tenure" in original_col:
                val = f"{int(val)}ヶ月"
            elif original_col == "Amount":
                val = f"{int(val):,}円"
            elif original_col == "Time":
                val = f"{int(val)}秒"
            elif "is_" in original_col:
                val = "Yes" if int(val) == 1 else "No"

            feature_values.append(str(val))
        else:
            feature_values.append("-")

    plot_df["display_label"] = (
        plot_df["feature"].astype(str)
        + "<br><span style='font-size:11px;color:gray'>"
        + pd.Series(feature_values, index=plot_df.index).astype(str)
        + "</span>"
    )

    plot_df["direction"] = plot_df["shap_value"].apply(
        lambda x: "不正方向" if x > 0 else "正常方向"
    )

    plot_df["color"] = plot_df["shap_value"].apply(
        lambda x: "#ff4b4b" if x > 0 else "#3b82f6"
    )

    plot_df = plot_df.sort_values("shap_value")

    fig = go.Figure(
        data=[
            go.Bar(
                x=[0] * len(plot_df),
                y=plot_df["display_label"],
                orientation="h",
                marker=dict(color=plot_df["color"]),
                text=plot_df["direction"],
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "SHAP値: %{x:.4f}<br>"
                    "%{text}<extra></extra>"
                ),
            )
        ],
        frames=[
            go.Frame(
                data=[
                    go.Bar(
                        x=plot_df["shap_value"],
                        y=plot_df["display_label"],
                        orientation="h",
                        marker=dict(color=plot_df["color"]),
                        text=plot_df["direction"],
                        textposition="outside",
                    )
                ]
            )
        ]
    )

    max_abs = max(abs(plot_df["shap_value"].min()), abs(plot_df["shap_value"].max()))

    fig.update_layout(
        title="SHAP Contribution by Feature",
        xaxis_title="SHAP value（右: 不正方向 / 左: 正常方向）",
        yaxis_title="",
        height=560,
        template="plotly_dark",
        margin=dict(l=20, r=60, t=70, b=40),
        showlegend=False,
        xaxis=dict(range=[-max_abs * 1.15, max_abs * 1.15]),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Replay",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 900, "redraw": True},
                                "transition": {"duration": 700, "easing": "cubic-out"},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    }
                ],
                "x": 1.0,
                "y": 1.15,
            }
        ],
    )

    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")

    return fig


def get_top_shap_text(shap_row, top_n=8):
    top = shap_row.abs().sort_values(ascending=False).head(top_n).index

    lines = []
    for feature in top:
        value = shap_row[feature]
        direction = "不正方向" if value > 0 else "正常方向"
        lines.append(f"{feature}: {value:.4f}（{direction}）")

    return "\n".join(lines)

st.set_page_config(
    page_title="Fraud Detection Review Mock",
    layout="wide"
)

st.title("💳 Fraud Detection Review Mock")
st.caption("カード不正検知の審査画面モック")

@st.cache_data
def load_data():
    result_df = pd.read_csv("fraud_detection_result.csv")
    shap_df = pd.read_csv("shap_values.csv")
    return result_df, shap_df

df, shap_df = load_data()

# サイドバー
st.sidebar.header("Filter")

risk_filter = st.sidebar.multiselect(
    "Risk Band",
    options=df["risk_band"].dropna().unique(),
    default=df["risk_band"].dropna().unique()
)

actual_filter = st.sidebar.multiselect(
    "Actual Label",
    options=df["actual_label"].unique(),
    default=df["actual_label"].unique()
)

filtered = df[
    (df["risk_band"].isin(risk_filter)) &
    (df["actual_label"].isin(actual_filter))
].copy()

# KPI
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", len(filtered))
col2.metric("Avg Fraud Score", round(filtered["fraud_score"].mean(), 4))
col3.metric("Predicted Fraud", int((filtered["predicted_label"] == "fraud").sum()))
col4.metric("Actual Fraud", int((filtered["actual_label"] == "fraud").sum()))

st.divider()

# 一覧
st.subheader("Transaction Review Queue")

show_cols = [
    "fraud_score",
    "risk_band",
    "actual_label",
    "predicted_label",
    "rule_reason",
    "Amount",
    "Time",
    "country",
    "channel",
    "entry_mode",
    "merchant_category",
    "customer_tenure_months",
]

st.dataframe(
    filtered.sort_values("fraud_score", ascending=False)[show_cols],
    use_container_width=True,
    height=350
)

st.divider()

# 詳細
st.subheader("Transaction Detail")

selected_index = st.selectbox(
    "Select transaction index",
    options=filtered.sort_values("fraud_score", ascending=False).index
)

row = df.loc[selected_index]
row_position = df.index.get_loc(selected_index)
shap_row = shap_df.iloc[row_position]
top_shap_text = get_top_shap_text(shap_row)

left, right = st.columns([1, 2])

with left:
    st.metric("Fraud Score", round(row["fraud_score"], 4))
    st.metric("Risk Band", row["risk_band"])
    st.metric("Actual", row["actual_label"])
    st.metric("Prediction", row["predicted_label"])

with right:
    st.write("### Rule-based Reason")
    st.info(row["rule_reason"])
    st.write("### SHAP-based Model Reason")
    st.warning(row["shap_reason"])
    st.write("### SHAP Contribution Chart")
    fig = plot_shap_contribution(shap_row, top_n=10)
    components.html(
        fig.to_html(
            include_plotlyjs="cdn",
            full_html=False,
            auto_play=True,
            config={"displayModeBar": False}
        ),
        height=620
    )

    st.write("### Azure OpenAI Review Comment")

    if st.button("Generate Review Comment"):
        with st.spinner("RAGで審査コメントを生成中..."):
            try:
                result = generate_review_comment_via_api(
                    row,
                    top_shap_text
                )

                st.success("Generated via RAG")

                st.write("### AI Review Comment")
                st.write(result["review_comment"])

                st.write("### Retrieved Manual Context")
                st.info(result["retrieved_context"])

            except Exception as e:
                st.error("RAG API call failed")
                st.exception(e)

    st.write("### Top SHAP Contributions")
    st.code(top_shap_text)

    st.write("### Key Attributes")
    detail_cols = [
        "Amount",
        "country",
        "channel",
        "entry_mode",
        "merchant_category",
        "is_foreign",
        "is_ec",
        "is_new_device",
        "is_high_risk_category",
        "customer_tenure_months",
        "authentication_result",
    ]

    detail_data = {
        col: row[col]
        for col in detail_cols
        if col in row.index
    }

    st.table(pd.DataFrame(detail_data.items(), columns=["feature", "value"]))

st.divider()
