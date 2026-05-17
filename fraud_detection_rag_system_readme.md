# Fraud Detection Review AI PoC

## Overview

カード不正検知をテーマにした、生成AI + RAG + Explainable AI を組み合わせた審査支援PoCです。

本PoCでは以下を実現しています。

- 不正検知モデルによる fraud score 推定
- SHAPによる特徴量寄与可視化
- Streamlitによる審査画面モック
- FastAPIによるBackend API化
- Azure OpenAIによる審査コメント生成
- Azure AI Searchを利用したRAG検索
- Docker Composeによる複数サービス統合

---

# Architecture

```text
Streamlit UI
↓
FastAPI Backend
↓
Azure AI Search (RAG)
↓
Azure OpenAI
```

---

# Features

## 1. Fraud Detection Dashboard

- Fraud Score表示
- Risk Band分類
- 実績ラベル表示
- Transaction Review Queue
- 詳細属性表示

---

## 2. SHAP Explainability

モデル予測に対する特徴量寄与を可視化。

- 不正方向寄与
- 正常方向寄与
- 実務向け表示名変換
- Plotlyによる動的グラフ表示

---

## 3. Azure OpenAI Review Comment

取引情報・SHAP寄与をもとに、審査担当者向けコメントを生成。

例:

- 不正可能性
- 確認ポイント
- 推奨Next Action

---

## 4. RAG (Retrieval Augmented Generation)

Azure AI Searchを利用し、社内審査マニュアルを検索。

取得内容:

- 不正パターン
- 審査観点
- 推奨Next Action

生成AIは検索されたマニュアルを根拠としてコメント生成を行う。

---

# Technologies

## Frontend

- Streamlit
- Plotly

## Backend

- FastAPI
- Uvicorn

## AI / ML

- XGBoost / LightGBM
- SHAP
- Azure OpenAI
- text-embedding-3-small

## RAG

- Azure AI Search
- LangChain

## Infra

- Docker
- Docker Compose

---

# Project Structure

```text
fraud-detection-project/
├── app.py
├── api.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── docs/
│   └── fraud_review_manual.md
├── shap_values.csv
├── fraud_detection_result.csv
├── azure_search_build.py
├── azure_search_test.py
└── .env
```

---

# Environment Variables

```env
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_DEPLOYMENT=
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=

AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_ADMIN_KEY=
AZURE_SEARCH_INDEX_NAME=
```

---

# Local Run

## Install

```bash
pip install -r requirements.txt
```

---

## Run FastAPI

```bash
uvicorn api:app --reload --port 8000
```

---

## Run Streamlit

```bash
streamlit run app.py
```

---

# Docker Run

```bash
docker compose up --build
```

Access:

```text
http://localhost:8501
```

---

# Azure AI Search Index Build

```bash
python azure_search_build.py
```

---

# Future Improvements

- Azure Container Apps deployment
- Authentication / RBAC
- Real-time fraud streaming
- Multi-agent investigation workflow
- Graph-based fraud analysis
- Cosmos DB integration
- Monitoring / Logging
- Human-in-the-loop review workflow

---

# Learning Points

本PoCを通じて以下を学習。

- Explainable AI
- SHAP
- RAG Architecture
- Vector Search
- Azure AI Search
- FastAPI API Design
- Docker-based multi-service architecture
- Azure OpenAI integration
- AI application architecture

---

# Notes

本プロジェクトは学習・PoC目的であり、実際のカード会社の審査ロジックを再現したものではありません。

