# Azure AIを活用した不正検知審査システム

## 概要

本プロジェクトは、以下を組み合わせた不正検知レビューアプリケーションです。

- 機械学習による不正検知
- SHAPによる説明可能AI
- RAG（Retrieval Augmented Generation）
- Azure OpenAI
- Azure AI Search
- Streamlitフロントエンド
- FastAPIバックエンド
- Azure Container Appsによるクラウドデプロイ

クレジットカード不正利用審査業務を模したPoCとして構築しています。

---

# 主な機能

## 不正検知スコアリング

- LightGBMによる不正検知
- Fraud Score算出
- Risk Band分類

## 説明可能AI

- SHAPによる特徴量寄与表示
- 不正判定理由の可視化

## RAGによる審査支援

- 審査マニュアル検索
- コンテキスト付きAIコメント生成
- 不正パターン別Next Action提案

## クラウド構成

- Azure OpenAIによるレビューコメント生成
- Azure AI Searchによるベクトル検索
- Azure Key VaultによるSecret管理
- Managed Identity認証
- Azure Container Appsによるデプロイ

---

# システム構成

```text
Browser
↓
fraud-web (Streamlit)
↓
fraud-api (FastAPI)
↓
Azure AI Search
↓
Azure OpenAI
```

## Secret管理

```text
Container Apps
↓ Managed Identity
Azure Key Vault
↓
Secrets
```

API Keyはコードへ直書きしていません。

---

# 技術スタック

## Frontend

- Streamlit

## Backend

- FastAPI
- Uvicorn

## AI / ML

- LightGBM
- SHAP
- Azure OpenAI
- RAG

## Search

- Azure AI Search
- Vector Search

## Infrastructure

- Docker
- Azure Container Registry (ACR)
- Azure Container Apps
- Azure Key Vault
- Managed Identity

---

# ローカル開発

## 仮想環境作成

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

## ライブラリインストール

```bash
pip install -r requirements.txt
```

## FastAPI起動

```bash
uvicorn api:app --reload --port 8000
```

## Streamlit起動

```bash
streamlit run app.py
```

---

# Docker

## Backend

```bash
docker build -t fraud-api -f Dockerfile.api .
```

## Frontend

```bash
docker build -t fraud-web -f Dockerfile.web .
```

---

# Azureデプロイ

## 利用サービス

- Azure OpenAI
- Azure AI Search
- Azure Container Registry
- Azure Container Apps
- Azure Key Vault
- Managed Identity

---

# Azure構成

```text
Azure Container Apps
├ fraud-web
└ fraud-api

fraud-api
├ Azure AI Search
├ Azure OpenAI
└ Key Vault
```

---

# セキュリティ

## Managed Identity

backendコンテナには system-assigned Managed Identity を付与しています。

## Key Vault

以下のSecretをAzure Key Vaultで管理しています。

- Azure OpenAI API Key
- Azure AI Search Admin Key

Container AppsからKey Vault参照でSecret取得しています。

---

# RAGフロー

```text
取引データ
↓
Vector Search
↓
関連審査マニュアル取得
↓
Azure OpenAI
↓
レビューコメント生成
```

---

# 処理フロー

1. ユーザーが疑わしい取引を選択
2. SHAP特徴量寄与を表示
3. FrontendからFastAPIへリクエスト
4. FastAPIがAzure AI Searchで関連文書検索
5. 検索結果をプロンプトへ付与
6. Azure OpenAIで審査コメント生成
7. Frontendへ結果表示

---

# Cold Startについて

Azure Container Appsでは以下の場合に初回応答が遅くなる場合があります。

- コンテナが0台へスケールしている
- 一定時間アクセスがない

推奨設定：

```bash
az containerapp update \
  --name fraud-api \
  --resource-group fraud-ai-rg \
  --min-replicas 1
```

---

# コスト最適化

検証後は以下でContainer Appsを0台へ戻せます。

```bash
az containerapp update \
  --name fraud-api \
  --resource-group fraud-ai-rg \
  --min-replicas 0
```

```bash
az containerapp update \
  --name fraud-web \
  --resource-group fraud-ai-rg \
  --min-replicas 0
```

---

# 今後の改善案

- Reviewer管理
- 認証機能
- 承認履歴
- アラート優先度管理
- マルチエージェント化
- CI/CD
- Application Insights監視
- GitHub Actions連携

---

# 学習ポイント

本プロジェクトで学習できる内容：

- 不正検知ML
- Explainable AI
- RAGアーキテクチャ
- Azure AI連携
- Secret管理
- クラウドネイティブ構成
- Frontend / Backend分離
- Dockerコンテナ化

---

# 作者

Azure AI・RAG・不正検知をテーマにした個人PoCプロジェクト

