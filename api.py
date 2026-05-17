import os
from typing import Optional
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from fastapi import FastAPI
from openai import AzureOpenAI
from pydantic import BaseModel

load_dotenv()

azure_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
)

search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY")),
)

AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

app = FastAPI(
    title="Fraud Detection API",
    description="カード不正検知PoC用API",
    version="0.1.0"
)


class ReviewRequest(BaseModel):
    fraud_score: float
    risk_band: str
    amount: Optional[float] = None
    country: Optional[str] = None
    channel: Optional[str] = None
    entry_mode: Optional[str] = None
    merchant_category: Optional[str] = None
    is_foreign: Optional[int] = None
    is_ec: Optional[int] = None
    is_new_device: Optional[int] = None
    customer_tenure_months: Optional[int] = None
    authentication_result: Optional[str] = None
    top_shap_text: Optional[str] = None

def build_rag_query(req: ReviewRequest) -> str:
    return f"""
    fraud_score: {req.fraud_score}
    risk_band: {req.risk_band}
    amount: {req.amount}
    country: {req.country}
    channel: {req.channel}
    entry_mode: {req.entry_mode}
    merchant_category: {req.merchant_category}
    is_foreign: {req.is_foreign}
    is_ec: {req.is_ec}
    is_new_device: {req.is_new_device}
    customer_tenure_months: {req.customer_tenure_months}
    authentication_result: {req.authentication_result}
    SHAP: {req.top_shap_text}
    この取引に該当する不正パターンとNext Action
    """

def embed_text(text: str):
    response = azure_client.embeddings.create(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        input=text
    )
    return response.data[0].embedding

def search_manual_context(query: str, top_k: int = 3) -> str:
    query_vector = embed_text(query)

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="content_vector"
    )

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        select=["content", "source"],
        top=top_k
    )

    contexts = []

    for result in results:
        contexts.append(
            f"source: {result['source']}\n{result['content']}"
        )

    return "\n\n---\n\n".join(contexts)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Fraud Detection API is running"}


@app.post("/review-comment")
def generate_review_comment(req: ReviewRequest):
    prompt = f"""
あなたはカード会社の不正検知アナリストです。
以下の取引について、審査担当者向けのコメントを日本語で作成してください。

条件:
- 200字以内
- 断定しすぎない
- 不正の可能性、確認ポイント、推奨アクションを含める
- 匿名特徴量V1〜V28は「潜在的な異常パターン」と表現する
- 社内審査担当者向けにする

取引情報:
fraud_score: {req.fraud_score}
risk_band: {req.risk_band}
Amount: {req.amount}
country: {req.country}
channel: {req.channel}
entry_mode: {req.entry_mode}
merchant_category: {req.merchant_category}
is_foreign: {req.is_foreign}
is_ec: {req.is_ec}
is_new_device: {req.is_new_device}
customer_tenure_months: {req.customer_tenure_months}
authentication_result: {req.authentication_result}

SHAP上位寄与:
{req.top_shap_text}
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
        max_tokens=300,
    )

    return {
        "review_comment": response.choices[0].message.content
    }

@app.post("/rag-review-comment")
def generate_rag_review_comment(req: ReviewRequest):
    query = build_rag_query(req)

    manual_context = search_manual_context(query, top_k=3)

    prompt = f"""
あなたはカード会社の不正検知アナリストです。
以下の取引情報と、審査マニュアルの関連抜粋をもとに、
審査担当者向けのコメントを日本語で作成してください。

条件:
- 300字以内
- 断定しすぎない
- 不正の可能性、確認ポイント、推奨Next Actionを含める
- マニュアルのNext Actionを優先して反映する
- 匿名特徴量V1〜V28は「潜在的な異常パターン」と表現する
- 社内審査担当者向けにする

取引情報:
fraud_score: {req.fraud_score}
risk_band: {req.risk_band}
Amount: {req.amount}
country: {req.country}
channel: {req.channel}
entry_mode: {req.entry_mode}
merchant_category: {req.merchant_category}
is_foreign: {req.is_foreign}
is_ec: {req.is_ec}
is_new_device: {req.is_new_device}
customer_tenure_months: {req.customer_tenure_months}
authentication_result: {req.authentication_result}

SHAP上位寄与:
{req.top_shap_text}

審査マニュアル関連抜粋:
{manual_context}
"""

    response = azure_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": "あなたはカード不正検知の審査支援AIです。検索された審査マニュアルを根拠に、実務的なNext Actionを提示してください。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=500,
    )

    return {
        "review_comment": response.choices[0].message.content,
        "retrieved_context": manual_context
    }