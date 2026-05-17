import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    openai_api_version="2024-12-01-preview",
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

query = "海外EC高額決済で新規端末、認証未実施の場合のNext Actionは？"

docs = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(docs, start=1):
    print(f"\n===== result {i} =====")
    print(doc.page_content[:1000])