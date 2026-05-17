import os

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader

load_dotenv()

# Azure OpenAI Embedding
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    openai_api_version="2024-12-01-preview",
)

# Markdown読み込み
loader = TextLoader("docs/fraud_review_manual.md", encoding="utf-8")
documents = loader.load()

# Chunk分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

docs = text_splitter.split_documents(documents)

print(f"chunk count: {len(docs)}")

# FAISS vector DB化
vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

# 保存
vectorstore.save_local("faiss_index")

print("saved: faiss_index")