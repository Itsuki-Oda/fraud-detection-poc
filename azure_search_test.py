import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

load_dotenv()

search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY")),
)

aoai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
)

def embed_text(text: str):
    response = aoai_client.embeddings.create(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        input=text
    )
    return response.data[0].embedding

query = "海外EC高額決済で新規端末、認証未実施の場合のNext Actionは？"
query_vector = embed_text(query)

vector_query = VectorizedQuery(
    vector=query_vector,
    k_nearest_neighbors=3,
    fields="content_vector"
)

results = search_client.search(
    search_text=query,
    vector_queries=[vector_query],
    select=["content", "source"],
    top=3
)

for i, result in enumerate(results, start=1):
    print(f"\n===== result {i} =====")
    print("source:", result["source"])
    print(result["content"][:1000])