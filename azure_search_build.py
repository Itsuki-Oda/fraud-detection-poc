import os
import uuid

from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from openai import AzureOpenAI

load_dotenv()

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")

azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

EMBEDDING_DIMENSIONS = 1536

credential = AzureKeyCredential(search_key)

index_client = SearchIndexClient(
    endpoint=search_endpoint,
    credential=credential
)

search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=index_name,
    credential=credential
)

aoai_client = AzureOpenAI(
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_key,
    api_version="2024-12-01-preview",
)


def embed_text(text: str) -> list[float]:
    response = aoai_client.embeddings.create(
        model=embedding_deployment,
        input=text
    )
    return response.data[0].embedding


def create_or_update_index():
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True
        ),
        SimpleField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="vector-profile"
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-config"
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="hnsw-config"
            )
        ]
    )

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search
    )

    index_client.create_or_update_index(index)

    print(f"created or updated index: {index_name}")


def load_and_chunk_docs():
    loader = TextLoader(
        "docs/fraud_review_manual.md",
        encoding="utf-8"
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = splitter.split_documents(documents)

    print(f"chunk count: {len(docs)}")

    return docs


def upload_documents(docs):
    upload_docs = []

    for i, doc in enumerate(docs):
        content = doc.page_content

        upload_docs.append({
            "id": str(uuid.uuid4()),
            "content": content,
            "source": f"fraud_review_manual_chunk_{i}",
            "content_vector": embed_text(content)
        })

    result = search_client.upload_documents(upload_docs)

    print(f"uploaded: {len(result)} documents")


if __name__ == "__main__":
    create_or_update_index()

    docs = load_and_chunk_docs()

    upload_documents(docs)