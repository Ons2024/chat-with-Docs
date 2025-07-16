import nltk
nltk.download('punkt')

from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv
import requests
import os
import shutil

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data"
EMBEDDING_ENDPOINT = os.getenv("LMSTUDIO_EMBEDDING_ENDPOINT")
API_KEY = os.getenv("LMSTUDIO_API_KEY")  # optional

# Custom embedding class for LM Studio
class LMStudioEmbeddings(Embeddings):
    def __init__(self, endpoint_url: str, api_key: str = None):
        self.endpoint_url = endpoint_url
        self.api_key = api_key

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": "text-embedding-nomic-embed-text-v1.5",  # Nom du modèle
            "input": texts  # C'est ici la clé attendue
        }

        response = requests.post(self.endpoint_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        embeddings = data.get("data") or data.get("embeddings") or data
        if not embeddings or len(embeddings) != len(texts):
            raise ValueError("Mismatch or missing embeddings from LM Studio response.")

        # Chaque élément de 'data' est généralement un dict contenant 'embedding'
        if isinstance(embeddings[0], dict) and "embedding" in embeddings[0]:
            embeddings = [e["embedding"] for e in embeddings]

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = LMStudioEmbeddings(endpoint_url=EMBEDDING_ENDPOINT, api_key=API_KEY)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
