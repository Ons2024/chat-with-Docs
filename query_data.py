import os
import requests
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"
LMSTUDIO_EMBEDDING_ENDPOINT = os.getenv("LMSTUDIO_EMBEDDING_ENDPOINT")
LMSTUDIO_CHAT_ENDPOINT = os.getenv("LMSTUDIO_CHAT_ENDPOINT")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY")  # optionnel

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

class LMStudioEmbeddings:
    def __init__(self, endpoint_url: str, api_key: str = None):
        self.endpoint_url = endpoint_url
        self.api_key = api_key

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": "text-embedding-nomic-embed-text-v1.5",
            "input": texts,
        }
        response = requests.post(self.endpoint_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        # Extraire la liste des embeddings
        raw_embeddings = data.get("data") or data.get("embeddings") or data

        # Si raw_embeddings est une liste de dicts avec clé 'embedding', extraire cette clé
        if isinstance(raw_embeddings, list) and len(raw_embeddings) > 0 and isinstance(raw_embeddings[0], dict) and "embedding" in raw_embeddings[0]:
            embeddings = [item["embedding"] for item in raw_embeddings]
        else:
            embeddings = raw_embeddings

        if not embeddings or len(embeddings) != len(texts):
            raise ValueError("Mismatch or missing embeddings from LM Studio response.")

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

def chat_with_lmstudio(prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    if LMSTUDIO_API_KEY:
        headers["Authorization"] = f"Bearer {LMSTUDIO_API_KEY}"

    payload = {
        "model": "deepseek/deepseek-r1-0528-qwen3-8b",  # ou ton modèle chat préféré
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 512,
    }
    response = requests.post(LMSTUDIO_CHAT_ENDPOINT, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

def main():
    # Tu peux aussi récupérer la question via input()
    query_text = "who is alice?"  # <-- modifie ici ta question

    embedding_function = LMStudioEmbeddings(LMSTUDIO_EMBEDDING_ENDPOINT, LMSTUDIO_API_KEY)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.3:
        print("Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("Prompt envoyé au modèle:\n", prompt)

    response_text = chat_with_lmstudio(prompt)
    sources = [doc.metadata.get("source", None) for doc, _ in results]
    print(f"Réponse: {response_text}\nSources: {sources}")

if __name__ == "__main__":
    main()
