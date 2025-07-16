# Chat with Docs

This project allows you to query a local document database using embeddings and a language model served via LM Studio. It uses [LangChain](https://python.langchain.com/) and [Chroma](https://docs.trychroma.com/) for vector search, and interacts with LM Studio for both embedding generation and chat completion.

## Features

- Query your own documents using natural language
- Retrieves relevant document chunks using vector search
- Uses LM Studio for both embeddings and chat completions
- Simple command-line interface



2. **Prepare your environment variables:**

   Create a `.env` file or set the following variables in your environment:
   ```
   LMSTUDIO_EMBEDDING_ENDPOINT= your link
   LMSTUDIO_CHAT_ENDPOINT= your link
   LMSTUDIO_API_KEY=your_api_key_if_needed
   ```

3. **Prepare your data:**
   - Place your markdown/text files in the `data/` directory.
   - Make sure the Chroma database is initialized in the `chroma/` directory.

## Usage

Run the main script to query your documents:

```sh
python query_data.py
```

