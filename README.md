# ☕ ChaiDocs Chatbot

A modern Streamlit chatbot for querying ChaiCode Docs using OpenAI GPT-4 and Pinecone vector search (RAG pipeline).  
Ask technical questions and get concise, referenced answers from ChaiCode documentation.

---

## 🚀 Features

- *Ask technical questions* about ChaiCode Docs and get concise, referenced answers.
- *Sidebar with sample questions* for quick exploration.
- *Reference links* to original documentation sources.
- *Modern, clean UI* with Streamlit.
- *RAG pipeline*: combines OpenAI LLM with Pinecone vector search for context-aware answers.

---

## 🛠 Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/LakshayJ17/rag-chaidocs.git
cd rag-chaidocs
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

- Fill in your keys in `.env`:
    ```
    OPENAI_API_KEY=your-openai-api-key
    PINECONE_API_KEY=your-pinecone-api-key
    ```
- **Never commit your real .env file!**

### 4. Index the Documentation (One-time setup)

This will fetch and embed ChaiCode Docs into Pinecone.

```bash
python indexing.py
```

### 5. Run the Chatbot App

```bash
streamlit run chat.py
```

---

## 📄 File Descriptions

### chat.py
- Loads environment variables and API keys.
- Sets up Streamlit UI (title, sidebar, input box).
- Handles sidebar sample question buttons (auto-fills main input).
- On user query:
    - Embeds the query using OpenAI.
    - Searches Pinecone for relevant document chunks.
    - Builds a context string from top results.
    - Sends context and query to OpenAI GPT-4 for answer generation.
    - Displays the answer and reference links.

### indexing.py
- Loads ChaiCode Docs URLs.
- Downloads and splits documents into chunks.
- Embeds chunks using OpenAI.
- Stores embeddings in Pinecone for fast semantic search.

### requirements.txt
- Lists all Python dependencies needed for the app and deployment.
- Optimized for Streamlit Cloud compatibility.

### .gitignore
- Ensures `.env` and `venv/` are not committed.

---

## 🧑‍💻 Tech Stack

- [Streamlit](https://streamlit.io/) – UI framework
- [OpenAI GPT-4](https://platform.openai.com/) – LLM for answer generation
- [Pinecone](https://www.pinecone.io/) – Vector database for semantic search
- [LangChain](https://www.langchain.com/) – RAG pipeline utilities

---

## 📝 Example Usage

1. Enter a technical question (or click a sample from the sidebar).
2. The chatbot retrieves relevant documentation chunks and generates a concise answer.
3. Reference links to the original docs are shown below the answer.
