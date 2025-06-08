from dotenv import load_dotenv
import os
import streamlit as st
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

client = OpenAI(api_key=openai_api_key)


st.set_page_config(page_title="ChaiDocs Chatbot", page_icon="☕")
st.markdown("<h1 style='text-align:center;'>☕ ChaiDocs Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Sidebar (sample ques)
with st.sidebar:
    st.title("💡 Try a Sample Question")
    sample_questions = [
        "What is stash in Git?",
        "How do I create a new branch in Git?",
        "What is normalization in SQL?",
        "How do I deploy Node.js with Nginx?",
        "What are HTML tags?",
        "How to use Tailwind CSS with Django?",
        "How to set up PostgreSQL with Docker?"
    ]
    for i, q in enumerate(sample_questions, 1):
        st.button(q, key=f"sample_{i}", on_click=lambda q=q: st.session_state.update(query=q))
    st.markdown("---")
    st.info("Ask any technical question related to ChaiCode Docs!")

# Main
query = st.text_input("🔍 Ask your question:", key="query", placeholder="e.g., What is stash in Git?")

if query:
    with st.spinner("Thinking..."):
        embeddings = OpenAIEmbeddings(
            model='text-embedding-3-large', 
            openai_api_key=openai_api_key
        )

        pc = PineconeClient(api_key=pinecone_api_key)

        index = pc.Index("chaidocs")
        vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")

        # Retrieve relevant chunks
        docs = vectorstore.similarity_search(query, k=10)

        # context
        seen = set()
        context = "\n\n---\n\n".join(
            f"Title: {doc.metadata.get('title', 'Untitled')}\n"
            f"Source: {doc.metadata.get('source', '')}\n"
            f"Content: {doc.page_content or doc.metadata.get('text', '')}"
            for doc in docs if not (doc.metadata.get("source", "") in seen or seen.add(doc.metadata.get("source", "")))
        )

        SYSTEM_PROMPT = f"""
You are a helpful assistant for ChaiCode Docs. Use only the following context to answer user queries.

Your job is to help the user quickly understand a concept and direct them to the exact source link for more details.

If you're unsure, say "Please explore the documentation for more details."

Context:
{context}
"""

        chat_completion = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]
        )

        answer = chat_completion.choices[0].message.content
        st.chat_message("assistant").markdown(answer)
