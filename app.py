import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from utils.document_processor import extract_text_from_pdf, chunk_text
from utils.vector_store import create_collection, add_chunks, search_chunks

load_dotenv()

st.set_page_config(page_title="RAG Document System", page_icon="📄", layout="wide")
st.title("📄 RAG Document System")
st.subheader("Upload any document and ask questions about it!")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("❌ Groq API key not found!")
    st.stop()

client = Groq(api_key=api_key)

# ── Session state ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ── Sidebar - Upload ──
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Reading and processing document..."):
                # Save file temporarily
                temp_path = f"data/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Extract and chunk
                text = extract_text_from_pdf(temp_path)
                chunks = chunk_text(text)

                # Store in ChromaDB
                doc_name = uploaded_file.name.replace(".pdf", "").replace(" ", "_")
                collection = create_collection(doc_name)
                add_chunks(collection, chunks, doc_name)

                st.session_state.collection = collection
                st.session_state.doc_name = uploaded_file.name
                st.session_state.messages = []

                st.success(f"✅ Processed {len(chunks)} chunks from {uploaded_file.name}")

    if st.session_state.doc_name:
        st.info(f"📄 Active: {st.session_state.doc_name}")

# ── Main chat ──
if st.session_state.collection is None:
    st.info("👈 Upload a PDF from the sidebar to get started!")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask anything about your document...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                # Find relevant chunks
                relevant_chunks = search_chunks(
                    st.session_state.collection,
                    prompt,
                    n_results=3
                )
                context = "\n\n".join(relevant_chunks)

                system_prompt = f"""You are a helpful assistant that answers questions 
                based on the provided document context.
                
                DOCUMENT CONTEXT:
                {context}
                
                INSTRUCTIONS:
                - Answer only based on the document context above
                - If the answer is not in the context say "I couldn't find that in the document"
                - Be clear and concise
                - Quote relevant parts when helpful
                """

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500
                )
                answer = response.choices[0].message.content
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})