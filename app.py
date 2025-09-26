# import os
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# # Load environment variables from .env file
# load_dotenv()

# # Langsmith Tracking (optional, can remove if not needed)
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# # Set up Streamlit page configuration
# st.set_page_config(page_title="CareBot", page_icon="💊")
# st.title("💊 CareBot")
# st.write("Describe your symptoms or ask a medical question.")

# # Get the user's question
# user_input = st.text_input("Enter your question here:")

# # Build the prompt template
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful medical assistant. Answer carefully and clearly."),
#         ("user", "Question: {question}")
#     ]
# )

# # Initialize Groq model and output parser
# llm = ChatGroq(model="qwen/qwen3-32b", api_key=os.getenv("GROQ_API_KEY"))
# output_parser = StrOutputParser()

# # Create chain
# chain = prompt | llm | output_parser

# # Generate response on user input
# if user_input:
#     with st.spinner("Analyzing..."):
#         response = chain.invoke({"question": user_input})
#     st.subheader("Response")
#     st.write(response)

import os
import streamlit as st
import re  # For post-processing to remove <think>
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from .env file
load_dotenv()

# Langsmith Tracking (optional, can remove if not needed)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Paths (matching main.py)
output_dir = r"D:\Udemy\Lang_Chain\1.2-Ollama\output"
faiss_index_path = os.path.join(output_dir, "faiss_index")

# FIXED: Load RAG components FIRST (top-level, before UI) to avoid scoping issues
@st.cache_resource(show_spinner=False)
def load_vectorstore():
    if not os.path.exists(faiss_index_path):
        raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}. Please run main.py first to process the PDF!")
    
    # Primary embeddings (384 dims to match main.py)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}  # 'cuda' for GPU
    )
    
    # Dimension check (prevents AssertionError)
    test_query_emb = embeddings.embed_query("test")
    expected_dim = len(test_query_emb)
    
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    
    # Verify index dims match
    index_dim = vectorstore.index.d if hasattr(vectorstore.index, 'd') else None
    if index_dim != expected_dim:
        raise ValueError(f"Dimension mismatch! Query embeddings: {expected_dim}, Index: {index_dim}. Re-run main.py with matching model.")
    
    if vectorstore.index.ntotal == 0:
        raise ValueError("FAISS index is empty! Re-run main.py to ensure PDF chunks were added.")
    
    print(f"[DEBUG] Loaded {vectorstore.index.ntotal} vectors successfully.")  # Terminal debug
    return vectorstore

def clean_response(response):
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
    lines = response.split('\n')
    cleaned_lines = [line for line in lines if not line.strip().startswith(('Okay,', '<think>', '<<think')) and line.strip()]
    return '\n'.join(cleaned_lines).strip()

if 'rag_loaded' not in st.session_state:
    with st.spinner("Loading PDF knowledge base... This may take a moment on first run."):
        try:
            vectorstore = load_vectorstore()
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        #    prompt = ChatPromptTemplate.from_messages( [ ("system", """You are a helpful medical assistant specializing in common skin problems. Provide answers based ONLY on the provided context from the PDF. Do not use external knowledge or general medical advice. If the context doesn't cover it, say 'I don't have information on that from the document.' Format your response as a concise list of bullet points with no introductory text or internal markers (e.g., <think>, <<think>). Start directly with headings like 'Treatment Strategies' or bullet points. Use the structure from the context: headings followed by - Item 1, - Item 2. Output ONLY the final structured answer—no explanations, reasoning, or prefixes."""), ("user", """Context: {context} Question: {question}""") ] )
            
            prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful medical assistant specializing in common skin problems. 
Answer strictly based on the provided PDF context. 
- Do not use external medical knowledge. 
- If the answer is not in the context, reply: "I don't have information on that from the document." 
- Format the response as structured bullet points or headings (e.g., 'Treatment Strategies', 'Clinical Features'). 
- Do not include explanations, reasoning steps, or prefixes. 
- Start directly with the structured answer."""),
        ("user", """Context: {context}
Question: {question}""")
    ]
)


            llm = ChatGroq(model="qwen/qwen3-32b", api_key=os.getenv("GROQ_API_KEY"))
            #llm = ChatGroq(model = "xai.grok-3", api_key=os.getenv("GROQ_API_KEY"))

            output_parser = StrOutputParser()

            # Format retrieved docs as context
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            # Create RAG chain
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | output_parser
            )
            
            # Store in session state for persistence
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = retriever
            st.session_state.rag_chain = rag_chain
            st.session_state.rag_loaded = True
            
            print("[DEBUG] RAG chain initialized successfully.")  # Terminal debug
            
        except Exception as e:
            st.error(f"Error loading vectorstore: {e}")
            st.info("💡 Tip: Delete the faiss_index folder and re-run main.py with the updated code above.")
            st.stop()
else:
    vectorstore = st.session_state.vectorstore
    retriever = st.session_state.retriever
    rag_chain = st.session_state.rag_chain

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Skin CareBot", 
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Main Header (centered, bold, beautiful)
st.markdown(
    """
    <div style="text-align: center; padding: 10px 0;">
        <h1 style="font-size: 2.5rem; margin-bottom: 0;">💊 Skin CareBot</h1>
        <p style="font-size: 1.1rem; color: #bbb;">Ask about common skin problems or describe your symptoms.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Status Bar
st.success("✅ Environment OK")

st.markdown("---")

# Input Section (card style)
with st.container():
    st.subheader("Status Query")
    # Add custom CSS for responsiveness
    st.markdown(
        """
        <style>
        .query-box {
            max-width: 700px;   /* Limit width */
            margin: auto;       /* Center horizontally */
        }
        .query-box input {
            font-size: 16px;    /* Slightly bigger text */
            padding: 8px;
        }
        .stButton>button {
            height: 40px;
            font-size: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Wrap inside a centered div
    with st.container():
        st.markdown('<div class="query-box">', unsafe_allow_html=True)

        user_input = st.text_input(
            "Your Question",
            placeholder="Enter your question here...",
            label_visibility="collapsed",
            key="user_input"
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            ask_button = st.button("Ask", use_container_width=True, type="primary")
        with col2:
            clear_button = st.button("Clear", use_container_width=True, type="secondary")

        st.markdown('</div>', unsafe_allow_html=True)

# Conversation Section
st.subheader("Conversation")
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="💊"):
            st.markdown(message["content"])

# Handle input
if ask_button and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="💊"):
        with st.spinner("Retrieving from PDF and analyzing..."):
            try:
                relevant_docs = retriever.invoke(user_input)
                st.session_state.last_relevant_docs = relevant_docs

                # INTEGRATED: Get unique relevant images from retrieved chunks
                relevant_images = []
                seen_paths = set()
                for doc in relevant_docs:
                    images = doc.metadata.get('images', [])
                    for img_path in images:
                        if img_path not in seen_paths:
                            seen_paths.add(img_path)
                            relevant_images.append(img_path)

                # INTEGRATED: Display relevant images if any
                if relevant_images:
                    st.markdown("**Relevant Medical Images from PDF:**")
                    cols = st.columns(min(3, len(relevant_images)))
                    for idx, img_path in enumerate(relevant_images):
                        with cols[idx % 3]:
                            st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)

                # If no relevant chunks found, return default message
                if not relevant_docs or all(len(doc.page_content.strip()) == 0 for doc in relevant_docs):
                    response = "I don't have information on that from the document."
                else:
                    raw_response = rag_chain.invoke(user_input)
                    response = clean_response(raw_response)

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                error_msg = f"⚠️ Retrieval or generation error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

elif clear_button:
    st.session_state.messages = []
    st.session_state.last_relevant_docs = None
    st.rerun()

# Debug Section
with st.expander("🔍 Debug Info"):
    if 'last_relevant_docs' in st.session_state and st.session_state.last_relevant_docs:
        for i, doc in enumerate(st.session_state.last_relevant_docs, 1):
            with st.expander(f"📑 Chunk {i}"):
                st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
    else:
        st.info("No query yet.")

# Advanced Options
with st.expander("⚙️ Advanced"):
    if st.button("🔄 Clear Cache & Reload", key="clear_cache_reload"):
        for key in list(st.session_state.keys()):
            if key != 'messages':
                del st.session_state[key]
        st.rerun()