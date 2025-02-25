import streamlit as st
from legal_rag import LegalRAG  # Import your LegalRAG class
import tempfile
import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatHuggingFace

# --- Setup and Initialization ---

# API Keys - Fetch from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Model Options - Expanded with more choices
MODEL_OPTIONS = {
    "GPT-4o-mini (OpenAI)": {"provider": "openai", "model_name": "gpt-4o-mini"},
    "GPT-4o (OpenAI)": {"provider": "openai", "model_name": "gpt-4o"},
    "GPT-4 (OpenAI)": {"provider": "openai", "model_name": "gpt-4"},
    "GPT-3.5-turbo (OpenAI)": {"provider": "openai", "model_name": "gpt-3.5-turbo"},
    "GPT-3.5-turbo-16k (OpenAI)": {"provider": "openai", "model_name": "gpt-3.5-turbo-16k"},
}


def initialize_legal_rag(selected_model_option):
    model_config = MODEL_OPTIONS[selected_model_option]
    provider = model_config["provider"]
    model_name = model_config["model_name"]

    model_instance = None  # Initialize outside if-else blocks

    if provider == "openai":
        model_instance = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=model_name, temperature=0) # Use API Key from secrets
    elif provider == "groq":
        model_instance = ChatGroq(api_key=GROQ_API_KEY, model_name=model_name, temperature=0.1) # Use API Key from secrets
    elif provider == "huggingface_hub":
        model_instance = ChatHuggingFace(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, repo_id=model_name, model_kwargs={"temperature": 0.1, "max_new_tokens": 500}) # Use API Key from secrets
    else:
        raise ValueError(f"Unsupported model provider: {provider}")

    legal_rag_instance = LegalRAG()
    legal_rag_instance.model = model_instance  # Directly set the model in LegalRAG instance
    legal_rag_instance._initialize_chains()  # Re-initialize chains with the new model
    return legal_rag_instance

def main():
    st.title("Legal RAG Chatbot")
    st.sidebar.header("App Settings")

    # Model Selection in Sidebar
    selected_model_name = st.sidebar.selectbox("Choose a Model", list(MODEL_OPTIONS.keys()))
    selected_model_config = MODEL_OPTIONS[selected_model_name]

    # Initialize LegalRAG with selected model in session state
    if 'legal_rag_instance' not in st.session_state or st.session_state.get("selected_model") != selected_model_name:
        with st.spinner(f"Initializing LegalRAG with {selected_model_name}..."):
            try:
                st.session_state.legal_rag_instance = initialize_legal_rag(selected_model_name)
                st.session_state.selected_model = selected_model_name  # Track current model
                st.success(f"LegalRAG initialized with {selected_model_name}!", icon="✅")
            except Exception as e:
                st.error(f"Error initializing LegalRAG with {selected_model_name}: {e}", icon="🚨")
                st.stop()

    legal_rag_instance = st.session_state.legal_rag_instance

    # Document Upload in Sidebar
    st.sidebar.subheader("Upload Document")
    uploaded_file = st.sidebar.file_uploader("Legal Document (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])
    document_type = st.sidebar.selectbox("Document Type", ["Court Order", "Contract", "Compliance Document", "Government Notification", "Statutory Document", "Case-related Document", "Other"], index=6)  # Default to 'Other'

    if uploaded_file is not None:
        if st.sidebar.button("Process Document", key="process_doc_button"):  # Unique key for button
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.read())
                file_path = tmp_file.name
            try:
                with st.spinner("Processing document..."):
                    legal_rag_instance.feed(file_path, document_type)
                st.sidebar.success("Document processed!", icon="✅")
            except Exception as e:
                st.sidebar.error(f"Error processing document: {e}", icon="🚨")
            finally:
                os.remove(file_path)  # Clean up temp file

    # Chat History Display
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Query Input at the bottom
    query = st.chat_input("Enter your legal query here:")  # Chat-style input

    if query:
        st.chat_message("user").markdown(query)  # Display user message immediately
        st.session_state.chat_history.append({"role": "user", "content": query})

        try:
            with st.chat_message("assistant"):  # Assistant message container
                with st.spinner("Generating response..."):
                    response = legal_rag_instance.ask_combined(query)  # Using ask_combined
                st.markdown(response)  # Display assistant response
            st.session_state.chat_history.append({"role": "assistant", "content": response})  # Store assistant message
            legal_rag_instance.get_chat_history()  # Keep chat history updated in LegalRAG memory

        except Exception as e:
            st.error(f"Error during query: {e}", icon="🚨")

    # Clear Chat History Button (Optional in sidebar)
    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        legal_rag_instance.memory.clear()  # Clear Langchain memory as well
        st.rerun()  # Force rerun to clear chat messages from UI


if __name__ == "__main__":
    main()