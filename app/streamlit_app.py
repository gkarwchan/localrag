"""
Streamlit web application for the RAG system.
Provides a chat interface for asking questions about indexed documentation.
"""
import streamlit as st
from app.rag_engine import RAGEngine
from config.settings import settings


# Page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None

    if 'current_model' not in st.session_state:
        st.session_state.current_model = settings.DEFAULT_LLM_MODEL


def initialize_rag_engine(model_name: str):
    """Initialize or reinitialize the RAG engine with selected model."""
    if st.session_state.rag_engine is None or st.session_state.current_model != model_name:
        with st.spinner(f"Loading RAG engine with {model_name}..."):
            st.session_state.rag_engine = RAGEngine(model_name=model_name)
            st.session_state.current_model = model_name


def display_message(role: str, content: str, sources: List[Dict] = None):
    """Display a chat message with optional sources."""
    with st.chat_message(role):
        st.markdown(content)

        # Display sources if available
        if sources:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"""
**Source {i}** (Score: {source['score']:.3f})
**Title:** {source['title']}
**URL:** [{source['url']}]({source['url']})
**Excerpt:** {source['content'][:200]}...
---
                    """)


def main():
    """Main Streamlit application."""
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")

        # Model selection
        st.subheader("LLM Model")
        available_models = ["llama3.2", "mistral", "llama3.2:latest", "mistral:latest"]

        selected_model = st.selectbox(
            "Select Model",
            options=available_models,
            index=available_models.index(st.session_state.current_model)
            if st.session_state.current_model in available_models else 0,
            help="Choose the Ollama model for generating answers"
        )

        # Configuration display
        st.subheader("Configuration")
        st.info(f"""
**Vector DB:** Qdrant
**Collection:** {settings.QDRANT_COLLECTION_NAME}
**Embedding Model:** {settings.EMBEDDING_MODEL}
**Top K Results:** {settings.TOP_K_RESULTS}
**Temperature:** {settings.TEMPERATURE}
        """)

        # Clear conversation button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # About section
        st.subheader("About")
        st.markdown("""
This RAG (Retrieval-Augmented Generation) system allows you to ask questions about indexed website documentation.

**Features:**
- Semantic search using sentence-transformers
- Local LLM inference with Ollama
- Source citations for transparency
- Follow-up question suggestions
        """)

    # Main content
    st.title("🤖 RAG Q&A System")
    st.markdown("Ask questions about your indexed documentation!")

    # Initialize RAG engine
    try:
        initialize_rag_engine(selected_model)
    except Exception as e:
        st.error(f"Error initializing RAG engine: {str(e)}")
        st.info("Please make sure:")
        st.markdown("""
1. Docker containers are running: `docker-compose up -d`
2. Documents are indexed: `python indexer/run_indexing.py`
3. Ollama has the selected model: `ollama pull {selected_model}`
        """)
        return

    # Display chat history
    for message in st.session_state.messages:
        display_message(
            message["role"],
            message["content"],
            message.get("sources")
        )

    # Chat input
    if prompt := st.chat_input("Ask a question about the documentation..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message("user", prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get conversation history for context
                    conversation_history = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.messages[:-1]  # Exclude current message
                    ]

                    # Query the RAG engine
                    answer, sources, followup_questions = st.session_state.rag_engine.query(
                        prompt,
                        conversation_history=conversation_history
                    )

                    # Display answer
                    st.markdown(answer)

                    # Display sources
                    if sources:
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"""
**Source {i}** (Relevance: {source['score']:.3f})
**Title:** {source['title']}
**URL:** [{source['url']}]({source['url']})
**Excerpt:** {source['content'][:200]}...
---
                                """)

                    # Display follow-up questions
                    if followup_questions:
                        st.markdown("---")
                        st.markdown("**💡 Follow-up questions:**")

                        # Create columns for follow-up question buttons
                        cols = st.columns(len(followup_questions))
                        for col, question in zip(cols, followup_questions):
                            with col:
                                if st.button(
                                    question,
                                    key=f"followup_{len(st.session_state.messages)}_{question[:20]}",
                                    use_container_width=True
                                ):
                                    # Add follow-up question as new user message
                                    st.session_state.messages.append({
                                        "role": "user",
                                        "content": question
                                    })
                                    st.rerun()

                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    from typing import List, Dict
    main()
