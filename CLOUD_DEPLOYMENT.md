# Cloud Deployment Guide - Minimal Cost

This guide shows how to deploy the RAG system using **cloud APIs only** (no local LLM/embeddings).

## Total Cost: ~$0-2/month

- **LLM (Groq)**: FREE tier
- **Embeddings (Together.ai)**: ~$0.008/1M tokens (~$0.10-1/month typical)
- **Vector DB (Qdrant Cloud)**: FREE tier (1GB)
- **Web App (Streamlit Cloud)**: FREE

---

## Step 1: Get API Keys

### 1.1 Groq (FREE - for LLM)
```bash
# Go to: https://console.groq.com/
# Sign up and get API key
# Free tier: Very generous rate limits
```

### 1.2 Together.ai (Cheap - for embeddings)
```bash
# Go to: https://api.together.xyz/
# Sign up and get API key
# Add $5 credits (will last months for typical use)
# Cost: $0.008 per 1M tokens
```

### 1.3 Qdrant Cloud (FREE tier)
```bash
# Go to: https://cloud.qdrant.io/
# Create free cluster (1GB storage limit)
# Get cluster URL and API key
```

---

## Step 2: Update Code to Use Cloud APIs

### 2.1 Update requirements.txt

```bash
# Add OpenAI library for API calls
echo "openai>=1.0.0" >> requirements.txt
pip install openai
```

### 2.2 Create .env.cloud file

```bash
cat > .env.cloud << 'EOF'
# API Keys
GROQ_API_KEY=your-groq-api-key-here
TOGETHER_API_KEY=your-together-api-key-here
QDRANT_API_KEY=your-qdrant-api-key-here

# Qdrant Cloud
QDRANT_HOST=your-cluster.qdrant.io
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=website_docs

# Cloud Models
GROQ_MODEL=llama-3.2-90b-text-preview
CLOUD_EMBEDDING_MODEL=togethercomputer/m2-bert-80M-8k-retrieval

# Chunking (same as before)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
TEMPERATURE=0.7
MAX_TOKENS=512

# Scraping
TARGET_WEBSITE_URL=https://your-website-to-index.com
SCRAPED_DATA_DIR=scraped_data
SCRAPED_IMAGES_DIR=scraped_images
SCRAPER_DELAY=1.0
EOF
```

### 2.3 Use cloud-based files

The cloud versions are already created:
- `indexer/processor_cloud.py` - Cloud embeddings for indexing
- `app/rag_engine_cloud.py` - Cloud LLM + embeddings for queries

---

## Step 3: Index Your Data

```bash
# Load cloud environment
export $(cat .env.cloud | xargs)

# Create indexing script
cat > index_cloud.py << 'EOF'
from indexer.scraper import ScrapyBlogScraper
from indexer.processor_cloud import DocumentProcessorCloud
from config.settings import settings
import os

# Load environment
from dotenv import load_dotenv
load_dotenv('.env.cloud')

print("Starting cloud indexing...")
settings.display_settings()

# Step 1: Scrape
scraper = ScrapyBlogScraper(
    base_url=os.getenv('TARGET_WEBSITE_URL'),
    output_dir='scraped_data',
    images_dir='scraped_images',
    delay=1.0
)
documents = scraper.scrape()

# Step 2: Process and index with cloud embeddings
processor = DocumentProcessorCloud()
processor.process_and_index(documents)

print("✓ Cloud indexing complete!")
EOF

# Run indexing
python index_cloud.py
```

This will:
1. Scrape your target website
2. Generate embeddings via Together.ai API
3. Store vectors in Qdrant Cloud

**Cost**: ~$0.10-0.50 depending on content size

---

## Step 4: Test Locally

```bash
# Create test script
cat > test_cloud.py << 'EOF'
from app.rag_engine_cloud import RAGEngineCloud
from dotenv import load_dotenv

load_dotenv('.env.cloud')

rag = RAGEngineCloud()
answer, sources, followups = rag.query("What is this documentation about?")

print("Answer:", answer)
print("\nSources:", len(sources))
print("\nFollow-ups:", followups)
EOF

python test_cloud.py
```

---

## Step 5: Deploy to Streamlit Cloud

### 5.1 Update Streamlit app

Create `app/streamlit_app_cloud.py`:

```python
"""
Streamlit app using cloud-based RAG engine.
"""
from typing import List, Dict
import streamlit as st
from app.rag_engine_cloud import RAGEngineCloud
from config.settings import settings


st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
    layout="wide"
)


def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    if 'current_model' not in st.session_state:
        st.session_state.current_model = "llama-3.2-90b-text-preview"


def initialize_rag_engine(model_name: str):
    if st.session_state.rag_engine is None or st.session_state.current_model != model_name:
        with st.spinner(f"Loading RAG engine with {model_name}..."):
            st.session_state.rag_engine = RAGEngineCloud(model_name=model_name)
            st.session_state.current_model = model_name


def display_message(role: str, content: str, sources: List[Dict] = None):
    with st.chat_message(role):
        st.markdown(content)

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
    initialize_session_state()

    with st.sidebar:
        st.title("⚙️ Settings")

        st.subheader("LLM Model")
        available_models = [
            "llama-3.2-90b-text-preview",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768"
        ]

        selected_model = st.selectbox(
            "Select Groq Model",
            options=available_models,
            index=0,
            help="All models are free on Groq"
        )

        st.subheader("Configuration")
        st.info(f"""
**Vector DB:** Qdrant Cloud
**Collection:** {settings.QDRANT_COLLECTION_NAME}
**LLM:** Groq (Free)
**Embeddings:** Together.ai
**Top K:** {settings.TOP_K_RESULTS}
        """)

        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.subheader("About")
        st.markdown("""
Cloud-based RAG system:
- Groq for LLM (FREE)
- Together.ai for embeddings (~$0.10/month)
- Qdrant Cloud for vectors (FREE tier)
        """)

    st.title("🤖 RAG Q&A System (Cloud)")
    st.markdown("Ask questions about your indexed documentation!")

    try:
        initialize_rag_engine(selected_model)
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Check your API keys in Streamlit secrets")
        return

    for message in st.session_state.messages:
        display_message(
            message["role"],
            message["content"],
            message.get("sources")
        )

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message("user", prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    conversation_history = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.messages[:-1]
                    ]

                    answer, sources, followup_questions = st.session_state.rag_engine.query(
                        prompt,
                        conversation_history=conversation_history
                    )

                    st.markdown(answer)

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

                    if followup_questions:
                        st.markdown("---")
                        st.markdown("**💡 Follow-up questions:**")
                        cols = st.columns(len(followup_questions))
                        for col, question in zip(cols, followup_questions):
                            with col:
                                if st.button(
                                    question,
                                    key=f"followup_{len(st.session_state.messages)}_{question[:20]}",
                                    use_container_width=True
                                ):
                                    st.session_state.messages.append({
                                        "role": "user",
                                        "content": question
                                    })
                                    st.rerun()

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
    main()
```

### 5.2 Push to GitHub

```bash
git add .
git commit -m "Add cloud deployment support"
git push origin main
```

### 5.3 Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Select your repository
4. Set main file: `app/streamlit_app_cloud.py`
5. Click "Advanced settings" → "Secrets"
6. Add your secrets:

```toml
GROQ_API_KEY = "your-groq-api-key"
TOGETHER_API_KEY = "your-together-api-key"
QDRANT_API_KEY = "your-qdrant-api-key"
QDRANT_HOST = "your-cluster.qdrant.io"
QDRANT_PORT = "6333"
QDRANT_COLLECTION_NAME = "website_docs"
GROQ_MODEL = "llama-3.2-90b-text-preview"
CLOUD_EMBEDDING_MODEL = "togethercomputer/m2-bert-80M-8k-retrieval"
TOP_K_RESULTS = "5"
TEMPERATURE = "0.7"
MAX_TOKENS = "512"
```

7. Click "Deploy"

**Done!** Your app is now live at `https://yourapp.streamlit.app`

---

## Cost Analysis

### One-time indexing (example: 100 pages):
- Scraping: FREE
- Embeddings: ~100 pages × 1000 tokens/page = 100K tokens = $0.0008
- Qdrant storage: FREE (under 1GB)

**Total: Less than $0.01**

### Monthly usage (100 queries/day):
- LLM (Groq): FREE
- Query embeddings: 100 queries × 30 days × 20 tokens/query = 60K tokens = $0.0005
- Qdrant reads: FREE

**Total: ~$0.01/month**

---

## Alternative: Use OpenAI Instead

If you prefer OpenAI (better quality, higher cost):

```env
# Use OpenAI instead of Groq
OPENAI_API_KEY=your-openai-key
```

Update `rag_engine_cloud.py`:
```python
# Replace Groq client with:
self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
self.model_name = "gpt-4o-mini"  # Cheapest: $0.15/1M input tokens

# For embeddings, use OpenAI:
response = self.llm_client.embeddings.create(
    input=[query],
    model="text-embedding-3-small"  # $0.02/1M tokens, 1536 dims
)
```

**Cost with OpenAI**: ~$2-5/month for moderate use

---

## Monitoring Costs

### Together.ai
- Dashboard: https://api.together.xyz/settings/billing
- Set spending limits to avoid surprises

### Groq
- Dashboard: https://console.groq.com/
- Free tier has rate limits (not cost limits)

### Qdrant Cloud
- Dashboard: https://cloud.qdrant.io/
- Free tier: 1GB storage (monitor usage)

---

## Troubleshooting

### "Invalid API key"
Check your `.env.cloud` or Streamlit secrets have correct keys

### "Rate limit exceeded" (Groq)
Wait a few seconds and retry. Free tier has limits.

### "Insufficient credits" (Together.ai)
Add more credits at https://api.together.xyz/settings/billing

### "Collection not found" (Qdrant)
Re-run indexing: `python index_cloud.py`
