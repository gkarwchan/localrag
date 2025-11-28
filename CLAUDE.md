# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a local RAG (Retrieval-Augmented Generation) system that combines:
- Qdrant vector database for semantic search
- Ollama for local LLM inference (llama3.2, mistral)
- sentence-transformers for local embeddings
- Streamlit for the web UI

The system scrapes website documentation, chunks and embeds it, then provides a chat interface for Q&A with source citations.

## Development Commands

### Initial Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env

# Start Docker services (Qdrant + Ollama)
docker-compose up -d

# Pull LLM models
docker exec ollama ollama pull llama3.2
docker exec ollama ollama pull mistral
```

### Running the Application
```bash
# Index website content (one-time or when updating sources)
python indexer/run_indexing.py

# Launch the Streamlit UI
streamlit run app/streamlit_app.py
```

### Development Tasks
```bash
# Check Docker services status
docker-compose ps

# View Qdrant dashboard
# Visit http://localhost:6333/dashboard

# Test Ollama connection
curl http://localhost:11434/api/tags

# View Ollama logs
docker logs ollama

# Restart services
docker-compose restart
```

## Architecture

### Data Flow

**Indexing Pipeline:**
1. `indexer/scraper.py` - Scrapes HTML content from URLs using BeautifulSoup
2. `indexer/processor.py` - Chunks text with LangChain's RecursiveCharacterTextSplitter
3. `indexer/processor.py` - Generates embeddings via sentence-transformers (batch_size=32)
4. `indexer/processor.py` - Stores vectors in Qdrant with metadata (title, url, chunk_index)

**Query Pipeline:**
1. `app/rag_engine.py` - Embeds user query with same model as indexing
2. `app/rag_engine.py` - Searches Qdrant using cosine similarity
3. `app/rag_engine.py` - Passes top-k chunks to Ollama as context
4. `app/rag_engine.py` - Generates answer and follow-up questions
5. `app/streamlit_app.py` - Displays results with source citations

### Key Components

**config/settings.py**
- Centralized configuration using environment variables
- Settings singleton instance used throughout codebase
- All configuration values have sensible defaults

**indexer/processor.py**
- `DocumentProcessor` class handles chunking, embedding, and indexing
- Deletes and recreates Qdrant collection on each indexing run
- Uses MD5 hashing for chunk IDs but inserts with sequential indices
- Batch uploads to Qdrant (batch_size=100)

**app/rag_engine.py**
- `RAGEngine` class handles query embedding, search, and LLM generation
- Maintains conversation history (last 3 exchanges, 6 messages)
- Uses same embedding model as indexing (critical for vector search)
- Ollama integration with configurable temperature and max_tokens

**app/streamlit_app.py**
- Session state management for chat history and RAG engine
- Model selection dropdown (dynamically fetches from Ollama)
- Follow-up question buttons that populate chat input
- Source document display with scores and chunk indices

### Important Implementation Details

**Embedding Consistency:**
- The same embedding model MUST be used for both indexing and querying
- Model is specified in `settings.EMBEDDING_MODEL`
- Changing the model requires re-running the entire indexing process
- Model dimension is auto-detected and used for Qdrant vector config

**Qdrant Collection Management:**
- Collection is recreated (deleted + created) on each indexing run
- This means all existing data is lost when re-indexing
- Collection name is configurable via `settings.QDRANT_COLLECTION_NAME`
- Uses COSINE distance metric for similarity

**Ollama Model Management:**
- Models must be pulled into the Ollama container before use
- Default model is `llama3.2` but can be changed in `.env`
- Model switching is supported at runtime via RAGEngine.switch_model()
- The Streamlit app fetches available models dynamically

**Text Chunking:**
- Uses recursive splitting with separators: `["\n\n", "\n", ". ", " ", ""]`
- Default chunk_size=1000, chunk_overlap=200
- Each chunk preserves metadata: title, url, chunk_index, total_chunks
- Chunks are numbered sequentially within each document

**Web Scraping:**
- Currently only supports static HTML (no JavaScript rendering)
- Uses BeautifulSoup to extract text from specified HTML tags
- Preserves document title and URL for citation
- No rate limiting or robots.txt checking implemented

## Configuration Notes

**Environment Variables:**
All settings in `.env` are optional and have defaults in `config/settings.py`:
- `EMBEDDING_MODEL` - Must match between indexing and querying
- `CHUNK_SIZE` / `CHUNK_OVERLAP` - Affects retrieval quality
- `TOP_K_RESULTS` - Number of chunks returned for context
- `TEMPERATURE` - LLM creativity (0.0-1.0)
- `MAX_TOKENS` - LLM response length limit

**Docker Services:**
- Qdrant: ports 6333 (HTTP), 6334 (gRPC)
- Ollama: port 11434
- Both use named volumes for persistence
- GPU support for Ollama available via docker-compose.yml comments

## Common Development Scenarios

**Adding a new embedding model:**
1. Update `EMBEDDING_MODEL` in `.env`
2. Re-run `python indexer/run_indexing.py` to re-index all documents
3. The dimension will be auto-detected and Qdrant collection updated

**Testing with different LLM models:**
1. Pull model: `docker exec ollama ollama pull <model_name>`
2. Update `DEFAULT_LLM_MODEL` in `.env` OR select in Streamlit UI
3. No re-indexing required

**Debugging retrieval quality:**
- Check `context_docs` scores in rag_engine.py:217
- Adjust `TOP_K_RESULTS` and `CHUNK_SIZE` in `.env`
- View retrieved chunks in Streamlit UI source citations
- Use Qdrant dashboard to inspect vectors and payloads

**Updating indexed content:**
- Run `python indexer/run_indexing.py` again with new/updated URLs
- Note: This deletes the existing collection and all vectors
- Consider implementing incremental updates if needed

## Testing Considerations

This codebase has no automated tests. When making changes:
- Test indexing with a small set of URLs first
- Verify Qdrant collection is created correctly
- Test querying with known questions that should match indexed content
- Check Ollama model availability before running queries
- Verify source citations match the actual retrieved documents
