# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a local RAG (Retrieval-Augmented Generation) system for indexing and querying website content using:
- **Scrapy** for web crawling
- **Qdrant** for vector storage
- **Ollama** for local LLM inference
- **sentence-transformers** for embeddings
- **Streamlit** for the web UI

The system has two main phases:
1. **Indexing**: Crawl a website → Extract content → Chunk text → Generate embeddings → Store in Qdrant
2. **Query**: User question → Embed query → Search Qdrant → Generate answer with LLM → Display in UI

## Development Commands

### Environment Setup
```bash
# Install dependencies (using uv - recommended)
uv sync

# Or with pip
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Then edit .env to set TARGET_WEBSITE_URL and other config
```

### Docker Services
```bash
# Start Qdrant and Ollama
docker-compose up -d

# Check services are running
docker-compose ps

# Stop services
docker-compose down

# View logs
docker-compose logs -f
```

### Ollama Model Management
```bash
# Pull LLM models
docker exec ollama ollama pull llama3.2
docker exec ollama ollama pull mistral

# List installed models
docker exec ollama ollama list

# Test Ollama is working
curl http://localhost:11434/api/tags
```

### Indexing
```bash
# Run the indexing pipeline (scrape + embed + index)
python -m indexer.run_indexing

# Re-index from already scraped JSON files
python -c "from indexer.run_indexing import index_from_json; index_from_json()"

# Index a specific URL programmatically
python -c "from indexer.run_indexing import index_from_url; index_from_url('https://example.com')"
```

### Running the Application
```bash
# Start the Streamlit web app
streamlit run app/streamlit_app.py

# Access at http://localhost:8501
```

### Testing Qdrant
```bash
# Check collection exists
curl http://localhost:6333/collections/website_docs

# View Qdrant dashboard
# Open http://localhost:6333/dashboard
```

## Architecture

### Data Flow

1. **Indexing Pipeline** (`indexer/run_indexing.py`):
   - `ScrapyBlogScraper` → Crawls website, extracts text/images → Saves JSON files to `scraped_data/`
   - `DocumentProcessor.chunk_documents()` → Splits text into chunks with overlap
   - `DocumentProcessor.create_embeddings()` → Generates vectors using sentence-transformers
   - `DocumentProcessor.index_to_qdrant()` → Stores vectors + metadata in Qdrant

2. **Query Pipeline** (`app/rag_engine.py`):
   - `RAGEngine.embed_query()` → Convert question to vector
   - `RAGEngine.search_documents()` → Qdrant similarity search
   - `RAGEngine.generate_answer()` → Ollama LLM generates response
   - `RAGEngine.generate_followup_questions()` → Suggest next questions

### Key Components

**Configuration** (`config/settings.py`):
- Centralized settings loaded from `.env`
- All defaults defined in `Settings` class
- Access via `settings.SETTING_NAME`

**Scraper** (`indexer/scraper.py`):
- `BlogSpider`: Scrapy CrawlSpider that auto-discovers links
- `BlogImagesPipeline`: Downloads images with metadata
- `JsonExportPipeline`: Saves each page as JSON
- Uses `DOWNLOAD_DELAY` for rate limiting (default 1.0s)

**Processor** (`indexer/processor.py`):
- `DocumentProcessor.chunk_documents()`: Uses LangChain's RecursiveCharacterTextSplitter
- `DocumentProcessor.create_embeddings()`: Batch encoding with sentence-transformers
- `DocumentProcessor.index_to_qdrant()`: Creates/recreates collection, uploads points

**RAG Engine** (`app/rag_engine.py`):
- Must use same embedding model as indexing (consistency critical!)
- `query()` method: main entry point for complete RAG pipeline
- Conversation history limited to last 6 messages (3 exchanges)

**Streamlit App** (`app/streamlit_app.py`):
- Chat interface with session state
- Model selection in sidebar
- Displays sources with relevance scores
- Follow-up question buttons

## Important Implementation Details

### Embedding Model Consistency
The same embedding model MUST be used for both indexing and querying. Changing `EMBEDDING_MODEL` in `.env` requires re-running the entire indexing pipeline.

### Scrapy Spider Configuration
- `BlogSpider` uses CrawlSpider with `LinkExtractor`
- Only crawls same domain (`allowed_domains`)
- Respects robots.txt by default (`ROBOTSTXT_OBEY = True`)
- Content extraction prioritizes: `<main>` → `<article>` → `<body>`

### Qdrant Collection Management
The indexing pipeline deletes and recreates the collection each time. This is by design in `processor.py:111-126`. For incremental updates, you would need to modify this behavior.

### JSON File Naming
Pages are saved as JSON with URL path converted to filename:
- `https://example.com/` → `index.json`
- `https://example.com/blog/post` → `blog_post.json`
- Path separators replaced with underscores

### Chunking Strategy
- RecursiveCharacterTextSplitter with separators: `["\n\n", "\n", ". ", " ", ""]`
- Chunk size and overlap configurable via `CHUNK_SIZE` and `CHUNK_OVERLAP`
- Each chunk stores: content, title, url, chunk_index, total_chunks

### LLM Context Building
The prompt includes:
- System prompt: Instructions for answering based on context
- Retrieved context: Top K chunks with source URLs
- Conversation history: Last 6 messages
- User question

## Configuration Via Environment Variables

All settings in `config/settings.py` can be overridden via `.env`:

**Qdrant**:
- `QDRANT_HOST` (default: "localhost")
- `QDRANT_PORT` (default: "6333")
- `QDRANT_COLLECTION_NAME` (default: "website_docs")

**Ollama**:
- `OLLAMA_HOST` (default: "http://localhost:11434")
- `DEFAULT_LLM_MODEL` (default: "llama3.2")

**Embeddings**:
- `EMBEDDING_MODEL` (default: "all-MiniLM-L6-v2")

**Chunking**:
- `CHUNK_SIZE` (default: "1000")
- `CHUNK_OVERLAP` (default: "200")

**RAG**:
- `TOP_K_RESULTS` (default: "5")
- `TEMPERATURE` (default: "0.7")
- `MAX_TOKENS` (default: "512")

**Scraping**:
- `TARGET_WEBSITE_URL` (required for automatic indexing)
- `SCRAPER_DELAY` (default: "1.0")
- `SCRAPED_DATA_DIR` (default: "scraped_data")
- `SCRAPED_IMAGES_DIR` (default: "scraped_images")

## Common Development Patterns

### Adding New Embedding Models
1. Update `EMBEDDING_MODEL` in `.env`
2. First run will auto-download the model
3. Must re-run indexing pipeline
4. Qdrant collection dimension must match new model

### Customizing Scraper Behavior
Edit `indexer/scraper.py` `LinkExtractor` rules to:
- Exclude URL patterns: `deny=(r'/tag/', r'/category/')`
- Include only patterns: `allow=(r'/docs/', r'/api/')`
- Adjust delay: `DOWNLOAD_DELAY` in settings

### Modifying Chunk Strategy
For different content types, adjust in `.env`:
- Technical docs: `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200`
- Blog posts: `CHUNK_SIZE=1500`, `CHUNK_OVERLAP=300`
- API docs: `CHUNK_SIZE=500`, `CHUNK_OVERLAP=100`

### Re-indexing Without Re-scraping
If you've already scraped and just want to re-index with different settings:
```python
from indexer.run_indexing import index_from_json
index_from_json("scraped_data")
```

## Troubleshooting

### "No module named 'scrapy'" or similar
Dependencies still installing. Wait for `uv sync` to complete.

### "Failed to connect to Qdrant"
```bash
docker-compose ps  # Check qdrant is running
curl http://localhost:6333/collections  # Test connection
```

### "Error generating answer: connection refused"
```bash
docker-compose ps  # Check ollama is running
docker exec ollama ollama list  # Verify model is installed
docker exec ollama ollama pull llama3.2  # Pull if missing
```

### Scraper finds only 1 page
- Check robots.txt blocking
- Verify links use same domain
- Look for JavaScript-heavy sites (Scrapy doesn't render JS)

### Memory issues during indexing
- Reduce `CHUNK_SIZE`
- Use smaller embedding model (all-MiniLM-L6-v2)
- Increase Docker memory limits

## File Structure Notes

- `config/`: Centralized settings management
- `indexer/`: Scraping and document processing
  - `scraper.py`: Scrapy spider and pipelines
  - `processor.py`: Chunking, embedding, Qdrant indexing
  - `run_indexing.py`: Main CLI and programmatic interfaces
- `app/`: Query-time application
  - `rag_engine.py`: RAG logic (embed, search, generate)
  - `streamlit_app.py`: Web UI
- `scraped_data/`: JSON files (gitignored)
- `scraped_images/`: Downloaded images (gitignored)

## Dependencies

- **Python 3.12+** required
- **sentence-transformers**: Large package (~800MB PyTorch)
- **Scrapy**: Async web scraping framework
- **LangChain**: Document processing utilities
- **Qdrant**: Vector database (via Docker)
- **Ollama**: LLM inference (via Docker)
- **Streamlit**: Web UI framework
