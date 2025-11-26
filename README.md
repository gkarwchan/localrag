# Local RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for indexing and querying website content using local LLMs and vector databases. Built with Scrapy for robust web crawling, Qdrant for vector search, and Ollama for local LLM inference.

## Features

- **Automatic Web Crawling**: Scrapy-based crawler that automatically discovers and indexes entire websites
- **Local LLM Inference**: Uses Ollama with Llama 3.2, Mistral, or other local models
- **Local Embeddings**: sentence-transformers for fast, privacy-preserving embedding generation
- **Vector Database**: Qdrant for efficient semantic search and retrieval
- **Image Extraction**: Downloads and indexes images with metadata (alt text, captions)
- **Interactive UI**: Streamlit-based chat interface with real-time responses
- **Source Citations**: Transparent answers with clickable source URLs
- **Follow-up Questions**: AI-generated contextual follow-up suggestions
- **Docker Support**: Containerized Qdrant and Ollama services for easy deployment
- **JSON Export**: Each page saved as structured JSON for inspection and re-indexing

## Tech Stack

### Core Technologies
- **Python 3.12+** - Modern Python with type hints
- **Scrapy 2.11.0** - Professional-grade web crawling framework
- **LangChain 0.1.0** - Document processing and RAG orchestration
- **sentence-transformers 2.2.2** - Local embedding generation (384-768 dimensions)
- **Qdrant 1.7.0** - High-performance vector database
- **Ollama 0.1.6** - Local LLM inference server
- **Streamlit 1.29.0** - Interactive web UI framework
- **Pillow 10.1.0** - Image processing and metadata extraction

### Supporting Libraries
- **PyTorch** - Deep learning framework (via sentence-transformers)
- **lxml** - Fast XML/HTML parsing
- **Twisted** - Asynchronous networking (Scrapy dependency)
- **python-dotenv** - Environment variable management
- **tqdm** - Progress bars for long-running operations

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Local RAG System                          │
└─────────────────────────────────────────────────────────────────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
         ┌──────▼──────┐  ┌─────▼─────┐  ┌──────▼──────┐
         │   Indexing  │  │  Storage  │  │   Query     │
         │   Pipeline  │  │   Layer   │  │  Pipeline   │
         └─────────────┘  └───────────┘  └─────────────┘
```

### 1. Indexing Pipeline (One-Time Setup)

```
┌─────────────────────────────────────────────────────────────────┐
│                      INDEXING PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘

Step 1: Web Crawling (Scrapy)
────────────────────────────────────────────────────────────────
    Target URL
        │
        ▼
    ┌─────────────────┐
    │  BlogSpider     │  • Automatic link discovery
    │  (CrawlSpider)  │  • Same-domain filtering
    │                 │  • Robots.txt compliance
    └────────┬────────┘  • Rate limiting (1s delay)
             │
             ▼
    ┌─────────────────┐
    │ Link Extractor  │  • Follows <a> tags
    │                 │  • Excludes external domains
    └────────┬────────┘  • Deduplicates URLs
             │
             ▼

Step 2: Content Extraction
────────────────────────────────────────────────────────────────
    ┌─────────────────┐
    │  parse_page()   │  • Extracts <title>
    │                 │  • Prioritizes <main>, <article>
    │                 │  • Removes nav, footer, scripts
    └────────┬────────┘  • Normalizes whitespace
             │
             ├──────────────────────┐
             │                      │
             ▼                      ▼
    ┌─────────────────┐    ┌──────────────────┐
    │  Text Content   │    │ Image Extraction │
    │                 │    │                  │
    │ • Title         │    │ • Image URLs     │
    │ • Main text     │    │ • Alt text       │
    │ • Cleaned HTML  │    │ • Captions       │
    └────────┬────────┘    │ • Titles         │
             │             └────────┬──────────┘
             │                      │
             ▼                      ▼
    ┌─────────────────┐    ┌──────────────────┐
    │ BlogPageItem    │    │ ImagesPipeline   │
    │                 │◄───│                  │
    │ • url           │    │ Downloads images │
    │ • title         │    │ to disk          │
    │ • text          │    └──────────────────┘
    │ • images[]      │
    │ • scraped_at    │
    └────────┬────────┘
             │
             ▼

Step 3: Data Persistence
────────────────────────────────────────────────────────────────
    ┌─────────────────┐
    │ JsonExportPipe  │  Saves to: scraped_data/
    │                 │  Format: {url_path}.json
    └────────┬────────┘  One file per page
             │
             ▼
    ┌─────────────────┐
    │  JSON Files     │
    │  ├─ index.json  │
    │  ├─ about.json  │
    │  └─ post1.json  │
    └────────┬────────┘
             │
             ▼

Step 4: Text Processing (LangChain)
────────────────────────────────────────────────────────────────
    ┌─────────────────┐
    │ DocumentLoader  │  Loads JSON documents
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ TextSplitter    │  • Recursive character splitting
    │                 │  • Chunk size: 1000 tokens
    │                 │  • Overlap: 200 tokens
    └────────┬────────┘  • Preserves context
             │
             ▼
    [Chunk 1] [Chunk 2] [Chunk 3] ... [Chunk N]

Step 5: Embedding Generation (sentence-transformers)
────────────────────────────────────────────────────────────────
    ┌─────────────────┐
    │ Embedding Model │  Model: all-MiniLM-L6-v2
    │                 │  Dimensions: 384
    │ (Local CPU/GPU) │  Speed: ~500 docs/sec (CPU)
    └────────┬────────┘  Batch size: 32
             │
             ▼
    [Vector 1] [Vector 2] [Vector 3] ... [Vector N]
    (384-dim)  (384-dim)  (384-dim)      (384-dim)

Step 6: Vector Storage (Qdrant)
────────────────────────────────────────────────────────────────
    ┌─────────────────┐
    │ Qdrant Client   │  • HNSW index for fast search
    │                 │  • Cosine similarity metric
    │ Collection:     │  • Metadata: url, title, text
    │ "website_docs"  │  • Persistent storage
    └─────────────────┘
```

### 2. Query Pipeline (Real-Time)

```
┌─────────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                             │
└─────────────────────────────────────────────────────────────────┘

Step 1: User Input
────────────────────────────────────────────────────────────────
    User Question: "How do I configure the scraper?"
                            │
                            ▼

Step 2: Query Embedding
────────────────────────────────────────────────────────────────
    ┌─────────────────┐
    │ Same Embedding  │  • Uses SAME model as indexing
    │ Model           │  • all-MiniLM-L6-v2
    │                 │  • Ensures consistency
    └────────┬────────┘
             │
             ▼
    Query Vector (384-dim)

Step 3: Semantic Search (Qdrant)
────────────────────────────────────────────────────────────────
    ┌─────────────────┐
    │ Vector Search   │  • Cosine similarity
    │                 │  • Top-K retrieval (K=5)
    │ HNSW Algorithm  │  • Sub-millisecond search
    └────────┬────────┘  • Score threshold: 0.7
             │
             ▼
    [Result 1: 0.92] [Result 2: 0.87] ... [Result 5: 0.75]
         │                 │                     │
         └─────────────────┴─────────────────────┘
                           │
                           ▼

Step 4: Context Assembly
────────────────────────────────────────────────────────────────
    ┌─────────────────┐
    │ RAG Engine      │  • Combines top results
    │                 │  • Preserves source URLs
    │ Retrieved Docs: │  • Formats as context
    │ • Doc 1 + URL   │  • Deduplicates content
    │ • Doc 2 + URL   │
    │ • Doc 3 + URL   │
    └────────┬────────┘
             │
             ▼
    Context: "From {url1}: content... From {url2}: content..."

Step 5: LLM Generation (Ollama)
────────────────────────────────────────────────────────────────
    ┌─────────────────┐
    │  Ollama LLM     │  Model: llama3.2 or mistral
    │                 │
    │ Prompt:         │  Temperature: 0.7
    │ "Context:       │  Max tokens: 512
    │  {context}      │  Streaming: Yes
    │                 │
    │  Question:      │  System prompt: Answer based
    │  {question}     │  on context, cite sources
    │                 │
    │  Answer:"       │
    └────────┬────────┘
             │
             ▼
    Generated Answer with inline citations

Step 6: Follow-up Generation
────────────────────────────────────────────────────────────────
    ┌─────────────────┐
    │ Follow-up Gen   │  • Analyzes conversation
    │                 │  • Suggests 3 questions
    │ Based on:       │  • Context-aware
    │ • Question      │  • Encourages exploration
    │ • Answer        │
    │ • Context       │
    └────────┬────────┘
             │
             ▼
    ["What about X?", "How does Y work?", "Can you explain Z?"]

Step 7: UI Display (Streamlit)
────────────────────────────────────────────────────────────────
    ┌─────────────────┐
    │ Streamlit App   │  • Streaming display
    │                 │  • Clickable sources
    │ Displays:       │  • Follow-up buttons
    │ • Answer        │  • Chat history
    │ • Sources       │  • Model selection
    │ • Follow-ups    │
    └─────────────────┘
```

### 3. Data Flow Diagram

```
                    ┌─────────────────────┐
                    │   Target Website    │
                    └──────────┬──────────┘
                               │
                               │ HTTP Requests
                               ▼
                    ┌─────────────────────┐
                    │  Scrapy Spider      │
                    │  - BlogSpider       │
                    │  - LinkExtractor    │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │   Images    │  │    Text     │  │  Metadata   │
    │ Pipeline    │  │  Content    │  │  (URLs)     │
    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
           │                │                │
           ▼                ▼                ▼
    ┌─────────────────────────────────────────────┐
    │         JSON Files (scraped_data/)          │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐       │
    │  │ page1   │ │ page2   │ │ page3   │  ...  │
    │  └─────────┘ └─────────┘ └─────────┘       │
    └────────────────────┬────────────────────────┘
                         │
                         │ Load & Process
                         ▼
              ┌──────────────────────┐
              │   Document Loader    │
              │   (LangChain)        │
              └──────────┬───────────┘
                         │
                         │ Split into chunks
                         ▼
              ┌──────────────────────┐
              │   Text Splitter      │
              │   (1000/200 tokens)  │
              └──────────┬───────────┘
                         │
                         │ [Chunk 1, Chunk 2, ...]
                         ▼
              ┌──────────────────────┐
              │  Embedding Model     │
              │  (sentence-trans.)   │
              └──────────┬───────────┘
                         │
                         │ [Vector 1, Vector 2, ...]
                         ▼
              ┌──────────────────────┐
              │   Qdrant Database    │
              │   Collection:        │
              │   "website_docs"     │
              └──────────┬───────────┘
                         │
                         │ Query time
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
┌───────────────┐              ┌──────────────────┐
│ User Question │              │   Streamlit UI   │
└───────┬───────┘              └──────────────────┘
        │                                ▲
        │ Embed query                    │
        ▼                                │
┌───────────────┐                        │
│ Query Vector  │                        │
└───────┬───────┘                        │
        │                                │
        │ Search                         │
        ▼                                │
┌───────────────┐                        │
│ Top-K Results │                        │
│ (with URLs)   │                        │
└───────┬───────┘                        │
        │                                │
        │ Format context                 │
        ▼                                │
┌───────────────┐                        │
│  Ollama LLM   │                        │
│  (llama3.2)   │                        │
└───────┬───────┘                        │
        │                                │
        │ Generate answer                │
        └────────────────────────────────┘
```

## Project Structure

```
localrag/
├── docker-compose.yml          # Qdrant + Ollama services
├── pyproject.toml              # Python project configuration
├── requirements.txt            # Python dependencies
├── uv.lock                     # UV dependency lock file
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
├── SCRAPER_USAGE.md           # Detailed scraper guide
│
├── config/
│   ├── __init__.py
│   └── settings.py            # Centralized configuration
│                              # - Qdrant settings
│                              # - Ollama settings
│                              # - Embedding config
│                              # - Scraping config
│
├── indexer/
│   ├── __init__.py
│   ├── scraper.py            # Scrapy-based web crawler
│   │                         # - BlogSpider (CrawlSpider)
│   │                         # - BlogImagesPipeline
│   │                         # - JsonExportPipeline
│   │                         # - ScrapyBlogScraper
│   │
│   ├── processor.py          # Document processing
│   │                         # - Text chunking (LangChain)
│   │                         # - Embedding generation
│   │                         # - Qdrant indexing
│   │
│   └── run_indexing.py       # Main indexing script
│                             # - Interactive CLI
│                             # - Programmatic API
│                             # - JSON re-indexing
│
├── app/
│   ├── __init__.py
│   ├── rag_engine.py         # RAG query logic
│   │                         # - Query embedding
│   │                         # - Semantic search
│   │                         # - LLM generation
│   │                         # - Follow-up generation
│   │
│   └── streamlit_app.py      # Streamlit UI
│                             # - Chat interface
│                             # - Model selection
│                             # - Source display
│                             # - Follow-up buttons
│
├── scraped_data/             # JSON output (gitignored)
│   ├── index.json
│   ├── about.json
│   └── blog_post.json
│
└── scraped_images/           # Downloaded images (gitignored)
    └── full/
        ├── abc123.jpg
        └── def456.png
```

## Installation

### Prerequisites

- **Python 3.12+** (3.8+ works but 3.12 recommended)
- **Docker and Docker Compose** for Qdrant and Ollama
- **Git** for version control
- **10GB+ free disk space** (for models and data)
- **8GB+ RAM** recommended (4GB minimum)

### Step 1: Clone or Setup Project

```bash
cd localrag
```

### Step 2: Install Python Dependencies

Using UV (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` to configure your target website:
```env
# Required: Set your target blog/website URL
TARGET_WEBSITE_URL=https://yourblog.com

# Optional: Customize scraping behavior
SCRAPER_DELAY=1.0
SCRAPED_DATA_DIR=scraped_data
SCRAPED_IMAGES_DIR=scraped_images

# Optional: Customize RAG behavior
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
```

### Step 4: Start Docker Services

```bash
docker-compose up -d
```

This starts:
- **Qdrant** on port 6333 (vector database)
- **Ollama** on port 11434 (LLM server)

Verify services are running:
```bash
docker-compose ps
```

Expected output:
```
NAME          IMAGE                    STATUS
qdrant        qdrant/qdrant:latest     Up
ollama        ollama/ollama:latest     Up
```

### Step 5: Pull Ollama Models

Pull the LLM models you want to use:

```bash
# Pull Llama 3.2 (recommended, ~2GB)
docker exec ollama ollama pull llama3.2

# Pull Mistral (alternative, ~4GB)
docker exec ollama ollama pull mistral

# List available models
docker exec ollama ollama list
```

## Usage

### Method 1: Automatic Website Crawling (Recommended)

The new Scrapy-based scraper automatically discovers and crawls all pages on a website:

```bash
python -m indexer.run_indexing
```

You'll see:
```
==================================================
RAG System - Website Indexing Pipeline (Scrapy)
==================================================
Current Configuration:
...
Target Website: https://yourblog.com

The crawler will automatically discover and scrape all pages
on the target website (same domain only).

Proceed with crawling and indexing? (yes/no):
```

Type `yes` and the crawler will:
1. Start at the target URL
2. Automatically discover all internal links
3. Extract text content and images from each page
4. Download images with metadata
5. Save each page as JSON
6. Process and index everything into Qdrant

### Method 2: Programmatic Crawling

```python
from indexer.run_indexing import index_from_url

# Crawl and index entire website
index_from_url("https://docs.example.com")
```

### Method 3: Re-index from Existing JSON

If you've already scraped a website and want to re-index:

```python
from indexer.run_indexing import index_from_json

# Re-index from scraped JSON files
index_from_json("scraped_data")
```

### Launch the RAG Application

Start the Streamlit web app:

```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Application

1. **Select Model**: Choose Llama 3.2 or Mistral from the sidebar
2. **Ask Questions**: Type your question in the chat input
3. **View Answers**: See AI-generated answers with source citations
4. **Explore Sources**: Click on source URLs to visit original pages
5. **Follow-up**: Click suggested follow-up questions for deeper exploration
6. **Clear History**: Use sidebar button to start fresh conversation

## Advanced Configuration

### Embedding Models

Available sentence-transformers models:

| Model | Dimensions | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | Default, balanced |
| `all-mpnet-base-v2` | 768 | Medium | Better | Higher quality |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | Fast | Good | Q&A optimized |

Change in `.env`:
```env
EMBEDDING_MODEL=all-mpnet-base-v2
```

**Important**: After changing embedding model, re-run indexing!

### Chunking Strategy

Adjust chunk size and overlap for different content types:

| Content Type | Chunk Size | Overlap | Rationale |
|-------------|-----------|---------|-----------|
| Technical docs | 1000 | 200 | Default, preserves context |
| Blog posts | 1500 | 300 | Longer content, more overlap |
| API docs | 500 | 100 | Short, focused chunks |
| Narrative content | 2000 | 400 | Long-form content |

Edit in `.env`:
```env
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Scraping Customization

Edit `indexer/scraper.py` to customize crawling behavior:

```python
# Exclude certain URL patterns
rules = (
    Rule(
        LinkExtractor(
            allow=(),
            deny=(r'/tag/', r'/category/', r'/admin/'),
        ),
        callback='parse_page',
        follow=True
    ),
)

# Adjust crawl delay
DOWNLOAD_DELAY = 2.0  # 2 seconds between requests

# Increase concurrent requests (use carefully!)
CONCURRENT_REQUESTS_PER_DOMAIN = 2
```

### RAG Parameters

Fine-tune retrieval and generation:

```env
# Number of chunks to retrieve
TOP_K_RESULTS=5

# LLM creativity (0.0-1.0)
TEMPERATURE=0.7

# Maximum response length
MAX_TOKENS=512
```

## Scraped Data Structure

Each page is saved as JSON in `scraped_data/`:

```json
{
  "url": "https://example.com/blog/post-title",
  "title": "Post Title Here",
  "text": "Full extracted text content with normalized whitespace...",
  "images": [
    {
      "url": "https://example.com/images/photo.jpg",
      "local_path": "full/abc123def456.jpg",
      "alt": "Photo description from alt attribute",
      "title": "Photo title from title attribute",
      "caption": "Figure caption if available"
    }
  ],
  "scraped_at": "2025-11-24T12:34:56.789012"
}
```

Images are downloaded to `scraped_images/full/` with content-hashed filenames.

## Troubleshooting

### Dependencies Installation Slow

The first `uv sync` or `pip install` can take 5-10 minutes due to large ML packages:
- PyTorch (~800MB)
- CUDA libraries (~3GB total for GPU support)
- sentence-transformers models (downloaded on first use)

This is normal. Subsequent installs use cached packages.

### Ollama Connection Error

**Error**: `Error generating answer: connection refused`

**Solutions**:
1. Check Ollama is running:
   ```bash
   docker-compose ps
   curl http://localhost:11434/api/tags
   ```

2. Pull the model:
   ```bash
   docker exec ollama ollama pull llama3.2
   ```

3. Check model is loaded:
   ```bash
   docker exec ollama ollama list
   ```

### Qdrant Connection Error

**Error**: `Failed to connect to Qdrant`

**Solutions**:
1. Verify Qdrant is running:
   ```bash
   docker-compose ps
   ```

2. Access Qdrant dashboard: http://localhost:6333/dashboard

3. Restart services:
   ```bash
   docker-compose restart qdrant
   ```

4. Check collection exists:
   ```bash
   curl http://localhost:6333/collections/website_docs
   ```

### Scraper Finds No Pages

**Issue**: Crawler only finds 1 page or stops early

**Causes & Solutions**:
1. **Robots.txt blocking**: Check `https://yoursite.com/robots.txt`
   - Solution: Set `ROBOTSTXT_OBEY = False` in `scraper.py` (use responsibly!)

2. **JavaScript-heavy site**: Scrapy doesn't render JS
   - Solution: Use Scrapy-Splash or Selenium for JS-rendered sites

3. **Different domain**: Links point to external sites
   - Solution: Check `allowed_domains` in spider output

4. **Links not in `<a>` tags**: Some sites use JS navigation
   - Solution: Manually provide URL list or use browser automation

### Memory Issues

**Issue**: System runs out of memory during indexing

**Solutions**:
1. Reduce chunk size:
   ```env
   CHUNK_SIZE=500
   ```

2. Use smaller embedding model:
   ```env
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   ```

3. Index in batches:
   ```python
   # Scrape first, then index later
   from indexer.run_indexing import index_from_json
   index_from_json("scraped_data")
   ```

4. Close other applications
5. Increase Docker memory limit in Docker Desktop settings

### Scrapy ImportError

**Error**: `ModuleNotFoundError: No module named 'scrapy'`

**Solution**:
Dependencies are still installing. Wait for `uv sync` to complete, then try again.

## Performance Optimization

### GPU Acceleration

If you have an NVIDIA GPU:

1. Install NVIDIA Docker runtime
2. Modify `docker-compose.yml`:
   ```yaml
   ollama:
     image: ollama/ollama:latest
     deploy:
       resources:
         reservations:
           devices:
             - driver: nvidia
               count: 1
               capabilities: [gpu]
   ```

3. Restart Ollama:
   ```bash
   docker-compose up -d ollama
   ```

### Embedding Performance

- **CPU**: ~100-500 docs/sec (depends on CPU)
- **GPU**: ~1000-5000 docs/sec with CUDA
- **Batch size**: Increase to 64 for faster processing (more memory)

### Query Performance

- **Qdrant search**: <10ms for most collections
- **LLM generation**: 1-5 seconds depending on model and length
- **Embedding**: <50ms per query

### Scaling for Large Sites

For websites with 10,000+ pages:

1. **Use disk-based Qdrant** (default is in-memory)
2. **Implement incremental indexing** (only index new/changed pages)
3. **Consider distributed crawling** with Scrapy Cloud or Scrapyd
4. **Use compression** for JSON storage
5. **Implement caching** for frequently accessed documents

## Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-cov

# Run tests
pytest

# With coverage
pytest --cov=. --cov-report=html
```

### Code Quality

```bash
# Format code
black .

# Lint
flake8 .
pylint app/ indexer/ config/

# Type checking
mypy .
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes
3. Test locally
4. Submit pull request

## Deployment

### Docker Production Deployment

Build custom image with your code:

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app/streamlit_app.py"]
```

### Environment Variables for Production

```env
# Use production URLs
QDRANT_HOST=qdrant.yourdomain.com
OLLAMA_HOST=https://ollama.yourdomain.com

# Security
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## License

MIT License - feel free to use for personal or commercial projects.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

See `CONTRIBUTING.md` for detailed guidelines (if available).

## Acknowledgments

- [Scrapy](https://scrapy.org/) - Fast, powerful web scraping framework
- [LangChain](https://www.langchain.com/) - Document processing and RAG orchestration
- [sentence-transformers](https://www.sbert.net/) - State-of-the-art embeddings
- [Qdrant](https://qdrant.tech/) - High-performance vector search
- [Ollama](https://ollama.ai/) - Easy local LLM deployment
- [Streamlit](https://streamlit.io/) - Fast web app framework

## Support

- **Documentation**: See `SCRAPER_USAGE.md` for detailed scraper guide
- **Issues**: Report bugs or request features on GitHub Issues
- **Discussions**: Join community discussions on GitHub Discussions

## Roadmap

Planned features:

- [ ] Multi-language support for embeddings
- [ ] PDF and DOCX scraping support
- [ ] Advanced filters in UI (date range, source filtering)
- [ ] Export chat history
- [ ] API endpoint for programmatic access
- [ ] Incremental indexing (only update changed pages)
- [ ] Multi-site indexing with namespace support
- [ ] Image search using vision models

## Version History

- **v0.2.0** (Current) - Scrapy integration, automatic crawling, image extraction
- **v0.1.0** - Initial release with BeautifulSoup scraper

---

Built with ❤️ for the open-source community
