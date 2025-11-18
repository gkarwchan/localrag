# Local RAG System

A Retrieval-Augmented Generation (RAG) system for querying website documentation using local LLMs and vector databases.

## Features

- **Local LLM Inference**: Uses Ollama with Llama 3.2 or Mistral models
- **Local Embeddings**: sentence-transformers for fast, local embedding generation
- **Vector Database**: Qdrant for efficient semantic search
- **Web Scraping**: Extract content from static HTML websites
- **Interactive UI**: Streamlit-based chat interface
- **Source Citations**: Transparent answers with source attribution
- **Follow-up Questions**: AI-generated contextual follow-up suggestions
- **Docker Support**: Containerized Qdrant and Ollama services

## Tech Stack

- **Python 3.8+**
- **LangChain**: Document processing and text splitting
- **sentence-transformers**: Local embedding generation
- **Qdrant**: Vector database
- **Ollama**: Local LLM inference
- **Streamlit**: Web UI
- **BeautifulSoup4**: Web scraping

## Project Structure

```
localrag/
├── docker-compose.yml          # Qdrant + Ollama services
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
├── README.md                  # This file
├── config/
│   └── settings.py           # Centralized configuration
├── indexer/
│   ├── scraper.py           # Web scraping
│   ├── processor.py         # Chunking & vectorization
│   └── run_indexing.py      # Main indexing script
└── app/
    ├── rag_engine.py        # RAG query logic
    └── streamlit_app.py     # Streamlit UI
```

## Installation

### 1. Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose
- Git

### 2. Clone or Setup Project

```bash
cd localrag
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` if you want to customize settings (optional).

### 5. Start Docker Services

```bash
docker-compose up -d
```

This starts:
- **Qdrant** on port 6333
- **Ollama** on port 11434

Verify services are running:
```bash
docker-compose ps
```

### 6. Pull Ollama Models

Pull the LLM models you want to use:

```bash
# Pull Llama 3.2
docker exec ollama ollama pull llama3.2

# Pull Mistral (optional)
docker exec ollama ollama pull mistral
```

## Usage

### Step 1: Index Website Content (One-Time)

Run the indexing script to scrape, chunk, and vectorize your website content:

```bash
python indexer/run_indexing.py
```

You'll be prompted to enter:
1. **Base website URL** (e.g., `https://docs.example.com`)
2. **List of page URLs** to index (one per line, press Enter twice when done)

Example:
```
Enter the base website URL: https://python.langchain.com
Enter the list of page URLs to index (one per line):
https://python.langchain.com/docs/introduction
https://python.langchain.com/docs/use_cases/question_answering
https://python.langchain.com/docs/modules/data_connection/vectorstores

[Press Enter twice to finish]
```

The script will:
1. Scrape the specified pages
2. Split content into chunks
3. Generate embeddings using sentence-transformers
4. Store in Qdrant vector database

### Step 2: Launch the RAG Application

Start the Streamlit web app:

```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Step 3: Ask Questions

1. Select your preferred LLM model from the sidebar (Llama 3.2 or Mistral)
2. Type your question in the chat input
3. View the AI-generated answer with source citations
4. Click on follow-up questions for deeper exploration

## Advanced Usage

### Programmatic Indexing

You can also index documents programmatically:

```python
from indexer.run_indexing import index_from_list

base_url = "https://docs.example.com"
urls = [
    "https://docs.example.com/guide/intro",
    "https://docs.example.com/guide/setup",
    "https://docs.example.com/api/reference"
]

index_from_list(base_url, urls)
```

### Custom Configuration

Edit `.env` to customize:

```env
# Embedding model (options: all-MiniLM-L6-v2, all-mpnet-base-v2)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Chunking parameters
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# RAG parameters
TOP_K_RESULTS=5
TEMPERATURE=0.7
MAX_TOKENS=512
```

### Using Different Embedding Models

Available sentence-transformers models:
- `all-MiniLM-L6-v2` (default) - Fast, 384 dimensions
- `all-mpnet-base-v2` - Higher quality, 768 dimensions
- `multi-qa-MiniLM-L6-cos-v1` - Optimized for Q&A

Change in `.env`:
```env
EMBEDDING_MODEL=all-mpnet-base-v2
```

**Note**: After changing the embedding model, you must re-run the indexing process.

## Troubleshooting

### Ollama Connection Error

If you see "Error generating answer: connection refused":

1. Check if Ollama container is running:
   ```bash
   docker-compose ps
   ```

2. Pull the model if not already available:
   ```bash
   docker exec ollama ollama pull llama3.2
   ```

3. Test Ollama directly:
   ```bash
   curl http://localhost:11434/api/tags
   ```

### Qdrant Connection Error

1. Check if Qdrant container is running:
   ```bash
   docker-compose ps
   ```

2. Access Qdrant UI at http://localhost:6333/dashboard

3. Restart services:
   ```bash
   docker-compose restart
   ```

### No Documents Found

If the indexing script doesn't find content:

1. Check if the URLs are accessible
2. Verify the website doesn't require JavaScript rendering
3. Check scraper logs for errors

### Memory Issues

If you encounter memory issues with large datasets:

1. Reduce `CHUNK_SIZE` in `.env`
2. Index fewer pages at a time
3. Use a smaller embedding model (e.g., `all-MiniLM-L6-v2`)

## Architecture

### Indexing Pipeline

```
Website URLs
    ↓
Web Scraper (BeautifulSoup)
    ↓
Text Chunking (LangChain)
    ↓
Embedding Generation (sentence-transformers)
    ↓
Vector Storage (Qdrant)
```

### Query Pipeline

```
User Question
    ↓
Query Embedding (sentence-transformers)
    ↓
Semantic Search (Qdrant)
    ↓
Context Retrieval
    ↓
LLM Answer Generation (Ollama)
    ↓
Follow-up Question Generation
```

## Performance Tips

1. **Use GPU**: If available, Ollama will automatically use GPU acceleration
2. **Batch Processing**: The indexer processes embeddings in batches of 32
3. **Caching**: sentence-transformers caches models locally after first download
4. **Model Selection**: Llama 3.2 is faster but Mistral may give better answers

## License

MIT License - feel free to use for personal or commercial projects.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- [LangChain](https://www.langchain.com/) for document processing
- [sentence-transformers](https://www.sbert.net/) for embeddings
- [Qdrant](https://qdrant.tech/) for vector search
- [Ollama](https://ollama.ai/) for local LLM inference
- [Streamlit](https://streamlit.io/) for the UI framework
