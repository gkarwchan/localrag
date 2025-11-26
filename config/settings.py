"""
Centralized configuration management for the RAG system.
Loads settings from environment variables with sensible defaults.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # Qdrant Configuration
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "website_docs")

    # Ollama Configuration
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "llama3.2")

    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Chunking Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    # RAG Configuration
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))

    # Web Scraping Configuration
    TARGET_WEBSITE_URL = os.getenv("TARGET_WEBSITE_URL", "")
    SCRAPER_DELAY = float(os.getenv("SCRAPER_DELAY", "1.0"))
    SCRAPER_USER_AGENT = os.getenv(
        "SCRAPER_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
    SCRAPED_DATA_DIR = os.getenv("SCRAPED_DATA_DIR", "scraped_data")
    SCRAPED_IMAGES_DIR = os.getenv("SCRAPED_IMAGES_DIR", "scraped_images")

    @classmethod
    def get_qdrant_url(cls):
        """Get the full Qdrant URL."""
        return f"http://{cls.QDRANT_HOST}:{cls.QDRANT_PORT}"

    @classmethod
    def display_settings(cls):
        """Display current settings (useful for debugging)."""
        print("=" * 50)
        print("Current Configuration:")
        print("=" * 50)
        print(f"Qdrant URL: {cls.get_qdrant_url()}")
        print(f"Qdrant Collection: {cls.QDRANT_COLLECTION_NAME}")
        print(f"Ollama Host: {cls.OLLAMA_HOST}")
        print(f"Default LLM Model: {cls.DEFAULT_LLM_MODEL}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"Chunk Size: {cls.CHUNK_SIZE}")
        print(f"Chunk Overlap: {cls.CHUNK_OVERLAP}")
        print(f"Top K Results: {cls.TOP_K_RESULTS}")
        print(f"\nScraping Configuration:")
        print(f"Target Website: {cls.TARGET_WEBSITE_URL or '(not set)'}")
        print(f"Scraper Delay: {cls.SCRAPER_DELAY}s")
        print(f"Scraped Data Dir: {cls.SCRAPED_DATA_DIR}")
        print(f"Scraped Images Dir: {cls.SCRAPED_IMAGES_DIR}")
        print("=" * 50)


# Create a singleton instance
settings = Settings()
