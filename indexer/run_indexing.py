"""
Main indexing script to scrape, process, and index website content.
This is a one-time script that should be run before starting the RAG application.
"""
import sys
from typing import List

from indexer.scraper import ScrapyBlogScraper
from indexer.processor import DocumentProcessor
from config.settings import settings


def main():
    """Main indexing pipeline using Scrapy for automatic crawling."""
    print("=" * 60)
    print("RAG System - Website Indexing Pipeline (Scrapy)")
    print("=" * 60)

    # Display current settings
    settings.display_settings()

    # Get website URL from settings or user
    base_url = settings.TARGET_WEBSITE_URL

    if not base_url:
        print("\nNo TARGET_WEBSITE_URL set in environment.")
        base_url = input("Enter the website URL to scrape (e.g., https://example.com): ").strip()

    if not base_url:
        print("Error: Website URL is required!")
        sys.exit(1)

    print(f"\nTarget Website: {base_url}")
    print(f"Scraped Data Directory: {settings.SCRAPED_DATA_DIR}")
    print(f"Scraped Images Directory: {settings.SCRAPED_IMAGES_DIR}")
    print(f"Crawler Delay: {settings.SCRAPER_DELAY}s between requests")

    # Confirm before proceeding
    print("\nThe crawler will automatically discover and scrape all pages")
    print("on the target website (same domain only).")
    confirm = input("\nProceed with crawling and indexing? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Indexing cancelled.")
        sys.exit(0)

    # Step 1: Scrape website using Scrapy
    print("\n" + "=" * 60)
    print("STEP 1: Crawling and Scraping Website")
    print("=" * 60)

    scraper = ScrapyBlogScraper(
        base_url=base_url,
        output_dir=settings.SCRAPED_DATA_DIR,
        images_dir=settings.SCRAPED_IMAGES_DIR,
        delay=settings.SCRAPER_DELAY
    )

    try:
        documents = scraper.scrape()
    except Exception as e:
        print(f"\nError during scraping: {e}")
        sys.exit(1)

    if not documents:
        print("\nError: No documents were successfully scraped!")
        sys.exit(1)

    # Step 2: Process and index documents
    print("\n" + "=" * 60)
    print("STEP 2: Processing and Indexing Documents")
    print("=" * 60)

    processor = DocumentProcessor()
    processor.process_and_index(documents)

    print("\n" + "=" * 60)
    print("✓ Indexing Pipeline Complete!")
    print("=" * 60)
    print(f"\nScraped {len(documents)} pages successfully.")
    print(f"Data saved to: {settings.SCRAPED_DATA_DIR}")
    print(f"Images saved to: {settings.SCRAPED_IMAGES_DIR}")
    print("\nYou can now run the RAG application using:")
    print("  streamlit run app/streamlit_app.py")
    print("=" * 60)


def index_from_url(base_url: str):
    """
    Programmatic method to index a website by automatically crawling it.

    Args:
        base_url: The base URL of the website to crawl

    Example:
        >>> from indexer.run_indexing import index_from_url
        >>> index_from_url("https://docs.example.com")
    """
    print("=" * 60)
    print("RAG System - Programmatic Indexing")
    print("=" * 60)

    settings.display_settings()

    print(f"\nCrawling: {base_url}")

    # Scrape using Scrapy
    scraper = ScrapyBlogScraper(
        base_url=base_url,
        output_dir=settings.SCRAPED_DATA_DIR,
        images_dir=settings.SCRAPED_IMAGES_DIR,
        delay=settings.SCRAPER_DELAY
    )
    documents = scraper.scrape()

    if not documents:
        print("Error: No documents scraped!")
        return

    # Process and index
    processor = DocumentProcessor()
    processor.process_and_index(documents)

    print(f"\n✓ Indexing complete! Indexed {len(documents)} pages.")


def index_from_json(json_dir: str = None):
    """
    Index documents from already scraped JSON files.
    Useful if you've already run the scraper and just want to re-index.

    Args:
        json_dir: Directory containing scraped JSON files
                  (defaults to settings.SCRAPED_DATA_DIR)

    Example:
        >>> from indexer.run_indexing import index_from_json
        >>> index_from_json("scraped_data")
    """
    import json
    from pathlib import Path

    json_dir = json_dir or settings.SCRAPED_DATA_DIR
    json_path = Path(json_dir)

    print("=" * 60)
    print("RAG System - Indexing from JSON Files")
    print("=" * 60)
    print(f"Loading from: {json_path}")

    if not json_path.exists():
        print(f"Error: Directory {json_path} does not exist!")
        return

    # Load all JSON files
    documents = []
    for json_file in sorted(json_path.glob('*.json')):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            documents.append(data)
            print(f"Loaded: {data.get('title', 'Untitled')[:50]}...")

    if not documents:
        print(f"Error: No JSON files found in {json_path}!")
        return

    print(f"\nFound {len(documents)} documents.")

    # Process and index
    processor = DocumentProcessor()
    processor.process_and_index(documents)

    print(f"\n✓ Indexing complete! Indexed {len(documents)} pages.")


if __name__ == "__main__":
    main()
