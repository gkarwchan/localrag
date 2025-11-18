"""
Main indexing script to scrape, process, and index website content.
This is a one-time script that should be run before starting the RAG application.
"""
import sys
from typing import List

from indexer.scraper import WebScraper
from indexer.processor import DocumentProcessor
from config.settings import settings


def main():
    """Main indexing pipeline."""
    print("=" * 60)
    print("RAG System - Website Indexing Pipeline")
    print("=" * 60)

    # Display current settings
    settings.display_settings()

    # Get website URL and pages from user
    print("\nPlease provide the website details:")
    base_url = input("Enter the base website URL (e.g., https://example.com): ").strip()

    if not base_url:
        print("Error: Base URL is required!")
        sys.exit(1)

    print("\nEnter the list of page URLs to index (one per line).")
    print("Press Enter twice when done:")

    urls = []
    while True:
        url = input().strip()
        if not url:
            break
        # Normalize URL if relative
        if url.startswith('/'):
            url = base_url.rstrip('/') + url
        elif not url.startswith('http'):
            url = base_url.rstrip('/') + '/' + url
        urls.append(url)

    if not urls:
        print("Error: At least one URL is required!")
        sys.exit(1)

    print(f"\n{len(urls)} URLs to index:")
    for i, url in enumerate(urls, 1):
        print(f"  {i}. {url}")

    # Confirm before proceeding
    confirm = input("\nProceed with indexing? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Indexing cancelled.")
        sys.exit(0)

    # Step 1: Scrape websites
    print("\n" + "=" * 60)
    print("STEP 1: Scraping Websites")
    print("=" * 60)

    scraper = WebScraper(base_url=base_url, delay=1.0)
    documents = scraper.scrape_pages(urls)

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
    print("\nYou can now run the RAG application using:")
    print("  streamlit run app/streamlit_app.py")
    print("=" * 60)


def index_from_list(base_url: str, urls: List[str]):
    """
    Alternative method to index from a Python list.
    Useful for programmatic indexing or testing.

    Args:
        base_url: The base URL of the website
        urls: List of URLs to index

    Example:
        >>> from indexer.run_indexing import index_from_list
        >>> base_url = "https://docs.example.com"
        >>> urls = [
        ...     "https://docs.example.com/guide/intro",
        ...     "https://docs.example.com/guide/setup"
        ... ]
        >>> index_from_list(base_url, urls)
    """
    print("=" * 60)
    print("RAG System - Programmatic Indexing")
    print("=" * 60)

    settings.display_settings()

    # Scrape
    scraper = WebScraper(base_url=base_url, delay=1.0)
    documents = scraper.scrape_pages(urls)

    # Process and index
    processor = DocumentProcessor()
    processor.process_and_index(documents)

    print("\n✓ Indexing complete!")


if __name__ == "__main__":
    main()
