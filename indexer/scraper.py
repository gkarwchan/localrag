"""
Web scraper module for extracting content from website pages.
Uses requests and BeautifulSoup for static HTML parsing.
"""
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from urllib.parse import urljoin, urlparse
import time


class WebScraper:
    """Scraper for extracting text content from web pages."""

    def __init__(self, base_url: str, delay: float = 1.0):
        """
        Initialize the web scraper.

        Args:
            base_url: The base URL of the website
            delay: Delay between requests in seconds (be respectful!)
        """
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def scrape_page(self, url: str) -> Dict[str, str]:
        """
        Scrape a single page and extract its text content.

        Args:
            url: The URL to scrape

        Returns:
            Dictionary with 'url', 'title', and 'content'
        """
        try:
            # Add delay to be respectful
            time.sleep(self.delay)

            # Fetch the page
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.content, 'lxml')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Get title
            title = soup.title.string if soup.title else url

            # Extract text content
            # Focus on main content areas
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)

            # Clean up whitespace
            text = ' '.join(text.split())

            return {
                'url': url,
                'title': title.strip() if title else '',
                'content': text
            }

        except requests.RequestException as e:
            print(f"Error scraping {url}: {e}")
            return {
                'url': url,
                'title': '',
                'content': ''
            }

    def scrape_pages(self, urls: List[str]) -> List[Dict[str, str]]:
        """
        Scrape multiple pages.

        Args:
            urls: List of URLs to scrape

        Returns:
            List of dictionaries with scraped content
        """
        print(f"Starting to scrape {len(urls)} pages...")
        documents = []

        for i, url in enumerate(urls, 1):
            print(f"Scraping page {i}/{len(urls)}: {url}")
            doc = self.scrape_page(url)

            if doc['content']:
                documents.append(doc)
                print(f"  ✓ Successfully scraped: {doc['title'][:50]}...")
            else:
                print(f"  ✗ Failed to scrape content")

        print(f"\nCompleted! Successfully scraped {len(documents)}/{len(urls)} pages.")
        return documents

    def normalize_url(self, url: str) -> str:
        """Normalize a URL relative to base_url if needed."""
        return urljoin(self.base_url, url)
