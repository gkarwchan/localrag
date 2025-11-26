"""
Scrapy-based web scraper for extracting content from blog websites.
Includes automatic crawling, image downloading, and JSON export.
"""
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy.pipelines.images import ImagesPipeline
from scrapy import Request
from urllib.parse import urlparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


class BlogPageItem(scrapy.Item):
    """Item definition for scraped blog pages."""
    url = scrapy.Field()
    title = scrapy.Field()
    text = scrapy.Field()
    images = scrapy.Field()
    scraped_at = scrapy.Field()


class BlogImagesPipeline(ImagesPipeline):
    """Custom pipeline for downloading images with metadata."""

    def get_media_requests(self, item, info):
        """Download images found in the page."""
        for image_data in item.get('images', []):
            yield Request(
                image_data['url'],
                meta={'image_data': image_data}
            )

    def item_completed(self, results, item, info):
        """Update item with downloaded image paths."""
        image_paths = []
        for ok, result in results:
            if ok:
                image_data = result['request'].meta['image_data']
                image_data['local_path'] = result['path']
                image_paths.append(image_data)

        item['images'] = image_paths
        return item


class JsonExportPipeline:
    """Pipeline to save each scraped page as a separate JSON file."""

    def open_spider(self, spider):
        """Create output directory when spider starts."""
        self.output_dir = Path(spider.settings.get('JSON_OUTPUT_DIR', 'scraped_data'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Clear existing JSON files
        for json_file in self.output_dir.glob('*.json'):
            json_file.unlink()

    def process_item(self, item, spider):
        """Save item as JSON file."""
        # Create a safe filename from the URL
        url_path = urlparse(item['url']).path
        filename = url_path.strip('/').replace('/', '_') or 'index'
        if not filename.endswith('.json'):
            filename = f"{filename}.json"

        filepath = self.output_dir / filename

        # Save as JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dict(item), f, indent=2, ensure_ascii=False)

        spider.logger.info(f"Saved page data to {filepath}")
        return item


class BlogSpider(CrawlSpider):
    """Spider for crawling blog websites and extracting content."""

    name = 'blog_spider'

    # These will be set when initializing the spider
    start_urls = []
    allowed_domains = []

    # Define rules for following links
    rules = (
        Rule(
            LinkExtractor(
                allow=(),  # Follow all links
                deny=(),   # Can add patterns to exclude
                tags=('a',),
                attrs=('href',),
            ),
            callback='parse_page',
            follow=True
        ),
    )

    def __init__(self, start_url=None, *args, **kwargs):
        """Initialize spider with the start URL."""
        super(BlogSpider, self).__init__(*args, **kwargs)

        if start_url:
            self.start_urls = [start_url]
            # Extract domain for allowed_domains
            parsed = urlparse(start_url)
            self.allowed_domains = [parsed.netloc]

    def parse_page(self, response):
        """Parse a blog page and extract content."""
        self.logger.info(f"Scraping: {response.url}")

        # Extract title
        title = response.css('title::text').get()
        if not title:
            title = response.css('h1::text').get() or response.url

        # Extract main text content
        # Try to find main content areas first
        main_content = (
            response.css('main ::text').getall() or
            response.css('article ::text').getall() or
            response.css('body ::text').getall()
        )

        # Clean and join text
        text = ' '.join([t.strip() for t in main_content if t.strip()])
        text = ' '.join(text.split())  # Normalize whitespace

        # Extract images with metadata
        images = []
        for img in response.css('img'):
            img_url = img.css('::attr(src)').get()
            if img_url:
                # Convert relative URLs to absolute
                img_url = response.urljoin(img_url)

                images.append({
                    'url': img_url,
                    'alt': img.css('::attr(alt)').get() or '',
                    'title': img.css('::attr(title)').get() or '',
                    'caption': self._extract_caption(img),
                    'local_path': None  # Will be filled by ImagesPipeline
                })

        # Create item
        item = BlogPageItem()
        item['url'] = response.url
        item['title'] = title.strip() if title else ''
        item['text'] = text
        item['images'] = images
        item['scraped_at'] = datetime.now().isoformat()

        yield item

    def _extract_caption(self, img_selector):
        """Try to extract image caption from common patterns."""
        # Look for figcaption if image is in a figure
        figure = img_selector.xpath('./ancestor::figure')
        if figure:
            caption = figure.css('figcaption ::text').get()
            if caption:
                return caption.strip()

        # Look for caption in image's alt or title
        return ''


class ScrapyBlogScraper:
    """Main scraper class that configures and runs Scrapy."""

    def __init__(self, base_url: str, output_dir: str = 'scraped_data',
                 images_dir: str = 'scraped_images', delay: float = 1.0):
        """
        Initialize the Scrapy-based scraper.

        Args:
            base_url: The base URL of the blog to scrape
            output_dir: Directory to save JSON files
            images_dir: Directory to save downloaded images
            delay: Delay between requests in seconds
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.images_dir = images_dir
        self.delay = delay

    def scrape(self) -> List[Dict[str, Any]]:
        """
        Run the scraper and return all scraped pages.

        Returns:
            List of dictionaries with scraped content
        """
        # Configure Scrapy settings
        settings = {
            'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'ROBOTSTXT_OBEY': True,
            'DOWNLOAD_DELAY': self.delay,
            'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
            'COOKIES_ENABLED': False,

            # Image pipeline settings
            'IMAGES_STORE': self.images_dir,
            'IMAGES_EXPIRES': 90,  # Days

            # JSON output directory
            'JSON_OUTPUT_DIR': self.output_dir,

            # Enable pipelines
            'ITEM_PIPELINES': {
                'indexer.scraper.BlogImagesPipeline': 1,
                'indexer.scraper.JsonExportPipeline': 300,
            },

            # Logging
            'LOG_LEVEL': 'INFO',
        }

        # Create crawler process
        process = CrawlerProcess(settings=settings)

        # Add spider to process
        process.crawl(BlogSpider, start_url=self.base_url)

        # Run crawler (blocks until complete)
        print(f"Starting to crawl: {self.base_url}")
        print(f"Output directory: {self.output_dir}")
        print(f"Images directory: {self.images_dir}")
        process.start()

        # After crawling, read all JSON files
        return self._load_scraped_data()

    def _load_scraped_data(self) -> List[Dict[str, Any]]:
        """Load all scraped data from JSON files."""
        output_path = Path(self.output_dir)
        documents = []

        if output_path.exists():
            for json_file in sorted(output_path.glob('*.json')):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    documents.append(data)
                    print(f"Loaded: {data.get('title', 'Untitled')[:50]}...")

        print(f"\nCompleted! Scraped {len(documents)} pages.")
        return documents
