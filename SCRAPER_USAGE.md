# Scrapy Blog Scraper Usage Guide

This guide explains how to use the new Scrapy-based web scraper to crawl and extract content from blog websites for RAG indexing.

## Features

- **Automatic Crawling**: Automatically discovers and crawls all pages within the same domain
- **Content Extraction**: Extracts page titles, text content, and images with metadata
- **Image Download**: Downloads images with metadata (alt text, captions, titles)
- **JSON Export**: Saves each page as a separate JSON file for easy inspection
- **RAG Integration**: Tracks URLs so search results can link back to source pages
- **Configurable**: Adjustable crawl delay, output directories, and more

## Configuration

### Environment Variables

Add these to your `.env` file (see `.env.example`):

```bash
# Required: The blog/website URL to scrape
TARGET_WEBSITE_URL=https://example.com/blog

# Optional: Scraping settings (with defaults shown)
SCRAPER_DELAY=1.0                    # Delay between requests in seconds
SCRAPED_DATA_DIR=scraped_data        # Directory for JSON output
SCRAPED_IMAGES_DIR=scraped_images    # Directory for downloaded images
```

## Usage Methods

### Method 1: Interactive CLI (Recommended for first-time use)

```bash
python -m indexer.run_indexing
```

This will:
1. Display current configuration
2. Prompt for the target website URL (if not in .env)
3. Ask for confirmation before starting
4. Crawl the website and extract all content
5. Index the content into Qdrant for RAG queries

### Method 2: Programmatic Usage

```python
from indexer.run_indexing import index_from_url

# Scrape and index a website
index_from_url("https://example.com/blog")
```

### Method 3: Index from Existing JSON Files

If you've already scraped a website and just want to re-index:

```python
from indexer.run_indexing import index_from_json

# Re-index from previously scraped JSON files
index_from_json("scraped_data")
```

## Output Structure

### JSON Files

Each scraped page is saved as a separate JSON file in `scraped_data/` with this structure:

```json
{
  "url": "https://example.com/blog/post-title",
  "title": "Post Title",
  "text": "Full extracted text content...",
  "images": [
    {
      "url": "https://example.com/images/photo.jpg",
      "local_path": "full/abc123.jpg",
      "alt": "Image alt text",
      "title": "Image title attribute",
      "caption": "Figure caption if available"
    }
  ],
  "scraped_at": "2025-11-24T12:34:56.789"
}
```

### Downloaded Images

Images are downloaded to `scraped_images/full/` with hashed filenames to prevent conflicts.

## How the Scraper Works

1. **Start URL**: Begins at the configured `TARGET_WEBSITE_URL`
2. **Link Discovery**: Finds all links on each page
3. **Domain Filtering**: Only follows links within the same domain
4. **Content Extraction**:
   - Prioritizes `<main>`, `<article>` tags for cleaner content
   - Removes navigation, footer, header, scripts, and styles
   - Extracts image metadata (alt text, titles, captions)
5. **Respects robots.txt**: Obeys crawl rules automatically
6. **Polite Crawling**: Adds configurable delay between requests

## Customization

### Adjusting Crawl Behavior

Edit `indexer/scraper.py` to customize:

- **Link patterns**: Modify `LinkExtractor` in `BlogSpider.rules` to include/exclude URL patterns
- **Content extraction**: Adjust CSS selectors in `parse_page()` method
- **Rate limiting**: Change `DOWNLOAD_DELAY` in Scrapy settings

### Example: Exclude Certain URL Patterns

```python
rules = (
    Rule(
        LinkExtractor(
            allow=(),
            deny=(r'/tag/', r'/category/'),  # Exclude tag/category pages
        ),
        callback='parse_page',
        follow=True
    ),
)
```

## Troubleshooting

### No pages found
- Check that `TARGET_WEBSITE_URL` is accessible
- Verify the website doesn't block automated requests
- Review `robots.txt` for crawl restrictions

### Images not downloading
- Ensure sufficient disk space
- Check image URLs are accessible
- Verify `scraped_images/` directory has write permissions

### Memory issues with large sites
- Increase crawl delay to reduce load
- Consider crawling in smaller batches
- Add URL patterns to exclude non-essential pages

## Data Structure for RAG

The JSON format is designed for RAG indexing:

- **`url`**: Used to link search results back to source
- **`title`**: Provides context for search results
- **`text`**: Main content indexed for similarity search
- **`images`**: Image metadata can be indexed for visual content search

When you query the RAG system, it will return the source URL with each result, allowing users to navigate to the original page.

## Next Steps

After scraping:

1. Start Qdrant: `docker-compose up -d qdrant`
2. Start Ollama: `docker-compose up -d ollama`
3. Run the indexing: `python -m indexer.run_indexing`
4. Launch the UI: `streamlit run app/streamlit_app.py`

Happy scraping!
