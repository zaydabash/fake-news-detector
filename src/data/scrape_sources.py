"""Multi-niche data scraping pipeline."""

import sys
import time
import yaml
import requests
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse
import re

import pandas as pd
from bs4 import BeautifulSoup
from newspaper import Article
from tqdm import tqdm
import argparse

from src.utils.logging import get_logger

logger = get_logger(__name__)


class WebScraper:
    """Web scraper for collecting text data from various sources."""
    
    def __init__(self, rate_limit: float = 1.0, timeout: int = 10, workers: int = 1):
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.workers = workers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape a single URL and extract text content."""
        try:
            time.sleep(self.rate_limit)  # Rate limiting
            
            # Try using newspaper3k first
            article = Article(url)
            article.download()
            article.parse()
            
            if article.text and len(article.text.strip()) > 50:
                return {
                    'url': url,
                    'title': article.title or '',
                    'text': article.text.strip(),
                    'source': urlparse(url).netloc,
                    'success': True
                }
            
            # Fallback to BeautifulSoup
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            if len(text) > 50:
                return {
                    'url': url,
                    'title': soup.title.string if soup.title else '',
                    'text': text,
                    'source': urlparse(url).netloc,
                    'success': True
                }
            
            return {'url': url, 'success': False, 'error': 'No content found'}
            
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            return {'url': url, 'success': False, 'error': str(e)}
    
    def find_article_urls(self, domain: str, max_urls: int = 100) -> List[str]:
        """Find article URLs from a domain's main page."""
        try:
            response = self.session.get(f"https://{domain}", timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            urls = set()
            
            # Find links that might be articles
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(f"https://{domain}", href)
                
                # Filter for potential article URLs
                if self._is_article_url(full_url, domain):
                    urls.add(full_url)
                    if len(urls) >= max_urls:
                        break
            
            return list(urls)[:max_urls]
            
        except Exception as e:
            logger.warning(f"Failed to find URLs from {domain}: {e}")
            return []
    
    def _is_article_url(self, url: str, domain: str) -> bool:
        """Check if URL looks like an article."""
        parsed = urlparse(url)
        
        # Must be from the same domain
        if parsed.netloc != domain and not parsed.netloc.endswith(f".{domain}"):
            return False
        
        # Skip certain file types and paths
        skip_patterns = [
            r'\.(pdf|jpg|jpeg|png|gif|css|js|xml|rss)$',
            r'/tag/',
            r'/category/',
            r'/author/',
            r'/search',
            r'/login',
            r'/register',
            r'/contact',
            r'/about',
            r'/privacy',
            r'/terms'
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False
        
        return True


def load_niche_config(niche: str) -> Dict[str, Any]:
    """Load configuration for a specific niche."""
    config_path = Path(__file__).parent.parent.parent / "configs" / f"{niche}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def scrape_niche_data(niche: str, config: Dict[str, Any]) -> pd.DataFrame:
    """Scrape data for a specific niche."""
    logger.info(f"Scraping data for niche: {niche}")
    
    # Extract configuration parameters
    rate_limit = 60.0 / config.get('rate_limit_per_domain_per_min', 2)
    timeout = config.get('timeout_seconds', 10)
    workers = config.get('workers', 1)
    max_per_source = config.get('max_per_source', 200)
    
    scraper = WebScraper(rate_limit=rate_limit, timeout=timeout, workers=workers)
    
    all_data = []
    
    # Scrape positive sources
    logger.info(f"Scraping positive sources: {config['positive_domains']}")
    for domain in config['positive_domains']:
        logger.info(f"Processing domain: {domain}")
        
        urls = scraper.find_article_urls(domain, max_per_source)
        logger.info(f"Found {len(urls)} URLs from {domain}")
        
        for url in tqdm(urls, desc=f"Scraping {domain}"):
            result = scraper.scrape_url(url)
            if result['success']:
                all_data.append({
                    'url': result['url'],
                    'title': result['title'],
                    'text': result['text'],
                    'source': result['source'],
                    'domain_type': 'positive',
                    'niche': niche
                })
    
    # Scrape negative sources
    logger.info(f"Scraping negative sources: {config['negative_domains']}")
    for domain in config['negative_domains']:
        logger.info(f"Processing domain: {domain}")
        
        urls = scraper.find_article_urls(domain, max_per_source)
        logger.info(f"Found {len(urls)} URLs from {domain}")
        
        for url in tqdm(urls, desc=f"Scraping {domain}"):
            result = scraper.scrape_url(url)
            if result['success']:
                all_data.append({
                    'url': result['url'],
                    'title': result['title'],
                    'text': result['text'],
                    'source': result['source'],
                    'domain_type': 'negative',
                    'niche': niche
                })
    
    # Filter by text length
    min_length = config.get('min_text_length', 50)
    max_length = config.get('max_text_length', 2000)
    
    filtered_data = []
    for item in all_data:
        text_length = len(item['text'])
        if min_length <= text_length <= max_length:
            filtered_data.append(item)
    
    logger.info(f"Scraped {len(filtered_data)} articles (filtered from {len(all_data)})")
    
    return pd.DataFrame(filtered_data)


def main():
    """Main function for scraping niche data."""
    parser = argparse.ArgumentParser(description="Scrape data for a specific niche")
    parser.add_argument("--niche", required=True, help="Niche to scrape data for")
    
    args = parser.parse_args()
    
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "data" / args.niche / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load config
        config = load_niche_config(args.niche)
        
        # Scrape data
        df = scrape_niche_data(args.niche, config)
        
        if df.empty:
            logger.error(f"No data scraped for niche: {args.niche}")
            sys.exit(1)
        
        # Save raw data
        output_file = output_dir / "scraped_data.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"Saved {len(df)} articles to {output_file}")
        
        # Print summary
        logger.info(f"\nScraping Summary for {args.niche}:")
        logger.info(f"Total articles: {len(df)}")
        logger.info(f"Positive sources: {len(df[df['domain_type'] == 'positive'])}")
        logger.info(f"Negative sources: {len(df[df['domain_type'] == 'negative'])}")
        
        # Show sample domains
        logger.info(f"Sample domains:")
        for domain_type in ['positive', 'negative']:
            domains = df[df['domain_type'] == domain_type]['source'].unique()[:5]
            logger.info(f"  {domain_type}: {', '.join(domains)}")
        
    except Exception as e:
        logger.error(f"Error scraping data for {args.niche}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
