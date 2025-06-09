"""
Web Scraping Service using Playwright

Provides asynchronous web scraping capabilities with Redis caching.
Uses Playwright for reliable JavaScript-enabled scraping.
"""

import asyncio
import json
import logging
import hashlib
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

import redis
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from app.core.config import settings


logger = logging.getLogger(__name__)


class ScraperService:
    """Async web scraper service with Redis caching."""
    
    def __init__(self, cache_ttl: int = 21600):  # 6 hours = 21600 seconds
        """
        Initialize scraper service.
        
        Args:
            cache_ttl: Cache time-to-live in seconds (default: 6 hours)
        """
        self.cache_ttl = cache_ttl
        self._redis_client: Optional[redis.Redis] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup()
        
    async def _initialize(self):
        """Initialize Redis client and Playwright browser."""
        try:
            # Initialize Redis client
            if settings.redis_url:
                self._redis_client = redis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # Test Redis connection
                await asyncio.get_event_loop().run_in_executor(
                    None, self._redis_client.ping
                )
                logger.info("Redis client initialized successfully")
            else:
                logger.warning("Redis URL not configured, caching disabled")
                
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            self._redis_client = None
            
        try:
            # Initialize Playwright browser
            self.playwright = await async_playwright().start()
            self._browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--disable-gpu'
                ]
            )
            
            # Create browser context with reasonable defaults
            self._context = await self._browser.new_context(
                viewport={'width': 1280, 'height': 720},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            
            logger.info("Playwright browser initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Playwright browser: {e}")
            raise
            
    async def _cleanup(self):
        """Clean up resources."""
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if hasattr(self, 'playwright'):
                await self.playwright.stop()
            if self._redis_client:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._redis_client.close
                )
            logger.info("Scraper service cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
    def _generate_cache_key(self, url: str, selectors: List[str]) -> str:
        """
        Generate a cache key for URL and selectors.
        
        Args:
            url: URL to scrape
            selectors: List of CSS selectors
            
        Returns:
            str: Cache key
        """
        # Normalize URL
        parsed_url = urlparse(url)
        normalized_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
        
        # Create key from URL and selectors
        key_data = {
            'url': normalized_url,
            'selectors': sorted(selectors)  # Sort for consistency
        }
        key_json = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_json.encode()).hexdigest()
        
        return f"scraper:v1:{key_hash}"
        
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, str]]:
        """
        Get cached scraping result.
        
        Args:
            cache_key: Redis cache key
            
        Returns:
            Optional[Dict[str, str]]: Cached result or None
        """
        if not self._redis_client:
            return None
            
        try:
            cached_data = await asyncio.get_event_loop().run_in_executor(
                None, self._redis_client.get, cache_key
            )
            if cached_data:
                result = json.loads(cached_data)
                logger.debug(f"Cache hit for key: {cache_key}")
                return result
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
            
        return None
        
    async def _set_cached_result(self, cache_key: str, result: Dict[str, str]) -> None:
        """
        Cache scraping result.
        
        Args:
            cache_key: Redis cache key
            result: Result to cache
        """
        if not self._redis_client:
            return
            
        try:
            cached_data = json.dumps(result)
            await asyncio.get_event_loop().run_in_executor(
                None, self._redis_client.setex, cache_key, self.cache_ttl, cached_data
            )
            logger.debug(f"Cached result for key: {cache_key}")
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")
            
    async def scrape(self, url: str, selectors: List[str]) -> Dict[str, str]:
        """
        Scrape a webpage for specific selectors.
        
        Args:
            url: URL to scrape
            selectors: List of CSS selectors to extract text from
            
        Returns:
            Dict[str, str]: Dictionary mapping selector to extracted text
            
        Raises:
            ValueError: If URL is invalid or selectors are empty
            RuntimeError: If scraping fails
        """
        # Validate inputs
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")
            
        if not selectors:
            raise ValueError("Selectors list cannot be empty")
            
        # Normalize URL
        url = url.strip()
        if not (url.startswith('http://') or url.startswith('https://')):
            url = 'https://' + url
            
        logger.info(f"Scraping URL: {url} with {len(selectors)} selectors")
        
        # Check cache first
        cache_key = self._generate_cache_key(url, selectors)
        cached_result = await self._get_cached_result(cache_key)
        if cached_result is not None:
            logger.info(f"Returning cached result for {url}")
            return cached_result
            
        # Perform scraping
        result = {}
        page: Optional[Page] = None
        
        try:
            if not self._context:
                raise RuntimeError("Browser context not initialized")
                
            # Create new page
            page = await self._context.new_page()
            
            # Set reasonable timeouts
            page.set_default_timeout(30000)  # 30 seconds
            page.set_default_navigation_timeout(30000)
            
            # Navigate to URL
            logger.debug(f"Navigating to: {url}")
            response = await page.goto(url, wait_until='domcontentloaded')
            
            if not response or response.status >= 400:
                raise RuntimeError(f"Failed to load page: HTTP {response.status if response else 'No response'}")
                
            # Wait for network idle (no network activity for 500ms)
            try:
                await page.wait_for_load_state('networkidle', timeout=15000)
                logger.debug("Network idle state reached")
            except PlaywrightTimeoutError:
                logger.warning("Network idle timeout, continuing with current page state")
                
            # Extract text for each selector
            for selector in selectors:
                try:
                    # Wait for element to be present
                    await page.wait_for_selector(selector, timeout=5000)
                    
                    # Extract text content
                    element = await page.query_selector(selector)
                    if element:
                        text = await element.text_content()
                        result[selector] = (text or "").strip()
                        logger.debug(f"Extracted text for '{selector}': {len(result[selector])} chars")
                    else:
                        result[selector] = ""
                        logger.warning(f"Selector '{selector}' not found")
                        
                except PlaywrightTimeoutError:
                    result[selector] = ""
                    logger.warning(f"Timeout waiting for selector '{selector}'")
                except Exception as e:
                    result[selector] = ""
                    logger.error(f"Error extracting text for selector '{selector}': {e}")
                    
        except Exception as e:
            logger.error(f"Scraping failed for {url}: {e}")
            raise RuntimeError(f"Scraping failed: {str(e)}")
            
        finally:
            if page:
                try:
                    await page.close()
                except Exception as e:
                    logger.error(f"Error closing page: {e}")
                    
        # Cache the result
        await self._set_cached_result(cache_key, result)
        
        logger.info(f"Scraping completed for {url}: {len([r for r in result.values() if r])} successful extractions")
        return result


# Convenience function for one-off scraping
async def scrape(url: str, selectors: List[str]) -> Dict[str, str]:
    """
    Convenience function to scrape a URL with selectors.
    
    Args:
        url: URL to scrape
        selectors: List of CSS selectors
        
    Returns:
        Dict[str, str]: Selector to text mapping
    """
    async with ScraperService() as scraper:
        return await scraper.scrape(url, selectors)


# CLI-like function for testing
async def scrape_with_cache_info(url: str, selectors: List[str]) -> Dict[str, Any]:
    """
    Scrape with additional cache information for debugging.
    
    Args:
        url: URL to scrape
        selectors: List of CSS selectors
        
    Returns:
        Dict[str, Any]: Results with cache metadata
    """
    async with ScraperService() as scraper:
        cache_key = scraper._generate_cache_key(url, selectors)
        cached_result = await scraper._get_cached_result(cache_key)
        
        if cached_result is not None:
            return {
                'data': cached_result,
                'cache_hit': True,
                'cache_key': cache_key
            }
        else:
            result = await scraper.scrape(url, selectors)
            return {
                'data': result,
                'cache_hit': False,
                'cache_key': cache_key
            }


# Example usage and testing
if __name__ == "__main__":
    async def test_scraper():
        """Test the scraper service."""
        test_url = "https://example.com"
        test_selectors = ["h1", "p", ".content"]
        
        try:
            result = await scrape(test_url, test_selectors)
            print(f"Scraping result: {result}")
        except Exception as e:
            print(f"Scraping failed: {e}")
            
    # Run test
    asyncio.run(test_scraper()) 