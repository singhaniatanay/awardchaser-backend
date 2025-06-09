"""
Web Scraping API Router

Provides endpoints for scraping web content using Playwright with API key protection.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from typing import Optional

from app.models import ScrapeRequest, ScrapeResponse
from app.services.scraper import ScraperService
from app.core.config import settings


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["scraping"])


async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """
    Verify API key from request header.
    
    Args:
        x_api_key: API key from X-API-Key header
        
    Returns:
        str: Validated API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is required. Provide X-API-Key header."
        )
    
    if not settings.scraper_api_key:
        logger.error("SCRAPER_API_KEY not configured in environment")
        raise HTTPException(
            status_code=500,
            detail="Scraper API key not configured"
        )
    
    if x_api_key != settings.scraper_api_key:
        logger.warning("Invalid API key provided for scraper endpoint")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return x_api_key


@router.post("/scrape", response_model=ScrapeResponse)
async def scrape_content(
    request: ScrapeRequest,
    api_key: str = Depends(verify_api_key)
) -> ScrapeResponse:
    """
    Scrape content from a website using CSS selectors.
    
    This endpoint uses Playwright to navigate to the specified URL and extract
    content using the provided CSS selectors. Results are cached for performance.
    
    Args:
        request: Scraping request with URL and selectors
        api_key: API key for authentication (from X-API-Key header)
        
    Returns:
        ScrapeResponse: Scraped content mapped by selector
        
    Raises:
        HTTPException: If scraping fails or URL is invalid
    """
    try:
        logger.info(f"Scraping content from: {request.url}")
        logger.debug(f"Selectors: {request.selectors}")
        
        # Validate URL format
        if not request.url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=400,
                detail="URL must start with http:// or https://"
            )
        
        # Initialize scraper and perform scraping
        async with ScraperService() as scraper:
            try:
                scraped_data = await scraper.scrape(request.url, request.selectors)
                logger.info(f"Successfully scraped {len(scraped_data)} selectors")
            except Exception as e:
                logger.error(f"Scraping failed for {request.url}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to scrape content: {str(e)}"
                )
        
        # Check if data was found
        selectors_found = sum(1 for value in scraped_data.values() if value and value.strip())
        
        if selectors_found == 0:
            logger.warning(f"No content found for any selectors on {request.url}")
        
        # Determine if result was cached (simplified check)
        # In a real implementation, you'd get this from the scraper service
        cached = False  # ScraperService handles caching internally
        
        return ScrapeResponse(
            url=request.url,
            data=scraped_data,
            selectors_found=selectors_found,
            cached=cached
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in scrape_content: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during scraping"
        )


@router.get("/scrape/health")
async def scrape_health_check(api_key: str = Depends(verify_api_key)):
    """
    Health check endpoint for the scraping service.
    
    Args:
        api_key: API key for authentication (from X-API-Key header)
    
    Returns:
        dict: Service status and browser availability
    """
    try:
        # Test browser initialization
        async with ScraperService() as scraper:
            # Simple test to verify browser is working
            test_result = await scraper.scrape(
                "data:text/html,<div id='test'>Health Check</div>",
                ["#test"]
            )
            
            browser_working = test_result.get("#test") == "Health Check"
        
        return {
            "status": "healthy",
            "browser": {
                "status": "working" if browser_working else "degraded",
                "engine": "Playwright + Chromium"
            },
            "cache": {
                "status": "enabled",
                "backend": "Redis"
            }
        }
        
    except Exception as e:
        logger.error(f"Scrape health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "browser": {
                    "status": "error",
                    "engine": "Playwright + Chromium"
                }
            }
        ) 