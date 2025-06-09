"""
Agent Tools Module

Provides tool implementations for AI agents to interact with various services.
"""

import logging
from typing import Dict, List, Any, Optional
import json
import httpx
from langchain_core.tools import BaseTool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import Field

from app.core.config import settings


logger = logging.getLogger(__name__)


class ScrapeTool(BaseTool):
    """Web scraping tool for AI agents."""
    
    name: str = "scrape_web"
    description: str = """
    Scrape content from websites using CSS selectors. Use this tool when you need to extract 
    specific content from web pages for research or data collection.
    
    Input should be a JSON object with:
    - url: The website URL to scrape (must start with http:// or https://)
    - selectors: List of CSS selectors to extract content from
    
    Example: {"url": "https://example.com", "selectors": ["h1", ".price", "#description"]}
    
    Returns a dictionary mapping each CSS selector to the extracted text content.
    """
    
    base_url: str = Field(default="http://localhost:8000", description="Base URL of the FastAPI server")
    api_key: str = Field(default_factory=lambda: settings.scraper_api_key, description="API key for scraper endpoint")
    
    def _run(self, tool_input: str) -> str:
        """
        Execute the web scraping tool.
        
        Args:
            tool_input: JSON string containing url and selectors
            
        Returns:
            str: JSON string with scraped results
        """
        try:
            # Parse tool input
            try:
                input_data = json.loads(tool_input)
            except json.JSONDecodeError as e:
                return f"Error: Invalid JSON input - {str(e)}"
            
            # Validate required fields
            if "url" not in input_data:
                return "Error: Missing required field 'url'"
            if "selectors" not in input_data:
                return "Error: Missing required field 'selectors'"
            
            url = input_data["url"]
            selectors = input_data["selectors"]
            
            # Validate input types
            if not isinstance(url, str):
                return "Error: 'url' must be a string"
            if not isinstance(selectors, list):
                return "Error: 'selectors' must be a list"
            if not all(isinstance(sel, str) for sel in selectors):
                return "Error: All selectors must be strings"
            
            # Validate URL format
            if not url.startswith(("http://", "https://")):
                return "Error: URL must start with http:// or https://"
            
            # Make request to scrape endpoint
            with httpx.Client(timeout=60.0) as client:
                try:
                    response = client.post(
                        f"{self.base_url}/api/v1/scrape",
                        json={
                            "url": url,
                            "selectors": selectors
                        },
                        headers={
                            "X-API-Key": self.api_key,
                            "Content-Type": "application/json"
                        }
                    )
                    
                    if response.status_code == 401:
                        return "Error: Invalid or missing API key for scraper service"
                    elif response.status_code == 400:
                        return f"Error: Bad request - {response.json().get('detail', 'Invalid request')}"
                    elif response.status_code == 500:
                        error_detail = response.json().get('detail', 'Internal server error')
                        return f"Error: Scraping failed - {error_detail}"
                    elif response.status_code != 200:
                        return f"Error: HTTP {response.status_code} - {response.text}"
                    
                    # Parse successful response
                    result = response.json()
                    
                    # Format the response for the agent
                    scraped_data = result.get("data", {})
                    selectors_found = result.get("selectors_found", 0)
                    
                    # Create a human-readable summary
                    summary = f"Successfully scraped {selectors_found} out of {len(selectors)} selectors from {url}\n\n"
                    
                    for selector, content in scraped_data.items():
                        if content and content.strip():
                            summary += f"Selector '{selector}':\n{content[:200]}{'...' if len(content) > 200 else ''}\n\n"
                        else:
                            summary += f"Selector '{selector}': No content found\n\n"
                    
                    # Also return the raw data as JSON for programmatic use
                    return f"{summary}\nRaw JSON data:\n{json.dumps(result, indent=2)}"
                    
                except httpx.ConnectError:
                    return "Error: Could not connect to scraper service. Is the FastAPI server running?"
                except httpx.TimeoutException:
                    return "Error: Request timed out. The website may be slow to respond."
                except httpx.RequestError as e:
                    return f"Error: Request failed - {str(e)}"
                    
        except Exception as e:
            logger.error(f"Unexpected error in ScrapeTool: {e}")
            return f"Error: Unexpected error occurred - {str(e)}"
    
    async def _arun(self, tool_input: str) -> str:
        """
        Async version of the scraping tool.
        
        Args:
            tool_input: JSON string containing url and selectors
            
        Returns:
            str: JSON string with scraped results
        """
        try:
            # Parse tool input
            try:
                input_data = json.loads(tool_input)
            except json.JSONDecodeError as e:
                return f"Error: Invalid JSON input - {str(e)}"
            
            # Validate required fields
            if "url" not in input_data:
                return "Error: Missing required field 'url'"
            if "selectors" not in input_data:
                return "Error: Missing required field 'selectors'"
            
            url = input_data["url"]
            selectors = input_data["selectors"]
            
            # Validate input types
            if not isinstance(url, str):
                return "Error: 'url' must be a string"
            if not isinstance(selectors, list):
                return "Error: 'selectors' must be a list"
            if not all(isinstance(sel, str) for sel in selectors):
                return "Error: All selectors must be strings"
            
            # Validate URL format
            if not url.startswith(("http://", "https://")):
                return "Error: URL must start with http:// or https://"
            
            # Make async request to scrape endpoint
            async with httpx.AsyncClient(timeout=60.0) as client:
                try:
                    response = await client.post(
                        f"{self.base_url}/api/v1/scrape",
                        json={
                            "url": url,
                            "selectors": selectors
                        },
                        headers={
                            "X-API-Key": self.api_key,
                            "Content-Type": "application/json"
                        }
                    )
                    
                    if response.status_code == 401:
                        return "Error: Invalid or missing API key for scraper service"
                    elif response.status_code == 400:
                        return f"Error: Bad request - {response.json().get('detail', 'Invalid request')}"
                    elif response.status_code == 500:
                        error_detail = response.json().get('detail', 'Internal server error')
                        return f"Error: Scraping failed - {error_detail}"
                    elif response.status_code != 200:
                        return f"Error: HTTP {response.status_code} - {response.text}"
                    
                    # Parse successful response
                    result = response.json()
                    
                    # Format the response for the agent
                    scraped_data = result.get("data", {})
                    selectors_found = result.get("selectors_found", 0)
                    
                    # Create a human-readable summary
                    summary = f"Successfully scraped {selectors_found} out of {len(selectors)} selectors from {url}\n\n"
                    
                    for selector, content in scraped_data.items():
                        if content and content.strip():
                            summary += f"Selector '{selector}':\n{content[:200]}{'...' if len(content) > 200 else ''}\n\n"
                        else:
                            summary += f"Selector '{selector}': No content found\n\n"
                    
                    # Also return the raw data as JSON for programmatic use
                    return f"{summary}\nRaw JSON data:\n{json.dumps(result, indent=2)}"
                    
                except httpx.ConnectError:
                    return "Error: Could not connect to scraper service. Is the FastAPI server running?"
                except httpx.TimeoutException:
                    return "Error: Request timed out. The website may be slow to respond."
                except httpx.RequestError as e:
                    return f"Error: Request failed - {str(e)}"
                    
        except Exception as e:
            logger.error(f"Unexpected error in ScrapeTool async: {e}")
            return f"Error: Unexpected error occurred - {str(e)}"


# Tool registry for AI agents
def get_agent_tools() -> List[BaseTool]:
    """
    Get list of available tools for AI agents.
    
    Returns:
        List[BaseTool]: List of tool instances
    """
    tools = [
        ScrapeTool()
    ]
    
    logger.info(f"Registered {len(tools)} tools for agent use")
    return tools


# Convenience function to get tools by name
def get_tool_by_name(tool_name: str) -> Optional[BaseTool]:
    """
    Get a specific tool by its name.
    
    Args:
        tool_name: Name of the tool to retrieve
        
    Returns:
        BaseTool instance or None if not found
    """
    tools = get_agent_tools()
    for tool in tools:
        if tool.name == tool_name:
            return tool
    return None


# Tool descriptions for agent prompts
def get_tool_descriptions() -> str:
    """
    Get formatted descriptions of all available tools for use in agent prompts.
    
    Returns:
        str: Formatted tool descriptions
    """
    tools = get_agent_tools()
    descriptions = []
    
    for tool in tools:
        descriptions.append(f"Tool: {tool.name}\nDescription: {tool.description.strip()}\n")
    
    return "\n".join(descriptions)


# Agent creation helper
def create_research_agent(temperature: float = 0.3) -> AgentExecutor:
    """
    Create a research agent with web scraping capabilities.
    
    Args:
        temperature: Model temperature for response generation
        
    Returns:
        AgentExecutor: Configured agent with tools
    """
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        api_key=settings.openai_api_key
    )
    
    # Get available tools
    tools = get_agent_tools()
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research assistant with web scraping capabilities. 
        You can extract specific information from websites using CSS selectors.
        
        When scraping websites:
        1. Always identify the most specific CSS selectors for the data you need
        2. Be respectful of website resources and don't make excessive requests
        3. Provide clear summaries of what you found
        4. If scraping fails, explain why and suggest alternatives
        
        Available tools:
        {tool_descriptions}
        """),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Format tool descriptions
    formatted_prompt = prompt.partial(tool_descriptions=get_tool_descriptions())
    
    # Create agent
    agent = create_openai_functions_agent(llm, tools, formatted_prompt)
    
    # Create executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    logger.info("Created research agent with web scraping capabilities")
    return agent_executor


# Example usage function
def example_agent_usage():
    """
    Example of how to use the agent with scraping tools.
    This is for demonstration purposes.
    """
    try:
        # Create agent
        agent = create_research_agent()
        
        # Example query
        query = """
        Scrape the title and main heading from https://httpbin.org/html
        Use appropriate CSS selectors to extract this information.
        """
        
        # Run agent
        result = agent.invoke({"input": query})
        
        print("Agent Response:")
        print("=" * 50)
        print(result["output"])
        
    except Exception as e:
        logger.error(f"Error in example usage: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run example if script is executed directly
    example_agent_usage() 