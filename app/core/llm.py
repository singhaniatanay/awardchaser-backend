"""
LLM Integration Module

Provides functions for interacting with OpenAI's language models,
including document-aware prompting and response generation.
"""

import logging
from typing import List, Optional
import openai
from langchain_core.documents import Document

from .config import settings


logger = logging.getLogger(__name__)


def ask_llm(
    system_prompt: str,
    user_prompt: str,
    docs: Optional[List[Document]] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """
    Ask the LLM a question with optional document context.
    
    Args:
        system_prompt: System message to set context and behavior
        user_prompt: User's question or request
        docs: List of Document objects to include as context
        model: OpenAI model to use (default: gpt-4o-mini)
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens in response (None for model default)
        
    Returns:
        str: The LLM's response text
        
    Raises:
        ValueError: If OpenAI API key is not configured
        openai.OpenAIError: If API call fails
    """
    try:
        # Validate API key
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
        
        # Set the API key
        openai.api_key = settings.openai_api_key
        
        # Format the user prompt with document context
        formatted_user_prompt = _format_prompt_with_docs(user_prompt, docs)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_user_prompt}
        ]
        
        # Prepare API call parameters
        api_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False  # Explicitly disable streaming
        }
        
        # Add max_tokens if specified
        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens
        
        logger.debug(f"Calling OpenAI API with model: {model}, temperature: {temperature}")
        
        # Make API call
        response = openai.ChatCompletion.create(**api_params)
        
        # Extract response text
        response_text = response.choices[0].message.content
        
        # Log usage statistics
        if hasattr(response, 'usage'):
            usage = response.usage
            logger.info(f"OpenAI API usage - Prompt tokens: {usage.prompt_tokens}, "
                       f"Completion tokens: {usage.completion_tokens}, "
                       f"Total tokens: {usage.total_tokens}")
        
        logger.debug(f"LLM response length: {len(response_text)} characters")
        return response_text
        
    except openai.InvalidRequestError as e:
        logger.error(f"Invalid OpenAI API request: {e}")
        raise
    except openai.AuthenticationError as e:
        logger.error(f"OpenAI API authentication failed: {e}")
        raise
    except openai.RateLimitError as e:
        logger.error(f"OpenAI API rate limit exceeded: {e}")
        raise
    except openai.OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in ask_llm: {e}")
        raise


def _format_prompt_with_docs(user_prompt: str, docs: Optional[List[Document]] = None) -> str:
    """
    Format user prompt with document context sections.
    
    Args:
        user_prompt: The original user prompt
        docs: List of Document objects to include as context
        
    Returns:
        str: Formatted prompt with document sections
    """
    if not docs:
        return user_prompt
    
    # Start with the user prompt
    formatted_prompt = user_prompt
    
    # Add document sections
    doc_sections = []
    for i, doc in enumerate(docs, 1):
        # Create source section header
        source_header = f"### Source {i}"
        
        # Add metadata if available
        if doc.metadata:
            metadata_info = []
            # Add key metadata fields
            if "source" in doc.metadata:
                metadata_info.append(f"File: {doc.metadata['source']}")
            if "file_type" in doc.metadata:
                metadata_info.append(f"Type: {doc.metadata['file_type'].upper()}")
            if "chunk_index" in doc.metadata and "total_chunks" in doc.metadata:
                metadata_info.append(f"Section: {doc.metadata['chunk_index'] + 1}/{doc.metadata['total_chunks']}")
            
            if metadata_info:
                source_header += f" ({', '.join(metadata_info)})"
        
        # Add the document content
        doc_section = f"{source_header}\n{doc.page_content.strip()}"
        doc_sections.append(doc_section)
    
    # Combine user prompt with document sections
    if doc_sections:
        formatted_prompt += "\n\n" + "\n\n".join(doc_sections)
    
    return formatted_prompt


def ask_llm_with_retrieval(
    system_prompt: str,
    user_prompt: str,
    query_text: Optional[str] = None,
    k: int = 5,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """
    Ask the LLM a question with automatic document retrieval.
    
    Args:
        system_prompt: System message to set context and behavior
        user_prompt: User's question or request
        query_text: Text to search for (uses user_prompt if None)
        k: Number of documents to retrieve
        model: OpenAI model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        
    Returns:
        str: The LLM's response text with document context
    """
    try:
        from .retrieval import get_vector_client
        
        # Use user_prompt as query if no specific query provided
        search_query = query_text or user_prompt
        
        # Retrieve relevant documents
        vector_client = get_vector_client()
        docs = vector_client.query(search_query, k=k)
        
        logger.info(f"Retrieved {len(docs)} documents for query: '{search_query[:50]}...'")
        
        # Call LLM with retrieved documents
        return ask_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            docs=docs,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
    except Exception as e:
        logger.error(f"Error in ask_llm_with_retrieval: {e}")
        # Fallback to LLM without documents
        logger.warning("Falling back to LLM without document retrieval")
        return ask_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            docs=None,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )


# Convenience functions for common use cases
def ask_question(question: str, context_docs: Optional[List[Document]] = None) -> str:
    """
    Ask a general question with optional context documents.
    
    Args:
        question: The question to ask
        context_docs: Optional documents for context
        
    Returns:
        str: The answer from the LLM
    """
    system_prompt = """You are a helpful AI assistant. Answer questions clearly and accurately 
    based on the provided context. If you cannot find the answer in the context, say so clearly."""
    
    return ask_llm(
        system_prompt=system_prompt,
        user_prompt=question,
        docs=context_docs
    )


def summarize_documents(docs: List[Document], focus: Optional[str] = None) -> str:
    """
    Summarize a list of documents with optional focus area.
    
    Args:
        docs: Documents to summarize
        focus: Optional focus area for the summary
        
    Returns:
        str: Summary of the documents
    """
    system_prompt = """You are an expert at summarizing documents. Create clear, 
    concise summaries that capture the key information and main points."""
    
    user_prompt = "Please provide a comprehensive summary of the following documents."
    if focus:
        user_prompt += f" Focus particularly on: {focus}"
    
    return ask_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        docs=docs
    )


def analyze_credit_card_strategy(
    transaction_details: str,
    user_cards: List[str],
    context_docs: Optional[List[Document]] = None
) -> str:
    """
    Analyze optimal credit card strategy for a transaction.
    
    Args:
        transaction_details: Details about the transaction
        user_cards: List of user's credit cards
        context_docs: Relevant credit card documents
        
    Returns:
        str: Credit card strategy recommendation
    """
    system_prompt = """You are an expert credit card advisor. Analyze the transaction 
    and recommend the optimal credit card strategy to maximize rewards, considering 
    the user's available cards and current market conditions. Provide specific, 
    actionable advice with clear reasoning."""
    
    user_prompt = f"""
    Transaction Details: {transaction_details}
    
    Available Cards: {', '.join(user_cards)}
    
    Please recommend the best credit card strategy for this transaction, including:
    1. Which card to use and why
    2. Expected reward value
    3. Any alternative strategies to consider
    4. Tips to maximize rewards
    """
    
    return ask_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        docs=context_docs
    ) 