"""
Input sanitization and prompt injection guards.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Patterns that indicate potential prompt injection
INSTRUCTION_PATTERNS = [
    r'\[INST\]',
    r'\[/INST\]',
    r'</s>',
    r'<s>',
    r'### Instruction:',
    r'### Response:',
    r'System:',
    r'User:',
    r'Assistant:',
    r'Ignore previous instructions',
    r'Forget everything',
    r'You are now',
    r'Act as if',
    r'Pretend to be'
]


def sanitize_query(query: str, max_length: int = 1000) -> str:
    """
    Sanitize user query to prevent prompt injection.
    
    Args:
        query: User query string
        max_length: Maximum allowed length
    
    Returns:
        Sanitized query string
    """
    if not query:
        return ""
    
    # Strip leading/trailing whitespace
    query = query.strip()
    
    # Limit length
    if len(query) > max_length:
        logger.warning(f"Query truncated from {len(query)} to {max_length} characters")
        query = query[:max_length]
    
    # Remove instruction-like patterns
    for pattern in INSTRUCTION_PATTERNS:
        query = re.sub(pattern, '', query, flags=re.IGNORECASE)
    
    # Remove HTML tags
    query = re.sub(r'<[^>]+>', '', query)
    
    # Normalize whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    
    return query


def sanitize_context(context: str) -> str:
    """
    Sanitize context before sending to LLM.
    
    Args:
        context: Context block string
    
    Returns:
        Sanitized context string
    """
    if not context:
        return ""
    
    # Remove instruction patterns
    for pattern in INSTRUCTION_PATTERNS:
        context = re.sub(pattern, '', context, flags=re.IGNORECASE)
    
    # Remove HTML tags
    context = re.sub(r'<[^>]+>', '', context)
    
    # Remove excessive newlines
    context = re.sub(r'\n{4,}', '\n\n\n', context)
    
    return context
