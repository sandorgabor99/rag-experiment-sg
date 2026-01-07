"""
Query Refinement module for RAG pipeline.

This module refines user questions into RAG-optimized search queries
using LLaMA before embedding search.
"""

import logging
import os
from typing import List, Optional
from dotenv import load_dotenv

try:
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_core.language_models import BaseChatModel
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        from langchain_community.chat_models import ChatOllama
    from langchain_openai import ChatOpenAI
    QA_AVAILABLE = True
except ImportError:
    QA_AVAILABLE = False
    ChatOllama = None
    ChatOpenAI = None
    BaseChatModel = None

try:
    load_dotenv()
except:
    pass

logger = logging.getLogger(__name__)


class QueryRefiner:
    """
    Refines user questions into RAG-optimized search queries.
    
    Uses LLaMA to rewrite questions into queries optimized for
    semantic embedding retrieval.
    """
    
    def __init__(
        self,
        provider: str = 'ollama',
        model: str = 'llama3',
        temperature: float = 0.15,
        max_tokens: int = 100,
        base_url: Optional[str] = None
    ):
        """
        Initialize query refiner.
        
        Args:
            provider: LLM provider - 'ollama' or 'openai'
            model: Model name (default: llama3)
            temperature: Temperature for LLM (default: 0.15)
            max_tokens: Maximum tokens for refined query (default: 100)
            base_url: Custom base URL for API
        """
        if not QA_AVAILABLE:
            raise ImportError(
                "QueryRefiner requires langchain packages. "
                "Install with: pip install langchain langchain-openai langchain-community langchain-anthropic python-dotenv"
            )
        
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.llm = self._initialize_llm(provider, model, temperature, base_url)
        
        logger.info(f"Initialized QueryRefiner with {provider} (model={model}, temperature={temperature})")
    
    def _initialize_llm(
        self,
        provider: str,
        model: str,
        temperature: float,
        base_url: Optional[str]
    ) -> BaseChatModel:
        """Initialize LLM based on provider."""
        if provider == 'ollama':
            if model is None:
                model = 'llama3'
            logger.info(f"Using Ollama with model: {model}")
            return ChatOllama(
                model=model,
                temperature=temperature,
                base_url=base_url or 'http://localhost:11434'
            )
        
        elif provider == 'openai':
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")
            if model is None:
                model = 'gpt-3.5-turbo'
            
            llm = ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                temperature=temperature,
                max_tokens=self.max_tokens
            )
            
            if base_url:
                llm.openai_api_base = base_url
            
            return llm
        
        else:
            raise ValueError(f"Unknown provider: {provider}. Supported: 'ollama', 'openai'")
    
    def refine(self, question: str, n_variants: int = 1) -> List[str]:
        """
        Refine user question into RAG-optimized search queries.
        
        Args:
            question: User's original question
            n_variants: Number of query variants to generate (default: 1)
        
        Returns:
            List of refined search queries (Hungarian, plain text)
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        try:
            messages = self._build_prompt(question, n_variants)
            
            logger.info(f"Refining question: '{question[:50]}...'")
            response = self.llm.invoke(messages)
            
            refined_text = response.content.strip()
            
            # Remove markdown formatting if present
            if refined_text.startswith('```'):
                lines = refined_text.split('\n')
                refined_text = '\n'.join([l for l in lines if not l.strip().startswith('```')])
            refined_text = refined_text.strip()
            
            if n_variants > 1:
                queries = [q.strip() for q in refined_text.split('\n') if q.strip()]
            else:
                queries = [refined_text]
            
            logger.info(f"Generated {len(queries)} refined query variant(s)")
            for i, query in enumerate(queries, 1):
                logger.info(f"  Variant {i}: {query[:80]}...")
            
            return queries
            
        except Exception as e:
            logger.error(f"Query refinement failed: {e}", exc_info=True)
            logger.warning(f"Falling back to original question: {question}")
            return [question]
    
    def _build_prompt(self, question: str, n_variants: int) -> List:
        """Build prompt messages for query refinement."""
        if n_variants > 1:
            variant_instruction = f"\nGenerate {n_variants} refined queries, one per line."
        else:
            variant_instruction = ""
        
        system_prompt = f"""You are a query refiner for a Retrieval-Augmented Generation system.

Rewrite the user's question into a concise search query optimized for semantic embedding retrieval.

Rules:
- Use Hungarian
- Prefer explicit keywords over natural language
- Expand implicit intent into explicit terms
- Include entity types, document types, or section names when implied
- Do NOT answer the question
- Do NOT explain anything
- Output ONLY the refined query text{variant_instruction}

Examples:

User question:
"Kik az alapító tagok?"

Refined query:
"Alapító tagok személyek listája név alias SECTION: Alapító tagok ENTITY_TYPE: személy DOCUMENT_TYPE: alapitok.txt"

User question:
"Ki az Error?"

Refined query:
"Mészáros Mihály alias Error személy leírás ENTITY_TYPE: személy"

Now refine this question:"""
        
        user_prompt = question
        
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
