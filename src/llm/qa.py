"""
Question-Answering module using LLaMA.

This module generates answers from context + question using LLM.
Implements RAG (Retrieval-Augmented Generation) pattern.
"""

import logging
import os
import time
from typing import Optional, TYPE_CHECKING, Tuple, Any
from dotenv import load_dotenv

if TYPE_CHECKING:
    from src.config import Config

try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_core.language_models import BaseChatModel
    # LLM imports (reuse from agentic_refiner)
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        from langchain_community.chat_models import ChatOllama
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    QA_AVAILABLE = True
except ImportError:
    QA_AVAILABLE = False
    ChatOllama = None
    ChatOpenAI = None
    ChatAnthropic = None
    BaseChatModel = None

# Load environment variables
try:
    load_dotenv()
except:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Default RAG prompt template (Hungarian, hardened)
DEFAULT_PROMPT_TEMPLATE = """Te egy tényalapú asszisztens vagy.

Válaszolj a kérdésre KIZÁRÓLAG a megadott kontextus alapján.

FONTOS SZABÁLYOK:
1. Ha a kontextus tartalmazza a választ (akár közvetve is), HASZNÁLD azt!
2. A válasz akkor is a kontextusban lévőnek számít, ha:
   - felsorolásként szerepel (pl. "Bocskai, Kazinczy")
   - egyenruha/viselet/attribútum névként jelenik meg
   - a kérdés "mit viselnek" és a kontextus "egyenruhái" vagy "viselet" szavakat használ
   - a kérdés és a kontextus eltérő szavakat használ, de ugyanarra a tényre utal

3. PÉLDÁK:
   - Kérdés: "mit viselnek?" → Kontextus: "egyenruhái: X, Y" → Válasz: "X és Y egyenruhát viselnek"
   - Kérdés: "milyen egyenruhák?" → Kontextus: "egyenruhái: A, B" → Válasz: "A és B egyenruhák"
   - Kérdés: "mi a viselet?" → Kontextus: "viselet: Z" → Válasz: "Z viselet"

4. Ha a kontextus tartalmaz neveket, listákat, felsorolásokat, azok VÁLASZOK!
   - Ne várj teljes mondatokat - a felsorolások is válaszok!

5. Csak akkor mondd, hogy nincs válasz, ha a kontextus valóban nem tartalmaz semmilyen releváns információt.

A kérdés típusától függően a következő struktúrát alkalmazza a válaszhoz:

- Leíró (Hogy néz ki?): Összefoglalja a releváns tulajdonságokat.
- Anyag/Összetétel: Sorolja fel az anyag vagy az összetétel részleteit.
- Változatok/Opciók: Sorolja fel az összes elérhető változatot.
- Nevek/aliasok: Adja meg a fő nevet és az esetleges ismert aliast.
- Viselet/egyenruha: Sorolja fel a viselet/egyenruha neveit vagy típusait.

Ne használj külső tudást.
Ne találj ki új tényeket.
Ne adj hozzá olyan információt, ami nincs a kontextusban.

Kontextus:
{context}

Kérdés:
{question}

Válasz:"""


class QuestionAnswerer:
    """
    Generates answers from context + question using LLM.
    
    Implements RAG (Retrieval-Augmented Generation) pattern where:
    1. Context is retrieved from knowledge base
    2. Question is asked
    3. LLM generates answer based on context
    """
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        prompt_template: Optional[str] = None,
        config: Optional['Config'] = None
    ):
        """
        Initialize QA system.
        
        Args:
            provider: LLM provider - 'ollama', 'openai', 'anthropic', or 'custom' (overrides config)
            model: Model name (overrides config)
            temperature: Temperature for LLM (overrides config)
            max_tokens: Maximum tokens for answer (overrides config)
            api_key: API key for the provider (defaults to env vars)
            base_url: Custom base URL for API (overrides config)
            prompt_template: Custom prompt template (optional)
            config: Optional Config instance
        """
        # Get QA config with priority: function arg > config > default
        qa_config = {}
        if config:
            qa_config = config.llm.get('qa', {})
        
        provider = provider if provider else qa_config.get('provider', 'ollama')
        model = model if model else qa_config.get('model', 'llama3')
        temperature = temperature if temperature is not None else qa_config.get('temperature', 0.7)
        max_tokens = max_tokens if max_tokens is not None else qa_config.get('max_tokens', 500)
        base_url = base_url if base_url else qa_config.get('base_url', None)
        
        if not QA_AVAILABLE:
            raise ImportError(
                "QA module requires langchain packages. "
                "Install with: pip install langchain langchain-openai langchain-community langchain-anthropic python-dotenv"
            )
        
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        
        # Initialize LLM
        self.llm = self._initialize_llm(provider, api_key, model, base_url, temperature)
        
        logger.info(f"Initialized QuestionAnswerer with {provider} (model={model}, temperature={temperature})")
    
    def _initialize_llm(
        self,
        provider: str,
        api_key: Optional[str],
        model: Optional[str],
        base_url: Optional[str],
        temperature: float
    ) -> BaseChatModel:
        """
        Initialize LLM based on provider.
        
        Reuses the same pattern from agentic_refiner.py for consistency.
        """
        
        if provider == 'ollama':
            # Local Ollama models (free, no API key needed)
            if model is None:
                model = 'llama3'  # Default Ollama model
            logger.info(f"Using Ollama with model: {model}")
            return ChatOllama(
                model=model,
                temperature=temperature,
                base_url=base_url or 'http://localhost:11434'
            )
        
        elif provider == 'openai':
            # OpenAI API
            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter."
                )
            if model is None:
                model = 'gpt-3.5-turbo'
            
            llm = ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                temperature=temperature,
                max_tokens=self.max_tokens
            )
            
            # Support custom base URLs (for proxies, local OpenAI-compatible APIs)
            if base_url:
                llm.openai_api_base = base_url
            
            return llm
        
        elif provider == 'anthropic':
            # Anthropic Claude API
            if api_key is None:
                api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key parameter."
                )
            if model is None:
                model = 'claude-3-sonnet-20240229'
            
            return ChatAnthropic(
                model=model,
                anthropic_api_key=api_key,
                temperature=temperature,
                max_tokens=self.max_tokens
            )
        
        elif provider == 'custom':
            # Custom OpenAI-compatible API (e.g., local models, proxies)
            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY", "dummy-key")  # Some APIs don't need real key
            if model is None:
                model = 'gpt-3.5-turbo'
            if base_url is None:
                raise ValueError("base_url required for custom provider")
            
            logger.info(f"Using custom API at {base_url} with model: {model}")
            return ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=temperature,
                max_tokens=self.max_tokens
            )
        
        else:
            raise ValueError(
                f"Unknown provider: {provider}. "
                "Supported providers: 'ollama', 'openai', 'anthropic', 'custom'"
            )
    
    def answer(self, question: str, context: str, return_metrics: bool = False) -> Any:  # Returns str or Tuple[str, float]
        """
        Generate answer from question + context.
        
        Args:
            question: User's question
            context: Context text retrieved from knowledge base
        
        Returns:
            Generated answer string
        """
        if not question or not question.strip():
            raise ValueError("A kérdés nem lehet üres")
        
        if not context or not context.strip():
            logger.warning("Üres kontextus lett megadva. A válasz korlátozott lehet.")
            context = "Nincs elérhető kontextus."
        
        try:
            # Build prompt
            prompt = self._build_prompt(question, context)
            
            # Debug: Log context length
            logger.debug(f"Kontextus hossza: {len(context)} karakter, {len(context.split())} szó")
            logger.debug(f"Prompt hossza: {len(prompt)} karakter")
            
            # Generate answer using LLM
            logger.info(f"Válasz generálása a kérdésre: '{question[:50]}...'")
            logger.info(f"Kontextus használata: {len(context)} karakter")
            response = self.llm.invoke(prompt)
            
            # Extract answer text
            answer = response.content.strip()
            
            logger.info(f"Generált válasz: {len(answer)} karakter")
            return answer
            
        except Exception as e:
            logger.error(f"A válasz generálása sikertelen: {e}", exc_info=True)
            raise
    
    def _build_prompt(self, question: str, context: str) -> str:
        """
        Build prompt template from question and context.
        
        Args:
            question: User's question
            context: Context text
        
        Returns:
            Formatted prompt string
        """
        # Ensure context is not empty
        if not context or not context.strip():
            logger.warning("A kontextus üres, helyőrző használata")
            context = "Nincs elérhető kontextus."
        
        try:
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            # Debug: verify prompt contains context
            if "Nincs elérhető kontextus" in prompt or len(context) < 10:
                logger.warning(f"A kontextus üresnek vagy nagyon rövidnek tűnik: {len(context)} karakter")
            return prompt
        except KeyError as e:
            logger.error(f"Prompt sablon formázási hiba: {e}")
            # Fallback to simple format (Hungarian)
            return f"""Te egy segítőkész asszisztens vagy. Válaszolj a kérdésre a megadott kontextus alapján.

Kontextus:
{context}

Kérdés: {question}

Válasz:"""
    
    def answer_with_messages(self, question: str, context: str) -> str:
        """
        Alternative method using LangChain message format.
        
        This can be useful for models that prefer structured messages.
        
        Args:
            question: User's question
            context: Context text
        
        Returns:
            Generated answer string
        """
        if not question or not question.strip():
            raise ValueError("A kérdés nem lehet üres")
        
        if not context or not context.strip():
            logger.warning("Üres kontextus lett megadva. A válasz korlátozott lehet.")
            context = "Nincs elérhető kontextus."
        
        try:
            # Build messages (Hungarian)
            system_message = SystemMessage(
                content="Te egy segítőkész asszisztens vagy, aki a megadott kontextus alapján válaszol a kérdésekre."
            )
            human_message = HumanMessage(
                content=f"Kontextus:\n{context}\n\nKérdés: {question}\n\nVálasz:"
            )
            
            messages = [system_message, human_message]
            
            # Generate answer
            logger.info(f"Válasz generálása a kérdésre: '{question[:50]}...'")
            response = self.llm.invoke(messages)
            
            answer = response.content.strip()
            logger.info(f"Generált válasz: {len(answer)} karakter")
            return answer
            
        except Exception as e:
            logger.error(f"A válasz generálása sikertelen: {e}", exc_info=True)
            raise


def main():
    """CLI for testing QA module."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Question-Answering using LLM with context.'
    )
    parser.add_argument(
        '--question',
        type=str,
        required=True,
        help='Question to answer'
    )
    parser.add_argument(
        '--context',
        type=str,
        required=True,
        help='Context text for answering'
    )
    parser.add_argument(
        '--provider',
        type=str,
        default='ollama',
        choices=['ollama', 'openai', 'anthropic', 'custom'],
        help='LLM provider (default: ollama)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='llama3',
        help='Model name (default: llama3)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Temperature for LLM (default: 0.7)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=500,
        help='Maximum tokens for answer (default: 500)'
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default=None,
        help='Custom base URL for API'
    )
    
    args = parser.parse_args()
    
    try:
        qa = QuestionAnswerer(
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=args.base_url
        )
        
        answer = qa.answer(args.question, args.context)
        
        print(f"\n{'='*60}")
        print("QUESTION:")
        print(f"{'='*60}")
        print(args.question)
        print(f"\n{'='*60}")
        print("CONTEXT:")
        print(f"{'='*60}")
        print(args.context[:500] + "..." if len(args.context) > 500 else args.context)
        print(f"\n{'='*60}")
        print("ANSWER:")
        print(f"{'='*60}")
        print(answer)
        print(f"{'='*60}\n")
        
    except Exception as e:
        logger.error(f"QA failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
