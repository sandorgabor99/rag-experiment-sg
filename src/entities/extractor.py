"""
Entity extraction module for extracting structured information from chunks.

This module extracts entities like people, locations, dates, and organizations
from text chunks using NER (Named Entity Recognition).

Supports:
- spaCy (fast, accurate, requires model download)
- LLM-based extraction (more flexible, slower)
- Hybrid approach (spaCy + LLM refinement)
"""

import logging
from typing import Dict, List, Optional, Set, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from src.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Try to import spaCy
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Install with: pip install spacy huspacy && python -m huspacy download")

# Try to import HuSpaCy for Hungarian models
try:
    import huspacy
    HUSPACY_AVAILABLE = True
except ImportError:
    HUSPACY_AVAILABLE = False
    logger.warning("HuSpaCy not available. For Hungarian models, install with: pip install huspacy && python -m huspacy download")

# Try to import LLM-based extraction
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.language_models import BaseChatModel
    from pydantic import BaseModel, Field
    LLM_EXTRACTION_AVAILABLE = True
except ImportError:
    LLM_EXTRACTION_AVAILABLE = False


class ExtractedEntities(BaseModel):
    """Pydantic model for extracted entities."""
    people: List[str] = Field(default_factory=list, description="List of person names")
    locations: List[str] = Field(default_factory=list, description="List of location names")
    organizations: List[str] = Field(default_factory=list, description="List of organization names")
    dates: List[str] = Field(default_factory=list, description="List of dates mentioned")
    events: List[str] = Field(default_factory=list, description="List of events mentioned")


class EntityExtractor:
    """
    Extracts entities from text chunks.
    
    Supports multiple extraction methods:
    - spaCy NER (fast, deterministic)
    - LLM-based (flexible, can extract custom entities)
    - Hybrid (spaCy + LLM refinement)
    """
    
    def __init__(
        self,
        method: str = 'spacy',
        model: str = 'hu_core_news_lg',  # Default to Hungarian large model
        llm: Optional[BaseChatModel] = None
    ):
        """
        Initialize entity extractor.
        
        Args:
            method: Extraction method - 'spacy', 'llm', or 'hybrid'
            model: spaCy model name (default: 'hu_core_news_lg' for Hungarian)
                   Options: 'hu_core_news_lg', 'hu_core_news_md', 'hu_core_news_trf', 'en_core_web_sm'
            llm: LLM instance for LLM-based extraction (required if method is 'llm' or 'hybrid')
        """
        self.method = method
        self.llm = llm
        self.nlp = None
        
        if method in ['spacy', 'hybrid']:
            if not SPACY_AVAILABLE:
                raise ImportError(
                    "spaCy is required for 'spacy' or 'hybrid' methods. "
                    "Install with: pip install spacy huspacy && python -m huspacy download"
                )
            try:
                # Check if it's a Hungarian model
                if model.startswith('hu_core_news'):
                    if not HUSPACY_AVAILABLE:
                        logger.warning(
                            f"HuSpaCy not available but Hungarian model '{model}' requested. "
                            "Install with: pip install huspacy && python -m huspacy download"
                        )
                    self.nlp = spacy.load(model)
                    logger.info(f"Loaded Hungarian spaCy model: {model}")
                else:
                    self.nlp = spacy.load(model)
                    logger.info(f"Loaded spaCy model: {model}")
            except OSError as e:
                if model.startswith('hu_core_news'):
                    logger.error(
                        f"Hungarian spaCy model '{model}' not found.\n"
                        "Install with: pip install huspacy && python -m huspacy download\n"
                        "Or download specific model: pip install hu_core_news_lg@https://huggingface.co/huspacy/hu_core_news_lg/resolve/main/hu_core_news_lg-any-py3-none-any.whl"
                    )
                else:
                    logger.error(f"spaCy model '{model}' not found. Download with: python -m spacy download {model}")
                raise
        
        if method in ['llm', 'hybrid']:
            if llm is None:
                raise ValueError("LLM instance required for 'llm' or 'hybrid' methods")
            logger.info("Using LLM-based entity extraction")
        
        logger.info(f"Initialized EntityExtractor with method: {method}")
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text.
        
        Args:
            text: Input text to extract entities from
        
        Returns:
            Dictionary with entity types as keys and lists of entities as values
        """
        if self.method == 'spacy':
            return self._extract_with_spacy(text)
        elif self.method == 'llm':
            return self._extract_with_llm(text)
        elif self.method == 'hybrid':
            return self._extract_hybrid(text)
        else:
            raise ValueError(f"Unknown extraction method: {self.method}")
    
    def _extract_with_spacy(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using spaCy NER."""
        doc = self.nlp(text)
        
        entities = {
            'people': [],
            'locations': [],
            'organizations': [],
            'dates': [],
            'events': []
        }
        
        seen = set()  # Deduplicate entities
        
        # Common false positive locations (Hungarian words that are not locations)
        false_positive_locations = {
            'kegyelem', 'mercy', 'egyike', 'egyik', 'tag', 'tagok', 
            'alapító', 'összes', 'aktív', 'passzív', 'személy',
            'alias', 'becenevén', 'beceneveit', 'avagy', 'a.', 'alias:',
            'nem', 'iszok', 'csak', 'ha', 'vedelek', 'akkor', 'nincs'
        }
        
        # Hungarian models use different labels - support both English and Hungarian labels
        # English: PERSON, GPE, LOC, FAC, ORG, DATE, TIME, EVENT
        # Hungarian: typically uses same labels but may vary by model
        
        for ent in doc.ents:
            entity_text = ent.text.strip()
            if entity_text in seen:
                continue
            seen.add(entity_text)
            
            # Filter out false positives
            entity_lower = entity_text.lower()
            if entity_lower in false_positive_locations:
                continue
            
            # People: PERSON label (works for both English and Hungarian models)
            if ent.label_ in ['PERSON']:
                entities['people'].append(entity_text)
            # Locations: GPE (Geopolitical), LOC (Location), FAC (Facility)
            # Hungarian models should properly classify these
            elif ent.label_ in ['GPE', 'LOC', 'FAC']:
                # Additional validation: skip if it's a common Hungarian word or very short
                if len(entity_text) < 3 or entity_lower in false_positive_locations:
                    continue
                entities['locations'].append(entity_text)
            # Organizations: ORG label
            elif ent.label_ in ['ORG']:
                entities['organizations'].append(entity_text)
            # Dates and times
            elif ent.label_ in ['DATE', 'TIME']:
                entities['dates'].append(entity_text)
            # Events
            elif ent.label_ in ['EVENT']:
                entities['events'].append(entity_text)
        
        # Post-process: remove any locations that are clearly not geographical places
        filtered_locations = []
        for loc in entities['locations']:
            loc_lower = loc.lower()
            # Skip if it's in our false positive list or looks like a common word
            if loc_lower not in false_positive_locations and len(loc) >= 3:
                # Additional check: if it's all lowercase and short, it's probably not a location
                if not (loc.islower() and len(loc) < 5):
                    filtered_locations.append(loc)
        
        entities['locations'] = filtered_locations
        return entities
    
    def _extract_with_llm(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using LLM."""
        PROMPT = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                Extract entities from the given text. Return a structured list of:
                - People: Names of people mentioned
                - Locations: Places, cities, countries, geographical locations (NOT abstract concepts like "mercy", "kegyelem")
                - Organizations: Companies, institutions, groups
                - Dates: Specific dates or time periods mentioned
                - Events: Significant events or occurrences
                
                IMPORTANT: 
                - Only extract actual geographical locations (cities, countries, regions, buildings, addresses)
                - Do NOT extract abstract concepts, emotions, or common words as locations
                - For Hungarian text, be especially careful with words like "kegyelem" (mercy), "egyike" (one of), etc.
                - Only extract entities that are explicitly mentioned in the text.
                - Return empty lists if no entities of a type are found.
                """
            ),
            ("user", "Text:\n{text}")
        ])
        
        runnable = PROMPT | self.llm.with_structured_output(ExtractedEntities)
        
        try:
            result = runnable.invoke({"text": text[:2000]})  # Limit text length
            return {
                'people': result.people or [],
                'locations': result.locations or [],
                'organizations': result.organizations or [],
                'dates': result.dates or [],
                'events': result.events or []
            }
        except Exception as e:
            logger.warning(f"Error extracting entities with LLM: {e}")
            return {
                'people': [],
                'locations': [],
                'organizations': [],
                'dates': [],
                'events': []
            }
    
    def _extract_hybrid(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using spaCy first, then refine with LLM."""
        # Start with spaCy extraction
        spacy_entities = self._extract_with_spacy(text)
        
        # Use LLM to find additional entities or refine
        llm_entities = self._extract_with_llm(text)
        
        # Merge results (spaCy takes precedence, LLM adds missing)
        merged = {
            'people': list(set(spacy_entities['people'] + llm_entities['people'])),
            'locations': list(set(spacy_entities['locations'] + llm_entities['locations'])),
            'organizations': list(set(spacy_entities['organizations'] + llm_entities['organizations'])),
            'dates': list(set(spacy_entities['dates'] + llm_entities['dates'])),
            'events': list(set(spacy_entities['events'] + llm_entities['events']))
        }
        
        return merged
    
    def extract_from_chunks(
        self,
        chunks: List[Dict],
        batch_size: int = 10
    ) -> List[Dict]:
        """
        Extract entities from a list of chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            batch_size: Number of chunks to process at once (for LLM method)
        
        Returns:
            List of chunks with added 'entities' field
        """
        logger.info(f"Extracting entities from {len(chunks)} chunks...")
        
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '')
            if not text:
                chunk['entities'] = {
                    'people': [],
                    'locations': [],
                    'organizations': [],
                    'dates': [],
                    'events': []
                }
                enriched_chunks.append(chunk)
                continue
            
            entities = self.extract_entities(text)
            chunk['entities'] = entities
            enriched_chunks.append(chunk)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(chunks)} chunks")
        
        logger.info(f"Entity extraction complete")
        return enriched_chunks


def extract_entities_from_chunks(
    chunks: List[Dict],
    method: Optional[str] = None,
    model: Optional[str] = None,
    llm: Optional[BaseChatModel] = None,
    config: Optional['Config'] = None
) -> List[Dict]:
    """
    Convenience function to extract entities from chunks.
    
    Args:
        chunks: List of chunk dictionaries
        method: Extraction method - 'spacy', 'llm', or 'hybrid'
        model: spaCy model name
        llm: LLM instance (required for 'llm' or 'hybrid')
    
    Returns:
        Chunks with added 'entities' field
    """
    extractor = EntityExtractor(method=method, model=model, llm=llm, config=config)
    return extractor.extract_from_chunks(chunks)


if __name__ == "__main__":
    # Example usage
    from knowledge_layer import read_knowledge_entries
    
    chunks = read_knowledge_entries(Path("chunks/chunks.jsonl"))
    
    # Extract with spaCy (fast, free)
    extractor = EntityExtractor(method='spacy')
    enriched_chunks = extractor.extract_from_chunks(chunks[:10])  # Test on first 10
    
    # Print results
    for chunk in enriched_chunks[:3]:
        print(f"\nChunk ID: {chunk.get('id')}")
        print(f"People: {chunk.get('entities', {}).get('people', [])}")
        print(f"Locations: {chunk.get('entities', {}).get('locations', [])}")
