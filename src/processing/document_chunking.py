"""
Document chunking module for Hungarian descriptive texts.

This module handles chunking of descriptive documents (e.g., selmec.txt)
that contain structured text with headings, sections, and paragraphs.

Features:
- Sentence-aware chunking
- Document structure detection (headings, sections)
- Entity-aware chunking
- Context prefixing
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Hungarian document structure patterns
HEADING_PATTERN = re.compile(r'^([A-ZÁÉÍÓÖŐÚÜŰ][^:\n]+):?\s*$', re.MULTILINE)
LIST_ITEM_PATTERN = re.compile(r'^[-•*]\s+', re.MULTILINE)
NUMBERED_LIST_PATTERN = re.compile(r'^\d+[\.)]\s+', re.MULTILINE)

# Hungarian name pattern for entity detection
HUNGARIAN_NAME_PATTERN = re.compile(
    r'\b([A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+)\s+([A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+)'
    r'(?:\s+(?:a\.|alias|avagy)\s+[^.\n]+)?'
)


@dataclass
class HungarianEntity:
    """Represents a Hungarian entity (person, organization, etc.)"""
    normalized_name: str
    surface_forms: Set[str]
    entity_type: str  # 'személy', 'egyenruha', 'lista', 'szabályzat'
    context: str  # surrounding context


@dataclass
class DocumentStructure:
    """Represents Hungarian document structure"""
    headings: List[Tuple[int, str]]  # (position, heading_text)
    sections: List[Tuple[int, int, str]]  # (start, end, section_name)
    lists: List[Tuple[int, int, str]]  # (start, end, list_title)
    is_list: bool


def detect_hungarian_names(text: str) -> List[HungarianEntity]:
    """
    Detect Hungarian personal names in text, handling inflected forms.
    
    Args:
        text: Hungarian text to analyze
    
    Returns:
        List of HungarianEntity objects with normalized names
    """
    entities = {}
    
    # Find all potential name patterns
    for match in HUNGARIAN_NAME_PATTERN.finditer(text):
        full_match = match.group(0)
        surname = match.group(1)
        given_name = match.group(2)
        
        # Normalize the name
        normalized = f"{surname} {given_name}"
        
        # Extract alias/avagy if present
        alias_match = re.search(r'(?:a\.|alias|avagy)\s+([^.\n]+)', full_match)
        if alias_match:
            normalized += f" ({alias_match.group(1)})"
        
        # Store surface forms (inflected versions)
        if normalized not in entities:
            entities[normalized] = HungarianEntity(
                normalized_name=normalized,
                surface_forms=set(),
                entity_type='személy',
                context=''
            )
        
        entities[normalized].surface_forms.add(full_match)
    
    return list(entities.values())


def detect_document_structure(text: str) -> DocumentStructure:
    """
    Detect Hungarian document structure (headings, sections, lists).
    
    Args:
        text: Hungarian text to analyze
    
    Returns:
        DocumentStructure object
    """
    lines = text.split('\n')
    headings = []
    sections = []
    lists = []
    current_section = None
    current_list = None
    list_title = None
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # Detect headings (lines starting with capital, ending with colon or standalone)
        heading_match = HEADING_PATTERN.match(line_stripped)
        if heading_match and len(line_stripped) < 100:  # Reasonable heading length
            heading_text = heading_match.group(1).rstrip(':')
            headings.append((i, heading_text))
            
            # End current section/list
            if current_section is not None:
                sections.append((current_section, i, current_section_name))
            if current_list is not None:
                lists.append((current_list, i, list_title))
            
            # Start new section
            current_section = i
            current_section_name = heading_text
            current_list = None
            list_title = None
        
        # Detect list items
        elif LIST_ITEM_PATTERN.match(line_stripped) or NUMBERED_LIST_PATTERN.match(line_stripped):
            if current_list is None:
                # Start new list
                current_list = i
                list_title = current_section_name if current_section is not None else "Lista"
                if current_section is not None:
                    sections.append((current_section, i, current_section_name))
                    current_section = None
        else:
            # Regular text - end list if we were in one
            if current_list is not None:
                lists.append((current_list, i, list_title))
                current_list = None
    
    # Close open sections/lists
    if current_section is not None:
        sections.append((current_section, len(lines), current_section_name))
    if current_list is not None:
        lists.append((current_list, len(lines), list_title))
    
    is_list = len(lists) > 0 and len([l for l in lists if l[1] - l[0] > 3]) > 0
    
    return DocumentStructure(
        headings=headings,
        sections=sections,
        lists=lists,
        is_list=is_list
    )


def build_context_prefix(
    entity_type: str,
    entity_name: Optional[str],
    document_type: str,
    section: Optional[str],
    language: str = "HU"
) -> str:
    """
    Build Hungarian-optimized context prefix for chunk.
    
    Args:
        entity_type: Type of entity (személy, egyenruha, lista, szabályzat)
        entity_name: Normalized entity name
        document_type: Source document name
        section: Section name
        language: Language code (default: HU)
    
    Returns:
        Context prefix string
    """
    prefix_parts = [f"[LANGUAGE: {language}]"]
    prefix_parts.append(f"[ENTITY_TYPE: {entity_type}]")
    
    if entity_name:
        prefix_parts.append(f"[ENTITY_NAME: {entity_name}]")
    
    prefix_parts.append(f"[DOCUMENT_TYPE: {document_type}]")
    
    if section:
        prefix_parts.append(f"[SECTION: {section}]")
    
    return " ".join(prefix_parts) + "\n\n"


def generate_chunk_metadata(
    chunk_text: str,
    entities: List[HungarianEntity],
    document_type: str,
    section: Optional[str],
    is_list: bool
) -> Dict[str, Any]:
    """
    Generate comprehensive metadata for a chunk.
    
    Args:
        chunk_text: Chunk text content
        entities: Detected Hungarian entities
        document_type: Source document name
        section: Section name
        is_list: Whether chunk contains a list
    
    Returns:
        Metadata dictionary
    """
    person_names = [e for e in entities if e.entity_type == 'személy']
    
    metadata = {
        'language': 'hu',
        'entity_type': entities[0].entity_type if entities else 'szabályzat',
        'document_type': document_type,
        'contains_person_names': len(person_names) > 0,
        'is_list': is_list,
    }
    
    if person_names:
        metadata['entity_name_normalized'] = person_names[0].normalized_name
        metadata['entity_name_surface_forms'] = list(person_names[0].surface_forms)
    
    if section:
        metadata['section'] = section
    
    # Detect if chunk mentions uniforms/egyenruha
    if re.search(r'egyenruha|viselet|gruben|aufhauer|csákó|nadrág', chunk_text, re.IGNORECASE):
        metadata['entity_type'] = 'egyenruha'
    
    return metadata


def chunk_document(
    text: str,
    chunk_size: int,
    overlap: int,
    encoder,
    source_file: str = "unknown"
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Chunk a Hungarian descriptive document (e.g., selmec.txt).
    
    This function implements document-aware chunking:
    - Sentence-aware chunking
    - Structure-respecting (respects headings, sections)
    - Entity-aware chunking
    - Context prefixing
    - Metadata generation
    
    Args:
        text: Hungarian text to chunk
        chunk_size: Target chunk size in tokens (250-400 recommended)
        overlap: Overlap tokens (40-60 recommended)
        encoder: Tiktoken encoder instance
        source_file: Source document name for metadata
    
    Returns:
        List of (chunk_text, metadata) tuples
    """
    # Hard maximum: 600 tokens
    hard_max = min(600, chunk_size * 1.5)
    
    # Detect Hungarian entities
    entities = detect_hungarian_names(text)
    logger.debug(f"Detected {len(entities)} Hungarian entities")
    
    # Detect document structure
    structure = detect_document_structure(text)
    logger.debug(f"Detected {len(structure.headings)} headings, {len(structure.sections)} sections, {len(structure.lists)} lists")
    
    # Split into sentences (Hungarian-aware)
    # Hungarian sentence endings: . ! ? followed by space and capital or newline
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÖŐÚÜŰ])|(?<=[.!?])\n+(?=[A-ZÁÉÍÓÖŐÚÜŰ])'
    sentences = re.split(sentence_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    chunks_with_metadata = []
    current_chunk_sentences = []
    current_tokens = 0
    current_section = None
    current_list_title = None
    current_entities = set()
    
    def count_tokens(t: str) -> int:
        return len(encoder.encode(t))
    
    # Map sentence positions to line numbers for section/list detection
    lines = text.split('\n')
    sentence_to_line_map = {}
    current_char = 0
    
    for i, sentence in enumerate(sentences):
        # Find which line this sentence starts on
        sentence_start = text.find(sentence, current_char)
        if sentence_start >= 0:
            # Count newlines up to this position
            line_num = text[:sentence_start].count('\n')
            sentence_to_line_map[i] = line_num
            current_char = sentence_start + len(sentence)
        else:
            # Fallback: estimate based on previous
            sentence_to_line_map[i] = sentence_to_line_map.get(i-1, 0) + 1
    
    def get_section_for_position(sent_pos: int) -> Optional[str]:
        """Get section name for sentence position."""
        line_num = sentence_to_line_map.get(sent_pos, 0)
        for start, end, name in structure.sections:
            if start <= line_num <= end:
                # Fix: don't use personal names as section titles
                if HUNGARIAN_NAME_PATTERN.search(name):
                    return None
                return name
        return None
    
    def get_list_title_for_position(sent_pos: int) -> Optional[str]:
        """Get list title for sentence position."""
        line_num = sentence_to_line_map.get(sent_pos, 0)
        for start, end, title in structure.lists:
            if start <= line_num <= end:
                return title
        return None
    
    def finalize_chunk(sentences_list: List[str], section: Optional[str], list_title: Optional[str]) -> Tuple[str, Dict[str, Any]]:
        """Create chunk with prefix and metadata."""
        chunk_text = ' '.join(sentences_list)
        
        # Determine entity type and name
        chunk_entities = [e for e in entities if any(sf in chunk_text for sf in e.surface_forms)]
        entity_type = 'személy' if chunk_entities else ('lista' if list_title else 'szabályzat')
        entity_name = chunk_entities[0].normalized_name if chunk_entities else None
        
        # Build context prefix
        prefix = build_context_prefix(
            entity_type=entity_type,
            entity_name=entity_name,
            document_type=source_file,
            section=section or list_title
        )
        
        # Combine prefix with chunk text
        full_chunk = prefix + chunk_text
        
        # Generate metadata
        metadata = generate_chunk_metadata(
            chunk_text=chunk_text,
            entities=chunk_entities,
            document_type=source_file,
            section=section or list_title,
            is_list=list_title is not None
        )
        
        return full_chunk, metadata
    
    # Process sentences
    for i, sentence in enumerate(sentences):
        sent_tokens = count_tokens(sentence)
        
        # Detect entities in this sentence
        sent_entities = {e.normalized_name for e in entities if any(sf in sentence for sf in e.surface_forms)}
        
        # Get section and list context
        section = get_section_for_position(i)
        list_title = get_list_title_for_position(i)
        
        # Fix section: don't use personal names as section titles
        if section and sent_entities:
            for entity_name in sent_entities:
                if entity_name in section or any(part in section for part in entity_name.split()):
                    section = None
                    break
        
        # If sentence is too large, split it (last resort)
        if sent_tokens > hard_max:
            logger.warning(f"Sentence too large ({sent_tokens} tokens), splitting by tokens")
            if current_chunk_sentences:
                chunks_with_metadata.append(finalize_chunk(
                    current_chunk_sentences, current_section, current_list_title
                ))
                current_chunk_sentences = []
                current_tokens = 0
            
            # Split large sentence - preserve entity information in sub-chunks
            sent_entities_list = [e for e in entities if any(sf in sentence for sf in e.surface_forms)]
            entity_type_for_split = 'személy' if sent_entities_list else ('lista' if list_title else 'szabályzat')
            entity_name_for_split = sent_entities_list[0].normalized_name if sent_entities_list else None
            
            sent_tokens_list = encoder.encode(sentence)
            sent_start = 0
            while sent_start < len(sent_tokens_list):
                sent_end = min(sent_start + chunk_size, len(sent_tokens_list))
                sent_chunk_tokens = sent_tokens_list[sent_start:sent_end]
                sent_chunk_text = encoder.decode(sent_chunk_tokens).strip()
                if sent_chunk_text:
                    prefix = build_context_prefix(
                        entity_type=entity_type_for_split,
                        entity_name=entity_name_for_split,
                        document_type=source_file,
                        section=section or list_title
                    )
                    chunks_with_metadata.append((
                        prefix + sent_chunk_text,
                        generate_chunk_metadata(sent_chunk_text, sent_entities_list, source_file, section or list_title, False)
                    ))
                sent_start += chunk_size - overlap
            continue
        
        # Check if adding this sentence would exceed chunk size
        would_exceed = current_tokens + sent_tokens > chunk_size
        
        # If sentence introduces new entities, try to keep them together
        new_entities = sent_entities - current_entities
        
        if would_exceed and not new_entities and current_chunk_sentences:
            # Finalize current chunk
            chunks_with_metadata.append(finalize_chunk(
                current_chunk_sentences, current_section, current_list_title
            ))
            
            # Start new chunk with overlap
            if overlap > 0 and current_chunk_sentences:
                overlap_sentences = []
                overlap_tokens = 0
                for prev_sent in reversed(current_chunk_sentences):
                    prev_tokens = count_tokens(prev_sent)
                    if overlap_tokens + prev_tokens <= overlap:
                        overlap_sentences.insert(0, prev_sent)
                        overlap_tokens += prev_tokens
                    else:
                        break
                
                if overlap_sentences:
                    current_chunk_sentences = overlap_sentences + [sentence]
                    current_tokens = overlap_tokens + sent_tokens
                else:
                    current_chunk_sentences = [sentence]
                    current_tokens = sent_tokens
            else:
                current_chunk_sentences = [sentence]
                current_tokens = sent_tokens
            
            # Update section/list context
            current_section = section
            current_list_title = list_title
            current_entities = sent_entities
        else:
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_tokens += sent_tokens
            current_entities.update(sent_entities)
            
            # Update section/list if changed
            if section and section != current_section:
                current_section = section
            if list_title and list_title != current_list_title:
                current_list_title = list_title
            
            # If we're approaching hard max, finalize
            if current_tokens > hard_max and len(current_chunk_sentences) > 1:
                # Remove last sentence and finalize
                last_sent = current_chunk_sentences.pop()
                chunks_with_metadata.append(finalize_chunk(
                    current_chunk_sentences, current_section, current_list_title
                ))
                current_chunk_sentences = [last_sent]
                current_tokens = count_tokens(last_sent)
                current_entities = {e.normalized_name for e in entities if any(sf in last_sent for sf in e.surface_forms)}
    
    # Add final chunk
    if current_chunk_sentences:
        chunks_with_metadata.append(finalize_chunk(
            current_chunk_sentences, current_section, current_list_title
        ))
    
    logger.info(f"Created {len(chunks_with_metadata)} document chunks")
    return chunks_with_metadata
