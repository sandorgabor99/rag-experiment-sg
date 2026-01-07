"""
Person and uniform chunking module for Hungarian data.

This module handles chunking of:
1. Person lists (alapitok.txt, osszestag.txt, alapitok.json)
2. Uniform/equipment data (egyenruhak.json)

Features:
- Detects name lists (>3 Hungarian names)
- Creates one list chunk + one person chunk per person
- Handles Hungarian aliases ("a.", "avagy")
- Normalizes names and surface forms
- Supports both text and JSON input formats
- Chunks hierarchical uniform JSON structures with proper metadata
"""

import json
import logging
import re
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# Hungarian name pattern: Capitalized word + Capitalized word (with optional alias/avagy)
HUNGARIAN_NAME_PATTERN = re.compile(
    r'\b([A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+)\s+([A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+)'
    r'(?:\s+(?:a\.|alias|avagy)\s+[^.\n]+)?'
)


def detect_name_list(text: str) -> Tuple[bool, Optional[str], List[Dict[str, Any]]]:
    """
    Detect if a text paragraph contains a list of Hungarian personal names.
    
    A list is detected when:
    - Contains more than 3 Hungarian personal names
    - Names are separated by commas or periods
    - Aliases are introduced by "a." or "avagy"
    
    Args:
        text: Text paragraph to analyze
    
    Returns:
        Tuple of (is_list, list_title, list_of_persons)
        - is_list: True if this is a name list
        - list_title: Semantic title for the list (e.g., "Alapító tagok")
        - list_of_persons: List of dicts with person info (name, alias, full_text)
    """
    # Find all name patterns in the text
    name_matches = list(HUNGARIAN_NAME_PATTERN.finditer(text))
    
    if len(name_matches) <= 3:
        return False, None, []
    
    # Check if names are separated by commas or periods (list pattern)
    # Count comma/period separators between names
    text_between_names = []
    for i in range(len(name_matches) - 1):
        start = name_matches[i].end()
        end = name_matches[i + 1].start()
        between = text[start:end]
        text_between_names.append(between.strip())
    
    # If most separators contain commas or periods, it's likely a list
    comma_or_period_count = sum(1 for sep in text_between_names if ',' in sep or '.' in sep)
    if comma_or_period_count < len(text_between_names) * 0.5:
        return False, None, []
    
    # Extract list title (first line or heading before the list)
    lines = text.split('\n')
    list_title = None
    for line in lines[:3]:  # Check first few lines
        line_stripped = line.strip()
        if line_stripped and not HUNGARIAN_NAME_PATTERN.match(line_stripped):
            # This might be a title
            # Remove trailing punctuation
            potential_title = line_stripped.rstrip('.:')
            if len(potential_title) < 50 and not any(HUNGARIAN_NAME_PATTERN.finditer(potential_title)):
                list_title = potential_title
                break
    
    # Default titles if none found
    if not list_title:
        if 'alapító' in text.lower():
            list_title = "Alapító tagok"
        elif 'tag' in text.lower():
            list_title = "Tagok felsorolása"
        else:
            list_title = "Névsor"
    
    # Extract individual persons from the list
    # Split text by lines first (each line typically contains one person)
    lines = text.split('\n')
    persons = []
    seen_names = set()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip if this line is the title
        if not HUNGARIAN_NAME_PATTERN.search(line):
            continue
        
        # Pattern 1: Name a. Alias (with period at end)
        # e.g., "Csuhaj Péter a. Boromissza IhajCsuhaj Gergely."
        pattern1 = re.compile(
            r'^([A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+(?:\s+[A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+)+)\s+a\.\s+([^.,]+?)(?:\.|,|$)',
            re.IGNORECASE
        )
        
        # Pattern 2: Name avagy Alias (with period at end)
        # e.g., "Kiss Sándor a. Sanya, aki a tortát tapossa laposra, avagy a Láncravert Hajzuhatag."
        pattern2 = re.compile(
            r'^([A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+(?:\s+[A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+)+)\s+.*?avagy\s+([^.,]+?)(?:\.|,|$)',
            re.IGNORECASE
        )
        
        # Pattern 3: Simple name without alias
        # e.g., "Virág Zoltán a. Zozó."
        pattern3 = re.compile(
            r'^([A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+(?:\s+[A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+)+)\s+a\.\s+([^.,]+?)\.$',
            re.IGNORECASE
        )
        
        match1 = pattern1.match(line)
        match2 = pattern2.match(line)
        match3 = pattern3.match(line)
        
        if match1:
            # Pattern 1: Name a. Alias
            full_name = match1.group(1).strip()
            alias = match1.group(2).strip().rstrip('.,')
            full_text = line.rstrip('.,')
            
            if full_name not in seen_names:
                seen_names.add(full_name)
                persons.append({
                    'normalized_name': full_name,  # Base name without alias
                    'base_name': full_name,
                    'alias': alias,
                    'full_text': full_text,
                    'surface_forms': [
                        full_text,
                        f"{full_name} a. {alias}",
                        f"{full_name} avagy {alias}"
                    ]
                })
        elif match2:
            # Pattern 2: Name ... avagy Alias
            full_name = match2.group(1).strip()
            alias = match2.group(2).strip().rstrip('.,')
            full_text = line.rstrip('.,')
            
            if full_name not in seen_names:
                seen_names.add(full_name)
                persons.append({
                    'normalized_name': full_name,
                    'base_name': full_name,
                    'alias': alias,
                    'full_text': full_text,
                    'surface_forms': [
                        full_text,
                        f"{full_name} a. {alias}",
                        f"{full_name} avagy {alias}"
                    ]
                })
        elif match3:
            # Pattern 3: Simple Name a. Alias (shorter form)
            full_name = match3.group(1).strip()
            alias = match3.group(2).strip().rstrip('.,')
            full_text = line.rstrip('.,')
            
            if full_name not in seen_names:
                seen_names.add(full_name)
                persons.append({
                    'normalized_name': full_name,
                    'base_name': full_name,
                    'alias': alias,
                    'full_text': full_text,
                    'surface_forms': [
                        full_text,
                        f"{full_name} a. {alias}",
                        f"{full_name} avagy {alias}"
                    ]
                })
    
    # If we found persons, it's a list
    if len(persons) > 3:
        return True, list_title, persons
    
    return False, None, []


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


def chunk_person_list(
    text: str,
    source_file: str,
    normalized_section: str
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Chunk a Hungarian name list into list chunk + individual person chunks.
    
    Args:
        text: Text containing the name list
        source_file: Source document name
        normalized_section: Normalized section title
    
    Returns:
        List of (chunk_text, metadata) tuples
    """
    chunks_with_metadata = []
    
    # Detect if this is a name list
    is_name_list, list_title_detected, persons_in_list = detect_name_list(text)
    
    if not is_name_list or not persons_in_list:
        return []
    
    # A) Create ONE LIST chunk (describes what the list represents)
    list_chunk_text = f"{list_title_detected}. Ez a lista tartalmazza a tagok neveit és aliasait."
    doc_type = source_file.replace('.json', '').replace('.txt', '').strip()
    list_metadata = {
        'entity_type': 'lista',
        'name': list_title_detected,
        'aliases': [],
        'document_type': doc_type,
        'section': normalized_section,
        'text': list_chunk_text
    }
    chunks_with_metadata.append((list_chunk_text, list_metadata))
    logger.debug(f"Created list chunk: {list_title_detected} ({len(persons_in_list)} persons)")
    
    # B) Create ONE PERSON chunk PER PERSON
    for person in persons_in_list:
        # Extract canonical base name (ONLY the Hungarian personal name, no aliases)
        base_name = person['base_name'].strip()
        alias = person.get('alias', '').strip() if person.get('alias') else None
        
        # Normalize base name (trim whitespace, no trailing dots)
        base_name_normalized = ' '.join(base_name.split()).strip().rstrip('.')
        
        # Parse aliases: split by "avagy" and commas, clean up
        aliases_list = []
        if alias:
            # Remove "alias " prefix if present
            alias_clean = alias.replace('alias ', '', 1).strip() if alias.startswith('alias ') else alias.strip()
            # Split by "avagy" first
            parts = [p.strip() for p in alias_clean.split('avagy')]
            # Then split each part by commas
            all_aliases = []
            for part in parts:
                comma_parts = [p.strip().rstrip(',.') for p in part.split(',')]
                all_aliases.extend(comma_parts)
            # Clean and deduplicate
            for a in all_aliases:
                a_clean = a.strip().rstrip(',.')
                if a_clean and a_clean not in aliases_list:
                    aliases_list.append(a_clean)
        
        # Build text body: "<name> alias <first_alias>, becenevén <other_aliases>."
        if aliases_list:
            first_alias = aliases_list[0]
            if len(aliases_list) > 1:
                other_aliases = ", ".join(aliases_list[1:])
                person_text = f"{base_name_normalized} alias {first_alias}, becenevén {other_aliases}."
            else:
                person_text = f"{base_name_normalized} alias {first_alias}."
        else:
            person_text = f"{base_name_normalized}."
        
        # Extract document_type base name (without extension)
        doc_type = source_file.replace('.json', '').replace('.txt', '').strip()
        
        # Generate metadata in new format
        person_metadata = {
            'entity_type': 'személy',
            'name': base_name_normalized,
            'aliases': aliases_list,
            'document_type': doc_type,
            'section': normalized_section,
            'text': person_text
        }
        
        chunks_with_metadata.append((person_text, person_metadata))
    
    logger.info(f"Split list into 1 list chunk + {len(persons_in_list)} person chunks")
    return chunks_with_metadata


def chunk_person_json(
    json_data: Dict[str, Any],
    source_file: str,
    list_key: str = "founders",
    list_title: Optional[str] = None
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Chunk a JSON structure containing a list of persons into list chunk + individual person chunks.
    
    Expected JSON structure:
    {
        "founders": [
            {
                "full_name": "Csuhaj Péter",
                "alias": "Boromissza IhajCsuhaj Gergely"
            },
            ...
        ]
    }
    
    Args:
        json_data: JSON dictionary containing person list
        source_file: Source document name
        list_key: Key in JSON that contains the person list (default: "founders")
        list_title: Optional explicit list title (if None, will be inferred from list_key)
    
    Returns:
        List of (chunk_text, metadata) tuples
    """
    chunks_with_metadata = []
    
    # Extract person list from JSON
    if list_key not in json_data:
        logger.warning(f"Key '{list_key}' not found in JSON. Available keys: {list(json_data.keys())}")
        return []
    
    persons_list = json_data[list_key]
    if not isinstance(persons_list, list):
        logger.warning(f"Value for key '{list_key}' is not a list: {type(persons_list)}")
        return []
    
    if len(persons_list) == 0:
        logger.warning(f"Person list is empty")
        return []
    
    # Determine list title
    if not list_title:
        # Infer from list_key
        if list_key == "founders":
            list_title = "Alapító tagok"
        elif list_key == "all_members" or "osszestag" in source_file.lower():
            list_title = "Összes aktív és passzív tag"
        elif "tag" in list_key.lower():
            list_title = "Tagok felsorolása"
        else:
            list_title = "Névsor"
    
    normalized_section = list_title.rstrip('.,;:!?').strip()
    
    # A) Create ONE LIST chunk
    list_chunk_text = f"{normalized_section}. Ez a lista tartalmazza a tagok neveit és aliasait."
    doc_type = source_file.replace('.json', '').replace('.txt', '').strip()
    list_metadata = {
        'entity_type': 'lista',
        'name': normalized_section,
        'aliases': [],
        'document_type': doc_type,
        'section': normalized_section,
        'text': list_chunk_text
    }
    chunks_with_metadata.append((list_chunk_text, list_metadata))
    
    # B) Create ONE PERSON chunk PER PERSON
    for person_data in persons_list:
        if not isinstance(person_data, dict):
            logger.warning(f"Skipping invalid person data (not a dict): {person_data}")
            continue
        
        # Extract full_name/name and alias from JSON
        # Support both "full_name" and "name" fields
        full_name = person_data.get('full_name', '').strip() or person_data.get('name', '').strip()
        alias_raw = person_data.get('alias', '').strip() if person_data.get('alias') else None
        
        # Handle alias prefix (some JSON files have "alias " prefix)
        alias = None
        if alias_raw:
            # Remove "alias " prefix if present
            alias = alias_raw.replace('alias ', '', 1).strip() if alias_raw.startswith('alias ') else alias_raw.strip()
            if not alias:  # If alias becomes empty after stripping, set to None
                alias = None
        
        if not full_name:
            logger.warning(f"Skipping person with missing full_name/name: {person_data}")
            continue
        
        # Normalize base name (trim whitespace, no trailing dots)
        base_name_normalized = ' '.join(full_name.split()).strip().rstrip('.')
        
        # Parse aliases: split by "avagy" and commas, clean up
        aliases_list = []
        if alias:
            # Remove "alias " prefix if present
            alias_clean = alias.replace('alias ', '', 1).strip() if alias.startswith('alias ') else alias.strip()
            # Split by "avagy" first
            parts = [p.strip() for p in alias_clean.split('avagy')]
            # Then split each part by commas
            all_aliases = []
            for part in parts:
                comma_parts = [p.strip().rstrip(',.') for p in part.split(',')]
                all_aliases.extend(comma_parts)
            # Clean and deduplicate
            for a in all_aliases:
                a_clean = a.strip().rstrip(',.')
                if a_clean and a_clean not in aliases_list:
                    aliases_list.append(a_clean)
        
        # Build text body: "<name> alias <first_alias>, becenevén <other_aliases>."
        if aliases_list:
            first_alias = aliases_list[0]
            if len(aliases_list) > 1:
                other_aliases = ", ".join(aliases_list[1:])
                person_text = f"{base_name_normalized} alias {first_alias}, becenevén {other_aliases}."
            else:
                person_text = f"{base_name_normalized} alias {first_alias}."
        else:
            person_text = f"{base_name_normalized}."
        
        # Extract document_type base name (without extension)
        doc_type = source_file.replace('.json', '').replace('.txt', '').strip()
        
        # Generate metadata in new format
        person_metadata = {
            'entity_type': 'személy',
            'name': base_name_normalized,
            'aliases': aliases_list,
            'document_type': doc_type,
            'section': normalized_section,
            'text': person_text
        }
        
        chunks_with_metadata.append((person_text, person_metadata))
    
    logger.info(f"Split JSON list into 1 list chunk + {len(persons_list)} person chunks")
    return chunks_with_metadata


def chunk_uniform_json(
    json_data: Dict[str, Any],
    source_file: str
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Chunk a hierarchical JSON structure containing uniform/equipment data.
    
    Expected JSON structure:
    {
        "SelmeciUtodiskolak": {
            "osszeallito": "...",
            "banyaszMunkaruha": {...},
            "diszegyenruha": {...},
            "banyamernokHallgatok": {...},
            "egyebEgyenruhak": {...},
            "egyetemekEsKarok": {...}
        }
    }
    
    Args:
        json_data: JSON dictionary containing uniform data
        source_file: Source document name
    
    Returns:
        List of (chunk_text, metadata) tuples
    """
    chunks_with_metadata = []
    
    # Detect uniform JSON structure
    uniform_key = None
    possible_keys = ['SelmeciUtodiskolak', 'egyenruhak', 'uniforms']
    for key in possible_keys:
        if key in json_data:
            uniform_key = key
            break
    
    if not uniform_key:
        logger.warning(f"Uniform JSON structure not found. Available keys: {list(json_data.keys())}")
        return []
    
    uniform_data = json_data[uniform_key]
    if not isinstance(uniform_data, dict):
        logger.warning(f"Uniform data is not a dictionary: {type(uniform_data)}")
        return []
    
    # Extract compiler/creator if available
    compiler = uniform_data.get('osszeallito', '')
    base_section = "Egyenruhák"
    
    # Helper function to create uniform chunk
    def create_uniform_chunk(
        title: str,
        description: str,
        elements: Optional[List[str]] = None,
        additional_info: Optional[Dict[str, Any]] = None,
        section: str = base_section,
        entity_name: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Create a uniform chunk with proper metadata."""
        # Build text content
        text_parts = []
        
        if description:
            text_parts.append(description)
        
        if elements:
            if isinstance(elements, list):
                elements_text = ", ".join(elements)
                text_parts.append(f"Elemei: {elements_text}.")
            else:
                text_parts.append(f"Elemei: {elements}.")
        
        if additional_info:
            for key, value in additional_info.items():
                if isinstance(value, list):
                    value_text = ", ".join(str(v) for v in value)
                elif isinstance(value, dict):
                    value_text = ", ".join(f"{k}: {v}" for k, v in value.items())
                else:
                    value_text = str(value)
                text_parts.append(f"{key}: {value_text}.")
        
        chunk_text = " ".join(text_parts)
        
        # Use title as entity name if not provided
        entity_name_normalized = entity_name or title
        
        # Build context prefix
        prefix = build_context_prefix(
            entity_type='egyenruha',
            entity_name=entity_name_normalized,
            document_type=source_file,
            section=section
        )
        
        # Create metadata
        metadata = {
            'language': 'hu',
            'entity_type': 'egyenruha',
            'entity_name': title,
            'entity_name_normalized': entity_name_normalized,
            'entity_name_surface_forms': [title, entity_name_normalized],
            'document_type': source_file,
            'section': section,
            'contains_person_names': False,
            'is_list': False
        }
        
        return (prefix + chunk_text, metadata)
    
    # Process banyaszMunkaruha (miner work uniform)
    if 'banyaszMunkaruha' in uniform_data:
        munkaruha = uniform_data['banyaszMunkaruha']
        if isinstance(munkaruha, dict):
            desc = munkaruha.get('altalanosLeiras', '')
            elemei = munkaruha.get('elemei', [])
            tortenet = munkaruha.get('tortenet', '')
            
            # Create chunk for general description and elements
            if desc or elemei:
                chunk, metadata = create_uniform_chunk(
                    title="Bányászmunkaruha",
                    description=desc,
                    elements=elemei,
                    section="Bányászmunkaruha"
                )
                chunks_with_metadata.append((chunk, metadata))
            
            # Create separate chunk for history if available
            if tortenet:
                chunk, metadata = create_uniform_chunk(
                    title="Bányászmunkaruha története",
                    description=tortenet,
                    section="Bányászmunkaruha"
                )
                chunks_with_metadata.append((chunk, metadata))
    
    # Process diszegyenruha (ceremonial uniform)
    if 'diszegyenruha' in uniform_data:
        disz = uniform_data['diszegyenruha']
        if isinstance(disz, dict):
            desc = disz.get('altalanosLeiras', '')
            tipusok = disz.get('tipusok', [])
            
            # General description chunk
            if desc:
                chunk, metadata = create_uniform_chunk(
                    title="Díszegyenruha",
                    description=desc,
                    section="Díszegyenruha"
                )
                chunks_with_metadata.append((chunk, metadata))
            
            # One chunk per uniform type
            for tipus in tipusok:
                if isinstance(tipus, dict):
                    nev = tipus.get('nev', '')
                    jellemzok = tipus.get('jellemzok', [])
                    
                    if nev:
                        chunk, metadata = create_uniform_chunk(
                            title=nev,
                            description=f"A {nev} jellemzői:",
                            elements=jellemzok,
                            section="Díszegyenruha",
                            entity_name=nev
                        )
                        chunks_with_metadata.append((chunk, metadata))
    
    # Process xxSzazadiDiszegyenruhak (20th century ceremonial uniforms)
    if 'xxSzazadiDiszegyenruhak' in uniform_data:
        xx_szazad = uniform_data['xxSzazadiDiszegyenruhak']
        if isinstance(xx_szazad, dict):
            for key, value in xx_szazad.items():
                if isinstance(value, dict):
                    alapszin = value.get('alapszin', '')
                    elemei = value.get('elemei', [])
                    iparag = value.get('iparagMegkulonboztetes', {})
                    
                    desc_parts = []
                    if alapszin:
                        desc_parts.append(f"Alapszín: {alapszin}.")
                    
                    chunk, metadata = create_uniform_chunk(
                        title=key,
                        description=" ".join(desc_parts) if desc_parts else "",
                        elements=elemei,
                        additional_info={"Iparágak megkülönböztetése": iparag} if iparag else None,
                        section="XX. századi díszegyenruhák"
                    )
                    chunks_with_metadata.append((chunk, metadata))
    
    # Process banyamernokHallgatok (mining engineering students)
    if 'banyamernokHallgatok' in uniform_data:
        hallgatok = uniform_data['banyamernokHallgatok']
        if isinstance(hallgatok, dict):
            desc = hallgatok.get('altalanosLeiras', '')
            tipusok = hallgatok.get('tipusok', [])
            
            # General description
            if desc:
                chunk, metadata = create_uniform_chunk(
                    title="Bányamérnök-hallgatók egyenruhája",
                    description=desc,
                    section="Bányamérnök-hallgatók"
                )
                chunks_with_metadata.append((chunk, metadata))
            
            # One chunk per uniform type
            for tipus in tipusok:
                if isinstance(tipus, dict):
                    nev = tipus.get('nev', '')
                    anyag = tipus.get('anyag', '')
                    elemei = tipus.get('elemei', [])
                    kieg = tipus.get('kieg', [])
                    eredet = tipus.get('eredet', '')
                    variaciok = tipus.get('variaciok', [])
                    
                    if nev:
                        additional = {}
                        if anyag:
                            additional['anyag'] = anyag
                        if eredet:
                            additional['eredet'] = eredet
                        if kieg:
                            additional['kiegészítők'] = kieg
                        if variaciok:
                            additional['variációk'] = variaciok
                        
                        desc_text = f"A {nev} egyenruha"
                        if anyag:
                            desc_text += f" {anyag} anyagból készül"
                        if eredet:
                            desc_text += f", eredete: {eredet}"
                        desc_text += "."
                        
                        chunk, metadata = create_uniform_chunk(
                            title=nev,
                            description=desc_text,
                            elements=elemei,
                            additional_info=additional if additional else None,
                            section="Bányamérnök-hallgatók",
                            entity_name=nev
                        )
                        chunks_with_metadata.append((chunk, metadata))
    
    # Process egyebEgyenruhak (other uniforms)
    if 'egyebEgyenruhak' in uniform_data:
        egyeb = uniform_data['egyebEgyenruhak']
        if isinstance(egyeb, dict):
            for uniform_name, uniform_data_item in egyeb.items():
                if isinstance(uniform_data_item, dict):
                    eredet = uniform_data_item.get('eredet', '')
                    anyag = uniform_data_item.get('anyag', '')
                    szin = uniform_data_item.get('szin', [])
                    elemei = uniform_data_item.get('elemei', [])
                    hozzaTartozo = uniform_data_item.get('hozzaTartozo', [])
                    variaciok = uniform_data_item.get('variaciok', {})
                    
                    desc_parts = []
                    if eredet:
                        desc_parts.append(f"Eredet: {eredet}.")
                    if anyag:
                        desc_parts.append(f"Anyag: {anyag}.")
                    if szin:
                        szin_text = ", ".join(szin) if isinstance(szin, list) else str(szin)
                        desc_parts.append(f"Színek: {szin_text}.")
                    
                    additional = {}
                    if hozzaTartozo:
                        additional['hozzátartozó'] = hozzaTartozo
                    if variaciok:
                        additional['variációk'] = variaciok
                    
                    chunk, metadata = create_uniform_chunk(
                        title=uniform_name,
                        description=" ".join(desc_parts),
                        elements=elemei,
                        additional_info=additional if additional else None,
                        section="Egyéb egyenruhák",
                        entity_name=uniform_name
                    )
                    chunks_with_metadata.append((chunk, metadata))
    
    # Process egyetemekEsKarok (universities and faculties)
    if 'egyetemekEsKarok' in uniform_data:
        egyetemek = uniform_data['egyetemekEsKarok']
        if isinstance(egyetemek, dict):
            for egyetem_name, karok in egyetemek.items():
                if isinstance(karok, dict):
                    for kar_name, egyenruhak in karok.items():
                        if isinstance(egyenruhak, list):
                            egyenruhak_text = ", ".join(egyenruhak)
                            desc = f"A {egyetem_name} {kar_name} karának egyenruhái: {egyenruhak_text}."
                            
                            chunk, metadata = create_uniform_chunk(
                                title=f"{egyetem_name} - {kar_name}",
                                description=desc,
                                section=f"Egyetemek és karok - {egyetem_name}",
                                entity_name=f"{kar_name} ({egyetem_name})"
                            )
                            chunks_with_metadata.append((chunk, metadata))
    
    logger.info(f"Created {len(chunks_with_metadata)} uniform chunks from JSON")
    return chunks_with_metadata
