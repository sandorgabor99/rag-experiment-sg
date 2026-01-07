"""
Configuration management for Knowledge Layer RAG Pipeline.

Loads configuration from config.yaml with ENV variable override support.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

logger = logging.getLogger(__name__)


class Config:
    """
    Centralized configuration manager.
    
    Priority order:
    1. CLI arguments (handled in orchestrator)
    2. ENV variables (KL_<SECTION>_<KEY>)
    3. config.yaml values
    4. Code defaults
    """
    
    def __init__(self, config_dict: Dict[str, Any], config_path: Optional[Path] = None):
        """
        Initialize config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            config_path: Path to config file (for logging)
        """
        self._config = config_dict
        self._config_path = config_path
        self._source_log = {}  # Track where each value came from
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> 'Config':
        """
        Load configuration from YAML file with ENV overrides.
        
        Args:
            config_path: Path to config.yaml (default: project_root/config.yaml)
        
        Returns:
            Config instance
        """
        if config_path is None:
            # Find project root (where config.yaml should be)
            current = Path(__file__).parent
            while current.parent != current:
                potential_config = current / "config.yaml"
                if potential_config.exists():
                    config_path = potential_config
                    break
                current = current.parent
            
            if config_path is None:
                # Fallback: look in current working directory
                config_path = Path("config.yaml")
        
        config_dict = {}
        
        # Load from YAML if exists
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f) or {}
                
                # Validate structure
                cls._validate_config(config_dict)
                
                logger.info(f"Loaded configuration from {config_path}")
            except yaml.YAMLError as e:
                logger.error(f"Invalid YAML in config file {config_path}: {e}")
                raise ValueError(f"Invalid YAML syntax in config file: {e}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
                config_dict = {}
        else:
            logger.info(f"Config file not found at {config_path}. Using defaults and ENV variables.")
            config_dict = {}
        
        # Apply ENV overrides
        config_dict = cls._apply_env_overrides(config_dict)
        
        return cls(config_dict, config_path)
    
    @staticmethod
    def _validate_config(config_dict: Dict[str, Any]) -> None:
        """
        Validate configuration structure.
        
        Args:
            config_dict: Configuration dictionary
        
        Raises:
            ValueError: If config structure is invalid
        """
        # Check for required top-level sections (warn if missing, but don't fail)
        expected_sections = ['embedding', 'chunking', 'search', 'context', 'llm', 'paths']
        for section in expected_sections:
            if section not in config_dict:
                logger.debug(f"Config section '{section}' not found, will use defaults")
        
        # Validate nested structure for llm section
        if 'llm' in config_dict and isinstance(config_dict['llm'], dict):
            llm = config_dict['llm']
            expected_llm_subsections = ['qa', 'refiner', 'entity_extraction']
            for subsection in expected_llm_subsections:
                if subsection not in llm:
                    logger.debug(f"LLM subsection '{subsection}' not found, will use defaults")
        
        # Validate value types for critical settings
        if 'embedding' in config_dict:
            embedding = config_dict['embedding']
            if 'batch_size' in embedding and not isinstance(embedding['batch_size'], int):
                raise ValueError(f"embedding.batch_size must be an integer, got {type(embedding['batch_size'])}")
        
        if 'chunking' in config_dict:
            chunking = config_dict['chunking']
            if 'chunk_size' in chunking and not isinstance(chunking['chunk_size'], int):
                raise ValueError(f"chunking.chunk_size must be an integer, got {type(chunking['chunk_size'])}")
            if 'overlap' in chunking and not isinstance(chunking['overlap'], int):
                raise ValueError(f"chunking.overlap must be an integer, got {type(chunking['overlap'])}")
        
        if 'context' in config_dict:
            context = config_dict['context']
            if 'max_tokens_default' in context and not isinstance(context['max_tokens_default'], int):
                raise ValueError(f"context.max_tokens_default must be an integer, got {type(context['max_tokens_default'])}")
            if 'similarity_threshold' in context:
                threshold = context['similarity_threshold']
                if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                    raise ValueError(f"context.similarity_threshold must be a float between 0 and 1, got {threshold}")
    
    @staticmethod
    def _apply_env_overrides(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to config.
        
        ENV format: KL_<SECTION>_<KEY>
        Examples:
        - KL_EMBEDDING_MODEL -> embedding.model
        - KL_CHUNKING_CHUNK_SIZE -> chunking.chunk_size
        - KL_LLM_QA_PROVIDER -> llm.qa.provider
        
        Args:
            config_dict: Configuration dictionary
        
        Returns:
            Updated configuration dictionary
        """
        # Ensure nested structure exists
        if 'embedding' not in config_dict:
            config_dict['embedding'] = {}
        if 'chunking' not in config_dict:
            config_dict['chunking'] = {}
        if 'search' not in config_dict:
            config_dict['search'] = {}
        if 'context' not in config_dict:
            config_dict['context'] = {}
        if 'llm' not in config_dict:
            config_dict['llm'] = {}
        if 'llm' in config_dict:
            if 'qa' not in config_dict['llm']:
                config_dict['llm']['qa'] = {}
            if 'refiner' not in config_dict['llm']:
                config_dict['llm']['refiner'] = {}
            if 'entity_extraction' not in config_dict['llm']:
                config_dict['llm']['entity_extraction'] = {}
        if 'paths' not in config_dict:
            config_dict['paths'] = {}
        
        # Process all ENV variables starting with KL_
        for env_key, env_value in os.environ.items():
            if not env_key.startswith('KL_'):
                continue
            
            # Parse KL_<SECTION>_<KEY> format
            parts = env_key[3:].split('_')  # Remove KL_ prefix
            if len(parts) < 2:
                continue
            
            # Handle nested keys (e.g., LLM_QA_PROVIDER -> llm.qa.provider)
            section = parts[0].lower()
            key_parts = [p.lower() for p in parts[1:]]
            
            # Convert value to appropriate type
            value = cls._parse_env_value(env_value)
            
            # Set nested value
            if section == 'llm' and len(key_parts) >= 2:
                # Handle llm.qa.* and llm.refiner.* and llm.entity_extraction.*
                subsection = key_parts[0]
                subkey = '_'.join(key_parts[1:])
                if subsection not in config_dict['llm']:
                    config_dict['llm'][subsection] = {}
                config_dict['llm'][subsection][subkey] = value
                logger.info(f"ENV override: {env_key} -> llm.{subsection}.{subkey} = {value}")
            else:
                # Simple section.key format
                key = '_'.join(key_parts)
                if section not in config_dict:
                    config_dict[section] = {}
                config_dict[section][key] = value
                logger.info(f"ENV override: {env_key} -> {section}.{key} = {value}")
        
        return config_dict
    
    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try boolean
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        if value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try null/None
        if value.lower() in ('null', 'none', ''):
            return None
        
        # Return as string
        return value
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            section: Configuration section (e.g., 'embedding', 'chunking')
            key: Configuration key
            default: Default value if not found
        
        Returns:
            Configuration value
        """
        if section not in self._config:
            return default
        
        section_config = self._config[section]
        if isinstance(section_config, dict):
            return section_config.get(key, default)
        
        return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self._config.get(section, {})
    
    @property
    def embedding(self) -> Dict[str, Any]:
        """Embedding configuration."""
        return self.get_section('embedding')
    
    @property
    def chunking(self) -> Dict[str, Any]:
        """Chunking configuration."""
        return self.get_section('chunking')
    
    @property
    def search(self) -> Dict[str, Any]:
        """Search configuration."""
        return self.get_section('search')
    
    @property
    def context(self) -> Dict[str, Any]:
        """Context configuration."""
        return self.get_section('context')
    
    @property
    def llm(self) -> Dict[str, Any]:
        """LLM configuration."""
        return self.get_section('llm')
    
    @property
    def paths(self) -> Dict[str, Any]:
        """Paths configuration."""
        return self.get_section('paths')
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
    
    def show(self):
        """Print effective configuration for debugging."""
        import json
        print("=" * 80)
        print("Effective Configuration:")
        print("=" * 80)
        print(json.dumps(self._config, indent=2, default=str))
        print("=" * 80)


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Optional path to config.yaml
    
    Returns:
        Config instance
    """
    return Config.load(config_path)
