"""
Configuration loader for AI telop generation
"""
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class TelopConfig:
    """Configuration for AI telop generation"""
    
    DEFAULT_CONFIG_PATH = "configs/config_telop_generation.yaml"
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to configuration YAML file (optional)
        """
        if config_path is None:
            config_path = self.DEFAULT_CONFIG_PATH
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            logger.warning(f"Configuration file not found: {self.config_path}")
            logger.warning("Using default configuration")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.warning("Using default configuration")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'telop_generation': {
                'enabled': True,
                'speech': {
                    'enabled': True,
                    'model': 'whisper',
                    'model_size': 'small',
                    'language': 'ja',
                    'min_segment_duration': 0.5,
                    'max_segment_duration': 5.0,
                    'cache_results': True
                },
                'emotion': {
                    'enabled': True,
                    'confidence_threshold': 0.6,
                    'laughter': {
                        'enabled': True,
                        'text_short': 'w',
                        'text_medium': 'www',
                        'text_long': 'wwww',
                        'pitch_std_threshold': 50.0,
                        'energy_threshold': 0.3
                    },
                    'surprise': {
                        'enabled': True,
                        'text': '！',
                        'text_alt': 'えっ',
                        'pitch_delta_threshold': 100.0,
                        'max_duration': 1.0
                    },
                    'sadness': {
                        'enabled': True,
                        'text': '...',
                        'text_alt': '悲',
                        'pitch_mean_threshold': 150.0,
                        'energy_threshold': 0.1
                    }
                },
                'output': {
                    'separate_tracks': True,
                    'ocr_track_prefix': 'OCR',
                    'speech_track_prefix': 'Speech',
                    'emotion_track_prefix': 'Emotion'
                }
            }
        }
    
    def is_enabled(self) -> bool:
        """Check if telop generation is enabled"""
        return self.config.get('telop_generation', {}).get('enabled', False)
    
    def is_speech_enabled(self) -> bool:
        """Check if speech recognition is enabled"""
        return (self.is_enabled() and 
                self.config.get('telop_generation', {}).get('speech', {}).get('enabled', False))
    
    def is_emotion_enabled(self) -> bool:
        """Check if emotion detection is enabled"""
        return (self.is_enabled() and 
                self.config.get('telop_generation', {}).get('emotion', {}).get('enabled', False))
    
    def get_speech_config(self) -> Dict[str, Any]:
        """Get speech recognition configuration"""
        return self.config.get('telop_generation', {}).get('speech', {})
    
    def get_emotion_config(self) -> Dict[str, Any]:
        """Get emotion detection configuration"""
        return self.config.get('telop_generation', {}).get('emotion', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration"""
        return self.config.get('telop_generation', {}).get('output', {})
    
    def save(self, output_path: str = None):
        """
        Save configuration to YAML file
        
        Args:
            output_path: Output path (optional, defaults to config_path)
        """
        if output_path is None:
            output_path = self.config_path
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Saved configuration to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")


def load_telop_config(config_path: str = None) -> TelopConfig:
    """
    Load telop generation configuration
    
    Args:
        config_path: Path to configuration file (optional)
    
    Returns:
        TelopConfig instance
    """
    return TelopConfig(config_path)
