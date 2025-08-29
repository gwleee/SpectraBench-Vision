"""
Environment Variable Manager for SpectraVision
Handles loading and validation of environment variables from .env files
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EnvironmentConfig:
    """Container for environment configuration"""
    # Hugging Face
    hf_token: Optional[str] = None
    
    # OpenAI
    openai_api_key: Optional[str] = None
    
    # CUDA
    cuda_visible_devices: str = "0"
    
    # Evaluation settings
    vlmeval_batch_size: int = 1
    hf_datasets_offline: bool = False
    transformers_offline: bool = False
    
    # Logging
    log_level: str = "INFO"
    enable_monitoring: bool = False
    
    # Cache
    cache_cleanup_level: str = "light"
    enable_cache_cleanup: bool = True
    
    # Output
    output_dir: str = "outputs"
    max_log_size: int = 100
    log_file_count: int = 5

class EnvironmentManager:
    """Manages environment variables and configuration"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize Environment Manager
        
        Args:
            env_file: Path to .env file (defaults to project root/.env)
        """
        self.project_root = Path(__file__).parent.parent
        
        if env_file:
            self.env_file = Path(env_file)
        else:
            # Try multiple locations for .env file
            potential_env_files = [
                self.project_root / ".env",
                Path.cwd() / ".env",
                Path("~/.spectravision/.env").expanduser()
            ]
            
            self.env_file = None
            for env_path in potential_env_files:
                if env_path.exists():
                    self.env_file = env_path
                    break
            
            if not self.env_file:
                self.env_file = potential_env_files[0]  # Default to project root
        
        self.config = EnvironmentConfig()
        self.load_environment()
        
        logger.info(f"Environment Manager initialized")
        logger.info(f"   .env file: {self.env_file}")
        logger.info(f"   HF token configured: {'Yes' if self.config.hf_token else 'No'}")
    
    def load_environment(self) -> EnvironmentConfig:
        """Load environment variables from .env file and system environment"""
        
        # First load from .env file if it exists
        if self.env_file.exists():
            self._load_from_file(self.env_file)
            logger.debug(f"Loaded environment from: {self.env_file}")
        else:
            logger.warning(f".env file not found at: {self.env_file}")
            logger.info("Using system environment variables and defaults")
        
        # Override with system environment variables
        self._load_from_system_env()
        
        # Validate configuration
        self._validate_config()
        
        return self.config
    
    def _load_from_file(self, env_file: Path):
        """Load variables from .env file"""
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse KEY=VALUE format
                    if '=' not in line:
                        logger.warning(f"Invalid line in .env file (line {line_num}): {line}")
                        continue
                    
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # Set in environment (but don't override existing system vars)
                    if key not in os.environ:
                        os.environ[key] = value
                        
        except Exception as e:
            logger.error(f"Error reading .env file {env_file}: {e}")
    
    def _load_from_system_env(self):
        """Load configuration from system environment variables"""
        
        # Hugging Face
        self.config.hf_token = os.getenv('HF_TOKEN')
        
        # OpenAI
        self.config.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # CUDA
        self.config.cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', '0')
        
        # Evaluation settings
        self.config.vlmeval_batch_size = int(os.getenv('VLMEVAL_BATCH_SIZE', '1'))
        self.config.hf_datasets_offline = os.getenv('HF_DATASETS_OFFLINE', '0').lower() in ('1', 'true', 'yes')
        self.config.transformers_offline = os.getenv('TRANSFORMERS_OFFLINE', '0').lower() in ('1', 'true', 'yes')
        
        # Logging
        self.config.log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        self.config.enable_monitoring = os.getenv('ENABLE_MONITORING', 'false').lower() in ('true', 'yes', '1')
        
        # Cache
        self.config.cache_cleanup_level = os.getenv('CACHE_CLEANUP_LEVEL', 'light').lower()
        self.config.enable_cache_cleanup = os.getenv('ENABLE_CACHE_CLEANUP', 'true').lower() in ('true', 'yes', '1')
        
        # Output
        self.config.output_dir = os.getenv('OUTPUT_DIR', 'outputs')
        self.config.max_log_size = int(os.getenv('MAX_LOG_SIZE', '100'))
        self.config.log_file_count = int(os.getenv('LOG_FILE_COUNT', '5'))
    
    def _validate_config(self):
        """Validate loaded configuration"""
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        if self.config.log_level not in valid_log_levels:
            logger.warning(f"Invalid log level '{self.config.log_level}', using 'INFO'")
            self.config.log_level = 'INFO'
        
        # Validate cache cleanup level
        valid_cleanup_levels = ['light', 'moderate', 'aggressive']
        if self.config.cache_cleanup_level not in valid_cleanup_levels:
            logger.warning(f"Invalid cache cleanup level '{self.config.cache_cleanup_level}', using 'light'")
            self.config.cache_cleanup_level = 'light'
        
        # Validate numeric values
        if self.config.vlmeval_batch_size < 1:
            logger.warning(f"Invalid batch size {self.config.vlmeval_batch_size}, using 1")
            self.config.vlmeval_batch_size = 1
        
        if self.config.max_log_size < 1:
            logger.warning(f"Invalid max log size {self.config.max_log_size}, using 100")
            self.config.max_log_size = 100
        
        if self.config.log_file_count < 1:
            logger.warning(f"Invalid log file count {self.config.log_file_count}, using 5")
            self.config.log_file_count = 5
    
    def apply_environment_settings(self):
        """Apply environment settings to the system"""
        
        # Set CUDA device
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config.cuda_visible_devices
        
        # Set evaluation settings
        os.environ['VLMEVAL_BATCH_SIZE'] = str(self.config.vlmeval_batch_size)
        os.environ['HF_DATASETS_OFFLINE'] = '1' if self.config.hf_datasets_offline else '0'
        os.environ['TRANSFORMERS_OFFLINE'] = '1' if self.config.transformers_offline else '0'
        
        # Set API keys if available
        if self.config.hf_token:
            os.environ['HF_TOKEN'] = self.config.hf_token
        
        if self.config.openai_api_key:
            os.environ['OPENAI_API_KEY'] = self.config.openai_api_key
        
        logger.debug("Environment settings applied")
    
    def check_required_tokens(self, require_hf: bool = True) -> Dict[str, bool]:
        """
        Check if required tokens are configured
        
        Args:
            require_hf: Whether HF token is required
            
        Returns:
            Dictionary with token availability status
        """
        status = {
            'hf_token': self.config.hf_token is not None,
            'openai_api_key': self.config.openai_api_key is not None,
        }
        
        if require_hf and not status['hf_token']:
            logger.warning("HF_TOKEN is required for accessing gated models like MiniCPM-V-2.6")
            logger.info("Please set HF_TOKEN in your .env file or system environment")
            logger.info("Get your token from: https://huggingface.co/settings/tokens")
        
        return status
    
    def create_env_file_if_missing(self) -> bool:
        """
        Create .env file from template if it doesn't exist
        
        Returns:
            True if file was created, False if it already existed
        """
        if self.env_file.exists():
            return False
        
        template_file = self.project_root / ".env.template"
        if not template_file.exists():
            logger.error("Environment template file not found")
            return False
        
        try:
            # Copy template to .env
            with open(template_file, 'r', encoding='utf-8') as template:
                content = template.read()
            
            # Ensure parent directory exists
            self.env_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.env_file, 'w', encoding='utf-8') as env_file:
                env_file.write(content)
            
            logger.info(f"Created .env file from template: {self.env_file}")
            logger.info("Please edit the .env file and add your personal API keys")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create .env file: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging/debugging"""
        return {
            'env_file': str(self.env_file),
            'hf_token_configured': bool(self.config.hf_token),
            'openai_api_key_configured': bool(self.config.openai_api_key),
            'cuda_visible_devices': self.config.cuda_visible_devices,
            'log_level': self.config.log_level,
            'output_dir': self.config.output_dir,
            'cache_cleanup_level': self.config.cache_cleanup_level,
            'monitoring_enabled': self.config.enable_monitoring
        }

# Global environment manager instance
_env_manager = None

def get_environment_manager() -> EnvironmentManager:
    """Get global environment manager instance"""
    global _env_manager
    if _env_manager is None:
        _env_manager = EnvironmentManager()
    return _env_manager

def get_config() -> EnvironmentConfig:
    """Get current environment configuration"""
    return get_environment_manager().config