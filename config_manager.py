"""
Centralized Configuration Management for Buddy Memory System
Provides validated configuration with overrides and standardized interfaces
NO silent failures - all configuration errors are loud and proud
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import os

# Phase 2: Standard interfaces and result types
from foundation_interface import ConfigurationInterface
from result_types import OperationResult, DictOperationResult, BoolOperationResult
from exceptions import ConfigurationError
from debug_info import debug_tracker


@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password123"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "mem0_fixed"

    def validate(self) -> List[str]:
        """Validate database configuration"""
        errors = []
        
        if not self.neo4j_url:
            errors.append("neo4j_url cannot be empty")
        
        if not self.neo4j_username:
            errors.append("neo4j_username cannot be empty")
        
        if not self.neo4j_password:
            errors.append("neo4j_password cannot be empty")
        
        if not self.qdrant_host:
            errors.append("qdrant_host cannot be empty")
        
        if self.qdrant_port <= 0 or self.qdrant_port > 65535:
            errors.append(f"qdrant_port must be 1-65535, got {self.qdrant_port}")
        
        if not self.qdrant_collection:
            errors.append("qdrant_collection cannot be empty")
        
        return errors


@dataclass
class MemoryConfig:
    """Memory system configuration"""
    relevance_threshold: float = 0.55
    max_expansions: int = 3
    cache_size: int = 100
    cache_ttl_minutes: int = 60
    max_context_tokens: int = 4000
    expansion_timeout_ms: int = 500
    recency_days: int = 30

    def validate(self) -> List[str]:
        """Validate memory configuration"""
        errors = []
        
        if not 0.0 <= self.relevance_threshold <= 1.0:
            errors.append(f"relevance_threshold must be 0.0-1.0, got {self.relevance_threshold}")
        
        if self.max_expansions <= 0:
            errors.append(f"max_expansions must be positive, got {self.max_expansions}")
        
        if self.cache_size <= 0:
            errors.append(f"cache_size must be positive, got {self.cache_size}")
        
        if self.cache_ttl_minutes <= 0:
            errors.append(f"cache_ttl_minutes must be positive, got {self.cache_ttl_minutes}")
        
        if self.max_context_tokens <= 0:
            errors.append(f"max_context_tokens must be positive, got {self.max_context_tokens}")
        
        if self.expansion_timeout_ms <= 0:
            errors.append(f"expansion_timeout_ms must be positive, got {self.expansion_timeout_ms}")
        
        if self.recency_days <= 0:
            errors.append(f"recency_days must be positive, got {self.recency_days}")
        
        return errors


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_directory: str = "/var/log/buddy/full_context"
    max_log_files: int = 100
    max_operations_tracked: int = 1000
    debug_level: str = "INFO"

    def validate(self) -> List[str]:
        """Validate logging configuration"""
        errors = []
        
        if not self.log_directory:
            errors.append("log_directory cannot be empty")
        
        if self.max_log_files <= 0:
            errors.append(f"max_log_files must be positive, got {self.max_log_files}")
        
        if self.max_operations_tracked <= 0:
            errors.append(f"max_operations_tracked must be positive, got {self.max_operations_tracked}")
        
        if self.debug_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            errors.append(f"debug_level must be DEBUG/INFO/WARNING/ERROR, got {self.debug_level}")
        
        return errors


@dataclass
class InterfaceConfig:
    """User interface configuration"""
    silent_mode: bool = False
    show_startup_messages: bool = True
    show_command_help: bool = True

    def validate(self) -> List[str]:
        """Validate interface configuration"""
        errors = []
        
        # Boolean values don't need validation, but we can add future checks here
        if not isinstance(self.silent_mode, bool):
            errors.append(f"silent_mode must be boolean, got {type(self.silent_mode)}")
        
        if not isinstance(self.show_startup_messages, bool):
            errors.append(f"show_startup_messages must be boolean, got {type(self.show_startup_messages)}")
        
        if not isinstance(self.show_command_help, bool):
            errors.append(f"show_command_help must be boolean, got {type(self.show_command_help)}")
        
        return errors


@dataclass
class SystemConfig:
    """Complete system configuration"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    interface: InterfaceConfig = field(default_factory=InterfaceConfig)
    
    def validate(self) -> OperationResult[bool]:
        """
        Validate complete configuration
        Returns OperationResult with all validation errors
        """
        all_errors = []
        
        # Validate each section
        all_errors.extend(self.database.validate())
        all_errors.extend(self.memory.validate())
        all_errors.extend(self.logging.validate())
        all_errors.extend(self.interface.validate())
        
        if all_errors:
            return OperationResult.error_result(f"Configuration validation failed: {'; '.join(all_errors)}")
        
        return OperationResult.success_result(True, {"validation_passed": True})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "database": {
                "neo4j_url": self.database.neo4j_url,
                "neo4j_username": self.database.neo4j_username,
                "neo4j_password": self.database.neo4j_password,
                "qdrant_host": self.database.qdrant_host,
                "qdrant_port": self.database.qdrant_port,
                "qdrant_collection": self.database.qdrant_collection
            },
            "memory": {
                "relevance_threshold": self.memory.relevance_threshold,
                "max_expansions": self.memory.max_expansions,
                "cache_size": self.memory.cache_size,
                "cache_ttl_minutes": self.memory.cache_ttl_minutes,
                "max_context_tokens": self.memory.max_context_tokens,
                "expansion_timeout_ms": self.memory.expansion_timeout_ms,
                "recency_days": self.memory.recency_days
            },
            "logging": {
                "log_directory": self.logging.log_directory,
                "max_log_files": self.logging.max_log_files,
                "max_operations_tracked": self.logging.max_operations_tracked,
                "debug_level": self.logging.debug_level
            },
            "interface": {
                "silent_mode": self.interface.silent_mode,
                "show_startup_messages": self.interface.show_startup_messages,
                "show_command_help": self.interface.show_command_help
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """Create SystemConfig from dictionary"""
        database_config = DatabaseConfig(**data.get("database", {}))
        memory_config = MemoryConfig(**data.get("memory", {}))
        logging_config = LoggingConfig(**data.get("logging", {}))
        interface_config = InterfaceConfig(**data.get("interface", {}))
        
        return cls(
            database=database_config,
            memory=memory_config,
            logging=logging_config,
            interface=interface_config
        )


class ConfigManager(ConfigurationInterface):
    """
    Manages system configuration with validation and overrides
    Implements ConfigurationInterface for standardized API
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Optional config file path (default: buddy_config.yaml)
        """
        self.config_file = config_file or "buddy_config.yaml"
        self.config = SystemConfig()
        self._loaded_from_file = False
        
        debug_tracker.log_operation("config_manager", "initialization", {
            "config_file": self.config_file
        })
    
    def load_config(self, config_source: Optional[str] = None) -> DictOperationResult:
        """
        Load configuration from file with validation
        
        Args:
            config_source: Optional config file path override
            
        Returns:
            OperationResult[Dict] with configuration data
        """
        config_path_str = config_source or self.config_file
        
        debug_tracker.log_operation("config_manager", "load_config_start", {
            "config_file": config_path_str
        })
        
        try:
            config_path = Path(config_path_str)
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
                
                self.config = SystemConfig.from_dict(config_data)
                self._loaded_from_file = True
                
                debug_tracker.log_operation("config_manager", "config_loaded_from_file", {
                    "config_file": str(config_path.absolute()),
                    "sections": list(config_data.keys())
                })
            else:
                # Use defaults if no config file
                self.config = SystemConfig()
                self._loaded_from_file = False
                
                debug_tracker.log_operation("config_manager", "config_using_defaults", {
                    "config_file": str(config_path.absolute()),
                    "reason": "file_not_found"
                })
            
            # Validate configuration
            validation_result = self.config.validate()
            if not validation_result.success:
                debug_tracker.log_operation("config_manager", "load_config_failed", {
                    "error": validation_result.error
                }, success=False, error=validation_result.error)
                
                return OperationResult.error_result(f"Config validation failed: {validation_result.error}")
            
            result_data = self.config.to_dict()
            
            debug_tracker.log_operation("config_manager", "load_config_success", {
                "loaded_from_file": self._loaded_from_file,
                "sections_count": len(result_data)
            }, success=True)
            
            return OperationResult.success_result(result_data, {
                "loaded_from_file": self._loaded_from_file,
                "config_file": str(config_path.absolute()),
                "validation_passed": True
            })
            
        except Exception as e:
            debug_tracker.log_operation("config_manager", "load_config_failed", {
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Failed to load configuration: {e}")
    
    def save_config(self, config_data: Dict[str, Any], destination: Optional[str] = None) -> BoolOperationResult:
        """
        Save configuration to file
        
        Args:
            config_data: Configuration data to save
            destination: Optional save destination override
            
        Returns:
            OperationResult[bool] indicating success/failure
        """
        save_path_str = destination or self.config_file
        
        debug_tracker.log_operation("config_manager", "save_config_start", {
            "config_file": save_path_str
        })
        
        try:
            # Validate config data before saving
            temp_config = SystemConfig.from_dict(config_data)
            validation_result = temp_config.validate()
            
            if not validation_result.success:
                debug_tracker.log_operation("config_manager", "save_config_failed", {
                    "error": validation_result.error
                }, success=False, error=validation_result.error)
                
                return OperationResult.error_result(f"Cannot save invalid configuration: {validation_result.error}")
            
            # Save to file
            save_path = Path(save_path_str)
            
            with open(save_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            # Update current config if we saved successfully
            self.config = temp_config
            
            debug_tracker.log_operation("config_manager", "save_config_success", {
                "config_file": str(save_path.absolute()),
                "sections_saved": len(config_data)
            }, success=True)
            
            return OperationResult.success_result(True, {
                "config_file": str(save_path.absolute()),
                "validation_passed": True
            })
            
        except Exception as e:
            debug_tracker.log_operation("config_manager", "save_config_failed", {
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Failed to save configuration: {e}")
    
    def validate_config(self, config_data: Dict[str, Any]) -> BoolOperationResult:
        """
        Validate configuration data without loading it
        
        Args:
            config_data: Configuration data to validate
            
        Returns:
            OperationResult[bool] with validation result
        """
        debug_tracker.log_operation("config_manager", "validate_config_start", {
            "sections": list(config_data.keys())
        })
        
        try:
            temp_config = SystemConfig.from_dict(config_data)
            validation_result = temp_config.validate()
            
            if validation_result.success:
                debug_tracker.log_operation("config_manager", "validate_config_success", {
                    "sections_validated": len(config_data)
                }, success=True)
                
                return OperationResult.success_result(True, {
                    "validation_passed": True,
                    "sections_validated": len(config_data)
                })
            else:
                debug_tracker.log_operation("config_manager", "validate_config_failed", {
                    "error": validation_result.error
                }, success=False, error=validation_result.error)
                
                return OperationResult.error_result(validation_result.error)
        
        except Exception as e:
            debug_tracker.log_operation("config_manager", "validate_config_failed", {
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Configuration validation failed: {e}")
    
    def get_config(self, key: Optional[str] = None) -> DictOperationResult:
        """
        Get configuration value(s)
        
        Args:
            key: Optional specific config key (e.g., 'database.neo4j_url'), None for all config
            
        Returns:
            OperationResult[Dict] with config data
        """
        try:
            config_dict = self.config.to_dict()
            
            if key is None:
                # Return entire configuration
                debug_tracker.log_operation("config_manager", "get_config_all", {
                    "sections": list(config_dict.keys())
                })
                
                return OperationResult.success_result(config_dict, {
                    "config_type": "complete",
                    "sections_count": len(config_dict)
                })
            else:
                # Return specific key
                key_parts = key.split('.')
                current_value = config_dict
                
                for part in key_parts:
                    if isinstance(current_value, dict) and part in current_value:
                        current_value = current_value[part]
                    else:
                        return OperationResult.error_result(f"Configuration key '{key}' not found")
                
                debug_tracker.log_operation("config_manager", "get_config_key", {
                    "key": key,
                    "value_type": type(current_value).__name__
                })
                
                return OperationResult.success_result({key: current_value}, {
                    "config_type": "specific_key",
                    "key": key
                })
        
        except Exception as e:
            debug_tracker.log_operation("config_manager", "get_config_failed", {
                "key": key,
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Failed to get configuration: {e}")
    
    def set_config(self, key: str, value: Any) -> BoolOperationResult:
        """
        Set configuration value with validation
        
        Args:
            key: Configuration key (e.g., 'database.neo4j_url')
            value: Configuration value
            
        Returns:
            OperationResult[bool] indicating success/failure
        """
        debug_tracker.log_operation("config_manager", "set_config_start", {
            "key": key,
            "value_type": type(value).__name__
        })
        
        try:
            # Get current config as dict
            config_dict = self.config.to_dict()
            
            # Navigate to the key location
            key_parts = key.split('.')
            current_dict = config_dict
            
            # Navigate to parent of target key
            for part in key_parts[:-1]:
                if part not in current_dict:
                    current_dict[part] = {}
                current_dict = current_dict[part]
            
            # Set the value
            current_dict[key_parts[-1]] = value
            
            # Validate the modified configuration
            validation_result = self.validate_config(config_dict)
            if not validation_result.success:
                return OperationResult.error_result(f"Invalid configuration after setting {key}: {validation_result.error}")
            
            # Apply the change to current config
            self.config = SystemConfig.from_dict(config_dict)
            
            debug_tracker.log_operation("config_manager", "set_config_success", {
                "key": key,
                "value_type": type(value).__name__
            }, success=True)
            
            return OperationResult.success_result(True, {
                "key": key,
                "validation_passed": True
            })
        
        except Exception as e:
            debug_tracker.log_operation("config_manager", "set_config_failed", {
                "key": key,
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Failed to set configuration: {e}")
    
    def get_mem0_config(self) -> Dict[str, Any]:
        """
        Get mem0-compatible configuration
        
        Returns:
            Dictionary with mem0 configuration format
        """
        return {
            "version": "v1.1",
            "graph_store": {
                "provider": "neo4j",
                "config": {
                    "url": self.config.database.neo4j_url,
                    "username": self.config.database.neo4j_username,
                    "password": self.config.database.neo4j_password
                }
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": self.config.database.qdrant_host,
                    "port": self.config.database.qdrant_port,
                    "collection_name": self.config.database.qdrant_collection
                }
            }
        }
    
    def override_config(self, overrides: Dict[str, Any]) -> OperationResult[bool]:
        """
        Apply configuration overrides for specific use cases
        
        Args:
            overrides: Dictionary of configuration overrides
            
        Returns:
            OperationResult[bool] indicating success/failure
        """
        debug_tracker.log_operation("config_manager", "override_config_start", {
            "override_keys": list(overrides.keys())
        })
        
        try:
            # Create new config with overrides
            current_dict = self.config.to_dict()
            
            # Apply nested overrides
            def apply_overrides(base: Dict, overrides: Dict):
                for key, value in overrides.items():
                    if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                        apply_overrides(base[key], value)
                    else:
                        base[key] = value
            
            apply_overrides(current_dict, overrides)
            
            # Create new config and validate
            new_config = SystemConfig.from_dict(current_dict)
            validation_result = new_config.validate()
            
            if not validation_result.success:
                debug_tracker.log_operation("config_manager", "override_config_failed", {
                    "error": validation_result.error
                }, success=False, error=validation_result.error)
                
                return OperationResult.error_result(f"Override validation failed: {validation_result.error}")
            
            self.config = new_config
            
            debug_tracker.log_operation("config_manager", "override_config_success", {
                "overrides_applied": len(overrides)
            }, success=True)
            
            return OperationResult.success_result(True, {
                "overrides_applied": len(overrides),
                "validation_passed": True
            })
            
        except Exception as e:
            debug_tracker.log_operation("config_manager", "override_config_failed", {
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Failed to apply configuration overrides: {e}")


# Global configuration manager instance
config_manager = ConfigManager()