"""
Foundation Interfaces for Buddy Memory System
Defines standard contracts that all components must implement
Ensures predictable APIs for multi-agent use
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

# Import our standard result types
from result_types import (
    OperationResult, MemorySearchResult, MemoryAddResult, 
    HealthCheckResult, ContextLogResult, DebugInfo,
    MemorySearchOperationResult, MemoryAddOperationResult,
    HealthCheckOperationResult, ContextLogOperationResult,
    DebugInfoOperationResult, StringOperationResult,
    BoolOperationResult, DictOperationResult
)


class MemoryFoundationInterface(ABC):
    """
    Standard interface that all memory components must implement
    Provides consistent API for memory operations across the system
    """
    
    @abstractmethod
    def search_memories(self, query: str, user_id: str, **options) -> MemorySearchOperationResult:
        """
        Search memories with standardized result format
        
        Args:
            query: Search query string
            user_id: User identifier
            **options: Additional search options (limit, enable_expansion, etc.)
            
        Returns:
            OperationResult[MemorySearchResult] with success/error status
        """
        pass
    
    @abstractmethod
    def add_memory(self, data: Any, user_id: str, **options) -> MemoryAddOperationResult:
        """
        Add memory with standardized result format
        
        Args:
            data: Memory data (string, dict, or list of messages)
            user_id: User identifier
            **options: Additional options (metadata, etc.)
            
        Returns:
            OperationResult[MemoryAddResult] with success/error status
        """
        pass
    
    @abstractmethod
    def update_memory(self, memory_id: str, data: Any, user_id: str, **options) -> BoolOperationResult:
        """
        Update existing memory
        
        Args:
            memory_id: ID of memory to update
            data: New memory data
            user_id: User identifier
            **options: Additional update options
            
        Returns:
            OperationResult[bool] indicating success/failure
        """
        pass
    
    @abstractmethod
    def delete_memory(self, memory_id: str, user_id: str, **options) -> BoolOperationResult:
        """
        Delete/archive memory
        
        Args:
            memory_id: ID of memory to delete
            user_id: User identifier
            **options: Additional delete options
            
        Returns:
            OperationResult[bool] indicating success/failure
        """
        pass
    
    @abstractmethod
    def get_all_memories(self, user_id: str, **options) -> DictOperationResult:
        """
        Get all memories for a user
        
        Args:
            user_id: User identifier
            **options: Additional retrieval options
            
        Returns:
            OperationResult[Dict] with memories data
        """
        pass
    
    @abstractmethod
    def get_health_status(self) -> HealthCheckOperationResult:
        """
        Get component health status
        
        Returns:
            OperationResult[HealthCheckResult] with health information
        """
        pass
    
    @abstractmethod
    def get_debug_info(self) -> DebugInfoOperationResult:
        """
        Get debug information for troubleshooting
        
        Returns:
            OperationResult[DebugInfo] with debug data
        """
        pass


class ContextFoundationInterface(ABC):
    """
    Standard interface for context components
    Provides consistent API for context operations
    """
    
    @abstractmethod
    def expand_context(self, data: Dict[str, Any], query: str, **options) -> DictOperationResult:
        """
        Expand context with standardized result format
        
        Args:
            data: Data to expand (search results, memories, etc.)
            query: Original query for context
            **options: Expansion options
            
        Returns:
            OperationResult[Dict] with expanded data
        """
        pass
    
    @abstractmethod
    def log_interaction(self, interaction_data: Dict[str, Any], **options) -> ContextLogOperationResult:
        """
        Log interaction and return lookup code
        
        Args:
            interaction_data: Complete interaction data to log
            **options: Logging options
            
        Returns:
            OperationResult[ContextLogResult] with lookup code
        """
        pass
    
    @abstractmethod
    def retrieve_context(self, lookup_code: str, **options) -> DictOperationResult:
        """
        Retrieve context by lookup code
        
        Args:
            lookup_code: Context lookup code
            **options: Retrieval options
            
        Returns:
            OperationResult[Dict] with context data
        """
        pass
    
    @abstractmethod
    def get_health_status(self) -> HealthCheckOperationResult:
        """
        Get component health status
        
        Returns:
            OperationResult[HealthCheckResult] with health information
        """
        pass
    
    @abstractmethod
    def get_debug_info(self) -> DebugInfoOperationResult:
        """
        Get debug information for troubleshooting
        
        Returns:
            OperationResult[DebugInfo] with debug data
        """
        pass


class ConfigurationInterface(ABC):
    """
    Standard interface for configuration management
    Provides consistent API for configuration operations
    """
    
    @abstractmethod
    def load_config(self, config_source: Optional[str] = None) -> DictOperationResult:
        """
        Load configuration from source
        
        Args:
            config_source: Optional config source (file path, etc.)
            
        Returns:
            OperationResult[Dict] with configuration data
        """
        pass
    
    @abstractmethod
    def save_config(self, config_data: Dict[str, Any], destination: Optional[str] = None) -> BoolOperationResult:
        """
        Save configuration to destination
        
        Args:
            config_data: Configuration data to save
            destination: Optional save destination
            
        Returns:
            OperationResult[bool] indicating success/failure
        """
        pass
    
    @abstractmethod
    def validate_config(self, config_data: Dict[str, Any]) -> BoolOperationResult:
        """
        Validate configuration data
        
        Args:
            config_data: Configuration to validate
            
        Returns:
            OperationResult[bool] with validation result
        """
        pass
    
    @abstractmethod
    def get_config(self, key: Optional[str] = None) -> DictOperationResult:
        """
        Get configuration value(s)
        
        Args:
            key: Optional specific config key, None for all config
            
        Returns:
            OperationResult[Dict] with config data
        """
        pass
    
    @abstractmethod
    def set_config(self, key: str, value: Any) -> BoolOperationResult:
        """
        Set configuration value
        
        Args:
            key: Configuration key
            value: Configuration value
            
        Returns:
            OperationResult[bool] indicating success/failure
        """
        pass


class AgentInterface(ABC):
    """
    Standard interface for agent components
    Defines how agents interact with the foundation
    """
    
    @abstractmethod
    def process_request(self, request: Dict[str, Any], context: Dict[str, Any]) -> DictOperationResult:
        """
        Process agent request with context
        
        Args:
            request: Agent request data
            context: Request context (user_id, session, etc.)
            
        Returns:
            OperationResult[Dict] with response data
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> DictOperationResult:
        """
        Get agent capabilities and metadata
        
        Returns:
            OperationResult[Dict] with capability information
        """
        pass
    
    @abstractmethod
    def validate_request(self, request: Dict[str, Any]) -> BoolOperationResult:
        """
        Validate agent request format
        
        Args:
            request: Request to validate
            
        Returns:
            OperationResult[bool] with validation result
        """
        pass
    
    @abstractmethod
    def get_health_status(self) -> HealthCheckOperationResult:
        """
        Get agent health status
        
        Returns:
            OperationResult[HealthCheckResult] with health information
        """
        pass
    
    @abstractmethod
    def get_debug_info(self) -> DebugInfoOperationResult:
        """
        Get debug information for troubleshooting
        
        Returns:
            OperationResult[DebugInfo] with debug data
        """
        pass


class PluginInterface(ABC):
    """
    Standard interface for plugin components
    Defines how plugins extend the system
    """
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> BoolOperationResult:
        """
        Initialize plugin with configuration
        
        Args:
            config: Plugin configuration
            
        Returns:
            OperationResult[bool] indicating success/failure
        """
        pass
    
    @abstractmethod
    def execute(self, action: str, params: Dict[str, Any]) -> DictOperationResult:
        """
        Execute plugin action
        
        Args:
            action: Action to execute
            params: Action parameters
            
        Returns:
            OperationResult[Dict] with action result
        """
        pass
    
    @abstractmethod
    def get_actions(self) -> DictOperationResult:
        """
        Get available plugin actions
        
        Returns:
            OperationResult[Dict] with action definitions
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> BoolOperationResult:
        """
        Shutdown plugin cleanly
        
        Returns:
            OperationResult[bool] indicating success/failure
        """
        pass
    
    @abstractmethod
    def get_health_status(self) -> HealthCheckOperationResult:
        """
        Get plugin health status
        
        Returns:
            OperationResult[HealthCheckResult] with health information
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> DictOperationResult:
        """
        Get plugin metadata (name, version, etc.)
        
        Returns:
            OperationResult[Dict] with metadata
        """
        pass