"""
Plugin System for Buddy Memory System Phase 3: Extension Architecture
Provides protocols and management for custom memory processors, context expanders, and filters
NO silent failures - all operations are loud and proud
"""

from typing import Dict, List, Any, Optional, Type, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from result_types import OperationResult
from debug_info import debug_tracker
from exceptions import ConfigurationError
import time


class MemoryProcessor(Protocol):
    """Protocol for custom memory processors"""
    
    def process_memory_input(self, data: Any, user_id: str, **kwargs) -> Dict[str, Any]:
        """Process memory input before adding to storage"""
        ...
    
    def process_memory_output(self, results: List[Dict[str, Any]], query: str, **kwargs) -> List[Dict[str, Any]]:
        """Process memory results after retrieval"""
        ...


class ContextExpander(Protocol):
    """Protocol for custom context expanders"""
    
    def should_expand(self, memory_result: Dict[str, Any], query: str, **kwargs) -> bool:
        """Determine if context should be expanded"""
        ...
    
    def expand_context(self, memory_result: Dict[str, Any], query: str, **kwargs) -> Dict[str, Any]:
        """Expand context for memory result"""
        ...


class MemoryFilter(Protocol):
    """Protocol for custom memory filters"""
    
    def filter_memories(self, memories: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter memories based on criteria"""
        ...


@dataclass
class PluginInfo:
    """Information about a registered plugin"""
    name: str
    plugin_type: str
    plugin_class: Type
    instance: Any
    priority: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)
    registration_time: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "plugin_type": self.plugin_type,
            "plugin_class": self.plugin_class.__name__,
            "priority": self.priority,
            "metadata": self.metadata,
            "registration_time": self.registration_time
        }


class PluginManager:
    """
    Manages plugins and extensions for the memory system
    Provides registration, validation, and execution coordination for plugins
    """
    
    def __init__(self):
        self.plugins: Dict[str, PluginInfo] = {}
        self.processors: Dict[str, MemoryProcessor] = {}
        self.expanders: Dict[str, ContextExpander] = {}
        self.filters: Dict[str, MemoryFilter] = {}
        self.initialization_time = time.time()
        
        debug_tracker.log_operation("plugin_manager", "initialization", {})
    
    def register_memory_processor(self, name: str, processor: MemoryProcessor, 
                                priority: int = 100, metadata: Optional[Dict[str, Any]] = None) -> OperationResult[bool]:
        """
        Register a custom memory processor
        
        Args:
            name: Unique name for the processor
            processor: Instance implementing MemoryProcessor protocol
            priority: Execution priority (lower numbers execute first)
            metadata: Optional metadata about the processor
            
        Returns:
            OperationResult[bool] indicating success/failure
        """
        debug_tracker.log_operation("plugin_manager", "register_processor_start", {
            "name": name,
            "priority": priority,
            "processor_type": type(processor).__name__
        })
        
        try:
            # Validate processor name
            if not name or not name.strip():
                raise ConfigurationError("Processor name cannot be empty")
            
            if name in self.plugins:
                raise ConfigurationError(f"Plugin with name '{name}' already exists")
            
            # Validate processor implements required methods
            if not hasattr(processor, 'process_memory_input'):
                raise ConfigurationError(f"Processor {name} missing 'process_memory_input' method")
            
            if not hasattr(processor, 'process_memory_output'):
                raise ConfigurationError(f"Processor {name} missing 'process_memory_output' method")
            
            # Validate priority
            if not isinstance(priority, int) or priority < 0:
                raise ConfigurationError(f"Priority must be non-negative integer, got {priority}")
            
            # Test the processor methods are callable
            if not callable(getattr(processor, 'process_memory_input')):
                raise ConfigurationError(f"Processor {name} 'process_memory_input' is not callable")
            
            if not callable(getattr(processor, 'process_memory_output')):
                raise ConfigurationError(f"Processor {name} 'process_memory_output' is not callable")
            
            plugin_info = PluginInfo(
                name=name,
                plugin_type="memory_processor",
                plugin_class=type(processor),
                instance=processor,
                priority=priority,
                metadata=metadata or {}
            )
            
            self.plugins[name] = plugin_info
            self.processors[name] = processor
            
            debug_tracker.log_operation("plugin_manager", "register_processor_success", {
                "name": name,
                "priority": priority,
                "processor_type": type(processor).__name__,
                "total_processors": len(self.processors)
            }, success=True)
            
            return OperationResult.success_result(True, {"registered_name": name, "plugin_type": "memory_processor"})
            
        except Exception as e:
            debug_tracker.log_operation("plugin_manager", "register_processor_failed", {
                "name": name,
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Failed to register processor {name}: {e}")
    
    def register_context_expander(self, name: str, expander: ContextExpander,
                                 priority: int = 100, metadata: Optional[Dict[str, Any]] = None) -> OperationResult[bool]:
        """
        Register a custom context expander
        
        Args:
            name: Unique name for the expander
            expander: Instance implementing ContextExpander protocol
            priority: Execution priority (lower numbers execute first)
            metadata: Optional metadata about the expander
            
        Returns:
            OperationResult[bool] indicating success/failure
        """
        debug_tracker.log_operation("plugin_manager", "register_expander_start", {
            "name": name,
            "priority": priority,
            "expander_type": type(expander).__name__
        })
        
        try:
            # Validate expander name
            if not name or not name.strip():
                raise ConfigurationError("Expander name cannot be empty")
            
            if name in self.plugins:
                raise ConfigurationError(f"Plugin with name '{name}' already exists")
            
            # Validate expander implements required methods
            if not hasattr(expander, 'should_expand'):
                raise ConfigurationError(f"Expander {name} missing 'should_expand' method")
            
            if not hasattr(expander, 'expand_context'):
                raise ConfigurationError(f"Expander {name} missing 'expand_context' method")
            
            # Validate priority
            if not isinstance(priority, int) or priority < 0:
                raise ConfigurationError(f"Priority must be non-negative integer, got {priority}")
            
            # Test the expander methods are callable
            if not callable(getattr(expander, 'should_expand')):
                raise ConfigurationError(f"Expander {name} 'should_expand' is not callable")
            
            if not callable(getattr(expander, 'expand_context')):
                raise ConfigurationError(f"Expander {name} 'expand_context' is not callable")
            
            plugin_info = PluginInfo(
                name=name,
                plugin_type="context_expander",
                plugin_class=type(expander),
                instance=expander,
                priority=priority,
                metadata=metadata or {}
            )
            
            self.plugins[name] = plugin_info
            self.expanders[name] = expander
            
            debug_tracker.log_operation("plugin_manager", "register_expander_success", {
                "name": name,
                "priority": priority,
                "expander_type": type(expander).__name__,
                "total_expanders": len(self.expanders)
            }, success=True)
            
            return OperationResult.success_result(True, {"registered_name": name, "plugin_type": "context_expander"})
            
        except Exception as e:
            debug_tracker.log_operation("plugin_manager", "register_expander_failed", {
                "name": name,
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Failed to register expander {name}: {e}")
    
    def register_memory_filter(self, name: str, filter_instance: MemoryFilter,
                              priority: int = 100, metadata: Optional[Dict[str, Any]] = None) -> OperationResult[bool]:
        """
        Register a custom memory filter
        
        Args:
            name: Unique name for the filter
            filter_instance: Instance implementing MemoryFilter protocol
            priority: Execution priority (lower numbers execute first)
            metadata: Optional metadata about the filter
            
        Returns:
            OperationResult[bool] indicating success/failure
        """
        debug_tracker.log_operation("plugin_manager", "register_filter_start", {
            "name": name,
            "priority": priority,
            "filter_type": type(filter_instance).__name__
        })
        
        try:
            # Validate filter name
            if not name or not name.strip():
                raise ConfigurationError("Filter name cannot be empty")
            
            if name in self.plugins:
                raise ConfigurationError(f"Plugin with name '{name}' already exists")
            
            # Validate filter implements required methods
            if not hasattr(filter_instance, 'filter_memories'):
                raise ConfigurationError(f"Filter {name} missing 'filter_memories' method")
            
            # Validate priority
            if not isinstance(priority, int) or priority < 0:
                raise ConfigurationError(f"Priority must be non-negative integer, got {priority}")
            
            # Test the filter method is callable
            if not callable(getattr(filter_instance, 'filter_memories')):
                raise ConfigurationError(f"Filter {name} 'filter_memories' is not callable")
            
            plugin_info = PluginInfo(
                name=name,
                plugin_type="memory_filter",
                plugin_class=type(filter_instance),
                instance=filter_instance,
                priority=priority,
                metadata=metadata or {}
            )
            
            self.plugins[name] = plugin_info
            self.filters[name] = filter_instance
            
            debug_tracker.log_operation("plugin_manager", "register_filter_success", {
                "name": name,
                "priority": priority,
                "filter_type": type(filter_instance).__name__,
                "total_filters": len(self.filters)
            }, success=True)
            
            return OperationResult.success_result(True, {"registered_name": name, "plugin_type": "memory_filter"})
            
        except Exception as e:
            debug_tracker.log_operation("plugin_manager", "register_filter_failed", {
                "name": name,
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Failed to register filter {name}: {e}")
    
    def get_processors_by_priority(self) -> List[MemoryProcessor]:
        """Get memory processors sorted by priority (lowest first)"""
        try:
            processor_plugins = [p for p in self.plugins.values() if p.plugin_type == "memory_processor"]
            processor_plugins.sort(key=lambda p: p.priority)
            return [p.instance for p in processor_plugins]
        except Exception as e:
            debug_tracker.log_operation("plugin_manager", "get_processors_failed", {
                "error": str(e)
            }, success=False, error=str(e))
            raise RuntimeError(f"Failed to get processors by priority: {e}")
    
    def get_expanders_by_priority(self) -> List[ContextExpander]:
        """Get context expanders sorted by priority (lowest first)"""
        try:
            expander_plugins = [p for p in self.plugins.values() if p.plugin_type == "context_expander"]
            expander_plugins.sort(key=lambda p: p.priority)
            return [p.instance for p in expander_plugins]
        except Exception as e:
            debug_tracker.log_operation("plugin_manager", "get_expanders_failed", {
                "error": str(e)
            }, success=False, error=str(e))
            raise RuntimeError(f"Failed to get expanders by priority: {e}")
    
    def get_filters_by_priority(self) -> List[MemoryFilter]:
        """Get memory filters sorted by priority (lowest first)"""
        try:
            filter_plugins = [p for p in self.plugins.values() if p.plugin_type == "memory_filter"]
            filter_plugins.sort(key=lambda p: p.priority)
            return [p.instance for p in filter_plugins]
        except Exception as e:
            debug_tracker.log_operation("plugin_manager", "get_filters_failed", {
                "error": str(e)
            }, success=False, error=str(e))
            raise RuntimeError(f"Failed to get filters by priority: {e}")
    
    def unregister_plugin(self, name: str) -> OperationResult[bool]:
        """
        Unregister a plugin
        
        Args:
            name: Name of the plugin to unregister
            
        Returns:
            OperationResult[bool] indicating success/failure
        """
        debug_tracker.log_operation("plugin_manager", "unregister_plugin_start", {
            "name": name
        })
        
        try:
            if not name:
                raise ConfigurationError("Plugin name cannot be empty")
            
            if name not in self.plugins:
                raise ConfigurationError(f"Plugin '{name}' not found")
            
            plugin_info = self.plugins[name]
            plugin_type = plugin_info.plugin_type
            
            # Remove from appropriate collection
            if plugin_type == "memory_processor":
                del self.processors[name]
            elif plugin_type == "context_expander":
                del self.expanders[name]
            elif plugin_type == "memory_filter":
                del self.filters[name]
            else:
                raise ConfigurationError(f"Unknown plugin type: {plugin_type}")
            
            del self.plugins[name]
            
            debug_tracker.log_operation("plugin_manager", "unregister_plugin_success", {
                "name": name,
                "plugin_type": plugin_type,
                "remaining_plugins": len(self.plugins)
            }, success=True)
            
            return OperationResult.success_result(True, {"unregistered_name": name, "plugin_type": plugin_type})
            
        except Exception as e:
            debug_tracker.log_operation("plugin_manager", "unregister_plugin_failed", {
                "name": name,
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Failed to unregister plugin {name}: {e}")
    
    def get_plugin_info(self, name: str) -> OperationResult[PluginInfo]:
        """
        Get information about a specific plugin
        
        Args:
            name: Name of the plugin
            
        Returns:
            OperationResult[PluginInfo] with plugin information
        """
        try:
            if not name:
                raise ConfigurationError("Plugin name cannot be empty")
            
            if name not in self.plugins:
                raise ConfigurationError(f"Plugin '{name}' not found")
            
            return OperationResult.success_result(self.plugins[name])
            
        except Exception as e:
            return OperationResult.error_result(f"Failed to get plugin info for {name}: {e}")
    
    def list_plugins(self) -> OperationResult[Dict[str, List[Dict[str, Any]]]]:
        """
        List all registered plugins by type
        
        Returns:
            OperationResult[Dict] with plugins organized by type
        """
        try:
            plugin_summary = {
                "memory_processors": [],
                "context_expanders": [],
                "memory_filters": []
            }
            
            for plugin_info in self.plugins.values():
                plugin_data = {
                    "name": plugin_info.name,
                    "class": plugin_info.plugin_class.__name__,
                    "priority": plugin_info.priority,
                    "metadata": plugin_info.metadata,
                    "registration_time": plugin_info.registration_time
                }
                
                if plugin_info.plugin_type == "memory_processor":
                    plugin_summary["memory_processors"].append(plugin_data)
                elif plugin_info.plugin_type == "context_expander":
                    plugin_summary["context_expanders"].append(plugin_data)
                elif plugin_info.plugin_type == "memory_filter":
                    plugin_summary["memory_filters"].append(plugin_data)
            
            # Sort each type by priority
            for plugin_type in plugin_summary:
                plugin_summary[plugin_type].sort(key=lambda p: p["priority"])
            
            debug_tracker.log_operation("plugin_manager", "list_plugins_success", {
                "total_plugins": len(self.plugins),
                "processors": len(plugin_summary["memory_processors"]),
                "expanders": len(plugin_summary["context_expanders"]),
                "filters": len(plugin_summary["memory_filters"])
            }, success=True)
            
            return OperationResult.success_result(plugin_summary, {
                "total_plugins": len(self.plugins),
                "uptime_seconds": time.time() - self.initialization_time
            })
            
        except Exception as e:
            debug_tracker.log_operation("plugin_manager", "list_plugins_failed", {
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Failed to list plugins: {e}")
    
    def validate_all_plugins(self) -> OperationResult[Dict[str, Any]]:
        """
        Validate all registered plugins are still functional
        
        Returns:
            OperationResult[Dict] with validation results
        """
        try:
            validation_results = {
                "total_plugins": len(self.plugins),
                "valid_plugins": 0,
                "invalid_plugins": 0,
                "validation_errors": []
            }
            
            for name, plugin_info in self.plugins.items():
                try:
                    instance = plugin_info.instance
                    plugin_type = plugin_info.plugin_type
                    
                    # Validate based on plugin type
                    if plugin_type == "memory_processor":
                        if not (hasattr(instance, 'process_memory_input') and 
                                hasattr(instance, 'process_memory_output')):
                            raise RuntimeError(f"Processor {name} missing required methods")
                    elif plugin_type == "context_expander":
                        if not (hasattr(instance, 'should_expand') and 
                                hasattr(instance, 'expand_context')):
                            raise RuntimeError(f"Expander {name} missing required methods")
                    elif plugin_type == "memory_filter":
                        if not hasattr(instance, 'filter_memories'):
                            raise RuntimeError(f"Filter {name} missing required methods")
                    
                    validation_results["valid_plugins"] += 1
                    
                except Exception as e:
                    validation_results["invalid_plugins"] += 1
                    validation_results["validation_errors"].append({
                        "plugin_name": name,
                        "plugin_type": plugin_info.plugin_type,
                        "error": str(e)
                    })
            
            overall_valid = validation_results["invalid_plugins"] == 0
            
            debug_tracker.log_operation("plugin_manager", "validate_all_plugins", {
                "total_plugins": validation_results["total_plugins"],
                "valid_plugins": validation_results["valid_plugins"],
                "invalid_plugins": validation_results["invalid_plugins"],
                "overall_valid": overall_valid
            }, success=overall_valid)
            
            return OperationResult.success_result(validation_results, {
                "overall_valid": overall_valid
            })
            
        except Exception as e:
            return OperationResult.error_result(f"Plugin validation failed: {e}")
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get plugin manager statistics"""
        try:
            return {
                "total_plugins": len(self.plugins),
                "memory_processors": len(self.processors),
                "context_expanders": len(self.expanders),
                "memory_filters": len(self.filters),
                "initialization_time": self.initialization_time,
                "uptime_seconds": time.time() - self.initialization_time,
                "plugin_types_available": ["memory_processor", "context_expander", "memory_filter"]
            }
        except Exception as e:
            debug_tracker.log_operation("plugin_manager", "get_stats_failed", {
                "error": str(e)
            }, success=False, error=str(e))
            raise RuntimeError(f"Failed to get plugin manager stats: {e}")


# Global plugin manager instance
plugin_manager = PluginManager()