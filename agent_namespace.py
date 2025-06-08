"""
Agent Namespace System for Buddy Memory System Phase 3: Extension Architecture
Provides isolated namespaces for agent-specific memory operations with custom configurations
NO silent failures - all operations are loud and proud
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from result_types import OperationResult, MemorySearchOperationResult, MemoryAddOperationResult
from config_manager import SystemConfig, config_manager
from plugin_system import plugin_manager, PluginManager, MemoryProcessor, ContextExpander, MemoryFilter
from debug_info import debug_tracker
from exceptions import ConfigurationError
import copy
import time


@dataclass
class AgentConfig:
    """Configuration specific to an agent"""
    agent_id: str
    agent_type: str
    base_config: SystemConfig
    overrides: Dict[str, Any] = field(default_factory=dict)
    custom_processors: List[str] = field(default_factory=list)
    custom_expanders: List[str] = field(default_factory=list)
    custom_filters: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate agent configuration on creation"""
        if not self.agent_id or not self.agent_id.strip():
            raise ConfigurationError("Agent ID cannot be empty")
        
        if not self.agent_type or not self.agent_type.strip():
            raise ConfigurationError("Agent type cannot be empty")
        
        # Validate agent_id format (alphanumeric + underscores/hyphens only)
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', self.agent_id):
            raise ConfigurationError(f"Agent ID '{self.agent_id}' contains invalid characters. Use only alphanumeric, underscore, or hyphen.")
        
        if not isinstance(self.base_config, SystemConfig):
            raise ConfigurationError("Base config must be a SystemConfig instance")
    
    def get_effective_config(self) -> SystemConfig:
        """Get configuration with agent-specific overrides applied"""
        try:
            # Deep copy base config
            config_dict = self.base_config.to_dict()
            
            # Apply overrides using recursive merge
            def apply_overrides(base: Dict, overrides: Dict):
                for key, value in overrides.items():
                    if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                        apply_overrides(base[key], value)
                    else:
                        base[key] = value
            
            apply_overrides(config_dict, self.overrides)
            
            # Create new config from merged dictionary
            effective_config = SystemConfig.from_dict(config_dict)
            
            return effective_config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create effective config for agent {self.agent_id}: {e}")
    
    def validate_plugin_references(self, plugin_manager: PluginManager) -> OperationResult[bool]:
        """Validate that all referenced plugins exist"""
        try:
            missing_plugins = []
            
            # Check processors
            for processor_name in self.custom_processors:
                if processor_name not in plugin_manager.processors:
                    missing_plugins.append(f"processor: {processor_name}")
            
            # Check expanders
            for expander_name in self.custom_expanders:
                if expander_name not in plugin_manager.expanders:
                    missing_plugins.append(f"expander: {expander_name}")
            
            # Check filters
            for filter_name in self.custom_filters:
                if filter_name not in plugin_manager.filters:
                    missing_plugins.append(f"filter: {filter_name}")
            
            if missing_plugins:
                return OperationResult.error_result(f"Missing plugins: {', '.join(missing_plugins)}")
            
            return OperationResult.success_result(True)
            
        except Exception as e:
            return OperationResult.error_result(f"Plugin validation failed: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "overrides": self.overrides,
            "custom_processors": self.custom_processors,
            "custom_expanders": self.custom_expanders,
            "custom_filters": self.custom_filters,
            "metadata": self.metadata,
            "creation_time": self.creation_time
        }


class AgentNamespace:
    """
    Provides isolated namespace for agent-specific memory operations
    Applies agent-specific processing, filtering, and configuration
    """
    
    def __init__(self, agent_config: AgentConfig, enhanced_memory, plugin_manager: PluginManager):
        if not agent_config:
            raise ConfigurationError("Agent config cannot be None")
        
        if not enhanced_memory:
            raise ConfigurationError("Enhanced memory cannot be None")
        
        if not plugin_manager:
            raise ConfigurationError("Plugin manager cannot be None")
        
        self.agent_config = agent_config
        self.enhanced_memory = enhanced_memory
        self.plugin_manager = plugin_manager
        self.operation_count = 0
        self.initialization_time = time.time()
        
        # Get effective configuration
        try:
            self.effective_config = agent_config.get_effective_config()
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize agent namespace: {e}")
        
        # Validate plugin references
        validation_result = agent_config.validate_plugin_references(plugin_manager)
        if not validation_result.success:
            raise ConfigurationError(f"Agent namespace validation failed: {validation_result.error}")
        
        debug_tracker.log_operation("agent_namespace", "initialization", {
            "agent_id": agent_config.agent_id,
            "agent_type": agent_config.agent_type,
            "custom_processors": len(agent_config.custom_processors),
            "custom_expanders": len(agent_config.custom_expanders),
            "custom_filters": len(agent_config.custom_filters),
            "config_overrides": bool(agent_config.overrides)
        })
    
    def _get_agent_user_id(self, user_id: str = "default") -> str:
        """Generate agent-specific user ID with namespace isolation"""
        base_user_id = user_id or "default"
        return f"{self.agent_config.agent_id}_{base_user_id}"
    
    def search_memories(self, query: str, user_id: str = "default", **options) -> MemorySearchOperationResult:
        """
        Agent-specific memory search with custom processing
        
        Args:
            query: Search query
            user_id: User identifier (will be prefixed with agent_id)
            **options: Additional search options
            
        Returns:
            OperationResult[MemorySearchResult] with processed results
        """
        operation_id = f"agent_search_{int(time.time() * 1000)}"
        self.operation_count += 1
        
        debug_tracker.log_operation("agent_namespace", "search_start", {
            "agent_id": self.agent_config.agent_id,
            "operation_id": operation_id,
            "query": query[:100],
            "user_id": user_id,
            "operation_count": self.operation_count
        })
        
        try:
            if not query or not query.strip():
                raise ConfigurationError("Search query cannot be empty")
            
            # Apply agent-specific user_id prefix for namespace isolation
            agent_user_id = self._get_agent_user_id(user_id)
            
            # Get base search results from enhanced memory
            search_result = self.enhanced_memory.search_memories(query, agent_user_id, **options)
            
            if not search_result.success:
                debug_tracker.log_operation("agent_namespace", "search_base_failed", {
                    "agent_id": self.agent_config.agent_id,
                    "operation_id": operation_id,
                    "error": search_result.error
                }, success=False, error=search_result.error)
                
                return search_result
            
            # Apply agent-specific processing
            processed_results = search_result.data.results
            original_count = len(processed_results)
            
            # Apply custom memory filters in priority order
            for filter_name in self.agent_config.custom_filters:
                if filter_name in self.plugin_manager.filters:
                    filter_instance = self.plugin_manager.filters[filter_name]
                    
                    try:
                        processed_results = filter_instance.filter_memories(
                            processed_results, 
                            {
                                "agent_id": self.agent_config.agent_id,
                                "agent_type": self.agent_config.agent_type,
                                "query": query,
                                "user_id": user_id
                            }
                        )
                    except Exception as e:
                        raise RuntimeError(f"Filter '{filter_name}' failed: {e}")
            
            # Apply custom processors (output processing) in priority order
            for processor_name in self.agent_config.custom_processors:
                if processor_name in self.plugin_manager.processors:
                    processor = self.plugin_manager.processors[processor_name]
                    
                    try:
                        processed_results = processor.process_memory_output(
                            processed_results, 
                            query,
                            agent_id=self.agent_config.agent_id,
                            agent_type=self.agent_config.agent_type,
                            user_id=user_id
                        )
                    except Exception as e:
                        raise RuntimeError(f"Processor '{processor_name}' failed: {e}")
            
            # Update search result with processed data
            search_result.data.results = processed_results
            search_result.data.total_count = len(processed_results)
            
            # Add agent processing metadata
            search_result.metadata["agent_processing"] = {
                "agent_id": self.agent_config.agent_id,
                "agent_type": self.agent_config.agent_type,
                "filters_applied": self.agent_config.custom_filters,
                "processors_applied": self.agent_config.custom_processors,
                "original_count": original_count,
                "processed_count": len(processed_results),
                "operation_id": operation_id
            }
            
            debug_tracker.log_operation("agent_namespace", "search_complete", {
                "agent_id": self.agent_config.agent_id,
                "operation_id": operation_id,
                "query": query[:100],
                "original_results": original_count,
                "processed_results": len(processed_results),
                "filters_applied": len(self.agent_config.custom_filters),
                "processors_applied": len(self.agent_config.custom_processors)
            }, success=True)
            
            return search_result
            
        except Exception as e:
            debug_tracker.log_operation("agent_namespace", "search_failed", {
                "agent_id": self.agent_config.agent_id,
                "operation_id": operation_id,
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Agent search failed: {e}")
    
    def add_memory(self, data: Any, user_id: str = "default", **options) -> MemoryAddOperationResult:
        """
        Agent-specific memory addition with custom processing
        
        Args:
            data: Memory data to add
            user_id: User identifier (will be prefixed with agent_id)
            **options: Additional options
            
        Returns:
            OperationResult[MemoryAddResult] with addition results
        """
        operation_id = f"agent_add_{int(time.time() * 1000)}"
        self.operation_count += 1
        
        debug_tracker.log_operation("agent_namespace", "add_start", {
            "agent_id": self.agent_config.agent_id,
            "operation_id": operation_id,
            "data_type": type(data).__name__,
            "user_id": user_id,
            "operation_count": self.operation_count
        })
        
        try:
            if data is None:
                raise ConfigurationError("Memory data cannot be None")
            
            # Apply agent-specific user_id prefix for namespace isolation
            agent_user_id = self._get_agent_user_id(user_id)
            
            # Apply custom input processors in priority order
            processed_data = data
            for processor_name in self.agent_config.custom_processors:
                if processor_name in self.plugin_manager.processors:
                    processor = self.plugin_manager.processors[processor_name]
                    
                    try:
                        processed_data = processor.process_memory_input(
                            processed_data, 
                            agent_user_id,
                            agent_id=self.agent_config.agent_id,
                            agent_type=self.agent_config.agent_type,
                            user_id=user_id
                        )
                    except Exception as e:
                        raise RuntimeError(f"Input processor '{processor_name}' failed: {e}")
            
            # Add agent metadata to options
            if 'metadata' not in options:
                options['metadata'] = {}
            
            options['metadata'].update({
                'agent_id': self.agent_config.agent_id,
                'agent_type': self.agent_config.agent_type,
                'processed_by': self.agent_config.custom_processors,
                'agent_operation_id': operation_id,
                'agent_creation_time': self.agent_config.creation_time
            })
            
            # Add to memory using enhanced memory
            add_result = self.enhanced_memory.add_memory(processed_data, agent_user_id, **options)
            
            if add_result.success:
                # Add agent processing metadata
                add_result.metadata["agent_processing"] = {
                    "agent_id": self.agent_config.agent_id,
                    "agent_type": self.agent_config.agent_type,
                    "processors_applied": self.agent_config.custom_processors,
                    "operation_id": operation_id
                }
            
            debug_tracker.log_operation("agent_namespace", "add_complete", {
                "agent_id": self.agent_config.agent_id,
                "operation_id": operation_id,
                "data_type": type(data).__name__,
                "processors_applied": len(self.agent_config.custom_processors),
                "success": add_result.success
            }, success=add_result.success)
            
            return add_result
            
        except Exception as e:
            debug_tracker.log_operation("agent_namespace", "add_failed", {
                "agent_id": self.agent_config.agent_id,
                "operation_id": operation_id,
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Agent add memory failed: {e}")
    
    def get_agent_stats(self) -> OperationResult[Dict[str, Any]]:
        """Get statistics for this agent namespace"""
        try:
            stats = {
                "agent_id": self.agent_config.agent_id,
                "agent_type": self.agent_config.agent_type,
                "creation_time": self.agent_config.creation_time,
                "uptime_seconds": time.time() - self.initialization_time,
                "operation_count": self.operation_count,
                "effective_config": self.effective_config.to_dict(),
                "custom_components": {
                    "processors": self.agent_config.custom_processors,
                    "expanders": self.agent_config.custom_expanders,
                    "filters": self.agent_config.custom_filters
                },
                "config_overrides": self.agent_config.overrides,
                "metadata": self.agent_config.metadata,
                "recent_operations": debug_tracker.get_recent_operations(10, "agent_namespace")
            }
            
            return OperationResult.success_result(stats)
            
        except Exception as e:
            return OperationResult.error_result(f"Failed to get agent stats: {e}")
    
    def validate_health(self) -> OperationResult[Dict[str, Any]]:
        """Validate agent namespace health"""
        try:
            health_info = {
                "agent_id": self.agent_config.agent_id,
                "status": "healthy",
                "checks": {}
            }
            
            # Check plugin references
            plugin_validation = self.agent_config.validate_plugin_references(self.plugin_manager)
            health_info["checks"]["plugin_references"] = {
                "status": "healthy" if plugin_validation.success else "unhealthy",
                "details": plugin_validation.error if not plugin_validation.success else "All plugins available"
            }
            
            # Check effective config
            try:
                config_validation = self.effective_config.validate()
                health_info["checks"]["configuration"] = {
                    "status": "healthy" if config_validation.success else "unhealthy",
                    "details": config_validation.error if not config_validation.success else "Configuration valid"
                }
            except Exception as e:
                health_info["checks"]["configuration"] = {
                    "status": "unhealthy",
                    "details": f"Configuration validation failed: {e}"
                }
            
            # Check enhanced memory access
            try:
                if hasattr(self.enhanced_memory, 'get_health_status'):
                    memory_health = self.enhanced_memory.get_health_status()
                    health_info["checks"]["enhanced_memory"] = {
                        "status": "healthy" if memory_health.success else "unhealthy",
                        "details": memory_health.error if not memory_health.success else "Enhanced memory accessible"
                    }
                else:
                    health_info["checks"]["enhanced_memory"] = {
                        "status": "healthy",
                        "details": "Enhanced memory accessible (no health check method)"
                    }
            except Exception as e:
                health_info["checks"]["enhanced_memory"] = {
                    "status": "unhealthy",
                    "details": f"Enhanced memory check failed: {e}"
                }
            
            # Overall status
            failed_checks = [check for check in health_info["checks"].values() if check["status"] != "healthy"]
            if failed_checks:
                health_info["status"] = "unhealthy"
                health_info["failed_checks"] = len(failed_checks)
            
            return OperationResult.success_result(health_info)
            
        except Exception as e:
            return OperationResult.error_result(f"Agent health validation failed: {e}")


class AgentManager:
    """
    Manages multiple agent namespaces
    Provides creation, configuration, and lifecycle management for agents
    """
    
    def __init__(self, enhanced_memory, base_config: SystemConfig):
        if not enhanced_memory:
            raise ConfigurationError("Enhanced memory cannot be None")
        
        if not isinstance(base_config, SystemConfig):
            raise ConfigurationError("Base config must be a SystemConfig instance")
        
        self.enhanced_memory = enhanced_memory
        self.base_config = base_config
        self.agents: Dict[str, AgentNamespace] = {}
        self.creation_count = 0
        self.initialization_time = time.time()
        
        debug_tracker.log_operation("agent_manager", "initialization", {
            "base_config_valid": bool(base_config),
            "enhanced_memory_type": type(enhanced_memory).__name__
        })
    
    def create_agent_namespace(self, agent_id: str, agent_type: str, 
                              config_overrides: Optional[Dict[str, Any]] = None,
                              custom_processors: Optional[List[str]] = None,
                              custom_expanders: Optional[List[str]] = None,
                              custom_filters: Optional[List[str]] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> OperationResult[AgentNamespace]:
        """
        Create a new agent namespace
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type/category of the agent
            config_overrides: Agent-specific configuration overrides
            custom_processors: List of custom processor names to use
            custom_expanders: List of custom expander names to use
            custom_filters: List of custom filter names to use
            metadata: Additional metadata for the agent
            
        Returns:
            OperationResult[AgentNamespace] with the created namespace
        """
        debug_tracker.log_operation("agent_manager", "create_agent_start", {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "has_overrides": bool(config_overrides),
            "processor_count": len(custom_processors or []),
            "expander_count": len(custom_expanders or []),
            "filter_count": len(custom_filters or [])
        })
        
        try:
            if not agent_id or not agent_id.strip():
                raise ConfigurationError("Agent ID cannot be empty")
            
            if not agent_type or not agent_type.strip():
                raise ConfigurationError("Agent type cannot be empty")
            
            if agent_id in self.agents:
                raise ConfigurationError(f"Agent '{agent_id}' already exists")
            
            # Create agent configuration
            agent_config = AgentConfig(
                agent_id=agent_id,
                agent_type=agent_type,
                base_config=self.base_config,
                overrides=config_overrides or {},
                custom_processors=custom_processors or [],
                custom_expanders=custom_expanders or [],
                custom_filters=custom_filters or [],
                metadata=metadata or {}
            )
            
            # Validate effective configuration
            effective_config = agent_config.get_effective_config()
            validation_result = effective_config.validate()
            if not validation_result.success:
                raise ConfigurationError(f"Agent config validation failed: {validation_result.error}")
            
            # Create agent namespace
            agent_namespace = AgentNamespace(agent_config, self.enhanced_memory, plugin_manager)
            self.agents[agent_id] = agent_namespace
            self.creation_count += 1
            
            debug_tracker.log_operation("agent_manager", "agent_created", {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "config_overrides": bool(config_overrides),
                "custom_components": len(custom_processors or []) + len(custom_expanders or []) + len(custom_filters or []),
                "total_agents": len(self.agents),
                "creation_count": self.creation_count
            }, success=True)
            
            return OperationResult.success_result(agent_namespace, {
                "agent_id": agent_id,
                "total_agents": len(self.agents)
            })
            
        except Exception as e:
            debug_tracker.log_operation("agent_manager", "create_agent_failed", {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Failed to create agent namespace: {e}")
    
    def get_agent_namespace(self, agent_id: str) -> Optional[AgentNamespace]:
        """
        Get existing agent namespace
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            AgentNamespace if found, None otherwise
        """
        if not agent_id:
            return None
        
        return self.agents.get(agent_id)
    
    def list_agents(self) -> OperationResult[Dict[str, Dict[str, Any]]]:
        """
        List all agent namespaces
        
        Returns:
            OperationResult[Dict] with agent summaries
        """
        try:
            agent_summary = {}
            for agent_id, namespace in self.agents.items():
                agent_summary[agent_id] = {
                    "agent_type": namespace.agent_config.agent_type,
                    "creation_time": namespace.agent_config.creation_time,
                    "uptime_seconds": time.time() - namespace.initialization_time,
                    "operation_count": namespace.operation_count,
                    "custom_processors": namespace.agent_config.custom_processors,
                    "custom_expanders": namespace.agent_config.custom_expanders,
                    "custom_filters": namespace.agent_config.custom_filters,
                    "has_overrides": bool(namespace.agent_config.overrides),
                    "metadata": namespace.agent_config.metadata
                }
            
            debug_tracker.log_operation("agent_manager", "list_agents_success", {
                "total_agents": len(agent_summary)
            }, success=True)
            
            return OperationResult.success_result(agent_summary, {
                "total_agents": len(agent_summary),
                "manager_uptime": time.time() - self.initialization_time
            })
            
        except Exception as e:
            debug_tracker.log_operation("agent_manager", "list_agents_failed", {
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Failed to list agents: {e}")
    
    def remove_agent_namespace(self, agent_id: str) -> OperationResult[bool]:
        """
        Remove agent namespace
        
        Args:
            agent_id: ID of the agent to remove
            
        Returns:
            OperationResult[bool] indicating success/failure
        """
        debug_tracker.log_operation("agent_manager", "remove_agent_start", {
            "agent_id": agent_id
        })
        
        try:
            if not agent_id:
                raise ConfigurationError("Agent ID cannot be empty")
            
            if agent_id not in self.agents:
                raise ConfigurationError(f"Agent '{agent_id}' not found")
            
            # Get agent info before deletion for logging
            agent_namespace = self.agents[agent_id]
            agent_type = agent_namespace.agent_config.agent_type
            operation_count = agent_namespace.operation_count
            
            # Remove the agent
            del self.agents[agent_id]
            
            debug_tracker.log_operation("agent_manager", "agent_removed", {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "operation_count": operation_count,
                "remaining_agents": len(self.agents)
            }, success=True)
            
            return OperationResult.success_result(True, {
                "removed_agent_id": agent_id,
                "remaining_agents": len(self.agents)
            })
            
        except Exception as e:
            debug_tracker.log_operation("agent_manager", "remove_agent_failed", {
                "agent_id": agent_id,
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Failed to remove agent namespace: {e}")
    
    def validate_all_agents(self) -> OperationResult[Dict[str, Any]]:
        """Validate health of all agent namespaces"""
        try:
            validation_results = {
                "total_agents": len(self.agents),
                "healthy_agents": 0,
                "unhealthy_agents": 0,
                "agent_health": {}
            }
            
            for agent_id, namespace in self.agents.items():
                try:
                    health_result = namespace.validate_health()
                    validation_results["agent_health"][agent_id] = health_result.data if health_result.success else {"error": health_result.error}
                    
                    if health_result.success and health_result.data.get("status") == "healthy":
                        validation_results["healthy_agents"] += 1
                    else:
                        validation_results["unhealthy_agents"] += 1
                        
                except Exception as e:
                    validation_results["unhealthy_agents"] += 1
                    validation_results["agent_health"][agent_id] = {"error": str(e)}
            
            overall_healthy = validation_results["unhealthy_agents"] == 0
            
            return OperationResult.success_result(validation_results, {
                "overall_healthy": overall_healthy
            })
            
        except Exception as e:
            return OperationResult.error_result(f"Agent validation failed: {e}")
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get agent manager statistics"""
        try:
            return {
                "total_agents": len(self.agents),
                "creation_count": self.creation_count,
                "initialization_time": self.initialization_time,
                "uptime_seconds": time.time() - self.initialization_time,
                "agent_types": list(set(ns.agent_config.agent_type for ns in self.agents.values())),
                "total_operations": sum(ns.operation_count for ns in self.agents.values())
            }
        except Exception as e:
            debug_tracker.log_operation("agent_manager", "get_stats_failed", {
                "error": str(e)
            }, success=False, error=str(e))
            raise RuntimeError(f"Failed to get agent manager stats: {e}")