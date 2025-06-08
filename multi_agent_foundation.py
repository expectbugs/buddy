"""
Multi-Agent Foundation for Buddy Memory System Phase 3: Extension Architecture
Main interface for multi-agent memory system providing isolated namespaces and shared foundation capabilities
NO silent failures - all operations are loud and proud
"""

from typing import Dict, List, Any, Optional
from result_types import OperationResult, HealthCheckResult, DebugInfo
from agent_namespace import AgentManager, AgentNamespace, AgentConfig
from plugin_system import plugin_manager, PluginManager, MemoryProcessor, ContextExpander, MemoryFilter
from config_manager import config_manager
from enhanced_memory import EnhancedMemory
from debug_info import debug_tracker
from health_check import health_checker
from exceptions import ConfigurationError
import time


class MultiAgentMemoryFoundation:
    """
    Main interface for multi-agent memory system
    Provides isolated namespaces and shared foundation capabilities for multiple AI agents
    """
    
    def __init__(self, enhanced_memory: EnhancedMemory):
        if not enhanced_memory:
            raise ConfigurationError("Enhanced memory cannot be None")
        
        if not isinstance(enhanced_memory, EnhancedMemory):
            raise ConfigurationError(f"Enhanced memory must be EnhancedMemory instance, got {type(enhanced_memory)}")
        
        self.enhanced_memory = enhanced_memory
        self.agent_manager = AgentManager(enhanced_memory, config_manager.config)
        self.plugin_manager = plugin_manager
        self.start_time = time.time()
        self.shared_operation_count = 0
        
        debug_tracker.log_operation("multi_agent_foundation", "initialization", {
            "enhanced_memory_type": type(enhanced_memory).__name__,
            "base_config_loaded": bool(config_manager.config),
            "plugin_manager_ready": bool(plugin_manager)
        })
    
    def create_agent(self, agent_id: str, agent_type: str,
                    config_overrides: Optional[Dict[str, Any]] = None,
                    custom_processors: Optional[List[str]] = None,
                    custom_expanders: Optional[List[str]] = None,
                    custom_filters: Optional[List[str]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> OperationResult[AgentNamespace]:
        """
        Create a new agent with its own namespace and configuration
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type/category of the agent (e.g., 'scheduler', 'finance', 'general')
            config_overrides: Agent-specific configuration overrides
            custom_processors: List of custom processor names for this agent
            custom_expanders: List of custom expander names for this agent
            custom_filters: List of custom filter names for this agent
            metadata: Additional metadata for the agent
            
        Returns:
            OperationResult[AgentNamespace] with the created agent namespace
        """
        debug_tracker.log_operation("multi_agent_foundation", "create_agent_start", {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "has_overrides": bool(config_overrides),
            "custom_components": len(custom_processors or []) + len(custom_expanders or []) + len(custom_filters or [])
        })
        
        try:
            # Validate inputs
            if not agent_id or not agent_id.strip():
                raise ConfigurationError("Agent ID cannot be empty")
            
            if not agent_type or not agent_type.strip():
                raise ConfigurationError("Agent type cannot be empty")
            
            # Create agent through agent manager
            result = self.agent_manager.create_agent_namespace(
                agent_id=agent_id,
                agent_type=agent_type,
                config_overrides=config_overrides,
                custom_processors=custom_processors,
                custom_expanders=custom_expanders,
                custom_filters=custom_filters,
                metadata=metadata
            )
            
            if result.success:
                debug_tracker.log_operation("multi_agent_foundation", "create_agent_success", {
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "total_agents": len(self.agent_manager.agents)
                }, success=True)
            else:
                debug_tracker.log_operation("multi_agent_foundation", "create_agent_failed", {
                    "agent_id": agent_id,
                    "error": result.error
                }, success=False, error=result.error)
            
            return result
            
        except Exception as e:
            debug_tracker.log_operation("multi_agent_foundation", "create_agent_exception", {
                "agent_id": agent_id,
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Failed to create agent: {e}")
    
    def get_agent(self, agent_id: str) -> Optional[AgentNamespace]:
        """
        Get existing agent namespace
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            AgentNamespace if found, None otherwise
        """
        if not agent_id:
            return None
        
        return self.agent_manager.get_agent_namespace(agent_id)
    
    def list_agents(self) -> OperationResult[Dict[str, Dict[str, Any]]]:
        """
        List all agent namespaces
        
        Returns:
            OperationResult[Dict] with agent summaries
        """
        try:
            return self.agent_manager.list_agents()
        except Exception as e:
            return OperationResult.error_result(f"Failed to list agents: {e}")
    
    def remove_agent(self, agent_id: str) -> OperationResult[bool]:
        """
        Remove agent namespace
        
        Args:
            agent_id: ID of the agent to remove
            
        Returns:
            OperationResult[bool] indicating success/failure
        """
        debug_tracker.log_operation("multi_agent_foundation", "remove_agent_start", {
            "agent_id": agent_id
        })
        
        try:
            result = self.agent_manager.remove_agent_namespace(agent_id)
            
            if result.success:
                debug_tracker.log_operation("multi_agent_foundation", "remove_agent_success", {
                    "agent_id": agent_id,
                    "remaining_agents": len(self.agent_manager.agents)
                }, success=True)
            else:
                debug_tracker.log_operation("multi_agent_foundation", "remove_agent_failed", {
                    "agent_id": agent_id,
                    "error": result.error
                }, success=False, error=result.error)
            
            return result
            
        except Exception as e:
            debug_tracker.log_operation("multi_agent_foundation", "remove_agent_exception", {
                "agent_id": agent_id,
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Failed to remove agent: {e}")
    
    def register_processor(self, name: str, processor: MemoryProcessor, priority: int = 100, 
                          metadata: Optional[Dict[str, Any]] = None) -> OperationResult[bool]:
        """
        Register a memory processor available to all agents
        
        Args:
            name: Unique name for the processor
            processor: MemoryProcessor instance
            priority: Execution priority (lower numbers execute first)
            metadata: Optional metadata about the processor
            
        Returns:
            OperationResult[bool] indicating success/failure
        """
        debug_tracker.log_operation("multi_agent_foundation", "register_processor_start", {
            "name": name,
            "priority": priority,
            "processor_type": type(processor).__name__
        })
        
        try:
            result = self.plugin_manager.register_memory_processor(name, processor, priority, metadata)
            
            debug_tracker.log_operation("multi_agent_foundation", "register_processor_complete", {
                "name": name,
                "success": result.success,
                "total_processors": len(self.plugin_manager.processors)
            }, success=result.success, error=result.error if not result.success else None)
            
            return result
            
        except Exception as e:
            debug_tracker.log_operation("multi_agent_foundation", "register_processor_exception", {
                "name": name,
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Failed to register processor: {e}")
    
    def register_expander(self, name: str, expander: ContextExpander, priority: int = 100,
                         metadata: Optional[Dict[str, Any]] = None) -> OperationResult[bool]:
        """
        Register a context expander available to all agents
        
        Args:
            name: Unique name for the expander
            expander: ContextExpander instance
            priority: Execution priority (lower numbers execute first)
            metadata: Optional metadata about the expander
            
        Returns:
            OperationResult[bool] indicating success/failure
        """
        debug_tracker.log_operation("multi_agent_foundation", "register_expander_start", {
            "name": name,
            "priority": priority,
            "expander_type": type(expander).__name__
        })
        
        try:
            result = self.plugin_manager.register_context_expander(name, expander, priority, metadata)
            
            debug_tracker.log_operation("multi_agent_foundation", "register_expander_complete", {
                "name": name,
                "success": result.success,
                "total_expanders": len(self.plugin_manager.expanders)
            }, success=result.success, error=result.error if not result.success else None)
            
            return result
            
        except Exception as e:
            debug_tracker.log_operation("multi_agent_foundation", "register_expander_exception", {
                "name": name,
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Failed to register expander: {e}")
    
    def register_filter(self, name: str, filter_instance: MemoryFilter, priority: int = 100,
                       metadata: Optional[Dict[str, Any]] = None) -> OperationResult[bool]:
        """
        Register a memory filter available to all agents
        
        Args:
            name: Unique name for the filter
            filter_instance: MemoryFilter instance
            priority: Execution priority (lower numbers execute first)
            metadata: Optional metadata about the filter
            
        Returns:
            OperationResult[bool] indicating success/failure
        """
        debug_tracker.log_operation("multi_agent_foundation", "register_filter_start", {
            "name": name,
            "priority": priority,
            "filter_type": type(filter_instance).__name__
        })
        
        try:
            result = self.plugin_manager.register_memory_filter(name, filter_instance, priority, metadata)
            
            debug_tracker.log_operation("multi_agent_foundation", "register_filter_complete", {
                "name": name,
                "success": result.success,
                "total_filters": len(self.plugin_manager.filters)
            }, success=result.success, error=result.error if not result.success else None)
            
            return result
            
        except Exception as e:
            debug_tracker.log_operation("multi_agent_foundation", "register_filter_exception", {
                "name": name,
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Failed to register filter: {e}")
    
    def list_plugins(self) -> OperationResult[Dict[str, List[Dict[str, Any]]]]:
        """
        List all registered plugins
        
        Returns:
            OperationResult[Dict] with plugins organized by type
        """
        try:
            return self.plugin_manager.list_plugins()
        except Exception as e:
            return OperationResult.error_result(f"Failed to list plugins: {e}")
    
    def get_foundation_health(self) -> OperationResult[Dict[str, Any]]:
        """
        Get comprehensive health status of the foundation
        
        Returns:
            OperationResult[Dict] with comprehensive health information
        """
        debug_tracker.log_operation("multi_agent_foundation", "health_check_start", {})
        
        try:
            # Get component health
            system_health = health_checker.comprehensive_health_check()
            memory_health = self.enhanced_memory.get_health_status()
            agent_health = self.agent_manager.validate_all_agents()
            plugin_validation = self.plugin_manager.validate_all_plugins()
            
            # Get operational statistics
            foundation_stats = {
                "uptime_seconds": time.time() - self.start_time,
                "active_agents": len(self.agent_manager.agents),
                "registered_plugins": len(self.plugin_manager.plugins),
                "shared_operations": self.shared_operation_count,
                "system_health": system_health,
                "memory_health": memory_health.data.to_dict() if memory_health.success else {"error": memory_health.error},
                "agent_health": agent_health.data if agent_health.success else {"error": agent_health.error},
                "plugin_validation": plugin_validation.data if plugin_validation.success else {"error": plugin_validation.error},
                "configuration": {
                    "config_loaded": bool(config_manager.config),
                    "config_valid": config_manager.config.validate().success if config_manager.config else False
                }
            }
            
            # Determine overall health
            overall_healthy = (
                system_health.get("overall_status") == "healthy" and
                memory_health.success and
                memory_health.data.status == "healthy" and
                agent_health.success and
                agent_health.metadata.get("overall_healthy", False) and
                plugin_validation.success and
                plugin_validation.metadata.get("overall_valid", False)
            )
            
            debug_tracker.log_operation("multi_agent_foundation", "health_check_complete", {
                "overall_healthy": overall_healthy,
                "active_agents": len(self.agent_manager.agents),
                "registered_plugins": len(self.plugin_manager.plugins)
            }, success=True)
            
            return OperationResult.success_result(foundation_stats, {
                "overall_healthy": overall_healthy,
                "components_checked": 4
            })
            
        except Exception as e:
            debug_tracker.log_operation("multi_agent_foundation", "health_check_failed", {
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Foundation health check failed: {e}")
    
    def get_foundation_debug_info(self) -> OperationResult[Dict[str, Any]]:
        """
        Get comprehensive debug information
        
        Returns:
            OperationResult[Dict] with comprehensive debug data
        """
        debug_tracker.log_operation("multi_agent_foundation", "debug_info_start", {})
        
        try:
            debug_info = {
                "foundation_info": {
                    "uptime_seconds": time.time() - self.start_time,
                    "start_time": self.start_time,
                    "shared_operations": self.shared_operation_count
                },
                "debug_summary": debug_tracker.get_debug_summary(),
                "active_agents": self.agent_manager.list_agents().data if self.agent_manager.list_agents().success else {},
                "registered_plugins": self.plugin_manager.list_plugins().data if self.plugin_manager.list_plugins().success else {},
                "enhanced_memory_debug": self.enhanced_memory.get_debug_info().data if self.enhanced_memory.get_debug_info().success else {},
                "configuration": config_manager.config.to_dict() if config_manager.config else {},
                "manager_stats": {
                    "agent_manager": self.agent_manager.get_manager_stats(),
                    "plugin_manager": self.plugin_manager.get_manager_stats()
                },
                "recent_operations": debug_tracker.get_recent_operations(20, "multi_agent_foundation")
            }
            
            debug_tracker.log_operation("multi_agent_foundation", "debug_info_complete", {
                "info_sections": len(debug_info),
                "active_agents": len(self.agent_manager.agents),
                "total_plugins": len(self.plugin_manager.plugins)
            }, success=True)
            
            return OperationResult.success_result(debug_info, {
                "info_sections": len(debug_info)
            })
            
        except Exception as e:
            debug_tracker.log_operation("multi_agent_foundation", "debug_info_failed", {
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Foundation debug info failed: {e}")
    
    def shared_search(self, query: str, user_id: str = "shared", **options) -> OperationResult[Any]:
        """
        Shared memory search accessible to all agents
        
        Args:
            query: Search query
            user_id: User identifier for shared namespace
            **options: Additional search options
            
        Returns:
            OperationResult with search results
        """
        debug_tracker.log_operation("multi_agent_foundation", "shared_search_start", {
            "query": query[:100],
            "user_id": user_id,
            "options_count": len(options)
        })
        
        try:
            if not query or not query.strip():
                raise ConfigurationError("Search query cannot be empty")
            
            self.shared_operation_count += 1
            
            result = self.enhanced_memory.search_memories(query, user_id, **options)
            
            if result.success:
                result.metadata["shared_operation"] = True
                result.metadata["operation_count"] = self.shared_operation_count
            
            debug_tracker.log_operation("multi_agent_foundation", "shared_search_complete", {
                "query": query[:100],
                "success": result.success,
                "results_count": len(result.data.results) if result.success else 0
            }, success=result.success, error=result.error if not result.success else None)
            
            return result
            
        except Exception as e:
            debug_tracker.log_operation("multi_agent_foundation", "shared_search_failed", {
                "query": query[:100],
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Shared search failed: {e}")
    
    def shared_add(self, data: Any, user_id: str = "shared", **options) -> OperationResult[Any]:
        """
        Shared memory addition accessible to all agents
        
        Args:
            data: Memory data to add
            user_id: User identifier for shared namespace
            **options: Additional options
            
        Returns:
            OperationResult with addition results
        """
        debug_tracker.log_operation("multi_agent_foundation", "shared_add_start", {
            "data_type": type(data).__name__,
            "user_id": user_id,
            "options_count": len(options)
        })
        
        try:
            if data is None:
                raise ConfigurationError("Memory data cannot be None")
            
            self.shared_operation_count += 1
            
            # Add shared operation metadata
            if 'metadata' not in options:
                options['metadata'] = {}
            
            options['metadata'].update({
                'shared_operation': True,
                'foundation_operation_count': self.shared_operation_count,
                'foundation_uptime': time.time() - self.start_time
            })
            
            result = self.enhanced_memory.add_memory(data, user_id, **options)
            
            if result.success:
                result.metadata["shared_operation"] = True
                result.metadata["operation_count"] = self.shared_operation_count
            
            debug_tracker.log_operation("multi_agent_foundation", "shared_add_complete", {
                "data_type": type(data).__name__,
                "success": result.success,
                "memories_added": len(result.data.memories_added) if result.success else 0
            }, success=result.success, error=result.error if not result.success else None)
            
            return result
            
        except Exception as e:
            debug_tracker.log_operation("multi_agent_foundation", "shared_add_failed", {
                "data_type": type(data).__name__,
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Shared add failed: {e}")
    
    def get_foundation_stats(self) -> OperationResult[Dict[str, Any]]:
        """
        Get comprehensive foundation statistics
        
        Returns:
            OperationResult[Dict] with foundation statistics
        """
        try:
            stats = {
                "foundation": {
                    "uptime_seconds": time.time() - self.start_time,
                    "start_time": self.start_time,
                    "shared_operations": self.shared_operation_count
                },
                "agents": self.agent_manager.get_manager_stats(),
                "plugins": self.plugin_manager.get_manager_stats(),
                "enhanced_memory": self.enhanced_memory.get_enhanced_stats() if hasattr(self.enhanced_memory, 'get_enhanced_stats') else {},
                "configuration": {
                    "config_loaded": bool(config_manager.config),
                    "config_sections": len(config_manager.config.to_dict()) if config_manager.config else 0
                }
            }
            
            return OperationResult.success_result(stats)
            
        except Exception as e:
            return OperationResult.error_result(f"Failed to get foundation stats: {e}")
    
    def shutdown(self) -> OperationResult[bool]:
        """
        Graceful shutdown of the foundation
        
        Returns:
            OperationResult[bool] indicating shutdown success
        """
        debug_tracker.log_operation("multi_agent_foundation", "shutdown_start", {
            "active_agents": len(self.agent_manager.agents),
            "uptime_seconds": time.time() - self.start_time,
            "shared_operations": self.shared_operation_count
        })
        
        try:
            # Log final statistics
            final_stats = {
                "total_uptime": time.time() - self.start_time,
                "agents_created": self.agent_manager.creation_count,
                "shared_operations": self.shared_operation_count,
                "plugins_registered": len(self.plugin_manager.plugins)
            }
            
            debug_tracker.log_operation("multi_agent_foundation", "shutdown_complete", final_stats, success=True)
            
            return OperationResult.success_result(True, final_stats)
            
        except Exception as e:
            debug_tracker.log_operation("multi_agent_foundation", "shutdown_failed", {
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(f"Foundation shutdown failed: {e}")