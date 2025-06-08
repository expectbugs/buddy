"""
Standard Result Types for Buddy Memory System
Provides consistent, predictable result wrappers for multi-agent use
NO silent failures - all operations return explicit success/error results
"""

from typing import Optional, Dict, List, Any, Union, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid


T = TypeVar('T')


@dataclass
class OperationResult(Generic[T]):
    """
    Standard result wrapper for all foundation operations
    Ensures consistent return types across the entire system
    """
    success: bool
    data: Optional[T]
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    def __post_init__(self):
        """Validate result consistency"""
        if self.success and self.error:
            raise ValueError("Successful result cannot have an error message")
        if not self.success and self.data is not None:
            raise ValueError("Failed result should not have data")
        if not self.success and not self.error:
            raise ValueError("Failed result must have an error message")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "operation_id": self.operation_id
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def success_result(cls, data: T, metadata: Optional[Dict[str, Any]] = None) -> 'OperationResult[T]':
        """Create successful result with data"""
        return cls(
            success=True, 
            data=data, 
            error=None,
            metadata=metadata or {}
        )
    
    @classmethod
    def error_result(cls, error: str, metadata: Optional[Dict[str, Any]] = None) -> 'OperationResult[T]':
        """Create error result with message"""
        return cls(
            success=False, 
            data=None, 
            error=error,
            metadata=metadata or {}
        )
    
    def unwrap(self) -> T:
        """
        Unwrap data or raise exception if error
        For use when you want to fail loudly on errors
        """
        if not self.success:
            raise RuntimeError(f"Operation failed: {self.error}")
        return self.data
    
    def unwrap_or(self, default: T) -> T:
        """
        Unwrap data or return default if error
        For use when you want to handle errors gracefully
        """
        if self.success:
            return self.data
        return default
    
    def map(self, func) -> 'OperationResult':
        """
        Transform successful data, propagate errors
        Enables functional composition of results
        """
        if self.success:
            try:
                new_data = func(self.data)
                return OperationResult.success_result(new_data, self.metadata)
            except Exception as e:
                return OperationResult.error_result(str(e), self.metadata)
        return self


@dataclass
class MemorySearchResult:
    """Standard structure for memory search results"""
    results: List[Dict[str, Any]]
    total_count: int
    search_time_ms: float
    query: str
    expansion_applied: bool = False
    expansion_metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "results": self.results,
            "total_count": self.total_count,
            "search_time_ms": self.search_time_ms,
            "query": self.query,
            "expansion_applied": self.expansion_applied,
            "expansion_metadata": self.expansion_metadata
        }
    
    @property
    def has_results(self) -> bool:
        """Check if search returned any results"""
        return self.total_count > 0
    
    def get_top_result(self) -> Optional[Dict[str, Any]]:
        """Get the highest relevance result"""
        if self.results:
            return self.results[0]
        return None


@dataclass
class MemoryAddResult:
    """Standard structure for memory add results"""
    memories_added: List[Dict[str, Any]]
    relationships_extracted: List[Dict[str, Any]]
    operation_time_ms: float
    lookup_codes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "memories_added": self.memories_added,
            "relationships_extracted": self.relationships_extracted,
            "operation_time_ms": self.operation_time_ms,
            "lookup_codes": self.lookup_codes
        }
    
    @property
    def memory_count(self) -> int:
        """Get number of memories added"""
        return len(self.memories_added)
    
    @property
    def relationship_count(self) -> int:
        """Get number of relationships extracted"""
        return len(self.relationships_extracted)


@dataclass
class HealthCheckResult:
    """Standard structure for health check results"""
    component: str
    status: str  # "healthy", "unhealthy", "degraded"
    response_time_ms: Optional[float]
    details: Dict[str, Any]
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "component": self.component,
            "status": self.status,
            "response_time_ms": self.response_time_ms,
            "details": self.details,
            "error": self.error
        }
    
    @property
    def is_healthy(self) -> bool:
        """Check if component is healthy"""
        return self.status == "healthy"
    
    @property
    def is_degraded(self) -> bool:
        """Check if component is degraded but functional"""
        return self.status == "degraded"
    
    @property
    def is_unhealthy(self) -> bool:
        """Check if component is unhealthy"""
        return self.status == "unhealthy"


@dataclass
class ContextLogResult:
    """Standard structure for context logging results"""
    lookup_code: str
    log_file: str
    timestamp: str
    success: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "lookup_code": self.lookup_code,
            "log_file": self.log_file,
            "timestamp": self.timestamp,
            "success": self.success
        }


@dataclass
class DebugInfo:
    """Standard structure for debug information"""
    component: str
    recent_operations: List[Dict[str, Any]]
    component_state: Dict[str, Any]
    statistics: Dict[str, Any]
    configuration: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "component": self.component,
            "recent_operations": self.recent_operations,
            "component_state": self.component_state,
            "statistics": self.statistics,
            "configuration": self.configuration
        }
    
    def get_error_operations(self) -> List[Dict[str, Any]]:
        """Get only failed operations"""
        return [op for op in self.recent_operations if not op.get("success", True)]
    
    def get_success_rate(self) -> float:
        """Calculate success rate from recent operations"""
        if not self.recent_operations:
            return 100.0
        
        successful = sum(1 for op in self.recent_operations if op.get("success", True))
        return (successful / len(self.recent_operations)) * 100.0


# Type aliases for common result types
MemorySearchOperationResult = OperationResult[MemorySearchResult]
MemoryAddOperationResult = OperationResult[MemoryAddResult]
HealthCheckOperationResult = OperationResult[HealthCheckResult]
ContextLogOperationResult = OperationResult[ContextLogResult]
DebugInfoOperationResult = OperationResult[DebugInfo]
StringOperationResult = OperationResult[str]
BoolOperationResult = OperationResult[bool]
DictOperationResult = OperationResult[Dict[str, Any]]
ListOperationResult = OperationResult[List[Any]]