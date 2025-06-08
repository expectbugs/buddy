"""
Custom Exception Hierarchy for Buddy Memory System
Provides specific exceptions for Rule 3 compliance (NO silent failures)
"""


class BuddyMemoryError(Exception):
    """Base exception for all Buddy memory system errors"""
    pass


class MemoryOperationError(BuddyMemoryError):
    """Raised when memory operations fail"""
    pass


class ContextExpansionError(BuddyMemoryError):
    """Raised when context expansion fails"""
    pass


class ContextLoggingError(BuddyMemoryError):
    """Raised when context logging fails"""
    pass


class ResponseGenerationError(BuddyMemoryError):
    """Raised when AI response generation fails"""
    pass


class ConfigurationError(BuddyMemoryError):
    """Raised when system configuration is invalid"""
    pass


class HealthCheckError(BuddyMemoryError):
    """Raised when component health checks fail"""
    pass


class DebugSystemError(BuddyMemoryError):
    """Raised when debug system operations fail"""
    pass