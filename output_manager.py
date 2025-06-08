"""
Output Manager for Buddy Memory System
Provides centralized control over user-facing output with silent mode support
NO silent failures - maintains full logging while controlling console output
"""

import sys
import logging
from typing import Any, Optional
from config_manager import SystemConfig

logger = logging.getLogger(__name__)


class OutputManager:
    """
    Centralized output management with silent mode support
    
    Controls what gets displayed to the user while maintaining full logging.
    In silent mode, suppresses system messages but preserves conversation flow.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize output manager with configuration
        
        Args:
            config: SystemConfig instance containing interface settings
        """
        self.config = config
        self._silent_mode = config.interface.silent_mode
        
        logger.debug(f"OutputManager initialized with silent_mode={self._silent_mode}")
    
    @property
    def silent_mode(self) -> bool:
        """Get current silent mode state"""
        return self._silent_mode
    
    def set_silent_mode(self, enabled: bool) -> None:
        """
        Set silent mode state and update configuration
        
        Args:
            enabled: True to enable silent mode, False to disable
        """
        self._silent_mode = enabled
        self.config.interface.silent_mode = enabled
        logger.info(f"Silent mode {'enabled' if enabled else 'disabled'}")
    
    def print_conversation(self, text: str, end: str = '\n') -> None:
        """
        Print conversation text (user input prompts and assistant responses)
        
        This is ALWAYS displayed regardless of silent mode as it's core functionality.
        
        Args:
            text: Text to display
            end: Line ending (default newline)
        """
        print(text, end=end, flush=True)
        logger.debug(f"Conversation output: {text[:100]}...")
    
    def print_system(self, text: str, end: str = '\n') -> None:
        """
        Print system messages (startup, status, notifications)
        
        Suppressed in silent mode but always logged.
        
        Args:
            text: Text to display
            end: Line ending (default newline)
        """
        logger.info(f"System message: {text}")
        
        if not self._silent_mode:
            print(text, end=end, flush=True)
    
    def print_command_result(self, text: str, end: str = '\n') -> None:
        """
        Print command execution results (/debug, /health, etc.)
        
        Suppressed in silent mode but always logged.
        
        Args:
            text: Text to display
            end: Line ending (default newline)
        """
        logger.info(f"Command result: {text}")
        
        if not self._silent_mode:
            print(text, end=end, flush=True)
    
    def print_explicit_command_result(self, text: str, end: str = '\n') -> None:
        """
        Print results from explicitly requested commands
        
        ALWAYS displayed, even in silent mode, because user explicitly requested it.
        This is for commands like /memories, /relationships, etc.
        
        Args:
            text: Text to display
            end: Line ending (default newline)
        """
        logger.info(f"Explicit command result: {text}")
        print(text, end=end, flush=True)
    
    def print_status(self, text: str, end: str = '\n') -> None:
        """
        Print status updates and progress messages
        
        Suppressed in silent mode but always logged.
        
        Args:
            text: Text to display  
            end: Line ending (default newline)
        """
        logger.debug(f"Status: {text}")
        
        if not self._silent_mode:
            print(text, end=end, flush=True)
    
    def print_error(self, text: str, end: str = '\n') -> None:
        """
        Print error messages
        
        In silent mode, logs error but doesn't display to user.
        This maintains silent mode while ensuring errors are captured.
        
        Args:
            text: Error text to display
            end: Line ending (default newline)
        """
        logger.error(f"Error: {text}")
        
        if not self._silent_mode:
            print(f"Error: {text}", end=end, flush=True, file=sys.stderr)
    
    def print_warning(self, text: str, end: str = '\n') -> None:
        """
        Print warning messages
        
        Suppressed in silent mode but always logged.
        
        Args:
            text: Warning text to display
            end: Line ending (default newline)
        """
        logger.warning(f"Warning: {text}")
        
        if not self._silent_mode:
            print(f"Warning: {text}", end=end, flush=True)
    
    def input_prompt(self, prompt: str) -> str:
        """
        Display input prompt and get user input
        
        ALWAYS displayed regardless of silent mode as it's required for interaction.
        
        Args:
            prompt: Prompt text to display
            
        Returns:
            User input string
        """
        logger.debug(f"Input prompt: {prompt}")
        return input(prompt)
    
    def print_startup_messages(self, messages: list) -> None:
        """
        Print startup messages and help text
        
        Respects both silent_mode and show_startup_messages config options.
        
        Args:
            messages: List of startup message strings
        """
        if self._silent_mode or not self.config.interface.show_startup_messages:
            logger.info("Startup messages suppressed")
            return
        
        for message in messages:
            print(message)
            logger.debug(f"Startup message: {message}")
    
    def print_help(self, help_text: str) -> None:
        """
        Print command help text
        
        Respects both silent_mode and show_command_help config options.
        
        Args:
            help_text: Help text to display
        """
        if self._silent_mode or not self.config.interface.show_command_help:
            logger.info("Help text suppressed")
            return
        
        print(help_text)
        logger.debug("Help text displayed")
    
    def get_status_summary(self) -> dict:
        """
        Get current output manager status for debugging
        
        Returns:
            Dictionary with current configuration and state
        """
        return {
            "silent_mode": self._silent_mode,
            "show_startup_messages": self.config.interface.show_startup_messages,
            "show_command_help": self.config.interface.show_command_help,
            "config_silent_mode": self.config.interface.silent_mode
        }


# Global output manager instance (will be initialized in run.py)
output_manager: Optional[OutputManager] = None


def get_output_manager() -> OutputManager:
    """
    Get the global output manager instance
    
    Returns:
        OutputManager instance
        
    Raises:
        RuntimeError: If output manager not initialized
    """
    if output_manager is None:
        raise RuntimeError("OutputManager not initialized. Call initialize_output_manager() first.")
    
    return output_manager


def initialize_output_manager(config: SystemConfig) -> OutputManager:
    """
    Initialize the global output manager instance
    
    Args:
        config: SystemConfig instance
        
    Returns:
        Initialized OutputManager instance
    """
    global output_manager
    output_manager = OutputManager(config)
    logger.info("Global OutputManager initialized")
    return output_manager