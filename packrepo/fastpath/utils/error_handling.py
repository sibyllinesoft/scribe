"""Standardized error handling utilities.

This module provides consistent error handling patterns used across the
FastPath system, with proper fallback mechanisms and logging support.
"""

import sys
import traceback
from typing import Optional, Any, Callable, TypeVar, Generic, Dict, Type, Union
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import logging


# Set up logger for error handling
logger = logging.getLogger(__name__)

T = TypeVar('T')
E = TypeVar('E', bound=Exception)


class ErrorSeverity(Enum):
    """Error severity levels for consistent error classification."""
    LOW = "low"           # Non-critical errors with fallbacks
    MEDIUM = "medium"     # Important errors that affect functionality  
    HIGH = "high"         # Critical errors that may cause failures
    CRITICAL = "critical" # System-breaking errors requiring immediate attention


@dataclass
class ErrorContext:
    """Context information for error reporting and debugging."""
    operation: str
    component: str
    file_path: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None
    severity: ErrorSeverity = ErrorSeverity.MEDIUM


class Result(Generic[T, E]):
    """Result type for operations that may fail.
    
    Provides a clean way to handle operations that may fail without
    raising exceptions, allowing for explicit error handling.
    """
    
    def __init__(self, value: Optional[T] = None, error: Optional[E] = None):
        if value is not None and error is not None:
            raise ValueError("Result cannot have both value and error")
        if value is None and error is None:
            raise ValueError("Result must have either value or error")
            
        self._value = value
        self._error = error
    
    @property
    def is_success(self) -> bool:
        """True if the result represents success."""
        return self._error is None
    
    @property
    def is_failure(self) -> bool:
        """True if the result represents failure."""
        return self._error is not None
    
    @property
    def value(self) -> T:
        """Get the success value. Raises if this is a failure."""
        if self._error is not None:
            raise RuntimeError(f"Attempted to get value from failed result: {self._error}")
        return self._value
    
    @property 
    def error(self) -> E:
        """Get the error. Raises if this is a success."""
        if self._value is not None:
            raise RuntimeError("Attempted to get error from successful result")
        return self._error
    
    def unwrap_or(self, default: T) -> T:
        """Get the value or return default if this is a failure."""
        return self._value if self.is_success else default
    
    def map(self, func: Callable[[T], Any]) -> 'Result':
        """Apply function to the value if this is a success."""
        if self.is_success:
            try:
                return Result(value=func(self._value))
            except Exception as e:
                return Result(error=e)
        return Result(error=self._error)
    
    @classmethod
    def success(cls, value: T) -> 'Result[T, E]':
        """Create a successful result."""
        return cls(value=value)
    
    @classmethod
    def failure(cls, error: E) -> 'Result[T, E]':
        """Create a failed result."""
        return cls(error=error)


class ErrorHandler:
    """Centralized error handling with consistent patterns and fallbacks."""
    
    def __init__(self, component_name: str = "FastPath"):
        self.component_name = component_name
        self._error_counts: Dict[str, int] = {}
        self._fallback_handlers: Dict[Type[Exception], Callable] = {}
    
    def handle_with_fallback(
        self,
        operation: Callable[[], T],
        fallback_value: T,
        context: Optional[ErrorContext] = None,
        log_errors: bool = True
    ) -> T:
        """Execute operation with fallback on error.
        
        Args:
            operation: Function to execute
            fallback_value: Value to return if operation fails
            context: Error context for logging
            log_errors: Whether to log errors
            
        Returns:
            Result of operation or fallback value on error
        """
        try:
            return operation()
        except Exception as e:
            if log_errors:
                self._log_error(e, context)
            return fallback_value
    
    def safe_execute(
        self,
        operation: Callable[[], T],
        context: Optional[ErrorContext] = None,
        reraise: bool = False
    ) -> Result[T, Exception]:
        """Safely execute operation returning Result type.
        
        Args:
            operation: Function to execute
            context: Error context for logging
            reraise: Whether to reraise critical errors
            
        Returns:
            Result object with success value or error
        """
        try:
            result = operation()
            # Handle case where operation returns None
            if result is None:
                return Result.success(None)
            return Result.success(result)
        except Exception as e:
            self._log_error(e, context)
            
            # Check if this is a critical error that should be reraised
            if reraise and self._is_critical_error(e, context):
                raise
                
            return Result.failure(e)
    
    def with_retry(
        self,
        operation: Callable[[], T],
        max_attempts: int = 3,
        context: Optional[ErrorContext] = None,
        backoff_multiplier: float = 1.0
    ) -> Result[T, Exception]:
        """Execute operation with retry logic.
        
        Args:
            operation: Function to execute
            max_attempts: Maximum number of attempts
            context: Error context for logging
            backoff_multiplier: Multiplier for delay between retries
            
        Returns:
            Result of operation or final error
        """
        import time
        
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                result = operation()
                return Result.success(result)
            except Exception as e:
                last_error = e
                self._increment_error_count(str(type(e).__name__))
                
                if attempt == max_attempts - 1:
                    break
                    
                # Add backoff delay
                if backoff_multiplier > 0:
                    delay = backoff_multiplier * (2 ** attempt)
                    time.sleep(delay)
                    
                if context:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {context.operation}: {e}"
                    )
        
        self._log_error(last_error, context)
        return Result.failure(last_error)
    
    def register_fallback_handler(
        self,
        exception_type: Type[Exception],
        handler: Callable[[Exception, Optional[ErrorContext]], Any]
    ):
        """Register a fallback handler for specific exception types.
        
        Args:
            exception_type: Exception type to handle
            handler: Function to handle the exception
        """
        self._fallback_handlers[exception_type] = handler
    
    def safe_file_operation(
        self,
        file_path: str,
        operation: Callable[[str], T],
        fallback_value: T,
        encoding: str = 'utf-8'
    ) -> T:
        """Safely perform file operations with standard error handling.
        
        Args:
            file_path: Path to file
            operation: Operation to perform on file
            fallback_value: Value to return on error
            encoding: File encoding to use
            
        Returns:
            Result of operation or fallback value
        """
        context = ErrorContext(
            operation="file_operation",
            component=self.component_name,
            file_path=file_path
        )
        
        def file_op():
            try:
                return operation(file_path)
            except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
                # Check for registered fallback handler
                for exc_type, handler in self._fallback_handlers.items():
                    if isinstance(e, exc_type):
                        return handler(e, context)
                raise
        
        return self.handle_with_fallback(file_op, fallback_value, context)
    
    def safe_subprocess_call(
        self,
        command: list,
        context: Optional[ErrorContext] = None,
        timeout: Optional[float] = None
    ) -> Result[str, Exception]:
        """Safely execute subprocess commands.
        
        Args:
            command: Command to execute
            context: Error context
            timeout: Command timeout in seconds
            
        Returns:
            Result with command output or error
        """
        import subprocess
        
        def run_command():
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=timeout
                )
                return result.stdout
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Command failed: {' '.join(command)}\\nError: {e.stderr}")
            except subprocess.TimeoutExpired:
                raise TimeoutError(f"Command timed out after {timeout}s: {' '.join(command)}")
        
        return self.safe_execute(run_command, context)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered.
        
        Returns:
            Dictionary with error statistics
        """
        return {
            'total_errors': sum(self._error_counts.values()),
            'error_counts': self._error_counts.copy(),
            'registered_handlers': list(self._fallback_handlers.keys())
        }
    
    def _log_error(self, error: Exception, context: Optional[ErrorContext] = None):
        """Log error with context information."""
        if context:
            logger.error(
                f"Error in {context.component}.{context.operation}: {error}",
                extra={
                    'component': context.component,
                    'operation': context.operation,
                    'file_path': context.file_path,
                    'severity': context.severity.value,
                    'additional_info': context.additional_info
                }
            )
        else:
            logger.error(f"Error in {self.component_name}: {error}")
        
        # Log stack trace for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Stack trace: {traceback.format_exc()}")
    
    def _increment_error_count(self, error_type: str):
        """Track error frequency for analysis."""
        self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
    
    def _is_critical_error(self, error: Exception, context: Optional[ErrorContext]) -> bool:
        """Determine if an error should be considered critical."""
        # System-level errors that should not be suppressed
        critical_types = (SystemExit, KeyboardInterrupt, MemoryError, SystemError)
        
        if isinstance(error, critical_types):
            return True
            
        # Check context severity
        if context and context.severity == ErrorSeverity.CRITICAL:
            return True
            
        return False


def error_boundary(
    fallback_value: Any = None,
    component: str = "Unknown",
    log_errors: bool = True
):
    """Decorator for adding error boundaries to functions.
    
    Args:
        fallback_value: Value to return on error
        component: Component name for logging
        log_errors: Whether to log errors
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = ErrorHandler(component)
            context = ErrorContext(
                operation=func.__name__,
                component=component
            )
            
            return handler.handle_with_fallback(
                lambda: func(*args, **kwargs),
                fallback_value,
                context,
                log_errors
            )
        return wrapper
    return decorator


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely perform division with fallback for zero division."""
    handler = ErrorHandler("MathUtils")
    return handler.handle_with_fallback(
        lambda: numerator / denominator,
        default,
        ErrorContext(operation="division", component="MathUtils")
    )


def safe_list_access(lst: list, index: int, default: Any = None) -> Any:
    """Safely access list element with fallback for index errors."""
    handler = ErrorHandler("ListUtils")
    return handler.handle_with_fallback(
        lambda: lst[index],
        default,
        ErrorContext(operation="list_access", component="ListUtils")
    )