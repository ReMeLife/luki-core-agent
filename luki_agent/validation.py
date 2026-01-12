"""
Input validation framework for LUKi Core Agent
Provides centralized validation utilities for requests, responses, and data integrity
"""

import re
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from pydantic import BaseModel, ValidationError, field_validator
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Validation error severity levels"""
    ERROR = "error"      # Blocks execution
    WARNING = "warning"  # Logs but continues
    INFO = "info"        # Informational only


class ValidationResult(BaseModel):
    """Result of validation operation"""
    valid: bool
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    sanitized_data: Optional[Dict[str, Any]] = None


class InputValidator:
    """Centralized input validation for agent requests"""
    
    # Dangerous patterns that should be rejected
    DANGEROUS_PATTERNS = [
        r"(?i)<script[^>]*>.*?</script>",  # Script tags
        r"(?i)javascript:",                 # JavaScript protocol
        r"(?i)on\w+\s*=",                  # Event handlers
        r"(?i)eval\s*\(",                  # Eval calls
        r"(?i)exec\s*\(",                  # Exec calls
    ]
    
    # PII patterns for detection/redaction
    PII_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    }
    
    @classmethod
    def validate_chat_message(cls, message: str, user_id: str) -> ValidationResult:
        """
        Validate chat message input
        
        Args:
            message: User message content
            user_id: User identifier
        
        Returns:
            ValidationResult with any errors or warnings
        """
        errors = []
        warnings = []
        sanitized = message.strip()
        
        # Check message length
        if len(sanitized) == 0:
            errors.append({
                "field": "message",
                "error": "Message cannot be empty",
                "severity": ValidationSeverity.ERROR
            })
        
        if len(sanitized) > 10000:
            errors.append({
                "field": "message",
                "error": f"Message too long: {len(sanitized)} characters (max 10000)",
                "severity": ValidationSeverity.ERROR
            })
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, sanitized):
                errors.append({
                    "field": "message",
                    "error": "Message contains potentially dangerous content",
                    "severity": ValidationSeverity.ERROR,
                    "pattern": pattern
                })
        
        # Check for PII (warning only)
        for pii_type, pattern in cls.PII_PATTERNS.items():
            if re.search(pattern, sanitized):
                warnings.append({
                    "field": "message",
                    "warning": f"Message may contain {pii_type}",
                    "severity": ValidationSeverity.WARNING,
                    "pii_type": pii_type
                })
        
        # Validate user_id format
        if not user_id or len(user_id) < 3:
            errors.append({
                "field": "user_id",
                "error": "Invalid user_id format",
                "severity": ValidationSeverity.ERROR
            })
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data={"message": sanitized, "user_id": user_id}
        )
    
    @classmethod
    def validate_tool_input(cls, tool_name: str, parameters: Dict[str, Any]) -> ValidationResult:
        """
        Validate tool execution parameters
        
        Args:
            tool_name: Name of the tool being called
            parameters: Tool parameters
        
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        
        # Check tool name
        if not tool_name or not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', tool_name):
            errors.append({
                "field": "tool_name",
                "error": "Invalid tool name format",
                "severity": ValidationSeverity.ERROR
            })
        
        # Check parameters is a dict
        if not isinstance(parameters, dict):
            errors.append({
                "field": "parameters",
                "error": "Parameters must be a dictionary",
                "severity": ValidationSeverity.ERROR
            })
            return ValidationResult(valid=False, errors=errors)
        
        # Check parameter keys are valid
        for key in parameters.keys():
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
                errors.append({
                    "field": f"parameters.{key}",
                    "error": f"Invalid parameter name: {key}",
                    "severity": ValidationSeverity.ERROR
                })
        
        # Check for deeply nested structures (potential DoS)
        try:
            max_depth = cls._check_dict_depth(parameters)
            if max_depth > 10:
                warnings.append({
                    "field": "parameters",
                    "warning": f"Parameters are deeply nested (depth: {max_depth})",
                    "severity": ValidationSeverity.WARNING
                })
        except RecursionError:
            errors.append({
                "field": "parameters",
                "error": "Parameters contain circular reference or are too deeply nested",
                "severity": ValidationSeverity.ERROR
            })
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data={"tool_name": tool_name, "parameters": parameters}
        )
    
    @classmethod
    def validate_memory_query(cls, query: str, top_k: int = 5) -> ValidationResult:
        """
        Validate memory/ELR query parameters
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        sanitized_query = query.strip()
        
        # Check query length
        if len(sanitized_query) == 0:
            errors.append({
                "field": "query",
                "error": "Query cannot be empty",
                "severity": ValidationSeverity.ERROR
            })
        
        if len(sanitized_query) > 500:
            warnings.append({
                "field": "query",
                "warning": f"Very long query ({len(sanitized_query)} chars), may be inefficient",
                "severity": ValidationSeverity.WARNING
            })
        
        # Check top_k range
        if top_k < 1 or top_k > 50:
            errors.append({
                "field": "top_k",
                "error": f"top_k must be between 1 and 50, got {top_k}",
                "severity": ValidationSeverity.ERROR
            })
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data={"query": sanitized_query, "top_k": top_k}
        )
    
    @classmethod
    def sanitize_output(cls, text: str, redact_pii: bool = False) -> str:
        """
        Sanitize agent output before sending to user
        
        Args:
            text: Output text
            redact_pii: Whether to redact PII patterns
        
        Returns:
            Sanitized text
        """
        sanitized = text.strip()
        
        if redact_pii:
            # Redact PII patterns
            for pii_type, pattern in cls.PII_PATTERNS.items():
                sanitized = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", sanitized)
        
        return sanitized
    
    @staticmethod
    def _check_dict_depth(d: Dict, current_depth: int = 0) -> int:
        """Check maximum nesting depth of dictionary"""
        if not isinstance(d, dict):
            return current_depth
        
        if not d:
            return current_depth
        
        max_child_depth = current_depth
        for value in d.values():
            if isinstance(value, dict):
                child_depth = InputValidator._check_dict_depth(value, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        child_depth = InputValidator._check_dict_depth(item, current_depth + 1)
                        max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth


class ResponseValidator:
    """Validate agent responses before returning to client"""
    
    @staticmethod
    def validate_chat_response(response: Dict[str, Any]) -> ValidationResult:
        """
        Validate chat response structure and content
        
        Args:
            response: Agent response dictionary
        
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ["text"]
        for field in required_fields:
            if field not in response:
                errors.append({
                    "field": field,
                    "error": f"Missing required field: {field}",
                    "severity": ValidationSeverity.ERROR
                })
        
        # Validate text field
        if "text" in response:
            text = response["text"]
            if not isinstance(text, str):
                errors.append({
                    "field": "text",
                    "error": "Text field must be a string",
                    "severity": ValidationSeverity.ERROR
                })
            elif len(text) > 50000:
                warnings.append({
                    "field": "text",
                    "warning": f"Response text is very long: {len(text)} characters",
                    "severity": ValidationSeverity.WARNING
                })
        
        # Validate tool_calls if present
        if "tool_calls" in response:
            if not isinstance(response["tool_calls"], list):
                errors.append({
                    "field": "tool_calls",
                    "error": "tool_calls must be a list",
                    "severity": ValidationSeverity.ERROR
                })
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


def validate_and_log(validation_result: ValidationResult, operation: str):
    """
    Log validation results with appropriate severity
    
    Args:
        validation_result: Result from validation
        operation: Name of the operation being validated
    """
    if not validation_result.valid:
        logger.error(
            f"Validation failed for {operation}",
            extra={
                "operation": operation,
                "error_count": len(validation_result.errors),
                "errors": validation_result.errors
            }
        )
    
    if validation_result.warnings:
        logger.warning(
            f"Validation warnings for {operation}",
            extra={
                "operation": operation,
                "warning_count": len(validation_result.warnings),
                "warnings": validation_result.warnings
            }
        )


# Decorator for automatic validation
def validate_input(validator_func: Callable):
    """Decorator to automatically validate function inputs"""
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            # Extract relevant parameters for validation
            validation_result = validator_func(*args, **kwargs)
            validate_and_log(validation_result, func.__name__)
            
            if not validation_result.valid:
                raise ValueError(f"Input validation failed: {validation_result.errors}")
            
            return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            validation_result = validator_func(*args, **kwargs)
            validate_and_log(validation_result, func.__name__)
            
            if not validation_result.valid:
                raise ValueError(f"Input validation failed: {validation_result.errors}")
            
            return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator
