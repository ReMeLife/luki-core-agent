"""
Safety Chain for LUKi Agent

Implements safety filtering, PII redaction, and consent enforcement
to ensure responsible AI behavior and data protection.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .config import settings


class SafetyLevel(Enum):
    """Safety filtering levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class SafetyResult:
    """Result of safety filtering"""
    filtered_content: str
    was_filtered: bool
    violations: List[str]
    redacted_items: List[str]
    confidence: float


class SafetyChain:
    """
    Safety filtering and compliance enforcement for LUKi
    """
    
    def __init__(self):
        self.config = {
            "enable_safety_filter": getattr(settings, "enable_safety_filters", True),
            "enable_pii_redaction": getattr(settings, "enable_pii_redaction", True),
            "enable_consent_checking": getattr(settings, "enable_consent_checking", True)
        }
        
        # PII patterns for redaction
        self.pii_patterns = {
            'phone': re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'credit_card': re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            'address': re.compile(r'\b\d+\s+[A-Za-z0-9\s,]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Court|Ct)\b', re.IGNORECASE),
        }
        
        # Unsafe content patterns
        self.unsafe_patterns = [
            re.compile(r'\b(?:suicide|kill\s+myself|end\s+my\s+life|want\s+to\s+die)\b', re.IGNORECASE),
            re.compile(r'\b(?:medical\s+advice|diagnosis|prescription|medication\s+dosage)\b', re.IGNORECASE),
            re.compile(r'\b(?:legal\s+advice|sue|lawsuit|attorney|lawyer)\b', re.IGNORECASE),
            re.compile(r'\b(?:financial\s+advice|investment|stock\s+tip|crypto)\b', re.IGNORECASE),
        ]
        
        # Inappropriate content patterns
        self.inappropriate_patterns = [
            re.compile(r'\b(?:explicit|sexual|inappropriate)\b', re.IGNORECASE),
            re.compile(r'\b(?:hate\s+speech|discrimination|racist|sexist)\b', re.IGNORECASE),
        ]
    
    async def filter_input(self, content: str, user_id: str) -> Tuple[str, bool]:
        """
        Filter user input for safety and appropriateness
        
        Args:
            content: User input to filter
            user_id: User identifier for context
            
        Returns:
            Tuple of (filtered_content, was_filtered)
        """
        if not self.config.get("enable_safety_filter", True):
            return content, False
        
        original_content = content
        violations = []
        
        # Check for unsafe content
        for pattern in self.unsafe_patterns:
            if pattern.search(content):
                violations.append("unsafe_content")
                break
        
        # Check for inappropriate content
        for pattern in self.inappropriate_patterns:
            if pattern.search(content):
                violations.append("inappropriate_content")
                break
        
        # If violations found, filter the input
        if violations:
            return self._create_filtered_input_message(violations), True
        
        # Apply PII redaction if enabled
        if self.config.get("enable_pii_redaction", True):
            content, _ = self._redact_pii(content)
        
        return content, False
    
    async def filter_output(self, content: str, user_id: str) -> Tuple[str, bool]:
        """
        Filter agent output for safety and compliance
        
        Args:
            content: Agent output to filter
            user_id: User identifier for context
            
        Returns:
            Tuple of (filtered_content, was_filtered)
        """
        if not self.config.get("enable_safety_filter", True):
            return content, False
        
        original_content = content
        violations = []
        was_filtered = False
        
        # Check for unsafe advice in output
        unsafe_advice_patterns = [
            # Medical advice variants
            re.compile(r'\b(?:you\s+should\s+take|i\s+recommend\s+taking|try\s+this\s+medication|prescribe|dosage)\b', re.IGNORECASE),
            # Legal advice variants (cover file/filing lawsuit)
            re.compile(r'\b(?:legal\s+action|fil(?:e|ing)\s+a\s+lawsuit|contact\s+a\s+lawyer|sue\s+someone)\b', re.IGNORECASE),
            # Financial advice variants
            re.compile(r'\b(?:invest\s+in|buy\s+stocks|financial\s+decision|guaranteed\s+returns)\b', re.IGNORECASE),
        ]
        
        for pattern in unsafe_advice_patterns:
            if pattern.search(content):
                violations.append("unsafe_advice")
                break
        
        # Apply PII redaction if enabled
        if self.config.get("enable_pii_redaction", True):
            content, pii_redacted = self._redact_pii(content)
            if pii_redacted:
                was_filtered = True
        
        # If violations found, replace with safe response
        if violations:
            content = self._create_safe_response(violations)
            was_filtered = True
        
        return content, was_filtered
    
    def _redact_pii(self, content: str) -> Tuple[str, bool]:
        """
        Redact personally identifiable information from content
        
        Args:
            content: Content to redact
            
        Returns:
            Tuple of (redacted_content, was_redacted)
        """
        redacted_content = content
        was_redacted = False
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(redacted_content)
            if matches:
                if pii_type == 'phone':
                    redacted_content = pattern.sub('[PHONE REDACTED]', redacted_content)
                elif pii_type == 'email':
                    redacted_content = pattern.sub('[EMAIL REDACTED]', redacted_content)
                elif pii_type == 'ssn':
                    redacted_content = pattern.sub('[SSN REDACTED]', redacted_content)
                elif pii_type == 'credit_card':
                    redacted_content = pattern.sub('[CREDIT CARD REDACTED]', redacted_content)
                elif pii_type == 'address':
                    redacted_content = pattern.sub('[ADDRESS REDACTED]', redacted_content)
                
                was_redacted = True
        
        return redacted_content, was_redacted
    
    def _create_filtered_input_message(self, violations: List[str]) -> str:
        """Create message for filtered input"""
        if "unsafe_content" in violations:
            return "I'm concerned about your wellbeing. If you're having thoughts of self-harm, please reach out to a mental health professional or crisis helpline."
        elif "inappropriate_content" in violations:
            return "I'd prefer to keep our conversation appropriate and respectful."
        else:
            return "I'd like to discuss something else instead."
    
    def _create_safe_response(self, violations: List[str]) -> str:
        """Create safe response for filtered output"""
        if "unsafe_advice" in violations:
            return "I can't provide specific medical, legal, or financial advice. For important decisions like these, it's best to consult with qualified professionals who can give you personalized guidance."
        else:
            return "I want to be helpful while staying within appropriate boundaries. Let me know if there's another way I can assist you."
    
    async def check_consent(self, user_id: str, data_type: str) -> bool:
        """
        Check if user has given consent for specific data usage
        
        Args:
            user_id: User identifier
            data_type: Type of data being accessed
            
        Returns:
            True if consent is granted
        """
        if not self.config.get("enable_consent_checking", True):
            return True
        
        # TODO: Implement actual consent checking with memory service
        # For now, assume consent is granted for basic conversation
        allowed_types = ["conversation", "memory_retrieval", "activity_suggestion"]
        return data_type in allowed_types
    
    def validate_memory_access(self, user_id: str, memory_content: Dict[str, Any]) -> bool:
        """
        Validate that memory access is appropriate
        
        Args:
            user_id: User identifier
            memory_content: Memory content to validate
            
        Returns:
            True if access is allowed
        """
        # Check sensitivity level
        sensitivity = memory_content.get("metadata", {}).get("sensitivity", "low")
        
        # For now, allow low and medium sensitivity
        # High sensitivity would require additional verification
        return sensitivity in ["low", "medium"]
    
    def sanitize_for_logging(self, content: str) -> str:
        """
        Sanitize content for safe logging (remove PII)
        
        Args:
            content: Content to sanitize
            
        Returns:
            Sanitized content safe for logging
        """
        sanitized, _ = self._redact_pii(content)
        
        # Additional sanitization for logging
        sanitized = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[NAME]', sanitized)  # Names
        sanitized = re.sub(r'\b\d{1,2}/\d{1,2}/\d{4}\b', '[DATE]', sanitized)  # Dates
        
        return sanitized
    
    def get_safety_metrics(self) -> Dict[str, Any]:
        """Get safety filtering metrics"""
        # TODO: Implement actual metrics tracking
        return {
            "filters_applied": 0,
            "pii_redactions": 0,
            "consent_checks": 0,
            "safety_level": "standard"
        }
