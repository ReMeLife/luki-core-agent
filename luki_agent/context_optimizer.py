"""
Context builder optimization for LUKi Core Agent
Provides intelligent caching and optimization for context assembly operations
"""

import hashlib
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


@dataclass
class ContextCacheEntry:
    """Cached context entry"""
    user_id: str
    context_type: str
    context_data: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    cache_key: str
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if entry has expired"""
        age_seconds = (datetime.utcnow() - self.created_at).total_seconds()
        return age_seconds > ttl_seconds
    
    def is_stale(self, staleness_seconds: int) -> bool:
        """Check if entry should be refreshed"""
        age_seconds = (datetime.utcnow() - self.last_accessed).total_seconds()
        return age_seconds > staleness_seconds


class ContextCache:
    """LRU cache for context data with intelligent eviction"""
    
    def __init__(
        self,
        max_size_mb: int = 100,
        max_entries: int = 1000,
        default_ttl_seconds: int = 300
    ):
        """
        Initialize context cache
        
        Args:
            max_size_mb: Maximum cache size in megabytes
            max_entries: Maximum number of cache entries
            default_ttl_seconds: Default time-to-live for entries
        """
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, ContextCacheEntry] = OrderedDict()
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._max_entries = max_entries
        self._default_ttl = default_ttl_seconds
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._current_size_bytes = 0
    
    def get(
        self,
        user_id: str,
        context_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached context if available
        
        Args:
            user_id: User identifier
            context_type: Type of context (e.g., 'memory', 'conversation', 'profile')
            parameters: Additional parameters for cache key
        
        Returns:
            Cached context data or None
        """
        cache_key = self._generate_cache_key(user_id, context_type, parameters)
        
        with self._lock:
            if cache_key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[cache_key]
            
            # Check if expired
            if entry.is_expired(self._default_ttl):
                del self._cache[cache_key]
                self._current_size_bytes -= entry.size_bytes
                self._misses += 1
                logger.debug(
                    f"Cache entry expired: {cache_key}",
                    extra={"user_id": user_id, "context_type": context_type}
                )
                return None
            
            # Update access metadata
            entry.last_accessed = datetime.utcnow()
            entry.access_count += 1
            
            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            
            self._hits += 1
            
            logger.debug(
                f"Cache hit: {cache_key}",
                extra={
                    "user_id": user_id,
                    "context_type": context_type,
                    "access_count": entry.access_count
                }
            )
            
            return entry.context_data.copy()
    
    def set(
        self,
        user_id: str,
        context_type: str,
        context_data: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None
    ):
        """
        Store context in cache
        
        Args:
            user_id: User identifier
            context_type: Type of context
            context_data: Context data to cache
            parameters: Additional parameters for cache key
            ttl_seconds: Custom TTL for this entry
        """
        cache_key = self._generate_cache_key(user_id, context_type, parameters)
        
        # Estimate size (rough approximation)
        import sys
        size_bytes = sys.getsizeof(str(context_data))
        
        with self._lock:
            # Remove old entry if exists
            if cache_key in self._cache:
                old_entry = self._cache[cache_key]
                self._current_size_bytes -= old_entry.size_bytes
                del self._cache[cache_key]
            
            # Create new entry
            entry = ContextCacheEntry(
                user_id=user_id,
                context_type=context_type,
                context_data=context_data.copy(),
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=0,
                size_bytes=size_bytes,
                cache_key=cache_key
            )
            
            # Add to cache
            self._cache[cache_key] = entry
            self._current_size_bytes += size_bytes
            
            # Evict if necessary
            self._evict_if_needed()
            
            logger.debug(
                f"Cache set: {cache_key}",
                extra={
                    "user_id": user_id,
                    "context_type": context_type,
                    "size_bytes": size_bytes,
                    "ttl_seconds": ttl_seconds or self._default_ttl
                }
            )
    
    def invalidate(
        self,
        user_id: Optional[str] = None,
        context_type: Optional[str] = None
    ):
        """
        Invalidate cache entries
        
        Args:
            user_id: If provided, invalidate entries for this user
            context_type: If provided, invalidate entries of this type
        """
        with self._lock:
            keys_to_remove = []
            
            for key, entry in self._cache.items():
                should_remove = True
                
                if user_id and entry.user_id != user_id:
                    should_remove = False
                
                if context_type and entry.context_type != context_type:
                    should_remove = False
                
                if should_remove:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                entry = self._cache[key]
                self._current_size_bytes -= entry.size_bytes
                del self._cache[key]
            
            logger.info(
                f"Invalidated {len(keys_to_remove)} cache entries",
                extra={"user_id": user_id, "context_type": context_type}
            )
    
    def _evict_if_needed(self):
        """Evict least recently used entries if cache is full"""
        # Evict by count
        while len(self._cache) > self._max_entries:
            key, entry = self._cache.popitem(last=False)  # Remove oldest
            self._current_size_bytes -= entry.size_bytes
            self._evictions += 1
            logger.debug(f"Evicted entry by count: {key}")
        
        # Evict by size
        while self._current_size_bytes > self._max_size_bytes and self._cache:
            key, entry = self._cache.popitem(last=False)
            self._current_size_bytes -= entry.size_bytes
            self._evictions += 1
            logger.debug(f"Evicted entry by size: {key}")
    
    def _generate_cache_key(
        self,
        user_id: str,
        context_type: str,
        parameters: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key from parameters"""
        key_parts = [user_id, context_type]
        
        if parameters:
            # Sort parameters for consistent hashing
            sorted_params = sorted(parameters.items())
            param_str = str(sorted_params)
            key_parts.append(param_str)
        
        key_string = ":".join(key_parts)
        
        # Hash for shorter key
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        
        return f"{context_type}:{user_id[:8]}:{key_hash}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = 0.0
            if total_requests > 0:
                hit_rate = (self._hits / total_requests) * 100.0
            
            return {
                "entries": len(self._cache),
                "size_bytes": self._current_size_bytes,
                "size_mb": round(self._current_size_bytes / (1024 * 1024), 2),
                "max_entries": self._max_entries,
                "max_size_mb": self._max_size_bytes / (1024 * 1024),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 2),
                "evictions": self._evictions
            }
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            logger.info("Cache cleared")


class ContextOptimizer:
    """Optimize context assembly and retrieval"""
    
    def __init__(self, enable_caching: bool = True):
        """
        Initialize context optimizer
        
        Args:
            enable_caching: Whether to enable context caching
        """
        self.cache = ContextCache() if enable_caching else None
        self._assembly_times: List[float] = []
    
    def get_optimized_context(
        self,
        user_id: str,
        context_type: str,
        builder_func: Any,
        parameters: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Get context with caching optimization
        
        Args:
            user_id: User identifier
            context_type: Type of context
            builder_func: Function to build context if cache miss
            parameters: Additional parameters
            force_refresh: Force cache refresh
        
        Returns:
            Context data
        """
        start_time = time.time()
        
        # Try cache first
        if self.cache and not force_refresh:
            cached = self.cache.get(user_id, context_type, parameters)
            if cached is not None:
                logger.debug(
                    f"Using cached context for {context_type}",
                    extra={"user_id": user_id, "context_type": context_type}
                )
                return cached
        
        # Build context
        context_data = builder_func(user_id, **(parameters or {}))
        
        # Cache result
        if self.cache:
            self.cache.set(user_id, context_type, context_data, parameters)
        
        # Track assembly time
        assembly_time = time.time() - start_time
        self._assembly_times.append(assembly_time)
        if len(self._assembly_times) > 100:
            self._assembly_times = self._assembly_times[-100:]
        
        logger.debug(
            f"Built context for {context_type}",
            extra={
                "user_id": user_id,
                "context_type": context_type,
                "assembly_time_seconds": round(assembly_time, 3)
            }
        )
        
        return context_data
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get context optimization performance statistics"""
        stats = {
            "average_assembly_time_seconds": 0.0,
            "sample_count": len(self._assembly_times)
        }
        
        if self._assembly_times:
            stats["average_assembly_time_seconds"] = round(
                sum(self._assembly_times) / len(self._assembly_times), 3
            )
            stats["min_assembly_time_seconds"] = round(min(self._assembly_times), 3)
            stats["max_assembly_time_seconds"] = round(max(self._assembly_times), 3)
        
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        
        return stats


# Global context optimizer instance
_context_optimizer: Optional[ContextOptimizer] = None


def get_context_optimizer() -> ContextOptimizer:
    """Get the global context optimizer instance"""
    global _context_optimizer
    if _context_optimizer is None:
        _context_optimizer = ContextOptimizer()
    return _context_optimizer
