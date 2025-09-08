"""Intelligent caching system for the Vision Analysis System."""

import os
import pickle
import json
import threading
import time
import hashlib
import sqlite3
import numpy as np
from typing import Any, Optional, Dict, List, Callable, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from PIL import Image
import cv2

from .performance import LRUCache, ImageCache, CacheStats, PerformanceMonitor

@dataclass
class CacheEntry:
    """Cache entry metadata."""
    key: str
    data_type: str
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int
    expiry_time: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class PersistentCache:
    """Persistent cache using SQLite for metadata and file system for data."""
    
    def __init__(self, cache_dir: str, db_name: str = "cache.db"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.cache_dir / db_name
        self.data_dir = self.cache_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        self._lock = threading.RLock()
        self._init_database()
    
    def _init_database(self):
        """Initialize cache database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    data_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_accessed TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 1,
                    expiry_time TIMESTAMP,
                    metadata TEXT,
                    file_path TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expiry_time ON cache_entries(expiry_time)")
    
    def _generate_file_path(self, key: str, data_type: str) -> Path:
        """Generate file path for cache entry."""
        # Use hash of key to create filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        extension = self._get_extension(data_type)
        return self.data_dir / f"{key_hash}{extension}"
    
    def _get_extension(self, data_type: str) -> str:
        """Get file extension based on data type."""
        extensions = {
            'image': '.png',
            'numpy': '.npy',
            'json': '.json',
            'pickle': '.pkl',
            'text': '.txt'
        }
        return extensions.get(data_type, '.dat')
    
    def put(self, key: str, data: Any, data_type: str = 'pickle', 
            expiry_hours: Optional[int] = None, metadata: Optional[Dict] = None) -> bool:
        """Store data in persistent cache."""
        try:
            with self._lock:
                file_path = self._generate_file_path(key, data_type)
                
                # Save data to file
                self._save_data_to_file(data, file_path, data_type)
                
                # Get file size
                size_bytes = file_path.stat().st_size
                
                # Calculate expiry time
                expiry_time = None
                if expiry_hours:
                    expiry_time = datetime.now() + timedelta(hours=expiry_hours)
                
                # Store metadata in database
                now = datetime.now()
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO cache_entries 
                        (key, data_type, size_bytes, created_at, last_accessed, 
                         access_count, expiry_time, metadata, file_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        key, data_type, size_bytes, now, now, 1,
                        expiry_time, json.dumps(metadata) if metadata else None,
                        str(file_path)
                    ))
                
                return True
        except Exception as e:
            print(f"Error storing cache entry {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve data from persistent cache."""
        try:
            with self._lock:
                # Get entry metadata
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT data_type, file_path, expiry_time 
                        FROM cache_entries 
                        WHERE key = ?
                    """, (key,))
                    row = cursor.fetchone()
                    
                    if not row:
                        return None
                    
                    data_type, file_path, expiry_time = row
                    
                    # Check if expired
                    if expiry_time:
                        expiry_dt = datetime.fromisoformat(expiry_time)
                        if datetime.now() > expiry_dt:
                            self.delete(key)
                            return None
                    
                    # Update access information
                    now = datetime.now()
                    conn.execute("""
                        UPDATE cache_entries 
                        SET last_accessed = ?, access_count = access_count + 1
                        WHERE key = ?
                    """, (now, key))
                
                # Load data from file
                file_path = Path(file_path)
                if not file_path.exists():
                    self.delete(key)
                    return None
                
                return self._load_data_from_file(file_path, data_type)
        
        except Exception as e:
            print(f"Error retrieving cache entry {key}: {e}")
            return None
    
    def _save_data_to_file(self, data: Any, file_path: Path, data_type: str):
        """Save data to file based on data type."""
        if data_type == 'image':
            if isinstance(data, np.ndarray):
                cv2.imwrite(str(file_path), data)
            elif isinstance(data, Image.Image):
                data.save(file_path)
            else:
                raise ValueError(f"Unsupported image data type: {type(data)}")
        
        elif data_type == 'numpy':
            np.save(file_path, data)
        
        elif data_type == 'json':
            with open(file_path, 'w') as f:
                json.dump(data, f)
        
        elif data_type == 'text':
            with open(file_path, 'w') as f:
                f.write(str(data))
        
        else:  # pickle as default
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
    
    def _load_data_from_file(self, file_path: Path, data_type: str) -> Any:
        """Load data from file based on data type."""
        if data_type == 'image':
            if file_path.suffix in ['.png', '.jpg', '.jpeg']:
                return cv2.imread(str(file_path))
            else:
                return Image.open(file_path)
        
        elif data_type == 'numpy':
            return np.load(file_path)
        
        elif data_type == 'json':
            with open(file_path, 'r') as f:
                return json.load(f)
        
        elif data_type == 'text':
            with open(file_path, 'r') as f:
                return f.read()
        
        else:  # pickle as default
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    
    def delete(self, key: str) -> bool:
        """Delete cache entry."""
        try:
            with self._lock:
                # Get file path
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT file_path FROM cache_entries WHERE key = ?", (key,))
                    row = cursor.fetchone()
                    
                    if row:
                        file_path = Path(row[0])
                        if file_path.exists():
                            file_path.unlink()
                    
                    # Remove from database
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                
                return True
        except Exception as e:
            print(f"Error deleting cache entry {key}: {e}")
            return False
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries."""
        count = 0
        try:
            with self._lock:
                now = datetime.now()
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT key, file_path FROM cache_entries 
                        WHERE expiry_time IS NOT NULL AND expiry_time < ?
                    """, (now,))
                    
                    expired_entries = cursor.fetchall()
                    
                    for key, file_path in expired_entries:
                        file_path = Path(file_path)
                        if file_path.exists():
                            file_path.unlink()
                        count += 1
                    
                    if expired_entries:
                        keys = [entry[0] for entry in expired_entries]
                        placeholders = ','.join(['?' for _ in keys])
                        conn.execute(f"DELETE FROM cache_entries WHERE key IN ({placeholders})", keys)
        
        except Exception as e:
            print(f"Error cleaning up expired entries: {e}")
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_entries,
                        SUM(size_bytes) as total_size_bytes,
                        AVG(access_count) as avg_access_count,
                        MIN(created_at) as oldest_entry,
                        MAX(last_accessed) as most_recent_access
                    FROM cache_entries
                """)
                row = cursor.fetchone()
                
                if row:
                    return {
                        'total_entries': row[0],
                        'total_size_mb': (row[1] or 0) / (1024 * 1024),
                        'avg_access_count': row[2] or 0,
                        'oldest_entry': row[3],
                        'most_recent_access': row[4]
                    }
        except Exception as e:
            print(f"Error getting cache stats: {e}")
        
        return {}

class ModelCache:
    """Specialized cache for ML models and inference results."""
    
    def __init__(self, cache_dir: str, max_memory_mb: int = 1024):
        self.persistent_cache = PersistentCache(cache_dir)
        self.memory_cache = ImageCache(max_size=50, max_memory_mb=max_memory_mb)
        self._model_registry = {}
        self._inference_cache = LRUCache(max_size=200)
        
        # Register caches with performance monitor
        monitor = PerformanceMonitor.instance()
        monitor.register_cache("model_memory", self.memory_cache)
        monitor.register_cache("model_inference", self._inference_cache)
    
    def cache_model(self, model_name: str, model_data: Any, model_type: str = 'pickle') -> bool:
        """Cache a trained model."""
        key = f"model:{model_name}"
        return self.persistent_cache.put(key, model_data, model_type, expiry_hours=24*7)  # 1 week
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Retrieve cached model."""
        key = f"model:{model_name}"
        return self.persistent_cache.get(key)
    
    def cache_inference_result(self, image_hash: str, model_name: str, result: Any) -> bool:
        """Cache inference result."""
        key = f"inference:{model_name}:{image_hash}"
        self._inference_cache.put(key, result)
        return True
    
    def get_inference_result(self, image_hash: str, model_name: str) -> Optional[Any]:
        """Get cached inference result."""
        key = f"inference:{model_name}:{image_hash}"
        return self._inference_cache.get(key)
    
    def cache_processed_image(self, image_hash: str, processed_image: np.ndarray) -> bool:
        """Cache processed image."""
        key = f"processed:{image_hash}"
        self.memory_cache.put(key, processed_image)
        return True
    
    def get_processed_image(self, image_hash: str) -> Optional[np.ndarray]:
        """Get cached processed image."""
        key = f"processed:{image_hash}"
        return self.memory_cache.get(key)

class ConfigCache:
    """Cache for configuration and settings."""
    
    def __init__(self, cache_dir: str):
        self.persistent_cache = PersistentCache(cache_dir)
        self.memory_cache = LRUCache(max_size=100)
        
        # Register with performance monitor
        monitor = PerformanceMonitor.instance()
        monitor.register_cache("config_memory", self.memory_cache)
    
    def cache_config(self, config_key: str, config_data: Any, expiry_hours: int = 24) -> bool:
        """Cache configuration data."""
        # Store in memory for fast access
        self.memory_cache.put(config_key, config_data)
        
        # Store persistently for long-term
        return self.persistent_cache.put(config_key, config_data, 'json', expiry_hours)
    
    def get_config(self, config_key: str) -> Optional[Any]:
        """Get cached configuration."""
        # Try memory cache first
        result = self.memory_cache.get(config_key)
        if result is not None:
            return result
        
        # Fall back to persistent cache
        result = self.persistent_cache.get(config_key)
        if result is not None:
            # Cache in memory for next time
            self.memory_cache.put(config_key, result)
        
        return result

class CacheManager:
    """Central cache management system."""
    
    def __init__(self, base_cache_dir: str):
        self.base_cache_dir = Path(base_cache_dir)
        self.base_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize specialized caches
        self.model_cache = ModelCache(str(self.base_cache_dir / "models"))
        self.config_cache = ConfigCache(str(self.base_cache_dir / "config"))
        self.image_cache = ImageCache(max_size=100, max_memory_mb=512)
        
        # Generic persistent cache
        self.persistent_cache = PersistentCache(str(self.base_cache_dir / "general"))
        
        # Cache cleanup thread
        self._cleanup_thread = None
        self._cleanup_active = False
        
        # Register all caches
        monitor = PerformanceMonitor.instance()
        monitor.register_cache("image_cache", self.image_cache)
        
        # Start cleanup scheduler
        self.start_cleanup_scheduler()
    
    def start_cleanup_scheduler(self):
        """Start background cache cleanup."""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_active = True
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                name="CacheCleanup",
                daemon=True
            )
            self._cleanup_thread.start()
    
    def stop_cleanup_scheduler(self):
        """Stop background cache cleanup."""
        self._cleanup_active = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=1.0)
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while self._cleanup_active:
            try:
                # Clean up expired entries every hour
                self.cleanup_expired()
                time.sleep(3600)  # 1 hour
            except Exception as e:
                print(f"Cache cleanup error: {e}")
                time.sleep(60)  # Retry in 1 minute on error
    
    def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired cache entries."""
        results = {}
        try:
            results['models'] = self.model_cache.persistent_cache.cleanup_expired()
            results['config'] = self.config_cache.persistent_cache.cleanup_expired()
            results['general'] = self.persistent_cache.cleanup_expired()
        except Exception as e:
            print(f"Error during cache cleanup: {e}")
        
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            'model_cache': self.model_cache.persistent_cache.get_stats(),
            'config_cache': self.config_cache.persistent_cache.get_stats(),
            'general_cache': self.persistent_cache.get_stats(),
            'image_memory_cache': self.image_cache.stats(),
            'model_memory_cache': self.model_cache.memory_cache.stats(),
            'config_memory_cache': self.config_cache.memory_cache.stats(),
        }
    
    def clear_all_caches(self):
        """Clear all caches."""
        self.image_cache.clear()
        self.model_cache.memory_cache.clear()
        self.model_cache._inference_cache.clear()
        self.config_cache.memory_cache.clear()
    
    def warm_up_cache(self, config_data: Dict[str, Any]):
        """Pre-warm cache with commonly used data."""
        # Cache frequently used configurations
        for key, value in config_data.items():
            self.config_cache.cache_config(f"warmup:{key}", value, expiry_hours=24)

# Utility functions for cache key generation
def generate_image_hash(image: np.ndarray) -> str:
    """Generate hash for image data."""
    # Use image shape and a sample of pixels for hash
    h, w = image.shape[:2]
    sample_pixels = image[::h//10, ::w//10].flatten()[:100]  # Sample pixels
    data = f"{h}x{w}:{hash(sample_pixels.tobytes())}"
    return hashlib.md5(data.encode()).hexdigest()

def generate_config_hash(config_dict: Dict[str, Any]) -> str:
    """Generate hash for configuration dictionary."""
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

# Decorators for automatic caching
def cache_result(cache_manager: CacheManager, cache_type: str = 'general', expiry_hours: int = 24):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from appropriate cache
            if cache_type == 'config':
                result = cache_manager.config_cache.get_config(key)
            else:
                result = cache_manager.persistent_cache.get(key)
            
            if result is not None:
                return result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            if cache_type == 'config':
                cache_manager.config_cache.cache_config(key, result, expiry_hours)
            else:
                cache_manager.persistent_cache.put(key, result, 'pickle', expiry_hours)
            
            return result
        return wrapper
    return decorator