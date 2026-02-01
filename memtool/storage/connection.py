"""SQLite connection pool management

Provides thread-safe connection pooling for SQLite databases.
Each thread maintains its own connection to avoid cross-thread issues.
"""

import sqlite3
import threading
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SQLiteConnectionPool:
    """Thread-safe SQLite connection pool
    
    Each thread maintains its own connection (thread-local storage).
    Connections are lazily created on first access per thread.
    
    Args:
        db_path: Path to SQLite database file
        timeout: Database lock timeout in seconds (default: 30.0)
    
    Example:
        >>> pool = SQLiteConnectionPool("memory.db")
        >>> conn = pool.get_connection()
        >>> cursor = conn.execute("SELECT * FROM memory")
    """
    
    def __init__(self, db_path: str, timeout: float = 30.0):
        self.db_path = str(Path(db_path).resolve())
        self.timeout = timeout
        self._local = threading.local()
        logger.info(f"Connection pool initialized for {self.db_path}")
    
    def get_connection(self) -> sqlite3.Connection:
        """Get connection for current thread (lazy-loaded)
        
        Returns:
            sqlite3.Connection: Database connection for current thread
        """
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = self._create_connection()
            logger.debug(f"Created new connection for thread {threading.current_thread().name}")
        return self._local.conn
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create and configure a new SQLite connection
        
        Returns:
            sqlite3.Connection: Configured database connection
        """
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=self.timeout
        )
        
        # Enable WAL mode for better concurrency
        conn.execute('PRAGMA journal_mode=WAL')
        
        # Use NORMAL synchronous mode (balance between safety and performance)
        conn.execute('PRAGMA synchronous=NORMAL')
        
        # Enable foreign key constraints
        conn.execute('PRAGMA foreign_keys=ON')
        
        # Set busy timeout (5 seconds, matching original implementation)
        conn.execute('PRAGMA busy_timeout=5000')
        
        # Return rows as sqlite3.Row objects (dict-like access)
        conn.row_factory = sqlite3.Row
        
        logger.debug(f"Connection configured: WAL mode, timeout={self.timeout}s")
        return conn
    
    def close_current_thread_connection(self):
        """Close connection for current thread"""
        if hasattr(self._local, 'conn') and self._local.conn:
            try:
                self._local.conn.close()
                logger.debug(f"Closed connection for thread {threading.current_thread().name}")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
            finally:
                self._local.conn = None
    
    def close_all(self):
        """Close all connections (mainly for testing cleanup)
        
        Note: Can only close the current thread's connection.
        Other threads' connections will be closed when those threads exit.
        """
        self.close_current_thread_connection()
    
    def __del__(self):
        """Cleanup on pool destruction"""
        self.close_current_thread_connection()
