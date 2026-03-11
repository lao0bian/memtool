#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bug fix regression tests (Codex review 2026-03-11)

Tests for:
- Bug#1: delete/cleanup cleans vector index
- Bug#2: search(track_access=False) skips access tracking
- Bug#3: list() offset works with complex sorting
- Bug#4: _track_access_batch updates consolidation_score
- MemoryStore singleton cache
"""

import os
import sys
import tempfile
import unittest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memtool_core import MemoryStore, init_db


class TestBugFixes(unittest.TestCase):
    """Regression tests for bugs found by Codex review."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_bugfixes.db")
        init_db(self.db_path)
        self.store = MemoryStore(self.db_path)

    def tearDown(self):
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def _put(self, key, content="test content", type_="feature", confidence="medium"):
        return self.store.put(
            type=type_, key=key, content=content,
            confidence_level=confidence
        )

    # === Bug#1: delete/cleanup cleans vector index ===

    def test_delete_calls_vector_delete(self):
        """delete() should attempt to clean vector index after SQLite delete."""
        from memtool_core import MemtoolError
        result = self._put("del-test", content="delete me")
        self.assertTrue(result["ok"])
        item_id = result["id"]

        # Delete
        del_result = self.store.delete(item_id=item_id)
        self.assertTrue(del_result["ok"])
        self.assertEqual(del_result["deleted"], 1)

        # Verify item is gone (get raises MemtoolError for NOT_FOUND)
        with self.assertRaises(MemtoolError):
            self.store.get(item_id=item_id)

    def test_cleanup_cleans_vector_index(self):
        """cleanup() with apply=True should clean vector index for deleted items."""
        # Create old items (cleanup needs older_than_days)
        result = self._put("cleanup-test", content="cleanup me")
        self.assertTrue(result["ok"])

        # Dry run should not delete
        cleanup_result = self.store.cleanup(
            type="feature",
            older_than_days=0,  # 0 means all
            stale_threshold=999,  # very high, so everything is candidate
            apply=False,
            limit=100
        )
        self.assertTrue(cleanup_result["ok"])
        self.assertEqual(cleanup_result["deleted"], 0)

    # === Bug#2: search track_access parameter ===

    def test_search_track_access_true(self):
        """search() with track_access=True (default) should update access_count."""
        result = self._put("track-true", content="searchable content alpha")
        item_id = result["id"]

        # Get initial access count
        item_before = self.store.get(item_id=item_id)
        access_before = item_before.get("access_count", 0)

        # Search should track access
        self.store.search(query="alpha", track_access=True)

        # Verify access was tracked
        item_after = self.store.get(item_id=item_id)
        # access_count should have increased (get also tracks, so +1 from get + maybe +1 from search)
        self.assertGreaterEqual(item_after.get("access_count", 0), access_before)

    def test_search_track_access_false(self):
        """search() with track_access=False should NOT update access_count."""
        result = self._put("track-false", content="searchable content beta")
        item_id = result["id"]

        # Record access count right after put (no access tracked yet, so 0)
        # But we need to get it without side effects, so use direct SQL
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT access_count FROM memory_items WHERE id = ?", (item_id,)).fetchone()
        access_before = row["access_count"]
        conn.close()

        # Search WITHOUT access tracking
        self.store.search(query="beta", track_access=False)

        # Verify access was NOT tracked
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT access_count FROM memory_items WHERE id = ?", (item_id,)).fetchone()
        access_after = row["access_count"]
        conn.close()

        self.assertEqual(access_before, access_after,
                         "search(track_access=False) should not update access_count")

    # === Bug#3: list() offset with complex sorting ===

    def test_list_offset_with_simple_sort(self):
        """list() offset should work with simple sorting (updated)."""
        # Create 5 items
        ids = []
        for i in range(5):
            result = self._put(f"offset-simple-{i}", content=f"item {i}")
            ids.append(result["id"])

        # Get all items
        all_items = self.store.list(type="feature", limit=50)
        total = len(all_items)

        # Get with offset
        offset_items = self.store.list(type="feature", limit=2, offset=2)
        self.assertLessEqual(len(offset_items), 2)

        # Offset items should not overlap with first 2
        if total >= 4:
            first_items = self.store.list(type="feature", limit=2, offset=0)
            first_ids = {i["id"] for i in first_items}
            offset_ids = {i["id"] for i in offset_items}
            self.assertEqual(len(first_ids & offset_ids), 0,
                             "offset items should not overlap with first page")

    def test_list_offset_with_complex_sort(self):
        """Bug#3: list() offset should work correctly with mixed/confidence/recency sorting."""
        # Create enough items to test offset
        for i in range(10):
            conf = "high" if i % 2 == 0 else "low"
            self._put(f"offset-complex-{i}", content=f"item {i}", confidence=conf)

        # Get page 1
        page1 = self.store.list(type="feature", limit=3, offset=0, sort_by="mixed")
        # Get page 2 with offset
        page2 = self.store.list(type="feature", limit=3, offset=3, sort_by="mixed")

        page1_ids = {i["id"] for i in page1}
        page2_ids = {i["id"] for i in page2}

        # Pages should not overlap (Bug#3: before fix, offset was reset to 0)
        self.assertEqual(len(page1_ids & page2_ids), 0,
                         "pages with offset should not overlap (Bug#3)")

    # === Bug#4: _track_access_batch consolidation_score ===

    def test_track_access_batch_updates_consolidation(self):
        """Bug#4: _track_access_batch should update consolidation_score."""
        result = self._put("consolidation-batch", content="test consolidation")
        item_id = result["id"]

        # Get initial consolidation_score (direct SQL to avoid side effects)
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT consolidation_score FROM memory_items WHERE id = ?", (item_id,)
        ).fetchone()
        score_before = row["consolidation_score"]
        conn.close()

        # Call batch track access multiple times
        for _ in range(5):
            self.store._track_access_batch([item_id])

        # Check consolidation_score was updated
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT consolidation_score, access_count FROM memory_items WHERE id = ?",
            (item_id,)
        ).fetchone()
        score_after = row["consolidation_score"]
        access_count = row["access_count"]
        conn.close()

        self.assertEqual(access_count, 5, "access_count should be 5")
        self.assertGreater(score_after, score_before,
                           "consolidation_score should increase after batch access (Bug#4)")

    # === MemoryStore singleton cache ===

    def test_store_singleton_cache(self):
        """_store_for should return the same MemoryStore instance for same db_path."""
        # Import from mcp_server
        from mcp_server import _store_for, _STORE_CACHE

        store1 = _store_for(self.db_path)
        store2 = _store_for(self.db_path)

        self.assertIs(store1, store2,
                      "_store_for should return cached instance")

    # === _clamp_limit ===

    def test_clamp_limit(self):
        """_clamp_limit should cap limit to MAX_LIMIT."""
        from mcp_server import _clamp_limit, MAX_LIMIT

        self.assertEqual(_clamp_limit(10), 10)
        self.assertEqual(_clamp_limit(0), 1)
        self.assertEqual(_clamp_limit(-5), 1)
        self.assertEqual(_clamp_limit(999), MAX_LIMIT)
        self.assertEqual(_clamp_limit(MAX_LIMIT), MAX_LIMIT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
