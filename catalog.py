"""
catalog.py — The Retrieval Module
===================================
Loads the SHL Individual Test Solutions catalog from catalog.json and
provides a simple, fast keyword-overlap search.

Interview talking point:
  "We deliberately chose keyword overlap instead of vector embeddings.
   The catalog is ~35 domain-specific items. Users say things like 'Java',
   'personality test', or 'numerical reasoning' — exact domain terms that
   keyword matching handles perfectly. It requires zero external API calls,
   has deterministic output, and runs in microseconds — all important when
   the evaluation harness enforces a 30-second wall-clock timeout."
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Inline fallback catalog — used only if catalog.json is missing at runtime.
# Keeping a small fallback here means the module never hard-crashes on import.
# ---------------------------------------------------------------------------
_FALLBACK_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "Verify Numerical Reasoning",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-numerical-reasoning/",
        "test_type": "A",
        "description": "Measures ability to interpret numerical data including graphs, charts, and tables."
    },
    {
        "name": "Verify Verbal Reasoning",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-verbal-reasoning/",
        "test_type": "A",
        "description": "Assesses ability to understand written passages and draw logical conclusions."
    },
    {
        "name": "OPQ32r",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/opq32r/",
        "test_type": "P",
        "description": "The Occupational Personality Questionnaire measuring 32 personality characteristics relevant to workplace behaviour."
    },
    {
        "name": "Java 8 (New)",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
        "test_type": "K",
        "description": "Assesses practical knowledge of Java 8 features including streams, lambdas, and concurrency."
    },
    {
        "name": "Python (New)",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/python-new/",
        "test_type": "K",
        "description": "Evaluates Python programming skills including syntax, data structures, and standard libraries."
    },
]


class CatalogSearcher:
    """
    Wraps the catalog list and exposes a `search` method.

    Design decisions worth explaining in an interview:
    1. We load the catalog once at startup (in __init__), not per-request.
       This avoids repeated disk I/O on every API call.
    2. The search is a simple token-intersection count — O(n*m) where n is
       catalog size (~35) and m is query tokens (~10). Totally negligible.
    3. Every URL returned by `search` and `get_by_names` comes directly from
       this loaded catalog dict — the LLM never fabricates a URL.
    """

    def __init__(self, catalog_path: str = "catalog.json"):
        self.catalog = self._load(catalog_path)
        # Pre-build a set of valid URLs for O(1) integrity checks elsewhere
        self.valid_urls: set = {item["url"] for item in self.catalog}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self, path: str) -> List[Dict[str, Any]]:
        """Load catalog.json; fall back to the inline list if file is absent."""
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        print("[CatalogSearcher] catalog.json not found — using inline fallback.")
        return _FALLBACK_CATALOG

    @staticmethod
    def _tokenize(text: str) -> set:
        """
        Lowercase and extract alphanumeric tokens.
        e.g. "Java 8 (New)" → {"java", "8", "new"}
        Using a set means intersection is O(min(|A|,|B|)).
        """
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def _score(self, item: Dict[str, Any], query_tokens: set) -> int:
        """
        Count how many query tokens appear in the combined text of one item.
        We concatenate name + description + test_type to maximise match surface.
        """
        item_text = " ".join([
            item.get("name", ""),
            item.get("description", ""),
            item.get("test_type", ""),
        ])
        return len(query_tokens & self._tokenize(item_text))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """
        Return up to `top_k` catalog items ranked by keyword overlap with `query`.

        If no item scores > 0 (completely unrecognised query), we return the
        top-k items anyway so the LLM always has something to work with rather
        than receiving an empty catalog block (which would cause it to hallucinate).
        """
        if not query.strip():
            # Empty query → return the full catalog up to top_k
            return self.catalog[:top_k]

        query_tokens = self._tokenize(query)

        # Score every item and sort descending; use name as a stable tiebreaker
        scored = sorted(
            self.catalog,
            key=lambda item: (-self._score(item, query_tokens), item["name"])
        )

        # Prefer items with at least one matching token
        matches = [item for item in scored if self._score(item, query_tokens) > 0]

        # Fall back to best-effort (top of sorted list) if nothing matched
        return (matches if matches else scored)[:top_k]

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Return the complete catalog.
        Used when the agent needs all items (e.g., for Compare mode where
        the user asks about two specific tests by name).
        """
        return self.catalog

    def get_by_names(self, names: List[str]) -> List[Dict[str, Any]]:
        """
        Case-insensitive exact name lookup.
        Used by the agent to pull full descriptions of specific tests
        for a comparison answer grounded in catalog data.
        """
        target = {n.lower() for n in names}
        return [item for item in self.catalog if item["name"].lower() in target]
