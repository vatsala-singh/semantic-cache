"""
End-to-end demo script — mirrors the four demo cases in the blog.

Run with:
    ./semantic-cache/bin/python test_pipeline.py

Requires:
    - FastAPI server running on http://localhost:8000
    - Ollama running with nomic-embed-text + llama3.2 pulled
    - Redis running on localhost:6379
"""

import json
import sys
import httpx

BASE = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}


def post(client: httpx.Client, query: str, bypass_cache: bool = False) -> dict:
    r = client.post(
        f"{BASE}/ask",
        json={"query": query, "bypass_cache": bypass_cache},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def hr(label: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print("─" * 60)


def show(result: dict) -> None:
    print(json.dumps(result, indent=2))


def main():
    with httpx.Client() as client:

        # Flush cache before starting
        client.delete(f"{BASE}/internal/cache").raise_for_status()
        print("Cache flushed.")

        # ── Demo Case 1: Cold request (LLM fallback) ───────────────────
        hr("Demo Case 1 — Cold request (LLM fallback)")
        q1 = "What is semantic caching?"
        print(f"Query: {q1!r}")
        r1 = post(client, q1)
        show(r1)
        assert r1["from_cache"] is False, "Expected LLM fallback (from_cache=false)"
        assert r1["debug"]["miss_reason"] == "no_match"
        print("\n✓ Correct: from_cache=false, miss_reason=no_match")

        # ── Demo Case 2: Exact-match cache hit ─────────────────────────
        hr("Demo Case 2 — Exact-match cache hit (same query)")
        print(f"Query: {q1!r}")
        r2 = post(client, q1)
        show(r2)
        assert r2["from_cache"] is True, "Expected exact cache hit"
        assert r2["debug"]["cache_path"] == "exact_match"
        assert r2["similarity"] == 1.0
        print("\n✓ Correct: from_cache=true, cache_path=exact_match, similarity=1.0")

        # ── Optional: Whitespace normalisation ─────────────────────────
        hr("Optional — Whitespace normalisation")
        q_ws = "  What   is  semantic   caching?  "
        print(f"Query: {q_ws!r}")
        r_ws = post(client, q_ws)
        show(r_ws)
        assert r_ws["from_cache"] is True, "Expected exact match after normalisation"
        print("\n✓ Correct: whitespace-padded query hit exact cache")

        # ── Demo Case 3: Semantic cache hit (paraphrased query) ────────
        hr("Demo Case 3 — Semantic cache hit (paraphrased query)")
        q3 = "Can you explain how semantic caching works?"
        print(f"Query: {q3!r}")
        r3 = post(client, q3)
        show(r3)
        if r3["from_cache"]:
            assert r3["debug"]["cache_path"] == "semantic_match"
            print(f"\n✓ Correct: semantic hit, similarity={r3['similarity']:.4f}")
        else:
            print(
                "\n⚠ Cache miss on paraphrase — similarity may be below threshold. "
                "Try lowering SIMILARITY_THRESHOLD in .env."
            )

        # ── Demo Case 4: Forcing a cache miss with bypass_cache ────────
        hr("Demo Case 4 — Forced cache miss (bypass_cache=true)")
        print(f"Query: {q1!r}  bypass_cache=true")
        r4 = post(client, q1, bypass_cache=True)
        show(r4)
        assert r4["from_cache"] is False
        assert r4["debug"]["miss_reason"] == "bypass"
        print("\n✓ Correct: from_cache=false, miss_reason=bypass")

        # ── Metrics ────────────────────────────────────────────────────
        hr("Internal metrics")
        m = client.get(f"{BASE}/internal/metrics").json()
        show(m)

        hr("Cache stats")
        s = client.get(f"{BASE}/internal/cache/stats").json()
        show(s)

    print("\n\nAll demo cases passed.\n")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"\n✗ Assertion failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print(
            "\n✗ Could not connect to the server at http://localhost:8000.\n"
            "  Start it with:  uvicorn app.main:app --reload --port 8000",
            file=sys.stderr,
        )
        sys.exit(1)