"""
Test-wide isolation: the suite must never touch the real Groq / Supabase project or
load the 1.2 GB embedding model.

* Dummy credentials are set BEFORE any `app.*` import. `load_dotenv()` does not
  override variables that already exist, so the developer's real `.env` is ignored.
* Building the real Supabase client or the real embedding model raises, so a test that
  forgets to mock them fails loudly instead of silently writing rows to production
  (the /feedback tests used to insert fake corrections into the live feedback memory).
"""

import os

os.environ["GROQ_API_KEY"] = "test-groq-key"
os.environ["SUPABASE_URL"] = "http://127.0.0.1:54321"
os.environ["SUPABASE_KEY"] = "test-supabase-key"
os.environ["API_KEYS"] = "tester:test-api-key"
os.environ["ADMIN_API_KEY"] = "test-admin-key"
os.environ.pop("AUTH_DISABLED", None)
os.environ.pop("FEEDBACK_AUTO_APPROVE", None)

TEST_API_KEY = "test-api-key"
TEST_ADMIN_KEY = "test-admin-key"

import pytest  # noqa: E402


def _blocked(what: str):
    def _raise(*_args, **_kwargs):
        raise RuntimeError(f"Tests must mock {what}; the real one is blocked in tests/conftest.py")
    return _raise


@pytest.fixture(autouse=True)
def _block_external_services(monkeypatch):
    from app.db import supabase_client
    from app.tools import embedding_tool

    monkeypatch.setattr(supabase_client, "get_supabase_client", _blocked("the Supabase client"))
    monkeypatch.setattr(embedding_tool, "get_embedding_model", _blocked("the embedding model"))
    from app.tools import reranker
    monkeypatch.setattr(reranker, "get_reranker", _blocked("the reranker model"))


@pytest.fixture(autouse=True)
def _fresh_rate_limits():
    """Each test starts with empty rate-limit windows."""
    from app import security

    security.chat_limiter._hits.clear()
    security.feedback_limiter._hits.clear()
