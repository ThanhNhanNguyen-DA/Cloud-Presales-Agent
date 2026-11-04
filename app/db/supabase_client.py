"""
Supabase client, created lazily on first use.

Modules keep doing `from app.db.supabase_client import supabase` and calling
`supabase.table(...)` / `supabase.rpc(...)` as before; the real client is only built
(and the settings checked) the first time one of those is called. Importing the app
therefore needs no network and no .env, and tests can patch `<module>.supabase`.
"""

from functools import lru_cache

from supabase import Client, create_client

from app import config


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    config.validate_config()
    return create_client(config.SUPABASE_URL, config.SUPABASE_KEY)


class _LazySupabase:
    """Stand-in that forwards every attribute to the real client, built on first access."""

    def __getattr__(self, name: str):
        # Introspection (mock.patch / inspect probing `__func__`, `_is_coroutine_marker`, …)
        # must not build a real client; the client's public API has no leading underscore.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(get_supabase_client(), name)


supabase: Client = _LazySupabase()  # type: ignore[assignment]
