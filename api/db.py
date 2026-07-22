import os
import asyncpg

database = os.environ.get("POSTGRES_URL", os.environ.get("database_url", "postgresql://postgres:postgres@localhost:5433/tripmate"))
_pool: asyncpg.Pool | None=None

async def init_db()-> None:
    global _pool
    try:
        _pool = await asyncpg.create_pool(database, min_size=1, max_size=5)
        async with _pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS model_calls (
                id           BIGSERIAL PRIMARY KEY,
                request_id   UUID        NOT NULL,   
                model        TEXT        NOT NULL,
                provider     TEXT        NOT NULL,
                ok           BOOLEAN     NOT NULL,
                latency_ms   INTEGER,
                tokens       INTEGER,
                agreement    REAL,                   
                was_selected BOOLEAN     NOT NULL DEFAULT FALSE,
                error        TEXT,
                created_at   TIMESTAMPTZ NOT NULL DEFAULT now());""")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_model_calls_model ON model_calls (model);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_model_calls_request ON model_calls (request_id);")
    except Exception as e:
        print(f"Database initialization failed: {e}")
        _pool = None

async def close_db()->None:
    if _pool is not None:
        await _pool.close()

async def log_model_calls(request_id, prompt: str, candidates: list[dict]) -> None:
    if _pool is None:
        return
    rows = [
    (request_id, c["name"], c["provider"], c["ok"], c.get("latency_ms"),
    c.get("tokens"), c.get("agreement"), c.get("was_selected", False), c.get("error"))
    for c in candidates]

    try:
        async with _pool.acquire() as conn:
            await conn.executemany("""
                INSERT INTO model_calls
                (request_id, model, provider, ok, latency_ms,
                tokens, agreement, was_selected, error)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, rows)
    except Exception as e:
        print(f"[db] log_model_calls failed probably ignored: {e}")

async def model_stats() -> list[dict]:
    if _pool is None:
        return []
    
    async with _pool.acquire() as conn:
        rows = await conn.fetch("""
        SELECT
            model, provider,
            COUNT(*)                             AS calls,
            COUNT(*) FILTER (WHERE ok)           AS successes,
            COUNT(*) FILTER (WHERE was_selected) AS wins,
            AVG(latency_ms) FILTER (WHERE ok)    AS avg_latency_ms,
            AVG(agreement)  FILTER (WHERE ok)    AS avg_agreement
        FROM model_calls
        GROUP BY model, provider
        ORDER BY wins DESC, calls DESC""")

    stats = []
    for r in rows:
        successes, wins = r["successes"] or 0, r["wins"] or 0
        stats.append({
            "model": r["model"], "provider": r["provider"],
            "calls": r["calls"], "successes": successes, "wins": wins,
            "win_rate": round(wins / successes, 3) if successes else 0.0,
            "avg_latency_ms": round(r["avg_latency_ms"]) if r["avg_latency_ms"] is not None else None,
            "avg_agreement": round(r["avg_agreement"], 3) if r["avg_agreement"] is not None else None,})
    return stats