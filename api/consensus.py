import asyncio
import httpx
import time
import math
import uuid
from api.db import log_model_calls
from api.config import  ACTIVE_MODELS
from api.providers import call_model, embed_text
from api.cache import get_ask_cached, set_ask_cached
from api.judge import judge_best

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))     
    na = math.sqrt(sum(x * x for x in a))      
    nb = math.sqrt(sum(y * y for y in b)) 
    return dot / (na * nb) if na and nb else 0.0

async def pick_best(client: httpx.AsyncClient, successful: list[dict]) -> tuple[dict, dict]:
    if len(successful) == 1:
        r = successful[0]
        return r, {r["name"]: 1.0}
    embeddings = await asyncio.gather(
    *[embed_text(client, r["text"]) for r in successful])

    scores: dict[str, float] = {}
    for i, r in enumerate(successful):
        sims = [
        cosine(embeddings[i], embeddings[j])
        for j in range(len(successful))
            if j != i]
        scores[r["name"]] = sum(sims) / len(sims)  
    winner = max(successful, key=lambda r: (scores[r["name"]], -r["latency_ms"]))
    return winner, scores


async def run_consensus(prompt: str, use_cache: bool = True, strategy: str = "semantic") -> dict:
    request_id = uuid.uuid4()

    if use_cache:
        cached = await get_ask_cached(prompt, strategy)
        if cached is not None:
            cached["cached"] = True
            return cached

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[call_model(client, m, prompt) for m in ACTIVE_MODELS])
        working = [r for r in results if r["ok"]]

        if not working:
            for r in results:
                r["agreement"], r["was_selected"] = 0.0, False
            await log_model_calls(request_id, prompt, results)
            return {"answer": None, "chosen": None, "strategy": strategy,
                    "candidates": results, "cached": False, "request_id": str(request_id)}

        judge_meta = None
        try:
            if strategy == "judge":
                winner, scores, judge_meta = await judge_best(client, prompt, working)
                strategy_used = "llm-judge"
            else:
                winner, scores = await pick_best(client, working)
                strategy_used = "semantic-consensus"
        except Exception:
            winner = min(working, key=lambda r: r["latency_ms"])
            scores = {}
            strategy_used = "fastest-fallback"

    for r in results:
        r["agreement"] = round(scores.get(r["name"], 0.0), 3)
        r["was_selected"] = (r["name"] == winner["name"])

    result = {"answer": winner["text"], "chosen": winner["name"],
              "strategy": strategy_used, "candidates": results, "cached": False}
    if judge_meta:
        result["judge"] = judge_meta

    await log_model_calls(request_id, prompt, results)
    if use_cache:
        await set_ask_cached(prompt, result, strategy)
    result["request_id"] = str(request_id)
    return result