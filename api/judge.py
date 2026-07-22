import re
import json 
from api.config import ACTIVE_MODELS
from api.providers import call_model

def pick_judge_model():
    for hint in ("70b","gemini"):
        for m in ACTIVE_MODELS:
            if hint in m.model:
                return m
    
    return ACTIVE_MODELS[0]

def _parse_verdict(text: str, n:int)-> tuple[int,str]:
    match = re.search(r"{.*}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON found in judge output: {text[:200]!r}")
    data = json.loads(match.group(0))
    best = int(data.get("best", 1))
    best = min(max(best, 1), n)  
    return best - 1, str(data.get("reason", "")) 

async def judge_best(client, prompt: str, successful: list[dict]):
    if len(successful) == 1:
        r = successful[0]
        return r, {r["name"]: 1.0}, {"judge_model": None, "reason": "only one candidate"}
    
    judge_model = pick_judge_model()
    listing = "nn".join(f"[{i + 1}] {r['text']}" for i, r in enumerate(successful))
    judge_prompt = (
    "You are a strict evaluator. Given a user question and several candidate "
    "answers, choose the SINGLE best one by correctness, completeness, and "
    "relevance. Respond with ONLY minified JSON of the form "
    '{"best": <1-based number>, "reason": "<one short sentence>"}.nn'
    f"Question:n{prompt}nn"
    f"Candidate answers:n{listing}nn"
    "JSON verdict:")

    res = await call_model(client, judge_model, judge_prompt)
    if not res["ok"]:
        raise RuntimeError(f"judge model failed: {res['error']}")
    idx, reason = _parse_verdict(res["text"], len(successful))
    winner = successful[idx]
    scores = {r["name"]: (1.0 if r is winner else 0.0) for r in successful}
    return winner, scores, {"judge_model": judge_model.name, "reason": reason}