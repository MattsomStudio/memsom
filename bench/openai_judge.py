"""openai_judge — same prompts as bench/judge.py, routed to OpenAI chat/completions.

Faithful: synthesize() and judge_correct() reuse judge.py's exact prompt text so
the only change vs the local run is the model (gpt-4o), not the wording.
Adds 429/5xx backoff for a long 500-question run.
"""
from __future__ import annotations
import json, os, time, urllib.request, urllib.error

_URL = "https://api.openai.com/v1/chat/completions"
_MODEL = os.environ.get("OAI_MODEL", "gpt-4o")


def _chat(prompt: str, model: str = _MODEL, timeout: int = 120, max_tries: int = 6) -> str:
    key = os.environ["OPENAI_API_KEY"]
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0}).encode("utf-8")
    for attempt in range(max_tries):
        req = urllib.request.Request(_URL, data=body, headers={
            "Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read())
            return (resp["choices"][0]["message"]["content"] or "").strip()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < max_tries - 1:
                time.sleep(2 ** attempt); continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < max_tries - 1:
                time.sleep(2 ** attempt); continue
            raise
    return ""


def synthesize(question: str, memories: list, model: str = _MODEL) -> str:
    if not memories:
        return ""
    block = "\n".join(f"- {m}" for m in memories)
    prompt = ("Answer the question using ONLY the memories below. Be concise. If the "
              "memories conflict, answer from them as given; do not add outside facts.\n\n"
              f"Question: {question}\n\nMemories:\n{block}\n\nAnswer:")
    try:
        return _chat(prompt, model)
    except Exception:
        return ""


def judge_correct(question: str, gold: str, answer: str, model: str = _MODEL) -> bool:
    if not answer or not answer.strip():
        return False
    prompt = (f"Question: {question}\n"
              f"Reference (correct) answer: {gold}\n"
              f"Candidate answer: {answer}\n\n"
              "Does the candidate answer contain the correct answer to the question, "
              "consistent with the reference answer? Ignore extra or contradictory "
              "text -- answer YES if the correct information is present, NO if it is "
              "absent or wrong. Reply with exactly one word: YES or NO.")
    try:
        text = _chat(prompt, model).upper()
    except Exception:
        return False
    for tok in text.replace("\n", " ").split():
        t = tok.strip(".,:;!*\"'")
        if t.startswith("YES"):
            return True
        if t.startswith("NO"):
            return False
    return text.startswith("Y")


if __name__ == "__main__":
    # cheap self-test on gpt-4o-mini
    a = synthesize("What is the capital of France?",
                   ["The user lives in Paris, the capital of France."], model="gpt-4o-mini")
    ok = judge_correct("What is the capital of France?", "Paris", a, model="gpt-4o-mini")
    print("synth:", a)
    print("judge YES?:", ok)
