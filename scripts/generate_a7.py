"""A7 generation: rewrite the sampled spam with an LLM (or the fallback).

Runs anywhere the repo is cloned (laptop or Colab); needs only the Python
standard library plus numpy/pandas. Resumable: re-running skips messages
already in the output file, so it is safe to stop and restart.

Examples
--------
Ollama on a laptop (after `ollama pull llama3.1:8b`):
    python scripts/generate_a7.py --backend ollama --model llama3.1:8b

Any OpenAI-compatible API (key read from the environment, never the repo):
    export LLM_API_KEY=...            # Windows: set LLM_API_KEY=...
    # Groq
    python scripts/generate_a7.py --backend openai \
        --base-url https://api.groq.com/openai/v1 --model llama-3.1-8b-instant
    # Google Gemini (OpenAI-compatible endpoint)
    python scripts/generate_a7.py --backend openai \
        --base-url https://generativelanguage.googleapis.com/v1beta/openai \
        --model gemini-2.0-flash
    # OpenRouter
    python scripts/generate_a7.py --backend openai \
        --base-url https://openrouter.ai/api/v1 --model meta-llama/llama-3.1-8b-instruct

Rule-based fallback (no LLM):
    python scripts/generate_a7.py --backend rule

Output: data/a7/generations_<backend>_<model>.jsonl, one JSON object per
(message, variant) with the rewrite, refusal flag and safety checks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spam_nb.llm_camouflage import (  # noqa: E402
    PROMPTS,
    SYSTEM_PROMPT,
    clean_response,
    has_live_contact,
    is_refusal,
    placeholders_kept,
    rule_based_rewrite,
)

SAMPLE = ROOT / "data" / "a7" / "a7_sample.jsonl"
PROMPT_VERSION = "v1"


def _post(url: str, payload: dict, headers: dict, timeout: int = 180) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def call_llm(args, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if args.backend == "ollama":
        reply = _post(
            args.base_url.rstrip("/") + "/api/chat",
            {"model": args.model, "messages": messages, "stream": False,
             "options": {"temperature": args.temperature, "seed": args.seed}},
            {},
        )
        return reply["message"]["content"]

    key = os.environ.get(args.api_key_env, "")
    if not key:
        raise SystemExit(f"Set the {args.api_key_env} environment variable to your API key.")
    reply = _post(
        args.base_url.rstrip("/") + "/chat/completions",
        {"model": args.model, "messages": messages, "temperature": args.temperature,
         "max_tokens": args.max_tokens, "seed": args.seed},
        {"Authorization": f"Bearer {key}"},
    )
    return reply["choices"][0]["message"]["content"] or ""


def generate_one(args, message: str, variant: str, index: int) -> tuple[str, str]:
    """Return (raw response, cleaned rewrite)."""
    if args.backend == "rule":
        text = rule_based_rewrite(message, variant, seed=args.seed + index)
        return text, text
    prompt = PROMPTS[variant].format(message=message)
    delay = 5.0
    for attempt in range(6):
        try:
            raw = call_llm(args, prompt)
            return raw, clean_response(raw)
        except urllib.error.HTTPError as error:
            body = error.read().decode(errors="replace")[:300]
            if error.code in (429, 500, 502, 503, 504) and attempt < 5:
                print(f"    HTTP {error.code}; retrying in {delay:.0f}s")
                time.sleep(delay)
                delay *= 2
                continue
            raise SystemExit(f"HTTP {error.code} from the LLM: {body}")
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt < 5:
                print(f"    {error}; retrying in {delay:.0f}s")
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=["ollama", "openai", "rule"], required=True)
    parser.add_argument("--model", default="rule-based-v1")
    parser.add_argument("--base-url", default="http://localhost:11434",
                        help="Ollama server, or the OpenAI-compatible API base URL")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--variants", nargs="+", default=list(PROMPTS), choices=list(PROMPTS))
    parser.add_argument("--n", type=int, default=None, help="only the first n messages")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="pause between calls (free-tier rate limits)")
    parser.add_argument("--sample", type=Path, default=SAMPLE)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.backend == "rule":
        args.model = "rule-based-v1"
    tag = re.sub(r"[^A-Za-z0-9.-]+", "-", f"{args.backend}_{args.model}").strip("-")
    out = args.out or ROOT / "data" / "a7" / f"generations_{tag}.jsonl"

    sample = [json.loads(line) for line in open(args.sample)]
    if args.n:
        sample = sample[: args.n]

    done = set()
    if out.exists():
        done = {(r["id"], r["variant"]) for r in map(json.loads, open(out))}
    todo = [(i, row, v) for i, row in enumerate(sample) for v in args.variants
            if (row["id"], v) not in done]
    print(f"{len(done)} already done, {len(todo)} to go -> {out}")

    out.parent.mkdir(parents=True, exist_ok=True)
    refusals = 0
    with open(out, "a") as fh:
        for count, (i, row, variant) in enumerate(todo, 1):
            start = time.time()
            raw, rewrite = generate_one(args, row["original"], variant, i)
            refused = is_refusal(rewrite)
            refusals += refused
            record = {
                "id": row["id"], "variant": variant, "backend": args.backend,
                "model": args.model, "prompt_version": PROMPT_VERSION,
                "temperature": args.temperature, "seed": args.seed,
                "rewrite": rewrite, "raw": raw, "refused": refused,
                "placeholders_kept": placeholders_kept(row["original"], rewrite),
                "live_contact": has_live_contact(rewrite),
                "seconds": round(time.time() - start, 2),
            }
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            if count % 25 == 0 or count == len(todo):
                print(f"  {count}/{len(todo)}  refusals so far: {refusals}")
            if args.sleep:
                time.sleep(args.sleep)
    print(f"Done: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
