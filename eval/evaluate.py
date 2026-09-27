"""Measure how well the assistant works, so you can report real numbers.

    python -m eval.evaluate              # retrieval metrics (free, no API key needed)
    python -m eval.evaluate --answers    # also grade LLM answers (needs LLM_API_KEY)

Retrieval metrics (per method):
  Hit@k  - % of questions where the correct section is in the top k results
  Doc@3  - % where any section of the correct article is in the top 3
  MRR    - Mean Reciprocal Rank: 1 if correct section is ranked 1st, 0.5 if 2nd, ...

Answer metrics (--answers):
  Accuracy  - answer contains the key fact(s) for that question (e.g. "14" characters)
  Citations - answer cites at least one source like [1]
  Refusals  - for off-topic questions, the bot declines instead of making something up
"""
import argparse
import json
import re
import statistics
import time
from pathlib import Path

from src import config
from src.retriever import Retriever

HERE = Path(__file__).parent
METHODS = ["bm25", "dense", "hybrid"]
REFUSAL_PATTERNS = r"don.t have|do not have|not (?:covered|in the|available)|no information|unable to|can.t help|cannot help|raise a ticket|4400"


def load_questions() -> list[dict]:
    lines = (HERE / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def evaluate_retrieval(retriever: Retriever, questions: list[dict]) -> dict:
    in_scope = [q for q in questions if q["expected_chunk"]]
    results = {}
    for method in METHODS:
        hits = {1: 0, 3: 0, 5: 0}
        doc_hits, rr, times, misses = 0, [], [], []
        for q in in_scope:
            start = time.perf_counter()
            ranked = [c["id"] for c in retriever.search(q["question"], k=5, method=method)]
            times.append((time.perf_counter() - start) * 1000)
            target = q["expected_chunk"]
            rank = ranked.index(target) + 1 if target in ranked else None
            for k in hits:
                hits[k] += bool(rank and rank <= k)
            doc_hits += any(cid.split("#")[0] == target.split("#")[0] for cid in ranked[:3])
            rr.append(1 / rank if rank else 0)
            if not rank or rank > 3:
                misses.append((q["question"], target, ranked[0]))
        n = len(in_scope)
        results[method] = {
            "Hit@1": 100 * hits[1] / n, "Hit@3": 100 * hits[3] / n, "Hit@5": 100 * hits[5] / n,
            "Doc@3": 100 * doc_hits / n, "MRR": statistics.mean(rr),
            "ms/query": statistics.median(times), "misses": misses,
        }
    return results


def evaluate_answers(retriever: Retriever, questions: list[dict]) -> dict:
    from src.generator import generate_answer

    correct = cited = refused = 0
    in_scope = [q for q in questions if q["expected_chunk"]]
    off_topic = [q for q in questions if not q["expected_chunk"]]
    failures = []
    for i, q in enumerate(questions, 1):
        print(f"  answering {i}/{len(questions)}", end="\r")
        answer = generate_answer(q["question"], retriever.search(q["question"], method="hybrid"))
        low = answer.lower()
        if q["expected_chunk"]:
            # each fact may list alternatives separated by "|", e.g. "18:00|6pm"
            ok = all(any(alt in low for alt in fact.lower().split("|")) for fact in q["must_include"])
            correct += ok
            cited += bool(re.search(r"\[\d\]", answer))
            if not ok:
                failures.append((q["question"], q["must_include"], answer[:200]))
        else:
            refused += bool(re.search(REFUSAL_PATTERNS, low))
        time.sleep(0.5)  # stay under free-tier rate limits
    return {
        "Accuracy": 100 * correct / len(in_scope),
        "Citations": 100 * cited / len(in_scope),
        "Refusals": 100 * refused / len(off_topic),
        "failures": failures,
    }


def to_markdown(retrieval: dict, answers: dict | None, n: int) -> str:
    cols = ["Hit@1", "Hit@3", "Hit@5", "Doc@3", "MRR", "ms/query"]
    lines = [f"# Evaluation results\n\n{n} test questions · embedding model `{config.EMBED_MODEL}`\n",
             "## Retrieval\n", "| Method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for method, r in retrieval.items():
        cells = [f"{r[c]:.1f}%" if c.startswith(("Hit", "Doc")) else
                 f"{r[c]:.3f}" if c == "MRR" else f"{r[c]:.1f}" for c in cols]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    if answers:
        lines += ["\n## Answers (hybrid retrieval, top 4)\n", f"LLM: `{config.LLM_MODEL}`\n",
                  "| Metric | Score |", "|---|---|",
                  f"| Answer accuracy (key facts present) | {answers['Accuracy']:.1f}% |",
                  f"| Answers citing a source | {answers['Citations']:.1f}% |",
                  f"| Off-topic questions correctly declined | {answers['Refusals']:.1f}% |"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", action="store_true", help="also grade LLM answers")
    args = parser.parse_args()

    questions = load_questions()
    retriever = Retriever()
    retrieval = evaluate_retrieval(retriever, questions)
    answers = evaluate_answers(retriever, questions) if args.answers else None

    report = to_markdown(retrieval, answers, len([q for q in questions if q["expected_chunk"]]))
    print(report)
    best = max(retrieval, key=lambda m: retrieval[m]["MRR"])
    print(f"Questions {best} ranked outside the top 3 (useful for spotting gaps):")
    for question, target, got in retrieval[best]["misses"]:
        print(f"  - {question!r}\n      expected {target}, top result {got}")
    if answers:
        print("\nAnswers missing key facts:")
        for question, facts, answer in answers["failures"]:
            print(f"  - {question!r} (needed {facts})\n      {answer!r}")
    (HERE / "results.md").write_text(report, encoding="utf-8")
    print(f"\nSaved to {HERE / 'results.md'}")


if __name__ == "__main__":
    main()
