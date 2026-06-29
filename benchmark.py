"""
benchmark.py

Minimal benchmark for LangGraph Resume Copilot.

Dataset:
  - 1 resume (data/my_resume.txt)
  - 5 job descriptions (data/jd_1.txt ... jd_5.txt)
  - 3 repeated runs per JD
  - 15 total workflow runs

Comparison: Cache OFF vs Cache ON

Metrics per run:
  - total_latency_s
  - llm_calls
  - resume_cache_hit
  - jd_cache_hit
  - match_score
  - bullets_rewritten

Usage:
  python benchmark.py
  python benchmark.py --resume data/my_resume.txt
  python benchmark.py --runs 3 --output data/benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import time
import sqlite3
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(".env.example")


# ── JD files ──────────────────────────────────────────────────────────────────

JD_FILES = [
    "data/jd_1.txt",
    "data/jd_2.txt",
    "data/jd_3.txt",
    "data/jd_4.txt",
    "data/jd_5.txt",
]


# ── DB helpers ────────────────────────────────────────────────────────────────

def clear_cache(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM resume_cache")
    conn.execute("DELETE FROM jd_cache")
    conn.commit()
    conn.close()


# ── Single run ────────────────────────────────────────────────────────────────

def run_single(resume: str, jd_text: str, jd_title: str) -> dict:
    from src.graph.workflow import build_graph
    from src.graph.state import GraphState

    graph = build_graph()
    initial_state: GraphState = {
        "raw_resume":          resume,
        "raw_job_description": jd_text,
        "errors":              [],
        "retry_count":         0,
        "missing_keywords":    [],
        "weak_keywords":       [],
        "rewritten_bullets":   [],
        "resume_cache_hit":    False,
        "jd_cache_hit":        False,
        "education_entries":   [],
        "experience_entries":  [],
        "project_entries":     [],
    }

    t0 = time.perf_counter()
    result = graph.invoke(initial_state, config={"run_name": f"benchmark | {jd_title}"})
    total_latency = round(time.perf_counter() - t0, 2)

    resume_hit = result.get("resume_cache_hit", False)
    jd_hit     = result.get("jd_cache_hit", False)

    llm_calls = 3
    if resume_hit: llm_calls -= 1
    if jd_hit:     llm_calls -= 1

    return {
        "jd_title":          jd_title,
        "total_latency_s":   total_latency,
        "resume_cache_hit":  resume_hit,
        "jd_cache_hit":      jd_hit,
        "llm_calls":         llm_calls,
        "errors":            result.get("errors", []),
        "match_score":       getattr(result.get("match_score"), "overall_score", None),
        "bullets_rewritten": len(result.get("rewritten_bullets", [])),
    }


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(resume_path: str, runs_per_jd: int, db_path: str, output_path: str) -> None:
    resume = Path(resume_path).read_text()

    # Load JDs from files
    jds: list[dict] = []
    for jd_file in JD_FILES:
        p = Path(jd_file)
        if p.exists():
            title = p.stem.replace("jd_", "JD ").title()
            jds.append({"title": title, "text": p.read_text()})
        else:
            print(f"  ⚠️  {jd_file} not found — skipping")

    if not jds:
        print("❌ No JD files found. Check that data/jd_1.txt ... jd_5.txt exist.")
        return

    print(f"\n{'='*60}")
    print(f"LangGraph Resume Copilot — Benchmark")
    print(f"Resume:  {resume_path}")
    print(f"JDs:     {len(jds)}")
    print(f"Runs/JD: {runs_per_jd}")
    print(f"Total:   {len(jds) * runs_per_jd} workflow runs")
    print(f"{'='*60}\n")

    all_results: list[dict] = []

    # ── Phase 1: Cache OFF ────────────────────────────────────────────────────
    print("📊 Phase 1: Cache OFF (cold start)")
    print("-" * 40)

    cache_off_results: list[dict] = []
    for jd in jds:
        clear_cache(db_path)
        r = run_single(resume, jd["text"], jd["title"])
        r["phase"] = "cache_off"
        r["run_idx"] = 1
        cache_off_results.append(r)
        all_results.append(r)
        status = "✅" if not r["errors"] else "❌"
        print(f"  {status} {r['jd_title']:20s} {r['total_latency_s']:6.1f}s  "
              f"LLM={r['llm_calls']}  score={r['match_score'] or 'N/A'}")

    # ── Phase 2: Cache ON ─────────────────────────────────────────────────────
    print(f"\n📊 Phase 2: Cache ON (runs 2–{runs_per_jd} per JD)")
    print("-" * 40)

    clear_cache(db_path)
    print("  Warming cache...")
    for jd in jds:
        run_single(resume, jd["text"], jd["title"])

    cache_on_results: list[dict] = []
    for run_idx in range(2, runs_per_jd + 1):
        for jd in jds:
            r = run_single(resume, jd["text"], jd["title"])
            r["phase"] = "cache_on"
            r["run_idx"] = run_idx
            cache_on_results.append(r)
            all_results.append(r)
            hits = []
            if r["resume_cache_hit"]: hits.append("resume")
            if r["jd_cache_hit"]:     hits.append("JD")
            hit_str = f"cache_hit=[{', '.join(hits)}]" if hits else "no_hit"
            print(f"  run{run_idx} {r['jd_title']:20s} {r['total_latency_s']:6.1f}s  "
                  f"LLM={r['llm_calls']}  {hit_str}")

    # ── Summary ───────────────────────────────────────────────────────────────
    def avg(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    off_lat = [r["total_latency_s"] for r in cache_off_results if not r["errors"]]
    on_lat  = [r["total_latency_s"] for r in cache_on_results  if not r["errors"]]
    off_llm = [r["llm_calls"] for r in cache_off_results]
    on_llm  = [r["llm_calls"] for r in cache_on_results]

    cache_hit_rate = sum(
        1 for r in cache_on_results if r["resume_cache_hit"] or r["jd_cache_hit"]
    ) / max(len(cache_on_results), 1)

    avg_off = avg(off_lat)
    avg_on  = avg(on_lat)
    speedup      = round((avg_off - avg_on) / avg_off * 100, 1) if avg_off > 0 else 0
    llm_reduction = round((avg(off_llm) - avg(on_llm)) / max(avg(off_llm), 1) * 100, 1)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Cache OFF  avg latency:   {avg_off:.1f}s   avg LLM calls: {avg(off_llm):.1f}")
    print(f"  Cache ON   avg latency:   {avg_on:.1f}s   avg LLM calls: {avg(on_llm):.1f}")
    print(f"  Latency reduction:        {speedup:.0f}%")
    print(f"  LLM call reduction:       {llm_reduction:.0f}%")
    print(f"  Cache hit rate:           {cache_hit_rate:.0%}")
    print(f"  Total runs completed:     {len(all_results)}")
    print(f"  Errors:                   {sum(1 for r in all_results if r['errors'])}")
    print(f"\n{'─'*60}")
    print("RESUME-READY METRIC:")
    print(f"  Implemented SQLite caching reducing avg latency by {speedup:.0f}%")
    print(f"  ({avg_off:.0f}s → {avg_on:.0f}s) and LLM calls by {llm_reduction:.0f}%")
    print(f"  across {len(jds)} JDs with {cache_hit_rate:.0%} cache hit rate")
    print(f"{'='*60}\n")

    output = {
        "timestamp":   datetime.utcnow().isoformat(),
        "resume_path": resume_path,
        "jd_count":    len(jds),
        "runs_per_jd": runs_per_jd,
        "total_runs":  len(all_results),
        "summary": {
            "cache_off_avg_latency_s": avg_off,
            "cache_on_avg_latency_s":  avg_on,
            "latency_reduction_pct":   speedup,
            "llm_call_reduction_pct":  llm_reduction,
            "cache_hit_rate":          round(cache_hit_rate, 3),
            "cache_off_avg_llm_calls": avg(off_llm),
            "cache_on_avg_llm_calls":  avg(on_llm),
        },
        "runs": all_results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, indent=2))
    print(f"Results saved to: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph Resume Copilot Benchmark")
    parser.add_argument("--resume", default="data/my_resume.txt")
    parser.add_argument("--runs",   type=int, default=3)
    parser.add_argument("--db",     default="./data/results.db")
    parser.add_argument("--output", default="./data/benchmark_results.json")
    args = parser.parse_args()

    run_benchmark(
        resume_path=args.resume,
        runs_per_jd=args.runs,
        db_path=args.db,
        output_path=args.output,
    )