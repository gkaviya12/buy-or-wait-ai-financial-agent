"""
llm/usage_tracker.py — Accumulates API call metadata for usage_report.md.
Thread-safe via a simple lock (unused in single-threaded pipeline, but good practice).
"""

from __future__ import annotations

import threading
from pathlib import Path
from datetime import datetime


class UsageTracker:
    """
    Tracks every LLM API call made during the pipeline run.

    After the pipeline finishes, call write_report() to produce
    the required evaluation/usage_report.md.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Per-model stats: model_name → {calls, prompt_tokens, completion_tokens, cost}
        self._stats: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float = 0.0,
    ) -> None:
        with self._lock:
            if model not in self._stats:
                self._stats[model] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": 0.0,
                }
            s = self._stats[model]
            s["calls"] += 1
            s["prompt_tokens"] += prompt_tokens
            s["completion_tokens"] += completion_tokens
            s["cost_usd"] += cost_usd

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def total_calls(self) -> int:
        return sum(s["calls"] for s in self._stats.values())

    def total_prompt_tokens(self) -> int:
        return sum(s["prompt_tokens"] for s in self._stats.values())

    def total_completion_tokens(self) -> int:
        return sum(s["completion_tokens"] for s in self._stats.values())

    def total_tokens(self) -> int:
        return self.total_prompt_tokens() + self.total_completion_tokens()

    def total_cost_usd(self) -> float:
        return sum(s["cost_usd"] for s in self._stats.values())

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def write_report(self, path: Path, num_requests: int = 250) -> None:
        """Write evaluation/usage_report.md as required by AGENTS.md §6.5."""
        path.parent.mkdir(parents=True, exist_ok=True)

        avg_tokens = (
            self.total_tokens() / num_requests if num_requests > 0 else 0
        )
        avg_cost = (
            self.total_cost_usd() / num_requests if num_requests > 0 else 0
        )

        lines = [
            "# Token Usage Report",
            "",
            f"Generated: {datetime.utcnow().isoformat()}Z",
            f"Requests processed: {num_requests}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Total API calls | {self.total_calls()} |",
            f"| Total prompt tokens | {self.total_prompt_tokens():,} |",
            f"| Total completion tokens | {self.total_completion_tokens():,} |",
            f"| Total tokens | {self.total_tokens():,} |",
            f"| Avg tokens per request | {avg_tokens:,.1f} |",
            f"| Estimated total cost (USD) | ${self.total_cost_usd():.4f} |",
            f"| Estimated cost per request (USD) | ${avg_cost:.4f} |",
            "",
            "## Per-Model Breakdown",
            "",
            "| Model | Calls | Prompt tokens | Completion tokens | Cost (USD) |",
            "|---|---|---|---|---|",
        ]

        for model, s in sorted(self._stats.items()):
            lines.append(
                f"| {model} | {s['calls']} | {s['prompt_tokens']:,} | "
                f"{s['completion_tokens']:,} | ${s['cost_usd']:.4f} |"
            )

        lines += [
            "",
            "## Notes",
            "",
            "- Cost estimates use public Gemini API pricing at time of run.",
            "- Prompt tokens include system prompts + user data context.",
            "- No API keys or credentials are included in this report.",
        ]

        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Usage report written to {path}")
