"""
llm/usage_tracker.py — Thread-safe API call recorder and Markdown report generator.

Every call made through LLMClient is recorded here.  At the end of the pipeline
run, write_report() produces the evaluation/usage_report.md required by the spec.

Tracked per call:
    model, call_type ('text'|'vision'), purpose ('ocr'|'explanation'|'other'),
    request_id, prompt_tokens, completion_tokens, estimated cost (USD)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config


# ---------------------------------------------------------------------------
# Gemini pricing table  (USD per 1 M tokens, approximate)
# Update from: https://ai.google.dev/pricing
# ---------------------------------------------------------------------------

_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.0-flash":  {"input": 0.10,  "output": 0.40},
    "gemini-1.5-flash":  {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro":    {"input": 1.25,  "output": 5.00},
    "gemini-2.0-pro":    {"input": 1.25,  "output": 5.00},
}
_DEFAULT_PRICING = {"input": 0.10, "output": 0.40}


def estimate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return the estimated USD cost for a single Gemini API call."""
    lower = model_name.lower()
    pricing = _DEFAULT_PRICING
    for key, p in _PRICING.items():
        if key in lower:
            pricing = p
            break
    return (
        prompt_tokens * pricing["input"] / 1_000_000
        + completion_tokens * pricing["output"] / 1_000_000
    )


# ---------------------------------------------------------------------------
# CallRecord — one row per API call
# ---------------------------------------------------------------------------

@dataclass
class CallRecord:
    """A single recorded API call."""
    timestamp: str       # ISO-8601 UTC
    model: str
    call_type: str       # 'text' | 'vision'
    purpose: str         # 'ocr' | 'explanation' | 'other'
    request_id: str      # '' when not tied to a specific request
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


# ---------------------------------------------------------------------------
# UsageTracker
# ---------------------------------------------------------------------------

class UsageTracker:
    """
    Thread-safe accumulator for all Gemini API calls made in one pipeline run.

    Usage:
        tracker = UsageTracker()
        # LLMClient calls tracker.record_call(...) automatically
        tracker.write_report()   # at end of pipeline
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: list[CallRecord] = []
        # Legacy per-model aggregate dict kept for backward-compat
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
        call_type: str = "text",
        purpose: str = "other",
        request_id: str = "",
    ) -> None:
        """Record one completed API call (thread-safe)."""
        record = CallRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=model,
            call_type=call_type,
            purpose=purpose,
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
        )
        with self._lock:
            self._calls.append(record)
            # Update legacy per-model stats
            if model not in self._stats:
                self._stats[model] = {
                    "calls": 0, "prompt_tokens": 0,
                    "completion_tokens": 0, "cost_usd": 0.0,
                }
            s = self._stats[model]
            s["calls"] += 1
            s["prompt_tokens"] += prompt_tokens
            s["completion_tokens"] += completion_tokens
            s["cost_usd"] += cost_usd

    # ------------------------------------------------------------------
    # Aggregate properties
    # ------------------------------------------------------------------

    def total_calls(self) -> int:
        return len(self._calls)

    def total_prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self._calls)

    def total_completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self._calls)

    def total_tokens(self) -> int:
        return self.total_prompt_tokens() + self.total_completion_tokens()

    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self._calls)

    def calls_by_purpose(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self._calls:
            counts[c.purpose] = counts.get(c.purpose, 0) + 1
        return counts

    def cost_by_purpose(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for c in self._calls:
            totals[c.purpose] = totals.get(c.purpose, 0.0) + c.cost_usd
        return totals

    def summary_dict(self) -> dict:
        """Plain-dict summary for stdout printing."""
        return {
            "total_calls": self.total_calls(),
            "total_prompt_tokens": self.total_prompt_tokens(),
            "total_completion_tokens": self.total_completion_tokens(),
            "total_cost_usd": round(self.total_cost_usd(), 4),
            "by_purpose": self.calls_by_purpose(),
        }

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def write_report(
        self,
        path: Path = config.USAGE_REPORT_PATH,
        num_requests: int = 250,
    ) -> None:
        """Write evaluation/usage_report.md as required by AGENTS.md §6.5."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._build_report(num_requests), encoding="utf-8")
        print(f"  [UsageTracker] Usage report written -> {path}")

    def _build_report(self, num_requests: int) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        total_tok = self.total_tokens()
        total_cost = self.total_cost_usd()
        avg_tokens = total_tok / num_requests if num_requests > 0 else 0
        avg_cost = total_cost / num_requests if num_requests > 0 else 0

        lines: list[str] = [
            "# Token Usage Report",
            "",
            f"Generated: {now}  ",
            f"Requests processed: {num_requests}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total API calls | {self.total_calls():,} |",
            f"| Total prompt tokens | {self.total_prompt_tokens():,} |",
            f"| Total completion tokens | {self.total_completion_tokens():,} |",
            f"| Total tokens | {total_tok:,} |",
            f"| Avg tokens per request | {avg_tokens:,.1f} |",
            f"| Estimated total cost | ${total_cost:.4f} USD |",
            f"| Estimated cost per request | ${avg_cost:.5f} USD |",
            "",
        ]

        # Breakdown by purpose
        by_purpose = self.calls_by_purpose()
        cost_by_p = self.cost_by_purpose()
        if by_purpose:
            lines += [
                "## Breakdown by Purpose",
                "",
                "| Purpose | Calls | Est. Cost (USD) |",
                "|---------|-------|-----------------|",
            ]
            for purpose in sorted(by_purpose):
                lines.append(
                    f"| {purpose} | {by_purpose[purpose]:,} "
                    f"| ${cost_by_p.get(purpose, 0.0):.4f} |"
                )
            lines.append("")

        # Per-model breakdown
        if self._stats:
            lines += [
                "## Per-Model Breakdown",
                "",
                "| Model | Calls | Prompt Tokens | Completion Tokens | Cost (USD) |",
                "|-------|-------|---------------|-------------------|------------|",
            ]
            for model, s in sorted(self._stats.items()):
                lines.append(
                    f"| {model} | {s['calls']:,} | {s['prompt_tokens']:,} "
                    f"| {s['completion_tokens']:,} | ${s['cost_usd']:.4f} |"
                )
            lines.append("")

        # Call log (most recent 100)
        if self._calls:
            recent = self._calls[-100:]
            start_idx = max(1, self.total_calls() - 99)
            lines += [
                "## Call Log (most recent 100)",
                "",
                "| # | Timestamp | Model | Type | Purpose | Request | "
                "Prompt Tok | Compl Tok | Cost (USD) |",
                "|---|-----------|-------|------|---------|---------|"
                "-----------|-----------|------------|",
            ]
            for i, c in enumerate(recent, start=start_idx):
                lines.append(
                    f"| {i} | {c.timestamp[:19]}Z | {c.model} | {c.call_type} "
                    f"| {c.purpose} | {c.request_id or '—'} "
                    f"| {c.prompt_tokens:,} | {c.completion_tokens:,} "
                    f"| ${c.cost_usd:.5f} |"
                )
            lines.append("")

        lines += [
            "## Notes",
            "",
            "- Cost estimates use public Gemini API pricing at time of run.",
            "- Prompt tokens include system prompt + user data context.",
            "- No API keys or credentials are included in this report.",
        ]
        return "\n".join(lines)
