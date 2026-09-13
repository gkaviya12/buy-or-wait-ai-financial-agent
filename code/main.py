"""
main.py — Entry point for the Buy-or-Wait pipeline.

End-to-end execution flow
==========================
1.  Load all CSV datasets into a DataBundle.
2.  Initialise LLMClient + UsageTracker (skipped if GEMINI_API_KEY is absent).
3.  OCR pass: fill any blank event amounts from linked images (one Gemini vision
    call per unique image; results cached).
4.  For each request (250 in full dataset):
      a. Retrieve profile, events, messages, payment_options.
      b. Extract message amendments (MessageProcessor).
      c. Build FinancialState (EventProcessor) — incorporates OCR fills.
      d. Build 90-day cash-flow projection (CashFlowProjector).
      e. Run safety analysis (SafetyChecker).
      f. Select best payment plan (DecisionEngine).
      g. Generate explanation (Explainer) — LLM or deterministic fallback.
      h. Validate the row (Validator) — replace with fallback on any violation.
5.  Write output.csv (OutputWriter).
6.  Write evaluation/usage_report.md (UsageTracker).
7.  Append completion entry to log.txt.
8.  Print a summary to stdout.

Usage:
    cd hackerrank-orchestrate-september26
    python code/main.py

Environment:
    GEMINI_API_KEY  — required for OCR and explanation generation.
                      If absent, OCR is skipped and explanations use the
                      deterministic fallback (pipeline still produces valid CSV).
    MODEL_TEXT      — overrides DEFAULT_MODEL_TEXT in config.py
    MODEL_VISION    — overrides DEFAULT_MODEL_VISION in config.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path

# Ensure code/ is on the module search path when run from the repo root
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dotenv import load_dotenv

load_dotenv()

import config
from data.loader import DataLoader
from data.exchange import ExchangeRateTable
from data.models import DecisionResult, Request
from ingestion.message_processor import MessageProcessor
from ingestion.event_processor import EventProcessor
from ingestion.ocr import ImageOCR
from forecasting.cash_flow import CashFlowProjector
from forecasting.safety_check import SafetyChecker
from decision import DecisionEngine
from output.explainer import Explainer, _fallback_explanation
from output.writer import OutputWriter
from validation import Validator
from llm.usage_tracker import UsageTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_llm(tracker: UsageTracker):
    """
    Try to initialise LLMClient.  Returns None (with a warning) if no API key.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("  [main] WARNING: GEMINI_API_KEY not set.")
        print("         OCR will be SKIPPED. Explanations will use deterministic fallback.")
        return None
    try:
        from llm.client import LLMClient
        client = LLMClient(tracker)
        print("  [main] Gemini client initialised.")
        return client
    except Exception as exc:
        print(f"  [main] WARNING: Could not initialise LLM client: {exc}")
        return None


def _make_fallback_explainer(client):
    """Return an Explainer (LLM-backed) or a pure-fallback object."""
    if client is not None:
        return Explainer(client)

    # Lightweight no-API fallback that mirrors the Explainer interface
    class _FallbackOnly:
        def explain(self, result, request, profile):
            return _fallback_explanation(result, request, profile)

    return _FallbackOnly()


def _log_completion(
    num_requests: int,
    num_violations: int,
    elapsed: float,
    output_path: Path,
    report_path: Path,
) -> None:
    """Append a completion entry to log.txt."""
    log_path = config.REPO_ROOT / "log.txt"
    from datetime import timezone as _tz
    ts = datetime.now(_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = (
        f"## [{ts}] Layer 7 -- main.py (full pipeline run)\n\n"
        f"Agent: Processed {num_requests} requests in {elapsed:.1f}s. "
        f"Validation violations replaced with fallback: {num_violations}. "
        f"Output written to {output_path}. "
        f"Usage report written to {report_path}.\n"
    )
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(entry + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(use_sample: bool = False) -> list[DecisionResult]:
    """
    Run the full pipeline.

    Parameters
    ----------
    use_sample : bool
        If True, process only the sample_requests.csv subset (faster for testing).

    Returns
    -------
    List of validated DecisionResult objects.
    """
    t_start = time.perf_counter()
    print("=" * 60)
    print("  BUY OR WAIT — Financial Agent Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load dataset
    # ------------------------------------------------------------------
    print("\n[1/7] Loading dataset...")
    loader = DataLoader()
    bundle = loader.load_all()

    if use_sample:
        # Filter to only the request_ids listed in sample_requests.csv
        try:
            import csv as _csv
            with open(config.SAMPLE_REQUESTS_CSV, encoding="utf-8") as fh:
                sample_ids = {row["request_id"] for row in _csv.DictReader(fh)}
            bundle.requests = [
                r for r in bundle.requests if r.request_id in sample_ids
            ]
            print(f"      Sample mode: {len(bundle.requests)} requests.")
        except FileNotFoundError:
            print("      sample_requests.csv not found; using all requests.")

    requests_to_process = bundle.requests
    print(f"      {len(requests_to_process)} requests to process.")

    # ------------------------------------------------------------------
    # Step 2: Initialise shared components
    # ------------------------------------------------------------------
    print("\n[2/7] Initialising components...")
    tracker = UsageTracker()
    llm_client = _init_llm(tracker)

    fx = ExchangeRateTable(bundle.exchange_rates)
    msg_proc = MessageProcessor()
    ev_proc = EventProcessor(fx)
    projector = CashFlowProjector()
    checker = SafetyChecker(projector)
    engine = DecisionEngine()
    explainer = _make_fallback_explainer(llm_client)
    validator = Validator()
    writer = OutputWriter()

    # ------------------------------------------------------------------
    # Step 3: OCR pass (fill blank amounts from images)
    # ------------------------------------------------------------------
    print("\n[3/7] OCR pass (filling blank event amounts)...")
    all_events = [ev for evs in bundle.events.values() for ev in evs]
    all_image_records = [
        rec for recs in bundle.image_records.values() for rec in recs
    ]
    blank_count = sum(1 for ev in all_events if ev.amount is None)
    print(f"      {blank_count} events have blank amounts.")

    ocr_fills: dict[str, float] = {}
    if llm_client is not None and blank_count > 0:
        ocr = ImageOCR(client=llm_client, images_dir=config.IMAGES_DIR)
        ocr_fills = ocr.fill_blank_amounts(all_events, all_image_records)
        print(f"      OCR filled {len(ocr_fills)} event amounts.")
    else:
        print("      OCR skipped (no API key or no blank amounts).")

    # ------------------------------------------------------------------
    # Step 4 + 5 + 6 + 7: Per-request pipeline
    # ------------------------------------------------------------------
    print(f"\n[4/7] Processing {len(requests_to_process)} requests...")
    results: list[DecisionResult] = []
    violations = 0
    errors = 0

    for i, req in enumerate(requests_to_process, start=1):
        if i % 25 == 0 or i == len(requests_to_process):
            print(f"      {i}/{len(requests_to_process)} processed...")

        try:
            uid = req.user_id
            profile = bundle.profiles[uid]
            raw_events = bundle.events.get(uid, [])
            msgs = bundle.messages.get(uid, [])
            ev_index = {ev.event_id: ev for ev in raw_events}

            # 4a. Extract message amendments
            amendments = msg_proc.extract_amendments(msgs, ev_index)

            # 4b. Build FinancialState
            state = ev_proc.build_state(
                profile=profile,
                raw_events=raw_events,
                amendments=amendments,
                ocr_fills=ocr_fills,
                request_date=req.request_date,
            )

            # 4c. 90-day projection + safety
            projection = projector.build_projection(state, req.request_date)
            safety = checker.compute_safety(
                projection, state, req.request_date, req.requested_amount
            )

            # 4d. Decision
            options = bundle.options.get(req.request_id, [])
            decision = engine.decide(req, state, safety, options)

            # 4e. Explanation
            explanation = explainer.explain(decision, req, profile)
            decision = replace(decision, decision_explanation=explanation)

            # 4f. Validation
            validated = validator.validate(decision, req)
            if validated.recommended_payment_method == "not_recommended" \
                    and decision.recommended_payment_method != "not_recommended":
                violations += 1

        except Exception as exc:
            print(f"  [main] ERROR on {req.request_id}: {exc}")
            traceback.print_exc()
            errors += 1
            # Emit a safe fallback row so the output CSV is always complete
            from validation.validator import _not_recommended_fallback
            validated = _not_recommended_fallback(req)
            validated = replace(
                validated,
                decision_explanation=(
                    "An internal error occurred; no recommendation could be generated."
                ),
            )

        results.append(validated)

    # ------------------------------------------------------------------
    # Step 5: Write output.csv
    # ------------------------------------------------------------------
    print(f"\n[5/7] Writing output.csv...")
    written = writer.write(results, path=config.OUTPUT_PATH)

    # ------------------------------------------------------------------
    # Step 6: Write usage report
    # ------------------------------------------------------------------
    print(f"\n[6/7] Writing usage report...")
    tracker.write_report(
        path=config.USAGE_REPORT_PATH,
        num_requests=len(requests_to_process),
    )

    # ------------------------------------------------------------------
    # Step 7: Log + summary
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    _log_completion(
        num_requests=len(requests_to_process),
        num_violations=violations,
        elapsed=elapsed,
        output_path=config.OUTPUT_PATH,
        report_path=config.USAGE_REPORT_PATH,
    )

    print(f"\n[7/7] Done in {elapsed:.1f}s.")
    print("=" * 60)
    print(f"  Requests processed : {len(requests_to_process)}")
    print(f"  Rows written       : {written}")
    print(f"  Validation fixes   : {violations}")
    print(f"  Pipeline errors    : {errors}")
    print(f"  LLM API calls      : {tracker.total_calls()}")
    print(f"  Total tokens used  : {tracker.total_tokens():,}")
    print(f"  Estimated cost     : ${tracker.total_cost_usd():.4f} USD")
    print(f"  Output             : {config.OUTPUT_PATH}")
    print(f"  Usage report       : {config.USAGE_REPORT_PATH}")
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Buy-or-Wait financial agent pipeline"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Process only the sample_requests.csv subset (faster for testing).",
    )
    args = parser.parse_args()

    run(use_sample=args.sample)
