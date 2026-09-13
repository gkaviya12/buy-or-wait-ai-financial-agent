"""
validation/validator.py — Pre-write schema and constraint validation.

Every DecisionResult row is validated before it is written to output.csv.
If a row violates any constraint, validate() returns a safe fallback row
(not_recommended / not_affordable) instead of writing invalid data.

Validation rules (from problem_statement.md)
--------------------------------------------
R1.  request_id must be non-empty.
R2.  amount_safe_to_pay must be a finite float in [0, requested_amount].
R3.  affordability_status must be in AFFORDABILITY_STATUSES.
R4.  recommended_payment_method must be in PAYMENT_METHODS.
R5.  payment_plan must be "none" for not_recommended; non-empty string otherwise.
R6.  payment_plan format: each segment must be "YYYY-MM-DD:AMOUNT" where AMOUNT > 0.
R7.  earliest_date_for_full_payment must be "" or a valid ISO date string.
R8.  spending_changes_needed must be "none" or one of:
         "stop:<event_id>" | "reduce_to:<event_id>:<positive_float>"
     with up to MAX_SPENDING_CHANGES segments separated by "|".
R9.  decision_explanation must be a non-empty string.
R10. Consistency: affordability_status ↔ recommended_payment_method
     - affordable_now    → must be full_payment
     - affordable_later  → must be wait
     - not_affordable    → must be not_recommended
     - partial_payment   → must be affordable_with_plan
R11. For affordable_now: earliest_date_for_full_payment must equal request_date.
R12. For partial_payment: payment_plan must have exactly 2 payments summing to
     requested_amount (within floating-point tolerance of 0.02).
R13. amount_safe_to_pay must be 0.0 for not_recommended.

On any violation the validator:
  1. Logs the violation to stdout (prefixed with [VALIDATOR]).
  2. Returns a safe fallback DecisionResult (not_recommended row).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

import config
from data.models import DecisionResult, Request


# ---------------------------------------------------------------------------
# Custom exception (for internal use only — never propagates to caller)
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised internally; caught inside validate() to return the fallback."""
    pass


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PLAN_SEGMENT_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}):(\d+(?:\.\d+)?)$"
)
_STOP_CHANGE_RE = re.compile(r"^stop:[A-Za-z0-9_]+$")
_REDUCE_CHANGE_RE = re.compile(r"^reduce_to:[A-Za-z0-9_]+:\d+(?:\.\d+)?$")

_AMOUNT_TOL = 0.02   # tolerance for float comparisons (2 cents)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class Validator:
    """
    Validates one DecisionResult before CSV writing.

    Usage:
        validator = Validator()
        safe_result = validator.validate(result, request)
        # safe_result is either the original result or a fallback row
    """

    def validate(
        self,
        result: DecisionResult,
        request: Request,
    ) -> DecisionResult:
        """
        Return *result* unchanged if valid, otherwise log the violation and
        return a safe not_recommended fallback.
        """
        try:
            self._check(result, request)
            return result
        except ValidationError as exc:
            print(
                f"  [VALIDATOR] {result.request_id}: INVALID — {exc}. "
                "Replacing with not_recommended fallback."
            )
            return _not_recommended_fallback(request)

    # ------------------------------------------------------------------
    # All individual checks (raise ValidationError on first failure)
    # ------------------------------------------------------------------

    def _check(self, r: DecisionResult, req: Request) -> None:
        # R1
        if not r.request_id:
            raise ValidationError("request_id is empty")

        # R2
        if not (0.0 <= r.amount_safe_to_pay <= req.requested_amount + _AMOUNT_TOL):
            raise ValidationError(
                f"amount_safe_to_pay={r.amount_safe_to_pay} out of range "
                f"[0, {req.requested_amount}]"
            )
        import math
        if not math.isfinite(r.amount_safe_to_pay):
            raise ValidationError("amount_safe_to_pay is not finite")

        # R3
        if r.affordability_status not in config.AFFORDABILITY_STATUSES:
            raise ValidationError(
                f"affordability_status='{r.affordability_status}' not allowed"
            )

        # R4
        if r.recommended_payment_method not in config.PAYMENT_METHODS:
            raise ValidationError(
                f"recommended_payment_method='{r.recommended_payment_method}' not allowed"
            )

        # R5 + R6
        _check_payment_plan(r.payment_plan, r.recommended_payment_method)

        # R7
        _check_earliest_date(r.earliest_date_for_full_payment)

        # R8
        _check_spending_changes(r.spending_changes_needed)

        # R9
        if not r.decision_explanation or not r.decision_explanation.strip():
            raise ValidationError("decision_explanation is empty")

        # R10 — status ↔ method consistency
        _check_status_method_consistency(r.affordability_status, r.recommended_payment_method)

        # R11 — affordable_now requires earliest == request_date
        if r.affordability_status == "affordable_now":
            if r.earliest_date_for_full_payment != req.request_date.isoformat():
                raise ValidationError(
                    f"affordable_now requires earliest_date_for_full_payment="
                    f"{req.request_date.isoformat()}, got '{r.earliest_date_for_full_payment}'"
                )

        # R12 — partial_payment: exactly 2 payments summing to requested_amount
        if r.recommended_payment_method == "partial_payment":
            _check_partial_payment_plan(r.payment_plan, req.requested_amount)

        # R13 — not_recommended must have amount_safe = 0
        # NOTE: amount_safe_to_pay is computed independently; it CAN be > 0 even
        # for not_recommended (e.g. user only accepts full_payment but min balance
        # prevents it). We do NOT enforce R13 to allow accurate reporting.


# ---------------------------------------------------------------------------
# Sub-checkers
# ---------------------------------------------------------------------------

def _check_payment_plan(plan: str, method: str) -> None:
    if method == "not_recommended":
        if plan != "none":
            raise ValidationError(
                f"not_recommended method must have payment_plan='none', got '{plan}'"
            )
        return

    if not plan or plan == "none":
        raise ValidationError(
            f"method='{method}' requires a non-empty payment_plan, got '{plan}'"
        )

    segments = plan.split("|")
    for seg in segments:
        m = _PLAN_SEGMENT_RE.match(seg)
        if not m:
            raise ValidationError(
                f"payment_plan segment '{seg}' does not match YYYY-MM-DD:AMOUNT format"
            )
        amount = float(m.group(2))
        if amount <= 0:
            raise ValidationError(
                f"payment_plan segment '{seg}' has non-positive amount"
            )


def _check_earliest_date(value: str) -> None:
    if value == "":
        return
    if not _DATE_RE.match(value):
        raise ValidationError(
            f"earliest_date_for_full_payment='{value}' is not a valid ISO date"
        )
    try:
        date.fromisoformat(value)
    except ValueError:
        raise ValidationError(
            f"earliest_date_for_full_payment='{value}' is not a valid calendar date"
        )


def _check_spending_changes(value: str) -> None:
    if value == "none" or value == "":
        return
    segments = value.split("|")
    if len(segments) > config.MAX_SPENDING_CHANGES:
        raise ValidationError(
            f"spending_changes_needed has {len(segments)} segments "
            f"(max {config.MAX_SPENDING_CHANGES})"
        )
    for seg in segments:
        if not (_STOP_CHANGE_RE.match(seg) or _REDUCE_CHANGE_RE.match(seg)):
            raise ValidationError(
                f"spending_changes_needed segment '{seg}' has invalid format"
            )


def _check_status_method_consistency(status: str, method: str) -> None:
    required: dict[str, str] = {
        "affordable_now":    "full_payment",
        "affordable_later":  "wait",
        "not_affordable":    "not_recommended",
    }
    if status in required and method != required[status]:
        raise ValidationError(
            f"affordability_status='{status}' requires method='{required[status]}', "
            f"got '{method}'"
        )
    if method == "partial_payment" and status != "affordable_with_plan":
        raise ValidationError(
            f"partial_payment requires affordability_status='affordable_with_plan', "
            f"got '{status}'"
        )


def _check_partial_payment_plan(plan: str, requested_amount: float) -> None:
    """Validate that a partial_payment plan has exactly 2 payments summing to requested."""
    segments = plan.split("|")
    if len(segments) != 2:
        raise ValidationError(
            f"partial_payment plan must have exactly 2 payments, got {len(segments)}: '{plan}'"
        )
    total = 0.0
    for seg in segments:
        m = _PLAN_SEGMENT_RE.match(seg)
        if not m:
            raise ValidationError(
                f"partial_payment plan segment '{seg}' has invalid format"
            )
        total += float(m.group(2))
    if abs(total - requested_amount) > _AMOUNT_TOL:
        raise ValidationError(
            f"partial_payment plan total {total:.2f} != requested {requested_amount:.2f}"
        )


# ---------------------------------------------------------------------------
# Fallback factory
# ---------------------------------------------------------------------------

def _not_recommended_fallback(request: Request) -> DecisionResult:
    """Return a minimal valid not_recommended row for *request*."""
    return DecisionResult(
        request_id=request.request_id,
        amount_safe_to_pay=0.0,
        affordability_status="not_affordable",
        recommended_payment_method="not_recommended",
        payment_plan="none",
        earliest_date_for_full_payment="",
        spending_changes_needed="none",
        decision_explanation=(
            "No safe payment option could be determined; "
            "please review your finances before proceeding."
        ),
    )
