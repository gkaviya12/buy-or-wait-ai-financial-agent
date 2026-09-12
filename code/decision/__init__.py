"""
decision/__init__.py — Facade: DecisionEngine ties all decision-layer modules together.

Usage:
    engine = DecisionEngine()
    result = engine.decide(request, state, safety_result, options)

The returned DecisionResult is ready for CSV writing (all fields are strings/scalars).
No explanation text is generated here — that is Layer 6 (LLM).
"""

from __future__ import annotations

from forecasting.cash_flow import CashFlowProjector
from forecasting.safety_check import SafetyChecker
from decision.spending_advisor import SpendingAdvisor
from decision.payment_selector import PaymentSelector

from data.models import (
    CandidatePlan,
    DecisionResult,
    FinancialState,
    PaymentOption,
    Request,
    SafetyResult,
)

__all__ = ["DecisionEngine"]


class DecisionEngine:
    """
    Deterministic decision engine (no LLM, no OCR).

    Instantiate once and call decide() for each request.
    """

    def __init__(self) -> None:
        self._projector = CashFlowProjector()
        self._checker = SafetyChecker(self._projector)
        self._advisor = SpendingAdvisor(self._checker)
        self._selector = PaymentSelector(self._checker, self._advisor)

    def decide(
        self,
        request: Request,
        state: FinancialState,
        safety_result: SafetyResult,
        options: list[PaymentOption],
    ) -> DecisionResult:
        """
        Produce a DecisionResult for one request.

        Parameters
        ----------
        request:
            The purchase request being evaluated.
        state:
            FinancialState produced by EventProcessor for this user/request.
        safety_result:
            SafetyResult produced by SafetyChecker for this request.
        options:
            All PaymentOption rows for this request_id.
        """
        best: CandidatePlan = self._selector.select(
            request, state, safety_result, options
        )

        # Format spending_changes_needed
        if best.spending_changes:
            changes_str = "|".join(c.format() for c in best.spending_changes)
        else:
            changes_str = "none"

        # Format earliest_date_for_full_payment
        earliest = safety_result.earliest_full_payment_date
        earliest_str = earliest.isoformat() if earliest else ""

        return DecisionResult(
            request_id=request.request_id,
            amount_safe_to_pay=safety_result.amount_safe_to_pay,
            affordability_status=best.affordability_status,
            recommended_payment_method=best.payment_plan.method,
            payment_plan=best.payment_plan.format_string(),
            earliest_date_for_full_payment=earliest_str,
            spending_changes_needed=changes_str,
            decision_explanation="",   # filled by Layer 6 (LLM)
        )
