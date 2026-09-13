"""
decision/payment_selector.py — Master decision engine: generates, validates,
ranks, and selects the best payment plan for one (request, user) pair.

Decision flow
=============

Step 1 — Enumerate all candidate plans WITHOUT spending changes
--------------------------------------------------------------
  A. full_payment CSV options   (one candidate per option with method='full_payment')
  B. installment CSV options    (one candidate per option with method='installments',
                                 within max_installment_months limit)
  C. synthetic wait             (if earliest_full_payment_date <= deadline and
                                 user accepts full_payment)
  D. synthetic partial_payment  (if request allows it, user accepts it, and
                                 0 < amount_safe_to_pay < requested_amount)

Step 2 — Try spending changes if no full-coverage-by-deadline candidate found
----------------------------------------------------------------------------
  Use SpendingAdvisor to find a minimal change set that makes paying the full
  amount safe.  Then re-evaluate all option types on the modified projection.

Step 3 — Rank all candidates with a 6-key tuple
------------------------------------------------
  rank_key = (
      not completes_by_deadline,   # False < True -> prefer completing on time
      len(spending_changes) > 0,   # False < True -> prefer no changes
      total_payable,               # ascending -> prefer cheaper
      first_date,                  # ascending -> prefer paying earlier
      num_payments,                # ascending -> prefer fewer payments
      option_id_rank,              # ascending -> prefer lower option_id
  )

Step 4 — Select best; derive affordability_status
-------------------------------------------------
  affordable_now        : full_payment, no spending changes, first_date == request_date
  affordable_later      : wait plan
  affordable_with_plan  : everything else that covers the full amount
  not_affordable        : not_recommended fallback
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import config
from data.models import (
    CandidatePlan,
    DailyBalance,
    FinancialState,
    PaymentOption,
    PaymentPlan,
    Request,
    SafetyResult,
    SpendingChange,
)
from decision.plan_builder import PlanBuilder, _option_rank
from decision.spending_advisor import SpendingAdvisor, _all_ids_for_pattern
from forecasting.safety_check import SafetyChecker


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _option_payment_schedule(option: PaymentOption) -> list[tuple[date, float]]:
    """(date, per-payment amount) list for all payments in this option."""
    return [(d, option.payment_amount) for d in option.payment_dates()]


def _last_date(plan: PaymentPlan) -> date:
    """Return the last settlement date in a PaymentPlan."""
    return max(d for d, _ in plan.payments) if plan.payments else date.max


def _is_within_month_limit(option: PaymentOption, max_months: Optional[int]) -> bool:
    """True iff the plan fits within the user's installment month limit."""
    if max_months is None:
        return True
    return option.number_of_payments <= max_months


def _derive_status(
    method: str,
    spending_changes: list[SpendingChange],
    first_date: Optional[date],
    request_date: date,
) -> str:
    if method == "not_recommended":
        return "not_affordable"
    if method == "wait":
        return "affordable_later"
    if (
        method == "full_payment"
        and not spending_changes
        and first_date == request_date
    ):
        return "affordable_now"
    return "affordable_with_plan"


def _make_candidate(
    plan: PaymentPlan,
    rank: int,
    spending_changes: list[SpendingChange],
    total_payable: float,
    completes_by_deadline: bool,
    request_date: date,
) -> CandidatePlan:
    first = plan.first_date()
    status = _derive_status(plan.method, spending_changes, first, request_date)
    return CandidatePlan(
        payment_plan=plan,
        affordability_status=status,
        spending_changes=spending_changes,
        completes_by_deadline=completes_by_deadline,
        total_payable=total_payable,
        option_id_rank=rank,
    )


def _apply_changes_to_ids(
    changes: list[SpendingChange],
    state: FinancialState,
) -> tuple[set[str], dict[str, float]]:
    """Translate SpendingChange list → (exclude_ids, amount_overrides) dicts."""
    pattern_by_ref = {p.reference_event_id: p for p in state.recurring_patterns}
    exclude: set[str] = set()
    overrides: dict[str, float] = {}
    for change in changes:
        pattern = pattern_by_ref.get(change.event_id)
        if pattern is None:
            continue
        ids = _all_ids_for_pattern(pattern, state)
        if change.change_type == "stop":
            exclude |= ids
        elif change.change_type == "reduce_to" and change.new_amount is not None:
            for eid in ids:
                overrides[eid] = change.new_amount
    return exclude, overrides


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class PaymentSelector:
    """
    Selects the best CandidatePlan for one (request, state, options) tuple.
    """

    def __init__(
        self,
        safety_checker: SafetyChecker,
        spending_advisor: SpendingAdvisor,
    ) -> None:
        self._checker = safety_checker
        self._advisor = spending_advisor

    def select(
        self,
        request: Request,
        state: FinancialState,
        safety_result: SafetyResult,
        options: list[PaymentOption],
    ) -> CandidatePlan:
        """Generate all candidates, rank them, return the best one."""
        profile = state.profile
        projection = safety_result.base_projection
        minimum = profile.minimum_balance_to_keep
        accepted = set(profile.payment_methods_user_will_consider)
        deadline = request.desired_completion_date
        req_date = request.request_date
        requested = request.requested_amount

        candidates: list[CandidatePlan] = []

        # ---- A. Full-payment CSV options (no spending changes) ---------------
        if "full_payment" in accepted:
            for opt in options:
                if opt.payment_method != "full_payment":
                    continue
                schedule = _option_payment_schedule(opt)
                if self._checker.is_payment_schedule_safe(projection, minimum, schedule):
                    plan, rank = PlanBuilder.from_full_payment_option(opt)
                    candidates.append(_make_candidate(
                        plan=plan, rank=rank, spending_changes=[],
                        total_payable=opt.total_payable_amount,
                        completes_by_deadline=opt.first_payment_date <= deadline,
                        request_date=req_date,
                    ))

        # ---- B. Installment CSV options (no spending changes) ----------------
        if "installments" in accepted:
            for opt in options:
                if opt.payment_method != "installments":
                    continue
                if not _is_within_month_limit(opt, profile.max_installment_months):
                    continue
                schedule = _option_payment_schedule(opt)
                if self._checker.is_payment_schedule_safe(projection, minimum, schedule):
                    plan, rank = PlanBuilder.from_installment_option(opt)
                    last = _last_date(plan)
                    candidates.append(_make_candidate(
                        plan=plan, rank=rank, spending_changes=[],
                        total_payable=opt.total_payable_amount,
                        completes_by_deadline=last <= deadline,
                        request_date=req_date,
                    ))

        # ---- C. Synthetic wait (no spending changes) -------------------------
        earliest = safety_result.earliest_full_payment_date
        if "full_payment" in accepted and earliest is not None:
            if earliest <= deadline:
                plan, rank = PlanBuilder.wait(requested, earliest)
                candidates.append(_make_candidate(
                    plan=plan, rank=rank, spending_changes=[],
                    total_payable=requested,
                    completes_by_deadline=True,
                    request_date=req_date,
                ))

        # ---- D. Synthetic partial payment (no spending changes) --------------
        safe_amt = safety_result.amount_safe_to_pay
        earliest = safety_result.earliest_full_payment_date
        if (
            "partial_payment" in accepted
            and request.allows_partial_payment
            and 0 < safe_amt < requested
            and earliest is not None          # need earliest for the 2nd payment
        ):
            plan, rank = PlanBuilder.partial_payment(
                safe_amount=safe_amt,
                requested_amount=requested,
                on_date=req_date,
                earliest_date=earliest,
            )
            candidates.append(_make_candidate(
                plan=plan, rank=rank, spending_changes=[],
                total_payable=requested,
                completes_by_deadline=earliest <= deadline,
                request_date=req_date,
            ))

        # ---- Step 2: try spending changes if still no full-by-deadline -------
        full_by_deadline = [
            c for c in candidates
            if c.completes_by_deadline
            and c.payment_plan.method not in ("partial_payment", "not_recommended")
        ]
        if not full_by_deadline:
            candidates.extend(
                self._candidates_with_spending_changes(
                    request, state, options, safety_result,
                    deadline, minimum, accepted,
                )
            )

        # ---- Step 3: rank all candidates -------------------------------------
        candidates.sort(key=lambda c: c.rank_key())

        if candidates:
            return candidates[0]

        # ---- Fallback: not_recommended ---------------------------------------
        plan, rank = PlanBuilder.not_recommended()
        return _make_candidate(
            plan=plan, rank=rank, spending_changes=[],
            total_payable=0.0, completes_by_deadline=False,
            request_date=req_date,
        )

    # ------------------------------------------------------------------
    # Spending-change branch
    # ------------------------------------------------------------------

    def _candidates_with_spending_changes(
        self,
        request: Request,
        state: FinancialState,
        options: list[PaymentOption],
        safety_result: SafetyResult,
        deadline: date,
        minimum: float,
        accepted: set[str],
    ) -> list[CandidatePlan]:
        req_date = request.request_date
        requested = request.requested_amount

        changes = self._advisor.find_changes(state, request)
        if not changes:
            return []

        # Build modified projection with spending changes
        exclude, overrides = _apply_changes_to_ids(changes, state)
        mod_result = self._checker.compute_safety_with_spending_changes(
            state=state,
            request_date=req_date,
            requested_amount=requested,
            modified_events_filter=exclude if exclude else None,
            amount_overrides=overrides if overrides else None,
        )
        mod_proj = mod_result.base_projection
        candidates: list[CandidatePlan] = []

        # E. Full-payment CSV WITH changes
        if "full_payment" in accepted:
            for opt in options:
                if opt.payment_method != "full_payment":
                    continue
                schedule = _option_payment_schedule(opt)
                if self._checker.is_payment_schedule_safe(mod_proj, minimum, schedule):
                    plan, rank = PlanBuilder.from_full_payment_option(opt)
                    candidates.append(_make_candidate(
                        plan=plan, rank=rank, spending_changes=changes,
                        total_payable=opt.total_payable_amount,
                        completes_by_deadline=opt.first_payment_date <= deadline,
                        request_date=req_date,
                    ))

        # F. Installments WITH changes
        if "installments" in accepted:
            for opt in options:
                if opt.payment_method != "installments":
                    continue
                if not _is_within_month_limit(opt, state.profile.max_installment_months):
                    continue
                schedule = _option_payment_schedule(opt)
                if self._checker.is_payment_schedule_safe(mod_proj, minimum, schedule):
                    plan, rank = PlanBuilder.from_installment_option(opt)
                    last = _last_date(plan)
                    candidates.append(_make_candidate(
                        plan=plan, rank=rank, spending_changes=changes,
                        total_payable=opt.total_payable_amount,
                        completes_by_deadline=last <= deadline,
                        request_date=req_date,
                    ))

        # G. Synthetic full payment today WITH changes
        if "full_payment" in accepted and mod_result.amount_safe_to_pay >= requested:
            plan, rank = PlanBuilder.synthetic_full_payment(requested, req_date)
            candidates.append(_make_candidate(
                plan=plan, rank=rank, spending_changes=changes,
                total_payable=requested,
                completes_by_deadline=req_date <= deadline,
                request_date=req_date,
            ))

        # H. Wait on modified projection WITH changes
        earliest_mod = mod_result.earliest_full_payment_date
        if "full_payment" in accepted and earliest_mod is not None:
            if earliest_mod <= deadline:
                plan, rank = PlanBuilder.wait(requested, earliest_mod)
                candidates.append(_make_candidate(
                    plan=plan, rank=rank, spending_changes=changes,
                    total_payable=requested,
                    completes_by_deadline=True,
                    request_date=req_date,
                ))

        return candidates
