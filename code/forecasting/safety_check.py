"""
forecasting/safety_check.py — Safety analysis on top of a cash-flow projection.

Answers two questions:
    1.  amount_safe_to_pay   — the maximum single payment on request_date that
                               keeps every daily closing balance ≥ minimum.
    2.  earliest_full_payment_date — the first calendar day within the 90-day
                               window where the full requested_amount can be
                               paid safely as a single payment.

Core mathematical insight
-------------------------
Paying amount X on day d removes X from the bank account permanently.
Every closing balance from day d onwards decreases by exactly X.
Days before d are unaffected.

Therefore:

    is_safe_on_day_d(X) = min(closing[d], closing[d+1], ..., closing[N]) − X ≥ minimum
                        = suffix_min[d] − minimum ≥ X

Where suffix_min[d] = min(closing_balance[d..N]).

This means:
    max_payable_on_day_d = suffix_min[d] − minimum_balance

And for any payment schedule {(d1, P1), (d2, P2), ...}:
    We accumulate payments left-to-right as we scan the projection.
    At each day d, the effective closing balance = base_closing[d] − Σ(payments ≤ d).
    Safe iff this stays ≥ minimum for all d.

Binary search (amount_safe_to_pay)
-----------------------------------
Input range: [0, requested_amount]
Predicate:   is_safe_to_pay_on_day_0(X)  ← monotone decreasing in X
Stop when:   hi − lo < precision (0.01 by default)
Result:      round(lo, 2)

The binary search generalises cleanly to payment plans — the predicate just
changes to is_payment_schedule_safe().  The decision layer uses that variant.

Suffix-minimum scan (earliest_full_payment_date)
-------------------------------------------------
Precompute suffix_min[i] = min(closing[i..N]) in O(n).
Walk i from 0 to N.  The first i where suffix_min[i] − minimum ≥ requested_amount
is the answer.  O(n) total.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Optional

import config
from data.models import DailyBalance, FinancialState, SafetyResult
from forecasting.cash_flow import CashFlowProjector


class SafetyChecker:
    """
    Runs all safety analyses for a single (user, request) pair.

    Usage pattern:
        projector = CashFlowProjector()
        checker   = SafetyChecker(projector)

        projection = projector.build_projection(state, request_date)
        result     = checker.compute_safety(
                         projection, state, request_date, requested_amount
                     )
    """

    def __init__(self, projector: CashFlowProjector) -> None:
        self._proj = projector

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def compute_safety(
        self,
        projection: list[DailyBalance],
        state: FinancialState,
        request_date: date,
        requested_amount: float,
        precision: float = config.AMOUNT_SEARCH_PRECISION,
    ) -> SafetyResult:
        """
        Compute the full SafetyResult for one request.

        Parameters
        ----------
        projection:
            Base 90-day projection (no additional payment applied).
        state:
            The FinancialState this projection was built from.
        request_date:
            Day 0 of the forecast.
        requested_amount:
            The amount the user is asking to pay.
        precision:
            Binary-search stopping criterion (default 0.01).
        """
        minimum = state.profile.minimum_balance_to_keep

        # 1. Binary-search for maximum safe payment on day 0
        amount_safe = self._binary_search_safe_amount(
            projection, minimum, requested_amount, precision
        )

        # 2. Find earliest day where full amount is safe as one payment
        earliest = self._find_earliest_full_payment_date(
            projection, minimum, requested_amount
        )

        # 3. Minimum balance across the base projection (diagnostic)
        min_closing = (
            min(db.closing_balance for db in projection) if projection else 0.0
        )

        return SafetyResult(
            amount_safe_to_pay=amount_safe,
            earliest_full_payment_date=earliest,
            min_closing_balance=round(min_closing, 2),
            base_projection=projection,
        )

    # ------------------------------------------------------------------
    # General payment schedule validator (used by the decision layer)
    # ------------------------------------------------------------------

    def is_payment_schedule_safe(
        self,
        projection: list[DailyBalance],
        minimum_balance: float,
        payments: list[tuple[date, float]],
    ) -> bool:
        """
        Return True iff applying *payments* on top of *projection* keeps
        every closing balance ≥ *minimum_balance*.

        Parameters
        ----------
        payments:
            List of (payment_date, amount).  Multiple payments on the same
            date are summed.  Payments outside the projection window are
            treated as no-ops (they don't affect the 90-day window).
        """
        if not payments:
            return True

        # Accumulate payment amounts by date
        pay_by_date: dict[date, float] = defaultdict(float)
        for pdate, pamount in payments:
            pay_by_date[pdate] += pamount

        cumulative = 0.0
        for db in projection:
            cumulative += pay_by_date.get(db.day, 0.0)
            if db.closing_balance - cumulative < minimum_balance:
                return False
        return True

    def compute_safety_with_spending_changes(
        self,
        state: FinancialState,
        request_date: date,
        requested_amount: float,
        modified_events_filter: set[str] | None = None,
        amount_overrides: dict[str, float] | None = None,
        precision: float = config.AMOUNT_SEARCH_PRECISION,
    ) -> SafetyResult:
        """
        Rebuild the projection after removing or reducing flexible expense events,
        then run a full safety check on that modified projection.

        Parameters
        ----------
        modified_events_filter:
            Set of event_ids to exclude from the projection (spending stopped).
        amount_overrides:
            {event_id: new_amount} for events that are reduced (not stopped).
        """
        from data.models import FinancialState
        import copy

        # Build modified cashflow events
        def _filtered(events):
            result = []
            for ev in events:
                if modified_events_filter and ev.event_id in modified_events_filter:
                    continue
                if amount_overrides and ev.event_id in amount_overrides:
                    ev2 = copy.copy(ev)
                    ev2.amount = amount_overrides[ev.event_id]
                    result.append(ev2)
                else:
                    result.append(ev)
            return result

        modified_state = FinancialState(
            profile=state.profile,
            cashflow_events=_filtered(state.cashflow_events),
            projected_events=_filtered(state.projected_events),
            recurring_patterns=state.recurring_patterns,
            ocr_filled_event_ids=state.ocr_filled_event_ids,
            amendment_applied_ids=state.amendment_applied_ids,
        )

        modified_projection = self._proj.build_projection(modified_state, request_date)
        return self.compute_safety(
            modified_projection, modified_state, request_date, requested_amount, precision
        )

    # ------------------------------------------------------------------
    # Private algorithms
    # ------------------------------------------------------------------

    def _binary_search_safe_amount(
        self,
        projection: list[DailyBalance],
        minimum_balance: float,
        requested_amount: float,
        precision: float,
    ) -> float:
        """
        Binary search for the maximum X in [0, requested_amount] such that
        paying X on day 0 keeps all balances ≥ minimum_balance.

        The search predicate is:
            all(db.closing_balance − X ≥ minimum for db in projection)
        which is equivalent to:
            X ≤ min_closing_balance − minimum

        Binary search is used (rather than the direct analytical formula) so
        the same function generalises to non-trivial payment schedules when
        called from the decision layer.
        """
        if not projection:
            return 0.0

        lo: float = 0.0
        hi: float = requested_amount

        # Fast exit: full amount is already safe
        if self._is_safe_single_payment(projection, minimum_balance, hi):
            return round(hi, 2)

        # Fast exit: balance already at/below minimum (nothing is safe)
        if not self._is_safe_single_payment(projection, minimum_balance, 0.0):
            return 0.0

        # Standard binary search
        while hi - lo > precision:
            mid = (lo + hi) / 2.0
            if self._is_safe_single_payment(projection, minimum_balance, mid):
                lo = mid
            else:
                hi = mid

        return round(lo, 2)

    def _find_earliest_full_payment_date(
        self,
        projection: list[DailyBalance],
        minimum_balance: float,
        requested_amount: float,
    ) -> Optional[date]:
        """
        Return the first date d in the projection where paying requested_amount
        as a single payment on d is safe.

        Uses a suffix-minimum scan:
            suffix_min[i] = min(closing_balance[i], closing_balance[i+1], ..., closing_balance[N-1])

        Paying X on day i is safe iff suffix_min[i] − minimum ≥ X.

        Time: O(n)  (two passes — one to build suffix_min, one to scan).
        """
        n = len(projection)
        if n == 0:
            return None

        # Build suffix minimums
        suffix_min: list[float] = [0.0] * n
        suffix_min[n - 1] = projection[n - 1].closing_balance
        for i in range(n - 2, -1, -1):
            suffix_min[i] = min(projection[i].closing_balance, suffix_min[i + 1])

        # Find first day where max_payable >= requested_amount
        for i, db in enumerate(projection):
            max_payable_on_day = suffix_min[i] - minimum_balance
            if max_payable_on_day >= requested_amount:
                return db.day

        return None  # Never safe within the forecast window

    @staticmethod
    def _is_safe_single_payment(
        projection: list[DailyBalance],
        minimum_balance: float,
        amount: float,
    ) -> bool:
        """
        Return True iff paying *amount* on day 0 keeps every closing balance
        ≥ minimum_balance.

        Since a day-0 payment permanently reduces all subsequent balances,
        this is equivalent to checking whether the minimum closing balance
        over the full projection minus *amount* stays above minimum_balance.
        """
        return all(db.closing_balance - amount >= minimum_balance for db in projection)
