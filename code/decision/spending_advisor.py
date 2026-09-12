"""
decision/spending_advisor.py — Greedy minimal spending-change finder.

Given a FinancialState whose base projection cannot safely support the full
requested payment today, this module searches for the smallest set of spending
changes (stop / reduce_to) that, when applied, raise amount_safe_to_pay to
at least the requested amount.

Algorithm
---------
1.  Scan the user's recurring debit patterns.
2.  For each pattern whose category is in the user's willingness lists and
    whose flexibility allows the change, compute the *impact* = total cash freed
    over the 90-day window if the change were applied.
3.  Sort candidate changes by impact descending (biggest wins first).
4.  Greedily accumulate changes, calling compute_safety_with_spending_changes()
    after each addition, until:
      (a) the full requested amount is safe, OR
      (b) we have reached MAX_SPENDING_CHANGES (from config), OR
      (c) there are no more candidate changes.
5.  A change is kept only if it actually *improves* amount_safe_to_pay vs. the
    previous set (no no-op inclusions).
6.  Return the accumulated list (may be empty if nothing helps).

Event matching (pattern → actual event IDs)
------------------------------------------
Projected events are linked via linked_event_id == pattern.reference_event_id.
Cashflow events (scheduled real events from the CSV) are matched heuristically
by (category, direction, amount within +-25% of pattern avg_amount, status==scheduled).
"""

from __future__ import annotations

import config
from data.models import (
    FinancialEvent,
    FinancialState,
    Request,
    RecurringPattern,
    SpendingChange,
)
from forecasting.safety_check import SafetyChecker


# ---------------------------------------------------------------------------
# Internal event-matching helpers
# ---------------------------------------------------------------------------

def _projected_ids_for_pattern(
    pattern: RecurringPattern,
    projected_events: list[FinancialEvent],
) -> set[str]:
    """Return event_ids of synthetic projected events belonging to *pattern*."""
    return {
        ev.event_id
        for ev in projected_events
        if ev.linked_event_id == pattern.reference_event_id
    }


def _cashflow_ids_for_pattern(
    pattern: RecurringPattern,
    cashflow_events: list[FinancialEvent],
) -> set[str]:
    """
    Return event_ids of real scheduled events that look like occurrences of
    *pattern*.  Matched by category + direction + amount proximity + status.
    """
    result = set()
    avg = max(pattern.avg_amount, 1.0)
    for ev in cashflow_events:
        if (
            ev.category == pattern.category
            and ev.direction == pattern.direction
            and ev.status == "scheduled"
            and ev.amount is not None
            and abs(ev.amount - pattern.avg_amount) / avg <= 0.25
        ):
            result.add(ev.event_id)
    return result


def _all_ids_for_pattern(
    pattern: RecurringPattern,
    state: FinancialState,
) -> set[str]:
    """Union of projected + cashflow event IDs for a recurring pattern."""
    return (
        _projected_ids_for_pattern(pattern, state.projected_events)
        | _cashflow_ids_for_pattern(pattern, state.cashflow_events)
    )


def _estimate_impact(
    pattern: RecurringPattern,
    saving_per_occurrence: float,
) -> float:
    """
    Approximate cash freed over FORECAST_DAYS by applying a change to this pattern.
    """
    if pattern.interval_days <= 0:
        return 0.0
    occurrences = config.FORECAST_DAYS / pattern.interval_days
    return saving_per_occurrence * occurrences


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class SpendingAdvisor:
    """
    Finds the minimal spending-change set that makes paying *requested_amount*
    on *request_date* safe.
    """

    def __init__(self, safety_checker: SafetyChecker) -> None:
        self._checker = safety_checker

    def find_changes(
        self,
        state: FinancialState,
        request: Request,
    ) -> list[SpendingChange]:
        """
        Return the minimal list of SpendingChange objects (up to
        config.MAX_SPENDING_CHANGES) that make paying request.requested_amount
        on request.request_date safe.

        Returns an empty list if no beneficial change can be found.
        """
        requested = request.requested_amount
        request_date = request.request_date

        candidates = self._build_candidates(state)
        if not candidates:
            return []

        candidates.sort(key=lambda t: t[0], reverse=True)

        accumulated_exclude: set[str] = set()
        accumulated_overrides: dict[str, float] = {}
        selected_changes: list[SpendingChange] = []
        current_safe: float = -1.0  # sentinel so first change always accepted

        for _, change, pattern in candidates:
            if len(selected_changes) >= config.MAX_SPENDING_CHANGES:
                break

            pattern_ids = _all_ids_for_pattern(pattern, state)

            if change.change_type == "stop":
                test_exclude = accumulated_exclude | pattern_ids
                test_overrides = dict(accumulated_overrides)
            else:  # reduce_to
                test_exclude = set(accumulated_exclude)
                test_overrides = {
                    **accumulated_overrides,
                    **{eid: change.new_amount for eid in pattern_ids},
                }

            result = self._checker.compute_safety_with_spending_changes(
                state=state,
                request_date=request_date,
                requested_amount=requested,
                modified_events_filter=test_exclude if test_exclude else None,
                amount_overrides=test_overrides if test_overrides else None,
            )

            new_safe = result.amount_safe_to_pay

            # Keep only changes that strictly improve safety
            if new_safe <= current_safe and len(selected_changes) > 0:
                continue

            selected_changes.append(change)
            accumulated_exclude = test_exclude
            accumulated_overrides = test_overrides
            current_safe = new_safe

            if new_safe >= requested:
                break

        return selected_changes

    def _build_candidates(
        self,
        state: FinancialState,
    ) -> list[tuple[float, SpendingChange, RecurringPattern]]:
        """Return (impact, SpendingChange, RecurringPattern) for all eligible changes."""
        profile = state.profile
        stop_categories = set(profile.expense_categories_willing_to_stop)
        reduce_categories = set(profile.expense_categories_willing_to_reduce)
        candidates: list[tuple[float, SpendingChange, RecurringPattern]] = []

        for pattern in state.recurring_patterns:
            if pattern.direction != "debit":
                continue

            if (
                pattern.flexibility in ("stoppable", "reducible")
                and pattern.category in stop_categories
            ):
                impact = _estimate_impact(pattern, pattern.avg_amount)
                change = SpendingChange(
                    event_id=pattern.reference_event_id,
                    change_type="stop",
                    new_amount=None,
                )
                candidates.append((impact, change, pattern))

            elif (
                pattern.flexibility == "reducible"
                and pattern.category in reduce_categories
                and pattern.minimum_allowed_amount is not None
                and pattern.minimum_allowed_amount < pattern.avg_amount
            ):
                saving = pattern.avg_amount - pattern.minimum_allowed_amount
                impact = _estimate_impact(pattern, saving)
                change = SpendingChange(
                    event_id=pattern.reference_event_id,
                    change_type="reduce_to",
                    new_amount=round(pattern.minimum_allowed_amount, 2),
                )
                candidates.append((impact, change, pattern))

        return candidates
