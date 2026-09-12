 """
ingestion/event_processor.py — Build a FinancialState for one (user, request).

Pipeline (no LLM, no OCR):
    1. Apply OCR fills to events with blank amounts (may be empty at this layer)
    2. Apply deterministic amendments from MessageProcessor
    3. Convert all event amounts to the user's home_currency
    4. Filter events for cash-flow relevance (drop failed / cancelled / unrealized)
    5. Detect recurring patterns from settled historical events
    6. Project recurring events forward over the 90-day forecast horizon
    7. Return FinancialState

Key rules:
    - Debit statuses included:  settled, scheduled, pending
    - Credit statuses included: settled, scheduled  (NOT pending — unconfirmed income)
    - Excluded entirely:        failed, cancelled, unrealized
    - Only events with settlement_date >= request_date enter the 90-day forecast;
      history is used only for recurring-pattern detection.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from datetime import date, timedelta
from statistics import median
from typing import Optional

import config
from data.exchange import ExchangeRateTable
from data.models import (
    Amendment,
    FinancialEvent,
    FinancialState,
    RecurringPattern,
    UserProfile,
)

# ---------------------------------------------------------------------------
# Category aliases for salary / income matching
# ---------------------------------------------------------------------------

_SALARY_CATEGORIES = frozenset({"salary", "income", "payroll", "wages", "gaji"})
_RENT_CATEGORIES = frozenset({"rent", "housing", "sewa"})

# Statuses included in the cashflow projection, split by direction
_DEBIT_OK = config.CASHFLOW_DEBIT_STATUSES
_CREDIT_OK = config.CASHFLOW_CREDIT_STATUSES
_EXCLUDED = config.EXCLUDED_STATUSES


# ---------------------------------------------------------------------------
# Internal helpers (module-level, private)
# ---------------------------------------------------------------------------


def _copy_event(ev: FinancialEvent) -> FinancialEvent:
    return copy.copy(ev)


def _apply_ocr_fills(
    events: list[FinancialEvent],
    ocr_fills: dict[str, float],
) -> list[FinancialEvent]:
    """
    Fill blank amounts on events whose amounts were extracted by OCR.
    Returns a new list; originals are not mutated.
    """
    if not ocr_fills:
        return events
    result = []
    for ev in events:
        if ev.amount is None and ev.event_id in ocr_fills:
            ev2 = _copy_event(ev)
            ev2.amount = ocr_fills[ev.event_id]
            result.append(ev2)
        else:
            result.append(ev)
    return result


def _apply_amendments(
    events: list[FinancialEvent],
    amendments: list[Amendment],
    request_date: date,
) -> tuple[list[FinancialEvent], set[str]]:
    """
    Apply all amendments to the event list.

    Returns
    -------
    (modified_events, amendment_applied_ids)
    """
    # Work on a shallow-copy map so we can mutate by event_id
    event_map: dict[str, FinancialEvent] = {e.event_id: _copy_event(e) for e in events}
    applied: set[str] = set()

    for amend in amendments:
        atype = amend.amendment_type

        # ---- 1. Cancel specific event (internal_transfer, refund_pending) --------
        if atype in ("internal_transfer", "refund_pending") and amend.target_event_id:
            _cancel_event_and_linked(event_map, amend.target_event_id, applied)

        # ---- 2. Investment mark-to-market: not a real cash event ----------------
        elif atype == "investment_value_only" and amend.target_event_id:
            _cancel_event_and_linked(event_map, amend.target_event_id, applied)

        # ---- 3. Prize processing (not yet credited): cancel pending credit ------
        elif atype == "prize_processing" and amend.target_event_id:
            ev = event_map.get(amend.target_event_id)
            if ev and ev.status == "pending":
                event_map[ev.event_id] = _with_status(ev, "cancelled")
                applied.add(ev.event_id)

        # ---- 4. Reimbursement closed: mark the credit as one-off (no action) ---
        elif atype == "reimbursement_closed" and amend.target_event_id:
            ev = event_map.get(amend.target_event_id)
            if ev:
                applied.add(ev.event_id)  # already settled; just flag it

        # ---- 5. Salary change / temporary reduction / confirmed -----------------
        elif atype in ("salary_change", "salary_confirmed"):
            if amend.new_amount is not None:
                effective = amend.effective_date or request_date
                _update_future_salary_events(
                    event_map, amend.new_amount, effective, applied
                )

        # ---- 6. Salary resume (same as salary_change with a specific date) ------
        elif atype == "salary_resume":
            if amend.new_amount is not None and amend.effective_date:
                _update_future_salary_events(
                    event_map, amend.new_amount, amend.effective_date, applied
                )

        # ---- 7. First salary from new employer ----------------------------------
        elif atype == "first_salary":
            if amend.new_amount is not None and amend.effective_date:
                _update_future_salary_events(
                    event_map, amend.new_amount, amend.effective_date, applied
                )

        # ---- 8. Partial income ended: use remaining confirmed amount -------------
        elif atype == "partial_income_ended":
            if amend.new_amount is not None:
                _update_future_salary_events(
                    event_map, amend.new_amount, request_date, applied
                )

        # ---- 9. Salary date change ----------------------------------------------
        elif atype == "salary_date_change" and amend.effective_date:
            _move_next_salary_date(event_map, amend.effective_date, request_date, applied)

        # ---- 10. Bonus / commission pending: cancel specific event or category --
        elif atype in ("bonus_pending", "commission_pending"):
            if amend.target_event_id:
                _cancel_event_and_linked(event_map, amend.target_event_id, applied)
            else:
                # Cancel all future pending bonus / commission credits
                for ev in list(event_map.values()):
                    if (
                        ev.direction == "credit"
                        and ev.category in ("bonus", "commission", "bonus_quarterly")
                        and ev.status == "pending"
                        and ev.settlement_date >= request_date
                    ):
                        event_map[ev.event_id] = _with_status(ev, "cancelled")
                        applied.add(ev.event_id)

        # ---- 11. Failed debit retry: ensure the debit stays in the forecast -----
        elif atype == "failed_debit_retry" and amend.target_event_id:
            ev = event_map.get(amend.target_event_id)
            if ev and ev.status == "failed":
                event_map[ev.event_id] = _with_status(ev, "scheduled")
                applied.add(ev.event_id)

        # ---- 12. No income → handled at projection stage (flag carried separately)
        # (no_income, no_income is passed forward; projection layer checks amendments)

    return list(event_map.values()), applied


def _cancel_event_and_linked(
    event_map: dict[str, FinancialEvent],
    event_id: str,
    applied: set[str],
) -> None:
    ev = event_map.get(event_id)
    if ev is None:
        return
    event_map[event_id] = _with_status(ev, "cancelled")
    applied.add(event_id)
    if ev.linked_event_id and ev.linked_event_id in event_map:
        linked = event_map[ev.linked_event_id]
        event_map[ev.linked_event_id] = _with_status(linked, "cancelled")
        applied.add(ev.linked_event_id)


def _update_future_salary_events(
    event_map: dict[str, FinancialEvent],
    new_amount: float,
    effective_date: date,
    applied: set[str],
) -> None:
    """Update the amount on all future scheduled salary credit events >= effective_date."""
    for ev in list(event_map.values()):
        if (
            ev.direction == "credit"
            and (ev.category.lower() in _SALARY_CATEGORIES or ev.event_type.lower() == "salary")
            and ev.settlement_date >= effective_date
            and ev.status in ("scheduled", "pending")
        ):
            updated = _copy_event(ev)
            updated.amount = new_amount
            event_map[ev.event_id] = updated
            applied.add(ev.event_id)


def _move_next_salary_date(
    event_map: dict[str, FinancialEvent],
    new_date: date,
    request_date: date,
    applied: set[str],
) -> None:
    """Move the next upcoming salary credit to new_date."""
    salary_events = sorted(
        [
            ev for ev in event_map.values()
            if ev.direction == "credit"
            and (ev.category.lower() in _SALARY_CATEGORIES or ev.event_type.lower() == "salary")
            and ev.settlement_date >= request_date
            and ev.status in ("scheduled", "pending")
        ],
        key=lambda e: e.settlement_date,
    )
    if salary_events:
        ev = salary_events[0]
        updated = _copy_event(ev)
        updated.settlement_date = new_date
        updated.event_date = new_date
        event_map[ev.event_id] = updated
        applied.add(ev.event_id)


def _with_status(ev: FinancialEvent, status: str) -> FinancialEvent:
    updated = _copy_event(ev)
    updated.status = status
    return updated


def _convert_currencies(
    events: list[FinancialEvent],
    home_currency: str,
    exchange_table: ExchangeRateTable,
) -> list[FinancialEvent]:
    """
    Convert every event amount to home_currency using the rate closest to
    the event's settlement_date.  Events already in home_currency are unchanged.
    Events with None amount are left as-is.
    """
    result = []
    for ev in events:
        if ev.currency == home_currency or ev.amount is None:
            result.append(ev)
            continue
        try:
            converted = exchange_table.convert(
                ev.amount, ev.currency, home_currency, ev.settlement_date
            )
            ev2 = _copy_event(ev)
            ev2.amount = converted
            ev2.currency = home_currency
            result.append(ev2)
        except ValueError:
            # No rate available — keep original (will be visible in projection)
            result.append(ev)
    return result


def _filter_cashflow_events(
    events: list[FinancialEvent],
    request_date: date,
) -> list[FinancialEvent]:
    """
    Keep only future events that affect the 90-day forecast.

    Rules:
    - settlement_date >= request_date  (past events are already in the balance)
    - debits:  status in {settled, scheduled, pending}
    - credits: status in {settled, scheduled}  (NOT pending — unconfirmed income)
    - drop:    failed, cancelled, unrealized
    - drop:    events with None amount (can't forecast without a number)
    """
    cashflow: list[FinancialEvent] = []
    for ev in events:
        if ev.settlement_date < request_date:
            continue
        if ev.status in _EXCLUDED:
            continue
        if ev.amount is None:
            continue
        if ev.direction == "debit" and ev.status in _DEBIT_OK:
            cashflow.append(ev)
        elif ev.direction == "credit" and ev.status in _CREDIT_OK:
            cashflow.append(ev)
    return cashflow


# ---------------------------------------------------------------------------
# Recurring pattern detection
# ---------------------------------------------------------------------------

# Standard recurrence intervals to recognise (days)
_KNOWN_INTERVALS = [7, 14, 21, 28, 30, 31, 90, 91, 92]


def _nearest_known_interval(gap: float) -> Optional[int]:
    """Snap a raw gap to the nearest known interval within tolerance."""
    for ki in _KNOWN_INTERVALS:
        if abs(gap - ki) <= config.RECURRENCE_INTERVAL_TOLERANCE:
            return ki
    return None


def _detect_recurring_patterns(
    settled_events: list[FinancialEvent],
) -> list[RecurringPattern]:
    """
    Detect recurring income and expense patterns from settled historical events.

    Groups events by (user_id, category, direction) and requires:
    - At least MIN_RECURRENCE_COUNT events in the group
    - Consistent inter-event gaps (all within RECURRENCE_INTERVAL_TOLERANCE of median)

    Returns one RecurringPattern per detected pattern.
    """
    # Group by (category, direction) — user_id is implicit (single user per call)
    groups: dict[tuple[str, str], list[FinancialEvent]] = defaultdict(list)
    for ev in settled_events:
        if ev.amount is not None and ev.amount > 0:
            groups[(ev.category, ev.direction)].append(ev)

    patterns: list[RecurringPattern] = []

    for (category, direction), group in groups.items():
        if len(group) < config.MIN_RECURRENCE_COUNT:
            continue

        # Sort by settlement date
        group_sorted = sorted(group, key=lambda e: e.settlement_date)

        # Compute all inter-event gaps
        gaps = [
            (group_sorted[i].settlement_date - group_sorted[i - 1].settlement_date).days
            for i in range(1, len(group_sorted))
        ]

        # Try to find a dominant interval using ALL gaps
        med_gap = median(gaps)
        snapped = _nearest_known_interval(med_gap)

        if snapped is None:
            # Try recent events only (last MIN_RECURRENCE_COUNT)
            recent = group_sorted[-config.MIN_RECURRENCE_COUNT :]
            if len(recent) >= 2:
                recent_gaps = [
                    (recent[i].settlement_date - recent[i - 1].settlement_date).days
                    for i in range(1, len(recent))
                ]
                med_gap = median(recent_gaps)
                snapped = _nearest_known_interval(med_gap)
                group_sorted = recent

        if snapped is None:
            continue  # No recognisable interval

        # Verify consistency: all gaps within tolerance of the snapped interval
        consistent = all(
            abs(g - snapped) <= config.RECURRENCE_INTERVAL_TOLERANCE for g in gaps
        )
        if not consistent:
            # Relax: check last (MIN_RECURRENCE_COUNT - 1) gaps only
            recent_gaps = gaps[-(config.MIN_RECURRENCE_COUNT - 1) :]
            consistent = all(
                abs(g - snapped) <= config.RECURRENCE_INTERVAL_TOLERANCE
                for g in recent_gaps
            )
            if not consistent:
                continue

        # Average the most recent ≤6 amounts
        recent_events = group_sorted[-6:]
        amounts = [e.amount for e in recent_events if e.amount is not None]
        if not amounts:
            continue
        avg_amount = sum(amounts) / len(amounts)

        last_event = group_sorted[-1]

        patterns.append(
            RecurringPattern(
                user_id=last_event.user_id,
                category=category,
                direction=direction,
                avg_amount=round(avg_amount, 2),
                currency=last_event.currency,
                interval_days=snapped,
                last_settlement_date=last_event.settlement_date,
                flexibility=last_event.flexibility,
                reference_event_id=last_event.event_id,
                description=last_event.description,
                minimum_allowed_amount=last_event.minimum_allowed_amount,
            )
        )

    return patterns


def _apply_salary_amendments_to_patterns(
    patterns: list[RecurringPattern],
    amendments: list[Amendment],
    request_date: date,
) -> list[RecurringPattern]:
    """
    Update salary/income pattern amounts based on salary_change / salary_resume
    amendments so that projected recurring salary events use the correct amount.
    """
    result = []
    for pat in patterns:
        updated = copy.copy(pat)
        if pat.direction == "credit" and pat.category.lower() in _SALARY_CATEGORIES:
            # Apply the most recent salary amendment (by effective_date, or request_date)
            salary_amends = [
                a for a in amendments
                if a.amendment_type in ("salary_change", "salary_resume", "first_salary")
                and a.new_amount is not None
                and (a.effective_date is None or a.effective_date <= request_date)
            ]
            if salary_amends:
                # Use the amendment with the latest effective_date
                best = max(
                    salary_amends,
                    key=lambda a: a.effective_date or date.min,
                )
                updated.avg_amount = best.new_amount
        result.append(updated)
    return result


def _apply_rent_amendments_to_patterns(
    patterns: list[RecurringPattern],
    amendments: list[Amendment],
) -> list[RecurringPattern]:
    """Apply rent_increase percentage amendments to rent-category patterns."""
    rent_amends = [a for a in amendments if a.amendment_type == "rent_increase"]
    if not rent_amends:
        return patterns

    result = []
    for pat in patterns:
        updated = copy.copy(pat)
        if pat.direction == "debit" and pat.category.lower() in _RENT_CATEGORIES:
            for amend in rent_amends:
                # Extract percentage from notes field (e.g., "rent increased by 12%")
                import re
                m = re.search(r"(\d+(?:\.\d+)?)\s*%", amend.notes)
                if m:
                    pct = float(m.group(1))
                    updated.avg_amount = round(pat.avg_amount * (1 + pct / 100), 2)
                    break
        result.append(updated)
    return result


def _project_recurring(
    patterns: list[RecurringPattern],
    request_date: date,
    horizon_days: int,
    existing_events: list[FinancialEvent],
    suppress_income: bool,
) -> list[FinancialEvent]:
    """
    Generate synthetic future events from recurring patterns.

    Parameters
    ----------
    patterns:
        Detected recurring patterns (possibly amended).
    request_date:
        Start of the 90-day forecast window.
    horizon_days:
        Length of the forecast window (default: 90).
    existing_events:
        Already-known future events (to avoid double-counting).
    suppress_income:
        If True, skip projecting income (credit) events — triggered by
        "no_income" or "seasonal contract ended" messages.
    """
    end_date = request_date + timedelta(days=horizon_days)

    # Build a proximity index of existing future events for deduplication
    # Key: (category, direction) → sorted list of settlement_dates
    existing_index: dict[tuple[str, str], list[date]] = defaultdict(list)
    for ev in existing_events:
        existing_index[(ev.category, ev.direction)].append(ev.settlement_date)

    projected: list[FinancialEvent] = []
    counter = 0

    for pat in patterns:
        if suppress_income and pat.direction == "credit":
            continue

        # Find the most recent date for this pattern (settled OR already scheduled)
        existing_dates = existing_index.get((pat.category, pat.direction), [])
        all_dates = sorted(existing_dates + [pat.last_settlement_date])
        last_known = all_dates[-1]

        # Generate next occurrences from last_known + interval
        current = last_known + timedelta(days=pat.interval_days)

        while current <= end_date:
            if current >= request_date:
                # Check for a nearby existing event (within tolerance)
                nearby = any(
                    abs((d - current).days) <= config.RECURRENCE_INTERVAL_TOLERANCE
                    for d in existing_dates
                )
                if not nearby:
                    counter += 1
                    projected.append(
                        FinancialEvent(
                            event_id=f"proj_{counter}_{pat.reference_event_id}",
                            user_id=pat.user_id,
                            event_type=(
                                "expense" if pat.direction == "debit" else "income"
                            ),
                            description=f"[Projected] {pat.description}",
                            category=pat.category,
                            direction=pat.direction,
                            amount=pat.avg_amount,
                            currency=pat.currency,
                            event_date=current,
                            settlement_date=current,
                            status="scheduled",
                            linked_event_id=pat.reference_event_id,
                            flexibility=pat.flexibility,
                            minimum_allowed_amount=pat.minimum_allowed_amount,
                        )
                    )

            current += timedelta(days=pat.interval_days)

    return projected


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class EventProcessor:
    """
    Reconstructs a deterministic FinancialState for one (user, request_date).

    Usage:
        processor = EventProcessor(exchange_table)
        state = processor.build_state(
            profile        = profile,
            raw_events     = all_user_events,
            amendments     = message_processor_output,
            ocr_fills      = {event_id: amount},   # may be empty
            request_date   = request.request_date,
        )
    """

    def __init__(self, exchange_table: ExchangeRateTable) -> None:
        self._fx = exchange_table

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build_state(
        self,
        profile: UserProfile,
        raw_events: list[FinancialEvent],
        amendments: list[Amendment],
        ocr_fills: dict[str, float],
        request_date: date,
    ) -> FinancialState:
        """
        Full pipeline: OCR fills → amendments → currency conversion →
        cashflow filter → recurring detection → 90-day projection.
        """
        home = profile.home_currency
        suppress_income = any(a.amendment_type == "no_income" for a in amendments)

        # 1. Fill blank amounts from OCR (may be empty dict before OCR layer runs)
        events = _apply_ocr_fills(raw_events, ocr_fills)

        # 2. Apply deterministic amendments
        events, amendment_ids = _apply_amendments(events, amendments, request_date)

        # 3. Convert currencies to home_currency
        events = _convert_currencies(events, home, self._fx)

        # 4. Filter for cash-flow events in the forecast window (>= request_date)
        cashflow_events = _filter_cashflow_events(events, request_date)

        # 5. Detect recurring patterns from SETTLED HISTORICAL events only
        historical_settled = [
            ev for ev in events
            if ev.status == "settled"
            and ev.settlement_date < request_date
            and ev.amount is not None
        ]
        raw_patterns = _detect_recurring_patterns(historical_settled)

        # 5a. Update pattern amounts based on salary/rent amendments
        patterns = _apply_salary_amendments_to_patterns(
            raw_patterns, amendments, request_date
        )
        patterns = _apply_rent_amendments_to_patterns(patterns, amendments)

        # 6. Project recurring events into the 90-day window
        projected = _project_recurring(
            patterns=patterns,
            request_date=request_date,
            horizon_days=config.FORECAST_DAYS,
            existing_events=cashflow_events,
            suppress_income=suppress_income,
        )

        # 7. Convert projected events (they inherit the source event's currency
        #    which was already converted, but re-convert to be safe)
        projected = _convert_currencies(projected, home, self._fx)

        return FinancialState(
            profile=profile,
            cashflow_events=cashflow_events,
            projected_events=projected,
            recurring_patterns=patterns,
            ocr_filled_event_ids=set(ocr_fills.keys()),
            amendment_applied_ids=amendment_ids,
        )
