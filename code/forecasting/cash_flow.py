"""
forecasting/cash_flow.py — Day-by-day balance projection over a 90-day window.

Algorithm
---------
Starting balance is `profile.current_available_balance`.  This number already
reflects everything that has settled *before* request_date, so we never
re-apply historical events.

For each calendar day d in [request_date, request_date + FORECAST_DAYS):
    1.  Gather all cashflow events (from FinancialState.cashflow_events +
        projected_events) whose settlement_date == d.
    2.  Sum debits and credits separately.
    3.  closing_balance = opening_balance + credits − debits
    4.  Next day's opening = this day's closing.

The result is a list[DailyBalance] of length FORECAST_DAYS (90 entries,
one per calendar day, even on days with zero cash activity).

Spending-change simulation
--------------------------
The decision layer can call build_projection() with a modified FinancialState
(events removed or amounts reduced) to evaluate "what if user stops X?"
without any special branching logic here.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Optional

import config
from data.models import DailyBalance, FinancialEvent, FinancialState


class CashFlowProjector:
    """
    Builds a deterministic 90-day balance projection from a FinancialState.

    The projector is stateless — call build_projection() as many times as
    needed with different states or request dates.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_projection(
        self,
        state: FinancialState,
        request_date: date,
        horizon_days: int = config.FORECAST_DAYS,
    ) -> list[DailyBalance]:
        """
        Generate a day-by-day projection over *horizon_days* starting from
        *request_date*.

        Parameters
        ----------
        state:
            The FinancialState for this user at this request_date.
        request_date:
            Day 0 of the forecast.
        horizon_days:
            Number of days to project (default: 90 from config).

        Returns
        -------
        A list of DailyBalance objects, one per day, in chronological order.
        Length is always *horizon_days*.
        """
        # Build a fast lookup: date → (total_debits, total_credits, [event_ids])
        daily: dict[date, dict] = defaultdict(
            lambda: {"debits": 0.0, "credits": 0.0, "ids": []}
        )
        all_events = state.cashflow_events + state.projected_events
        self._index_events(all_events, daily)

        # Walk day by day
        balance = state.profile.current_available_balance
        projection: list[DailyBalance] = []
        end_date = request_date + timedelta(days=horizon_days)

        current = request_date
        while current < end_date:
            bucket = daily.get(current, {"debits": 0.0, "credits": 0.0, "ids": []})
            debits = bucket["debits"]
            credits = bucket["credits"]
            closing = balance + credits - debits

            projection.append(
                DailyBalance(
                    day=current,
                    opening_balance=round(balance, 2),
                    total_debits=round(debits, 2),
                    total_credits=round(credits, 2),
                    closing_balance=round(closing, 2),
                    events_on_day=list(bucket["ids"]),
                )
            )

            balance = closing
            current += timedelta(days=1)

        return projection

    def build_projection_with_extra_payments(
        self,
        state: FinancialState,
        request_date: date,
        extra_payments: list[tuple[date, float]],
        horizon_days: int = config.FORECAST_DAYS,
    ) -> list[DailyBalance]:
        """
        Build a projection and layer on top one or more additional payments.

        Used by SafetyChecker to simulate paying a specific amount on a
        specific date without modifying the FinancialState.

        Parameters
        ----------
        extra_payments:
            List of (payment_date, amount) tuples representing additional debits.
        """
        base = self.build_projection(state, request_date, horizon_days)

        # Group extra payments by date
        payment_by_date: dict[date, float] = defaultdict(float)
        for pdate, pamount in extra_payments:
            payment_by_date[pdate] += pamount

        # Re-walk adjusting closing balances
        adjusted: list[DailyBalance] = []
        cumulative_extra = 0.0

        for db in base:
            cumulative_extra += payment_by_date.get(db.day, 0.0)
            adjusted.append(
                DailyBalance(
                    day=db.day,
                    opening_balance=round(db.opening_balance - cumulative_extra + payment_by_date.get(db.day, 0.0), 2),
                    total_debits=round(db.total_debits + payment_by_date.get(db.day, 0.0), 2),
                    total_credits=db.total_credits,
                    closing_balance=round(db.closing_balance - cumulative_extra, 2),
                    events_on_day=db.events_on_day,
                )
            )

        return adjusted

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _index_events(
        events: list[FinancialEvent],
        daily: dict[date, dict],
    ) -> None:
        """Accumulate event amounts into the daily buckets."""
        for ev in events:
            if ev.amount is None or ev.amount == 0:
                continue
            bucket = daily[ev.settlement_date]
            if ev.direction == "debit":
                bucket["debits"] += ev.amount
            else:
                bucket["credits"] += ev.amount
            bucket["ids"].append(ev.event_id)
