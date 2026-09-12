"""
decision/plan_builder.py — Factories for concrete PaymentPlan objects.

Builds candidate plans from:
  - CSV PaymentOption rows  (full_payment / installments)
  - Synthetic constructions (partial_payment, wait, not_recommended)

No business logic here — just object construction and date arithmetic.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Optional

from data.models import PaymentOption, PaymentPlan, Request


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _option_rank(option_id: str) -> int:
    """Extract the numeric suffix from 'payment_option_42' -> 42."""
    digits = re.sub(r"\D", "", option_id)
    return int(digits) if digits else 999_999


def _payment_dates(option: PaymentOption) -> list[date]:
    """
    Generate the ordered list of settlement dates for a PaymentOption.

    For full_payment (frequency=None): single date = first_payment_date.
    For installments: first_payment_date, +freq, +2*freq, …, +(n-1)*freq.
    """
    return option.payment_dates()


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------

class PlanBuilder:
    """Static factory for all PaymentPlan variants."""

    # ------------------------------------------------------------------
    # CSV-option plans
    # ------------------------------------------------------------------

    @staticmethod
    def from_full_payment_option(option: PaymentOption) -> tuple[PaymentPlan, int]:
        """
        Build a single-payment plan from a full_payment PaymentOption.
        Returns (plan, option_id_rank).
        """
        rank = _option_rank(option.payment_option_id)
        plan = PaymentPlan(
            payments=[(option.first_payment_date, option.total_payable_amount)],
            method="full_payment",
            option_id=option.payment_option_id,
        )
        return plan, rank

    @staticmethod
    def from_installment_option(option: PaymentOption) -> tuple[PaymentPlan, int]:
        """
        Build an installment plan from an installments PaymentOption.
        Returns (plan, option_id_rank).
        """
        dates = _payment_dates(option)
        payments = [(d, option.payment_amount) for d in dates]
        rank = _option_rank(option.payment_option_id)
        plan = PaymentPlan(
            payments=payments,
            method="installments",
            option_id=option.payment_option_id,
        )
        return plan, rank

    @staticmethod
    def synthetic_full_payment(amount: float, on_date: date) -> tuple[PaymentPlan, int]:
        """Synthetic full_payment plan: pay amount on on_date. Returns (plan, rank)."""
        plan = PaymentPlan(
            payments=[(on_date, amount)],
            method="full_payment",
            option_id=None,
        )
        return plan, 999_998

    @staticmethod
    def partial_payment(amount: float, on_date: date) -> tuple[PaymentPlan, int]:
        """Partial payment plan. Returns (plan, rank)."""
        plan = PaymentPlan(
            payments=[(on_date, amount)],
            method="partial_payment",
            option_id=None,
        )
        return plan, 999_997

    @staticmethod
    def wait(requested_amount: float, earliest_date: date) -> tuple[PaymentPlan, int]:
        """'Wait and pay in full' plan. Returns (plan, rank)."""
        plan = PaymentPlan(
            payments=[(earliest_date, requested_amount)],
            method="wait",
            option_id=None,
        )
        return plan, 999_996

    @staticmethod
    def not_recommended() -> tuple[PaymentPlan, int]:
        """Fallback plan when no viable option exists. Returns (plan, rank)."""
        plan = PaymentPlan(
            payments=[],
            method="not_recommended",
            option_id=None,
        )
        return plan, 999_999
