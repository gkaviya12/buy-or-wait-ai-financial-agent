"""
data/models.py — All domain dataclasses.
Pure data containers; no business logic lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


# ---------------------------------------------------------------------------
# Input domain objects
# ---------------------------------------------------------------------------


@dataclass
class UserProfile:
    user_id: str
    home_currency: str
    current_available_balance: float
    minimum_balance_to_keep: float
    financial_priorities: list[str]          # ordered list
    expense_categories_to_protect: list[str]
    expense_categories_willing_to_reduce: list[str]
    expense_categories_willing_to_stop: list[str]
    payment_methods_user_will_consider: list[str]
    max_installment_months: Optional[int]    # None means installments not considered


@dataclass
class FinancialEvent:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str           # 'debit' | 'credit'
    amount: Optional[float]  # None = must be extracted from image
    currency: str
    event_date: date
    settlement_date: date
    status: str              # settled | pending | scheduled | failed | cancelled | unrealized
    linked_event_id: Optional[str]
    flexibility: str         # fixed | stoppable | reducible
    minimum_allowed_amount: Optional[float]


@dataclass
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str          # full_payment | installments
    payment_amount: float        # per-payment amount
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: Optional[int]  # None for full_payment
    financing_fee: float
    total_payable_amount: float

    def num_months(self) -> int:
        """Approximate number of months this installment plan spans."""
        if self.payment_frequency_days is None or self.number_of_payments <= 1:
            return 0
        return round((self.number_of_payments - 1) * self.payment_frequency_days / 30)

    def payment_dates(self) -> list[date]:
        """Generate all payment dates for this option."""
        from datetime import timedelta
        if self.number_of_payments == 1 or self.payment_frequency_days is None:
            return [self.first_payment_date]
        dates = []
        current = self.first_payment_date
        for _ in range(self.number_of_payments):
            dates.append(current)
            current = current + timedelta(days=self.payment_frequency_days)
        return dates


@dataclass
class Message:
    message_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    sent_at: str           # ISO-8601 string
    source_type: str       # employer | bank | merchant | service_provider
    message_text: str


@dataclass
class ImageRecord:
    image_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]


@dataclass
class ExchangeRate:
    rate_date: date
    from_currency: str
    to_currency: str
    rate: float


@dataclass
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: float
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str


# ---------------------------------------------------------------------------
# Intermediate processing objects
# ---------------------------------------------------------------------------


@dataclass
class Amendment:
    """A structured change extracted from a message."""
    source_message_id: str
    amendment_type: str          # salary_change | event_cancel | date_change | refund_pending | rate_change
    target_event_id: Optional[str]
    new_amount: Optional[float]
    effective_date: Optional[date]
    notes: str = ""


@dataclass
class RecurringPattern:
    """A detected recurring income or expense pattern for a user."""
    user_id: str
    category: str
    direction: str          # 'debit' | 'credit'
    avg_amount: float
    currency: str
    interval_days: int      # 7=weekly, 28-31=monthly, etc.
    last_settlement_date: date
    flexibility: str
    reference_event_id: str  # ID of most recent event in this pattern
    description: str
    minimum_allowed_amount: Optional[float] = None


@dataclass
class SpendingChange:
    """A recommended change to a recurring flexible expense."""
    event_id: str
    change_type: str         # 'stop' | 'reduce_to'
    new_amount: Optional[float] = None  # only for 'reduce_to'

    def format(self) -> str:
        if self.change_type == "stop":
            return f"stop:{self.event_id}"
        return f"reduce_to:{self.event_id}:{self.new_amount}"


@dataclass
class FinancialState:
    """Reconstructed financial state for one user at one request_date."""
    profile: UserProfile
    cashflow_events: list[FinancialEvent]        # filtered + amended + currency-converted
    projected_events: list[FinancialEvent]        # recurring events projected forward
    recurring_patterns: list[RecurringPattern]
    ocr_filled_event_ids: set[str]               # events whose amounts came from OCR
    amendment_applied_ids: set[str]              # events affected by messages


# ---------------------------------------------------------------------------
# Forecasting objects
# ---------------------------------------------------------------------------


@dataclass
class DailyBalance:
    """Balance snapshot for a single day in the 90-day projection."""
    day: date
    opening_balance: float
    total_debits: float
    total_credits: float
    closing_balance: float
    events_on_day: list[str] = field(default_factory=list)  # event_ids

    @property
    def net(self) -> float:
        return self.total_credits - self.total_debits


@dataclass
class SafetyResult:
    """Output of the 90-day safety check."""
    amount_safe_to_pay: float
    earliest_full_payment_date: Optional[date]   # None = never within 90 days
    min_closing_balance: float                   # lowest balance in base projection
    base_projection: list[DailyBalance]          # the unmodified 90-day projection


# ---------------------------------------------------------------------------
# Decision objects
# ---------------------------------------------------------------------------


@dataclass
class PaymentPlan:
    """A concrete payment schedule."""
    payments: list[tuple[date, float]]   # (date, amount) in chronological order
    method: str                          # full_payment | partial_payment | installments | wait
    option_id: Optional[str] = None     # set when method == 'installments'

    def total(self) -> float:
        return sum(amt for _, amt in self.payments)

    def first_date(self) -> Optional[date]:
        return self.payments[0][0] if self.payments else None

    def num_payments(self) -> int:
        return len(self.payments)

    def format_string(self) -> str:
        if not self.payments:
            return "none"
        return "|".join(f"{d.isoformat()}:{amt}" for d, amt in self.payments)


@dataclass
class CandidatePlan:
    """One evaluated candidate plan for ranking."""
    payment_plan: PaymentPlan
    affordability_status: str
    spending_changes: list[SpendingChange]
    completes_by_deadline: bool
    total_payable: float
    option_id_rank: int = 999999   # large = no option (partial_payment / wait)

    def rank_key(self) -> tuple:
        """Lower = better. Used for sorting."""
        return (
            not self.completes_by_deadline,   # True (fails) sorts after False (passes)
            len(self.spending_changes) > 0,   # prefer no spending changes
            self.total_payable,               # minimize total cost
            self.payment_plan.first_date(),   # start earlier
            self.payment_plan.num_payments(), # fewer payments
            self.option_id_rank,              # lowest option_id
        )


# ---------------------------------------------------------------------------
# Output object
# ---------------------------------------------------------------------------


@dataclass
class DecisionResult:
    """The complete output row for one request."""
    request_id: str
    amount_safe_to_pay: float
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str                          # formatted string or "none"
    earliest_date_for_full_payment: str        # YYYY-MM-DD or ""
    spending_changes_needed: str               # formatted string or "none"
    decision_explanation: str


# ---------------------------------------------------------------------------
# Bundle (all data loaded at startup)
# ---------------------------------------------------------------------------


@dataclass
class DataBundle:
    """All CSVs loaded into memory at startup."""
    requests: list[Request]
    profiles: dict[str, UserProfile]           # user_id → profile
    events: dict[str, list[FinancialEvent]]    # user_id → events
    options: dict[str, list[PaymentOption]]    # request_id → options
    messages: dict[str, list[Message]]         # user_id → messages
    image_records: dict[str, list[ImageRecord]] # user_id → image records
    exchange_rates: list[ExchangeRate]
