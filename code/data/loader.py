"""
data/loader.py — Load all CSVs once into a typed DataBundle.
Provides fast lookup methods keyed by user_id and request_id.
"""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import config
from data.models import (
    DataBundle,
    ExchangeRate,
    FinancialEvent,
    ImageRecord,
    Message,
    PaymentOption,
    Request,
    UserProfile,
)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_date(s: str) -> date:
    s = s.strip()
    if not s:
        raise ValueError(f"Expected a YYYY-MM-DD date, got empty string")
    return datetime.strptime(s, "%Y-%m-%d").date()


def _parse_optional_date(s: str) -> Optional[date]:
    s = s.strip()
    return _parse_date(s) if s else None


def _parse_float(s: str) -> Optional[float]:
    s = s.strip()
    if not s:
        return None
    # Remove commas in some locales
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_pipe_list(s: str) -> list[str]:
    s = s.strip()
    return [x.strip() for x in s.split("|") if x.strip()] if s else []


def _parse_bool(s: str) -> bool:
    return s.strip().lower() in {"true", "1", "yes"}


def _parse_optional_int(s: str) -> Optional[int]:
    s = s.strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of dicts, stripping BOM and whitespace."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------


def _load_profiles(path: Path) -> dict[str, UserProfile]:
    profiles: dict[str, UserProfile] = {}
    for row in _read_csv(path):
        uid = row["user_id"].strip()
        profiles[uid] = UserProfile(
            user_id=uid,
            home_currency=row["home_currency"].strip(),
            current_available_balance=float(row["current_available_balance"]),
            minimum_balance_to_keep=float(row["minimum_balance_to_keep"]),
            financial_priorities=_parse_pipe_list(row["financial_priorities"]),
            expense_categories_to_protect=_parse_pipe_list(
                row["expense_categories_to_protect"]
            ),
            expense_categories_willing_to_reduce=_parse_pipe_list(
                row.get("expense_categories_user_is_willing_to_reduce", "")
            ),
            expense_categories_willing_to_stop=_parse_pipe_list(
                row.get("expense_categories_user_is_willing_to_stop", "")
            ),
            payment_methods_user_will_consider=_parse_pipe_list(
                row["payment_methods_user_will_consider"]
            ),
            max_installment_months=_parse_optional_int(
                row.get("max_installment_months", "")
            ),
        )
    return profiles


def _load_events(path: Path) -> dict[str, list[FinancialEvent]]:
    events: dict[str, list[FinancialEvent]] = {}
    for row in _read_csv(path):
        uid = row["user_id"].strip()
        amount_str = row.get("amount", "").strip()
        event_date = _parse_date(row["event_date"])
        # settlement_date may be blank — fall back to event_date
        settlement_date_str = row.get("settlement_date", "").strip()
        settlement_date = (
            _parse_date(settlement_date_str) if settlement_date_str else event_date
        )
        ev = FinancialEvent(
            event_id=row["event_id"].strip(),
            user_id=uid,
            event_type=row["event_type"].strip(),
            description=row.get("description", "").strip(),
            category=row.get("category", "").strip(),
            direction=row["direction"].strip(),
            amount=_parse_float(amount_str),
            currency=row["currency"].strip(),
            event_date=event_date,
            settlement_date=settlement_date,
            status=row["status"].strip(),
            linked_event_id=row.get("linked_event_id", "").strip() or None,
            flexibility=row.get("flexibility", "fixed").strip() or "fixed",
            minimum_allowed_amount=_parse_float(
                row.get("minimum_allowed_amount", "")
            ),
        )
        events.setdefault(uid, []).append(ev)
    return events


def _load_options(path: Path) -> dict[str, list[PaymentOption]]:
    options: dict[str, list[PaymentOption]] = {}
    for row in _read_csv(path):
        rid = row["request_id"].strip()
        freq_str = row.get("payment_frequency_days", "").strip()
        opt = PaymentOption(
            payment_option_id=row["payment_option_id"].strip(),
            request_id=rid,
            payment_method=row["payment_method"].strip(),
            payment_amount=float(row["payment_amount"]),
            number_of_payments=int(row["number_of_payments"]),
            first_payment_date=_parse_date(row["first_payment_date"]),
            payment_frequency_days=int(freq_str) if freq_str else None,
            financing_fee=float(row.get("financing_fee", "0") or "0"),
            total_payable_amount=float(row["total_payable_amount"]),
        )
        options.setdefault(rid, []).append(opt)
    return options


def _load_messages(path: Path) -> dict[str, list[Message]]:
    messages: dict[str, list[Message]] = {}
    for row in _read_csv(path):
        uid = row["user_id"].strip()
        msg = Message(
            message_id=row["message_id"].strip(),
            user_id=uid,
            request_id=row.get("request_id", "").strip() or None,
            related_event_id=row.get("related_event_id", "").strip() or None,
            sent_at=row.get("sent_at", "").strip(),
            source_type=row.get("source_type", "").strip(),
            message_text=row.get("message_text", "").strip(),
        )
        messages.setdefault(uid, []).append(msg)
    return messages


def _load_image_records(path: Path) -> dict[str, list[ImageRecord]]:
    records: dict[str, list[ImageRecord]] = {}
    for row in _read_csv(path):
        uid = row["user_id"].strip()
        rec = ImageRecord(
            image_id=row["image_id"].strip(),
            user_id=uid,
            request_id=row.get("request_id", "").strip() or None,
            related_event_id=row.get("related_event_id", "").strip() or None,
        )
        records.setdefault(uid, []).append(rec)
    return records


def _load_exchange_rates(path: Path) -> list[ExchangeRate]:
    rates: list[ExchangeRate] = []
    for row in _read_csv(path):
        rates.append(
            ExchangeRate(
                rate_date=_parse_date(row["rate_date"]),
                from_currency=row["from_currency"].strip(),
                to_currency=row["to_currency"].strip(),
                rate=float(row["rate"]),
            )
        )
    return rates


def _load_requests(path: Path) -> list[Request]:
    requests: list[Request] = []
    for row in _read_csv(path):
        requests.append(
            Request(
                request_id=row["request_id"].strip(),
                user_id=row["user_id"].strip(),
                request_date=_parse_date(row["request_date"]),
                request_type=row["request_type"].strip(),
                requested_amount=float(row["requested_amount"]),
                desired_completion_date=_parse_date(row["desired_completion_date"]),
                allows_partial_payment=_parse_bool(row["allows_partial_payment"]),
                request_text=row.get("request_text", "").strip(),
            )
        )
    return requests


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------


class DataLoader:
    """
    Loads all CSVs once into a DataBundle.
    Call load_all() at startup; subsequent lookups are O(1) dict access.
    """

    def __init__(self, dataset_dir: Path = config.DATASET_DIR) -> None:
        self._dataset_dir = dataset_dir
        self._bundle: Optional[DataBundle] = None

    def load_all(self) -> DataBundle:
        """Load every input CSV and return the populated DataBundle."""
        print("Loading dataset…")
        self._bundle = DataBundle(
            requests=_load_requests(config.REQUESTS_CSV),
            profiles=_load_profiles(config.PROFILES_CSV),
            events=_load_events(config.EVENTS_CSV),
            options=_load_options(config.PAYMENT_OPTIONS_CSV),
            messages=_load_messages(config.MESSAGES_CSV),
            image_records=_load_image_records(config.IMAGES_CSV),
            exchange_rates=_load_exchange_rates(config.EXCHANGE_RATES_CSV),
        )
        n_events = sum(len(v) for v in self._bundle.events.values())
        print(
            f"  Loaded {len(self._bundle.requests)} requests, "
            f"{len(self._bundle.profiles)} profiles, "
            f"{n_events} events, "
            f"{len(self._bundle.exchange_rates)} exchange rates."
        )
        return self._bundle

    # ------------------------------------------------------------------
    # Convenience accessors (thin wrappers over DataBundle)
    # ------------------------------------------------------------------

    def get_profile(self, user_id: str) -> UserProfile:
        assert self._bundle, "Call load_all() first."
        return self._bundle.profiles[user_id]

    def get_events(self, user_id: str) -> list[FinancialEvent]:
        assert self._bundle, "Call load_all() first."
        return self._bundle.events.get(user_id, [])

    def get_options(self, request_id: str) -> list[PaymentOption]:
        assert self._bundle, "Call load_all() first."
        return self._bundle.options.get(request_id, [])

    def get_messages(self, user_id: str, request_id: Optional[str] = None) -> list[Message]:
        assert self._bundle, "Call load_all() first."
        msgs = self._bundle.messages.get(user_id, [])
        if request_id is not None:
            # Return messages scoped to this request OR with no request scope
            msgs = [m for m in msgs if m.request_id in {request_id, None}]
        return msgs

    def get_image_records(self, user_id: str, request_id: Optional[str] = None) -> list[ImageRecord]:
        assert self._bundle, "Call load_all() first."
        recs = self._bundle.image_records.get(user_id, [])
        if request_id is not None:
            recs = [r for r in recs if r.request_id in {request_id, None}]
        return recs
