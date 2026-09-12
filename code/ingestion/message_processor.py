"""
ingestion/message_processor.py — Deterministic, regex-based amendment extractor.

No Gemini / no OCR.  Works on raw message text in both English and Indonesian
by matching structural patterns (currency amounts, dates, signal phrases).

Amendment taxonomy extracted from all 215 messages in messages.csv:
  salary_change       — new salary amount (permanent or temporary)
  salary_date_change  — payroll date moved, no amount change
  no_income           — seasonal contract ended / employment ended
  commission_pending  — base salary is confirmed but commission/bonus is not
  bonus_pending       — entire bonus/bonus-event is unconfirmed
  refund_pending      — merchant/bank refund initiated but not yet credited
  internal_transfer   — debit + credit are an intra-account move (not cashflow)
  rent_increase       — recurring rent raised by a stated percentage
  prize_processing    — prize verified but not yet credited (exclude from income)
  prize_settled       — prize proceeds actually reached account (already settled)
  invoice_confirmed   — one freelance invoice confirmed; others still pending
  failed_debit_retry  — a debit attempt failed; a retry is expected
  fx_pending          — foreign-currency refund / charge; home amount TBD
  reimbursement_closed — one-off employer reimbursement; not recurring salary
  first_salary        — first salary from a new employer confirmed
  regular_salary_confirmed — base salary confirmed for next payroll (keep as-is)
  salary_resume       — salary resumes from a given date with a confirmed amount
  partial_income_ended — one income source ended; a confirmed amount remains
  card_dispute_open   — disputed charge under investigation; no reversal yet
  investment_value_only — investment mark-to-market move only; no cash event
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import date, datetime
from typing import Optional

from data.models import Amendment, FinancialEvent, Message

# ---------------------------------------------------------------------------
# Shared regex building-blocks
# ---------------------------------------------------------------------------

# Supported currency codes (from exchange_rates.csv)
_CURRENCIES = r"(?:IDR|EUR|USD|ZAR|INR)"

# Extracts (currency, amount) — handles "EUR 1037.52", "IDR 42750000", "42750000 IDR"
_AMT_RE = re.compile(
    rf"(?:({_CURRENCIES})\s*([\d,\.]+))"
    rf"|"
    rf"(?:([\d,\.]+)\s*({_CURRENCIES}))",
    re.IGNORECASE,
)

# ISO-8601 date: YYYY-MM-DD
_DATE_RE = re.compile(r"\b(\d{{4}}-\d{{2}}-\d{{2}})\b".format())

# Percentage: "12%" or "12.5%"
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_amount(text: str) -> Optional[tuple[float, str]]:
    """
    Return (amount_float, currency_code) for the first currency amount in *text*.
    Returns None if no recognisable amount is found.
    """
    for m in _AMT_RE.finditer(text):
        if m.group(1):          # CURRENCY AMOUNT form
            currency = m.group(1).upper()
            raw = m.group(2)
        else:                    # AMOUNT CURRENCY form
            raw = m.group(3)
            currency = m.group(4).upper()
        raw = raw.replace(",", "")
        try:
            return (float(raw), currency)
        except ValueError:
            continue
    return None


def _extract_date(text: str) -> Optional[date]:
    """Return the first ISO-8601 date found in *text*, or None."""
    m = _DATE_RE.search(text)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except ValueError:
            pass
    return None


def _extract_percentage(text: str) -> Optional[float]:
    """Return the first percentage value found in *text*, or None."""
    m = _PCT_RE.search(text)
    return float(m.group(1)) if m else None


def _contains_any(text: str, keywords: list[str]) -> bool:
    """Case-insensitive check for any keyword in text."""
    lower = text.lower()
    return any(kw in lower for kw in keywords)


# ---------------------------------------------------------------------------
# Pattern classifiers (one per message template)
# ---------------------------------------------------------------------------
# Each classifier returns an Amendment or None.
# The classifiers are tried in priority order in _classify().

_NO_INCOME_SIGNALS = [
    "seasonal contract has ended",
    "no off-season income",
    "kontrak musiman saat ini telah berakhir",
    "belum ada pendapatan di luar musim",
    "employment has ended",
    "no regular salary payments scheduled",
    "hubungan kerja anda telah berakhir",
    "tidak ada pembayaran gaji rutin",
]

_REFUND_PENDING_SIGNALS = [
    "refund has been initiated but has not reached",
    "pengembalian dana sudah diproses, tetapi belum masuk",
]

_INTERNAL_TRANSFER_SIGNALS = [
    "transfer between your two accounts",
    "debit dan kredit dengan jumlah yang sama berasal dari transfer antara dua rekening",
    "both accounts are registered under the same account holder",
    "kedua rekening terdaftar atas nama pemilik yang sama",
]

_RENT_INCREASE_SIGNALS = [
    "increases monthly rent by",
    "menaikkan biaya sewa bulanan sebesar",
    "perpanjangan sewa menaikkan",
    "renewed lease increases monthly rent",
]

_BONUS_PENDING_SIGNALS = [
    "quarterly bonus is still subject to",
    "bonus kuartalan anda masih menunggu",
    "bonus kuartalan anda masih menunggu hasil",
    "final amount and payment date have not been approved",
    "jumlah akhir dan tanggal pembayaran belum disetujui",
]

_COMMISSION_PENDING_SIGNALS = [
    "commission shown for open deals is still pending",
    "komisi dari transaksi yang masih berjalan belum disetujui",
    "commission is marked as earned",
]

_SALARY_DATE_CHANGE_SIGNALS = [
    "confirmed salary is now expected on",
    "gaji yang sudah dikonfirmasi kini diperkirakan masuk pada",
    "replaces the payroll date",
    "revised date",
]

_SALARY_INCREASE_SIGNALS = [
    "salary has increased to",
    "monthly salary has increased to",
    "naik menjadi",
    "gaji bulanan anda naik menjadi",
]

_SALARY_DECREASE_SIGNALS = [
    "next salary is reduced to",
    "temporary monthly pay is",
    "gaji bulanan sementara anda adalah",
    "gaji bulanan sementara anda",
    "your next salary is reduced to",
]

_SALARY_RESUME_SIGNALS = [
    "regular salary of",
    "salary of",
    "resumes on",
    "gaji pokok",
]

_FIRST_SALARY_SIGNALS = [
    "first salary will be",
    "first salary from the new employer is",
    "your first salary of",
    "gaji pertama dari perusahaan baru adalah",
    "gaji pertama anda sebesar",
]

_PARTIAL_INCOME_ENDED_SIGNALS = [
    "one household employment record has ended",
    "remaining confirmed monthly salary is",
    "salah satu sumber pendapatan kerja rumah tangga telah berakhir",
    "sisa gaji bulanan yang dikonfirmasi adalah",
]

_PRIZE_PROCESSING_SIGNALS = [
    "prize claim has been verified and is still in payment processing",
    "payment has not been credited to your account yet",
    "klaim hadiah anda sudah diverifikasi dan masih dalam proses pembayaran",
]

_PRIZE_SETTLED_SIGNALS = [
    "prize proceeds have reached your account",
    "hasil penjualan investasi anda sudah masuk",
]

_INVOICE_CONFIRMED_SIGNALS = [
    "client approved an invoice payment of",
    "klien menyetujui pembayaran faktur sebesar",
]

_FAILED_DEBIT_SIGNALS = [
    "previous debit attempt failed",
    "another debit will be attempted",
]

_INVESTMENT_VALUE_SIGNALS = [
    "no units have been sold and no cash proceeds",
    "investasi tersebut belum dijual dan tidak ada transaksi tunai",
    "displayed value will continue to move with market prices",
    "displayed market value has increased substantially",
    "displayed value of the investment has fallen",
]

_CARD_DISPUTE_SIGNALS = [
    "extra card charge is still being investigated",
    "reversal has not been posted",
    "dispute is open",
    "tagihan kartu tambahan masih dalam penyelidikan",
]

_FX_PENDING_SIGNALS = [
    "foreign-currency refund is still processing",
    "bill was charged in a foreign currency",
    "pengembalian dana dalam mata uang asing",
]

_REIMBURSEMENT_CLOSED_SIGNALS = [
    "latest employer credit is the reimbursement for your earlier work expense",
    "dana terbaru dari perusahaan adalah penggantian atas biaya kerja",
    "claim is now closed and no additional reimbursement",
]

_SALARY_CONFIRMED_SIGNALS = [
    "regular salary for the next payroll is",
    "confirmed base salary is",
    "base salary is",
    "salary is confirmed for",
    "salary will be",
    "salary for the next payroll is",
    "gaji rutin untuk penggajian berikutnya sudah dikonfirmasi",
    "gaji pokok yang dikonfirmasi adalah",
    "gaji sebesar",
    "gaji rutin anda untuk penggajian berikutnya adalah",
]


def _classify(msg: Message, events: dict[str, FinancialEvent]) -> Optional[Amendment]:
    """
    Try each pattern family in priority order.
    Returns the first matching Amendment, or None for informational-only messages.
    """
    t = msg.message_text

    # ------------------------------------------------------------------ 1
    # Employment / seasonal contract ended → stop income projection
    if _contains_any(t, _NO_INCOME_SIGNALS):
        return Amendment(
            source_message_id=msg.message_id,
            amendment_type="no_income",
            target_event_id=None,
            new_amount=None,
            effective_date=None,
            notes="seasonal or employment contract ended; no recurring income projected",
        )

    # ------------------------------------------------------------------ 2
    # Internal bank transfer (own accounts) → exclude both sides from cashflow
    if _contains_any(t, _INTERNAL_TRANSFER_SIGNALS):
        return Amendment(
            source_message_id=msg.message_id,
            amendment_type="internal_transfer",
            target_event_id=msg.related_event_id,
            new_amount=None,
            effective_date=None,
            notes="intra-account transfer; not real cashflow",
        )

    # ------------------------------------------------------------------ 3
    # Merchant refund pending — do not count as income yet
    if _contains_any(t, _REFUND_PENDING_SIGNALS):
        return Amendment(
            source_message_id=msg.message_id,
            amendment_type="refund_pending",
            target_event_id=msg.related_event_id,
            new_amount=None,
            effective_date=None,
            notes="refund initiated but not yet credited; exclude pending credit",
        )

    # ------------------------------------------------------------------ 4
    # Partial income ended (one source gone, remainder confirmed)
    if _contains_any(t, _PARTIAL_INCOME_ENDED_SIGNALS):
        amt = _extract_amount(t)
        return Amendment(
            source_message_id=msg.message_id,
            amendment_type="partial_income_ended",
            target_event_id=None,
            new_amount=amt[0] if amt else None,
            effective_date=_extract_date(t),
            notes=(
                f"one income source ended; remaining confirmed salary "
                f"= {amt[0]} {amt[1]}" if amt else "one income source ended"
            ),
        )

    # ------------------------------------------------------------------ 5
    # Lease / rent increase by percentage
    if _contains_any(t, _RENT_INCREASE_SIGNALS):
        pct = _extract_percentage(t)
        return Amendment(
            source_message_id=msg.message_id,
            amendment_type="rent_increase",
            target_event_id=msg.related_event_id,
            new_amount=None,
            effective_date=_extract_date(t),
            notes=f"rent increased by {pct}%" if pct else "rent increased",
        )

    # ------------------------------------------------------------------ 6
    # Salary date moved (no amount change)
    if _contains_any(t, _SALARY_DATE_CHANGE_SIGNALS):
        new_date = _extract_date(t)
        if new_date:
            return Amendment(
                source_message_id=msg.message_id,
                amendment_type="salary_date_change",
                target_event_id=None,
                new_amount=None,
                effective_date=new_date,
                notes=f"payroll date moved to {new_date}",
            )

    # ------------------------------------------------------------------ 7
    # Salary increase
    if _contains_any(t, _SALARY_INCREASE_SIGNALS):
        amt = _extract_amount(t)
        if amt:
            return Amendment(
                source_message_id=msg.message_id,
                amendment_type="salary_change",
                target_event_id=None,
                new_amount=amt[0],
                effective_date=_extract_date(t),
                notes=f"salary increased to {amt[0]} {amt[1]}",
            )

    # ------------------------------------------------------------------ 8
    # Salary decrease / temporary reduction
    if _contains_any(t, _SALARY_DECREASE_SIGNALS):
        amt = _extract_amount(t)
        if amt:
            return Amendment(
                source_message_id=msg.message_id,
                amendment_type="salary_change",
                target_event_id=None,
                new_amount=amt[0],
                effective_date=_extract_date(t),
                notes=f"salary temporarily reduced to {amt[0]} {amt[1]}",
            )

    # ------------------------------------------------------------------ 9
    # Salary resume (with amount + date)
    if _contains_any(t, _SALARY_RESUME_SIGNALS) and "resumes on" in t.lower():
        amt = _extract_amount(t)
        resume_date = _extract_date(t)
        if amt and resume_date:
            return Amendment(
                source_message_id=msg.message_id,
                amendment_type="salary_resume",
                target_event_id=None,
                new_amount=amt[0],
                effective_date=resume_date,
                notes=f"salary of {amt[0]} {amt[1]} resumes on {resume_date}",
            )

    # ------------------------------------------------------------------ 10
    # First salary from a new employer
    if _contains_any(t, _FIRST_SALARY_SIGNALS):
        amt = _extract_amount(t)
        credit_date = _extract_date(t)
        if amt:
            return Amendment(
                source_message_id=msg.message_id,
                amendment_type="first_salary",
                target_event_id=None,
                new_amount=amt[0],
                effective_date=credit_date,
                notes=f"first salary {amt[0]} {amt[1]} on {credit_date}",
            )

    # ------------------------------------------------------------------ 11
    # Bonus not yet confirmed
    if _contains_any(t, _BONUS_PENDING_SIGNALS):
        return Amendment(
            source_message_id=msg.message_id,
            amendment_type="bonus_pending",
            target_event_id=msg.related_event_id,
            new_amount=None,
            effective_date=None,
            notes="quarterly bonus pending final review; do not include in forecast",
        )

    # ------------------------------------------------------------------ 12
    # Commission pending (base salary OK, commission excluded)
    if _contains_any(t, _COMMISSION_PENDING_SIGNALS):
        amt = _extract_amount(t)
        return Amendment(
            source_message_id=msg.message_id,
            amendment_type="commission_pending",
            target_event_id=None,
            new_amount=amt[0] if amt else None,
            effective_date=None,
            notes=(
                f"base salary confirmed at {amt[0]} {amt[1]}; commission pending"
                if amt else "base salary confirmed; commission excluded"
            ),
        )

    # ------------------------------------------------------------------ 13
    # Regular/base salary confirmed for next payroll — note only, no change needed
    if _contains_any(t, _SALARY_CONFIRMED_SIGNALS):
        amt = _extract_amount(t)
        if amt:
            return Amendment(
                source_message_id=msg.message_id,
                amendment_type="salary_confirmed",
                target_event_id=None,
                new_amount=amt[0],
                effective_date=_extract_date(t),
                notes=f"salary confirmed at {amt[0]} {amt[1]} for next payroll",
            )

    # ------------------------------------------------------------------ 14
    # Prize in payment processing (not yet credited — exclude from income)
    if _contains_any(t, _PRIZE_PROCESSING_SIGNALS):
        return Amendment(
            source_message_id=msg.message_id,
            amendment_type="prize_processing",
            target_event_id=msg.related_event_id,
            new_amount=None,
            effective_date=None,
            notes="prize verified but not yet credited; exclude from income",
        )

    # ------------------------------------------------------------------ 15
    # Prize proceeds actually settled (already in account; no action needed)
    if _contains_any(t, _PRIZE_SETTLED_SIGNALS):
        return Amendment(
            source_message_id=msg.message_id,
            amendment_type="prize_settled",
            target_event_id=msg.related_event_id,
            new_amount=None,
            effective_date=None,
            notes="prize proceeds settled; use event amount as-is",
        )

    # ------------------------------------------------------------------ 16
    # Invoice confirmed (one invoice; others still pending)
    if _contains_any(t, _INVOICE_CONFIRMED_SIGNALS):
        amt = _extract_amount(t)
        settle_date = _extract_date(t)
        return Amendment(
            source_message_id=msg.message_id,
            amendment_type="invoice_confirmed",
            target_event_id=None,
            new_amount=amt[0] if amt else None,
            effective_date=settle_date,
            notes=(
                f"one invoice of {amt[0]} {amt[1]} confirmed; others pending"
                if amt else "one invoice confirmed; others pending"
            ),
        )

    # ------------------------------------------------------------------ 17
    # Failed debit — a retry is expected; keep the debit in the forecast
    if _contains_any(t, _FAILED_DEBIT_SIGNALS):
        return Amendment(
            source_message_id=msg.message_id,
            amendment_type="failed_debit_retry",
            target_event_id=msg.related_event_id,
            new_amount=None,
            effective_date=None,
            notes="previous debit failed; a retry will be attempted (keep in forecast)",
        )

    # ------------------------------------------------------------------ 18
    # Investment mark-to-market — no cash; exclude the event from income
    if _contains_any(t, _INVESTMENT_VALUE_SIGNALS):
        return Amendment(
            source_message_id=msg.message_id,
            amendment_type="investment_value_only",
            target_event_id=msg.related_event_id,
            new_amount=None,
            effective_date=None,
            notes="portfolio value change only; no cash received",
        )

    # ------------------------------------------------------------------ 19
    # Card dispute / reversal not yet posted — keep the original debit in forecast
    if _contains_any(t, _CARD_DISPUTE_SIGNALS):
        return Amendment(
            source_message_id=msg.message_id,
            amendment_type="card_dispute_open",
            target_event_id=msg.related_event_id,
            new_amount=None,
            effective_date=None,
            notes="card dispute open; reversal not yet posted; keep original debit",
        )

    # ------------------------------------------------------------------ 20
    # FX refund/charge still pending settlement
    if _contains_any(t, _FX_PENDING_SIGNALS):
        return Amendment(
            source_message_id=msg.message_id,
            amendment_type="fx_pending",
            target_event_id=msg.related_event_id,
            new_amount=None,
            effective_date=None,
            notes="foreign-currency transaction; final home-currency amount not yet known",
        )

    # ------------------------------------------------------------------ 21
    # Reimbursement closed (one-off payment, not recurring salary)
    if _contains_any(t, _REIMBURSEMENT_CLOSED_SIGNALS):
        return Amendment(
            source_message_id=msg.message_id,
            amendment_type="reimbursement_closed",
            target_event_id=msg.related_event_id,
            new_amount=None,
            effective_date=None,
            notes="one-off employer reimbursement; not recurring salary",
        )

    # No match — informational message
    return None


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class MessageProcessor:
    """
    Deterministic amendment extractor.  Requires no external API.

    Usage:
        processor = MessageProcessor()
        amendments = processor.extract_amendments(user_messages, event_index)
    """

    def extract_amendments(
        self,
        messages: list[Message],
        event_index: dict[str, FinancialEvent] | None = None,
    ) -> list[Amendment]:
        """
        Process all messages for one user and return a list of amendments.

        Parameters
        ----------
        messages:
            All messages for this user (already filtered by user_id).
        event_index:
            Optional {event_id → FinancialEvent} for cross-referencing.
        """
        if event_index is None:
            event_index = {}

        amendments: list[Amendment] = []
        for msg in messages:
            amend = _classify(msg, event_index)
            if amend is not None:
                amendments.append(amend)
        return amendments
