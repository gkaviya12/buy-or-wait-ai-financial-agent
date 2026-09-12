"""
data/exchange.py — Fixed, dated exchange-rate lookups.
Rates are provided monthly (15th of each month).
We pick the closest available rate to the settlement date.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from data.models import ExchangeRate


class ExchangeRateTable:
    """
    Lookup table for fixed, dated currency conversion rates.

    Usage:
        table = ExchangeRateTable(rates_list)
        eur_amount = table.convert(100.0, "USD", "EUR", settlement_date)
    """

    def __init__(self, rates: list[ExchangeRate]) -> None:
        # Index: (from_currency, to_currency) → sorted list of (rate_date, rate)
        self._index: dict[tuple[str, str], list[tuple[date, float]]] = {}
        for r in rates:
            key = (r.from_currency, r.to_currency)
            self._index.setdefault(key, []).append((r.rate_date, r.rate))
        # Sort each list by date for binary-search / linear scan
        for key in self._index:
            self._index[key].sort()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
        on_date: date,
    ) -> float:
        """
        Convert *amount* from *from_currency* to *to_currency* using the rate
        closest to *on_date*.  Returns the amount unchanged when currencies are
        identical.  Raises ValueError if no rate is available for the pair.
        """
        if from_currency == to_currency:
            return amount
        rate = self.get_rate(from_currency, to_currency, on_date)
        return amount * rate

    def get_rate(
        self,
        from_currency: str,
        to_currency: str,
        on_date: date,
    ) -> float:
        """
        Return the exchange rate for (from_currency → to_currency) closest to
        *on_date*.  Tries direct pair first, then inverse, then raises.
        """
        direct = self._lookup(from_currency, to_currency, on_date)
        if direct is not None:
            return direct

        # Try inverse (e.g., if we have EUR→ZAR but need ZAR→EUR)
        inverse = self._lookup(to_currency, from_currency, on_date)
        if inverse is not None and inverse != 0:
            return 1.0 / inverse

        raise ValueError(
            f"No exchange rate available for {from_currency} → {to_currency} "
            f"near {on_date}"
        )

    def can_convert(self, from_currency: str, to_currency: str) -> bool:
        """Return True if a rate exists for this currency pair (or its inverse)."""
        return (
            (from_currency, to_currency) in self._index
            or (to_currency, from_currency) in self._index
            or from_currency == to_currency
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _lookup(
        self,
        from_currency: str,
        to_currency: str,
        on_date: date,
    ) -> Optional[float]:
        """Return the rate closest to *on_date* for the direct pair, or None."""
        key = (from_currency, to_currency)
        entries = self._index.get(key)
        if not entries:
            return None

        # Linear scan for closest date (list is short — typically < 50 entries)
        best_rate: Optional[float] = None
        best_delta = float("inf")
        for rate_date, rate in entries:
            delta = abs((rate_date - on_date).days)
            if delta < best_delta:
                best_delta = delta
                best_rate = rate
        return best_rate
