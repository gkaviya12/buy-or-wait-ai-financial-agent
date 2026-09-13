"""
output/writer.py — Write validated DecisionResult rows to output.csv.

Column order matches config.OUTPUT_COLUMNS exactly.

Amount formatting
-----------------
`amount_safe_to_pay` is written as a plain float with up to 2 decimal places
(e.g. 25256.0, 17229139.2 — no thousands separator, no currency symbol).
The problem statement sample output uses Python default float str() formatting,
so we strip trailing zeros after the decimal point unless they are the only digit
(e.g. 25256.0 → "25256.0", NOT "25256.00" or "25256").

Row ordering
------------
Rows are written in the same order as config requests list (i.e. the order they
appear in requests.csv), NOT sorted by request_id alphabetically.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import config
from data.models import DecisionResult


def _format_amount(value: float) -> str:
    """
    Format amount_safe_to_pay for CSV output.
    Matches the sample output format (e.g. '25256.0', '17229139.2').
    Uses Python default float repr stripped of unnecessary trailing zeros.
    """
    # Round to 2 decimal places, then let Python format it naturally
    rounded = round(value, 2)
    s = f"{rounded:.10f}".rstrip("0")
    # Ensure at least one decimal digit (Python's str(float) always has one)
    if s.endswith("."):
        s += "0"
    return s


class OutputWriter:
    """
    Writes a list of validated DecisionResult objects to output.csv.

    Usage:
        writer = OutputWriter()
        writer.write(results, path=config.OUTPUT_PATH)
    """

    def write(
        self,
        results: Sequence[DecisionResult],
        path: Path = config.OUTPUT_PATH,
    ) -> int:
        """
        Write all results to *path* in the required column order.

        Parameters
        ----------
        results:
            Ordered list of DecisionResult objects (one per request).
        path:
            Destination CSV path (created/overwritten).

        Returns
        -------
        Number of rows successfully written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        written = 0
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(config.OUTPUT_COLUMNS)

            for result in results:
                row = _result_to_row(result)
                writer.writerow(row)
                written += 1

        print(f"  [OutputWriter] Wrote {written} rows -> {path}")
        return written


# ---------------------------------------------------------------------------
# Row serialisation
# ---------------------------------------------------------------------------

def _result_to_row(result: DecisionResult) -> list[str]:
    """
    Convert a DecisionResult to a list of strings in OUTPUT_COLUMNS order.

    Column order (from config.OUTPUT_COLUMNS):
        request_id, amount_safe_to_pay, affordability_status,
        recommended_payment_method, payment_plan,
        earliest_date_for_full_payment, spending_changes_needed,
        decision_explanation
    """
    return [
        result.request_id,
        _format_amount(result.amount_safe_to_pay),
        result.affordability_status,
        result.recommended_payment_method,
        result.payment_plan,                      # already formatted string
        result.earliest_date_for_full_payment,    # YYYY-MM-DD or ""
        result.spending_changes_needed,           # formatted string or "none"
        result.decision_explanation,
    ]
