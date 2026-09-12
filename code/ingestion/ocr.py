"""
ingestion/ocr.py — Vision-LLM-based extraction of monetary amounts from images.

Images are payslips, rent receipts, invoices, and bank statements.
Results are cached so each image is only processed once.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import config
from data.models import FinancialEvent, ImageRecord
from llm.client import LLMClient

_SYSTEM_PROMPT = """You are a financial document parser. 
You will be shown a financial document image (payslip, receipt, invoice, or bank statement).
Your job is to extract the single most relevant monetary amount from the document.

Rules:
- Return ONLY the numeric value (no currency symbols, no commas, no spaces).
- Use dot as the decimal separator.
- For payslips: return the net pay / take-home amount.
- For receipts/invoices: return the total amount due or amount payable.
- For bank statements: return the transaction amount referenced.
- If you cannot determine the amount, return: UNKNOWN
- Do NOT include any explanation, only the number or UNKNOWN.

Examples of correct responses:
4365000
996.60
UNKNOWN
"""


class ImageOCR:
    """
    Extracts monetary amounts from financial document images using vision LLM.
    Results are cached: each image_id is only processed once per pipeline run.
    """

    def __init__(self, client: LLMClient, images_dir: Path = config.IMAGES_DIR) -> None:
        self._client = client
        self._images_dir = images_dir
        self._cache: dict[str, Optional[float]] = {}  # image_id → extracted amount

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_amount(
        self,
        image_id: str,
        event_context: str = "",
    ) -> Optional[float]:
        """
        Extract the numeric amount from the image identified by *image_id*.
        Returns None if extraction fails or image is missing.
        Results are cached.
        """
        if image_id in self._cache:
            return self._cache[image_id]

        image_path = self._images_dir / f"{image_id}.png"
        user_prompt = (
            f"Extract the monetary amount from this financial document.\n"
            f"Context: {event_context}" if event_context
            else "Extract the monetary amount from this financial document."
        )

        response = self._client.vision(_SYSTEM_PROMPT, user_prompt, image_path)
        amount = _parse_amount_response(response)
        self._cache[image_id] = amount

        status = f"{amount}" if amount is not None else "UNKNOWN"
        print(f"  [OCR] {image_id} → {status}")
        return amount

    def fill_blank_amounts(
        self,
        events: list[FinancialEvent],
        image_records: list[ImageRecord],
    ) -> dict[str, float]:
        """
        For every event with a blank amount that has a linked image record,
        extract the amount via OCR and return a mapping of {event_id → amount}.

        Call this ONCE upfront for all events across all users.
        """
        # Build a lookup: related_event_id → image_id
        event_to_image: dict[str, str] = {}
        for rec in image_records:
            if rec.related_event_id:
                event_to_image[rec.related_event_id] = rec.image_id

        filled: dict[str, float] = {}

        for event in events:
            if event.amount is None and event.event_id in event_to_image:
                image_id = event_to_image[event.event_id]
                context = (
                    f"This is a {event.event_type} event in the category '{event.category}'. "
                    f"Description: {event.description}. "
                    f"Currency: {event.currency}."
                )
                amount = self.extract_amount(image_id, context)
                if amount is not None:
                    filled[event.event_id] = amount
                else:
                    print(
                        f"  [OCR] WARNING: Could not extract amount for event "
                        f"{event.event_id} (image {image_id}). "
                        "This event will be excluded from cash-flow projection."
                    )

        return filled  # {event_id → extracted_amount}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_amount_response(response: str) -> Optional[float]:
    """Parse the LLM's raw text response into a float."""
    if not response:
        return None
    cleaned = response.strip()
    if cleaned.upper() == "UNKNOWN" or not cleaned:
        return None
    # Strip anything that isn't a digit, dot, or minus sign
    numeric = re.sub(r"[^\d.\-]", "", cleaned)
    try:
        value = float(numeric)
        return value if value > 0 else None
    except ValueError:
        return None
