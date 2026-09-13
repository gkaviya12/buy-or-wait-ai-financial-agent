"""
output/explainer.py — Generate concise decision explanations via Gemini.

The explanation is a single sentence (≤ 60 words) that states:
  1. The recommendation and why it is the safest option.
  2. The key financial fact behind it (amount safe, earliest date, spending change).

Prompt design
-------------
- System prompt: strict persona + length constraint (≤ 60 words, one sentence)
- User prompt: a structured summary of the financial context — NO raw CSV rows
  are passed; only the computed decision facts are included to minimise token use.
- Temperature = 0.0 (deterministic).
- Falls back to a hand-crafted explanation if the LLM returns empty string.

The fallback generator covers all five methods deterministically with no API call,
so the pipeline always produces a valid explanation even if the LLM is unavailable.
"""

from __future__ import annotations

from data.models import DecisionResult, Request, UserProfile
from llm.client import LLMClient

_SYSTEM_PROMPT = (
    "You are a financial advisor writing brief, factual payment recommendations. "
    "Reply with a SINGLE sentence of at most 60 words. "
    "Do not use bullet points, headers, or markdown. "
    "State what the user should do and why, referencing specific amounts and dates."
)


class Explainer:
    """
    Generates one-sentence decision explanations using Gemini.

    Parameters
    ----------
    client : LLMClient
        Shared LLM client (already configured with API key).
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def explain(
        self,
        result: DecisionResult,
        request: Request,
        profile: UserProfile,
    ) -> str:
        """
        Return a ≤60-word explanation for the given decision.

        The method calls the LLM and, if it returns an empty string,
        falls back to a deterministic hand-crafted sentence.
        """
        user_prompt = _build_user_prompt(result, request, profile)
        raw = self._client.chat(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
            purpose="explanation",
            request_id=result.request_id,
        )
        text = (raw or "").strip()
        if not text:
            text = _fallback_explanation(result, request, profile)
        # Enforce hard length limit: truncate at word boundary to ≤ 80 words
        return _trim_to_words(text, max_words=80)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_user_prompt(
    result: DecisionResult,
    request: Request,
    profile: UserProfile,
) -> str:
    """
    Build a compact structured prompt — just the decision facts, no raw CSV.
    """
    currency = profile.home_currency
    lines = [
        f"Request: {request.request_id}",
        f"Requested amount: {request.requested_amount:,.2f} {currency}",
        f"Request date: {request.request_date}",
        f"Deadline: {request.desired_completion_date}",
        f"Current safe-to-pay: {result.amount_safe_to_pay:,.2f} {currency}",
        f"Affordability status: {result.affordability_status}",
        f"Recommended method: {result.recommended_payment_method}",
        f"Payment plan: {result.payment_plan}",
    ]
    if result.earliest_date_for_full_payment:
        lines.append(
            f"Earliest date for full payment: {result.earliest_date_for_full_payment}"
        )
    if result.spending_changes_needed and result.spending_changes_needed != "none":
        lines.append(f"Spending changes required: {result.spending_changes_needed}")
    lines.append(
        "\nWrite ONE sentence explaining this recommendation to the user."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------

def _fallback_explanation(
    result: DecisionResult,
    request: Request,
    profile: UserProfile,
) -> str:
    """
    Produce a deterministic explanation without any API call.
    Used when the LLM is unavailable or returns an empty response.
    """
    method = result.recommended_payment_method
    currency = profile.home_currency
    amt = result.amount_safe_to_pay
    requested = request.requested_amount
    earliest = result.earliest_date_for_full_payment
    changes = result.spending_changes_needed

    has_changes = changes and changes != "none"

    if method == "full_payment":
        if has_changes:
            return (
                f"You can pay the full {requested:,.2f} {currency} today if you "
                f"apply the suggested spending changes ({changes})."
            )
        return (
            f"Your balance safely covers the full {requested:,.2f} {currency} "
            f"today, so a single payment is recommended."
        )

    elif method == "installments":
        return (
            f"An installment plan is recommended because paying "
            f"{requested:,.2f} {currency} in full today would risk your "
            f"minimum balance; spreading payments keeps your finances safe."
        )

    elif method == "partial_payment":
        remainder = round(requested - amt, 2)
        return (
            f"You can safely pay {amt:,.2f} {currency} now and the remaining "
            f"{remainder:,.2f} {currency} on {earliest} once your balance recovers."
        )

    elif method == "wait":
        return (
            f"Your current balance cannot safely cover {requested:,.2f} {currency}; "
            f"wait until {earliest} when your projected balance will be sufficient."
        )

    else:  # not_recommended
        return (
            f"No safe payment option is available within the 90-day forecast "
            f"window for {requested:,.2f} {currency}; we cannot recommend proceeding."
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _trim_to_words(text: str, max_words: int) -> str:
    """Truncate text at the last complete sentence within max_words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    trimmed = " ".join(words[:max_words])
    # Try to end at a sentence boundary
    for sep in (".", "!", "?"):
        idx = trimmed.rfind(sep)
        if idx > 0:
            return trimmed[: idx + 1]
    return trimmed + "."
