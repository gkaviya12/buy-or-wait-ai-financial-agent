"""
llm/client.py — Unified Gemini API client for text and vision calls.

Uses the current `google-genai` SDK (google.genai), NOT the deprecated
`google-generativeai` package.

Reads GEMINI_API_KEY from .env.  All calls automatically update UsageTracker
with real token counts (from response.usage_metadata when available) and
estimated cost.  Falls back to character-based estimation when the SDK does
not return metadata.

Design decisions
----------------
- Zero-temperature by default (deterministic outputs for explanation generation)
- Vision calls pass image bytes directly via types.Part
- Retry with exponential back-off on transient 429/503 errors (up to MAX_RETRIES)
- All errors surface as empty string rather than crashing the pipeline
- `purpose` and `request_id` are forwarded to UsageTracker for the usage report
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import google.genai as genai
from google.genai import types
from dotenv import load_dotenv

import config
from llm.usage_tracker import UsageTracker, estimate_cost

load_dotenv()

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0   # seconds; doubles on each retry


class LLMClient:
    """
    Unified wrapper around Google Gemini (google-genai SDK) for text and vision calls.

    All calls automatically update the shared UsageTracker.
    System-level errors result in a fallback empty string rather than crashing.
    """

    def __init__(
        self,
        tracker: UsageTracker,
        model_text: str = config.DEFAULT_MODEL_TEXT,
        model_vision: str = config.DEFAULT_MODEL_VISION,
    ) -> None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. Copy .env.example to .env and fill it in."
            )
        self._client = genai.Client(api_key=api_key)
        self._tracker = tracker
        self._model_text_name = os.environ.get("MODEL_TEXT", model_text)
        self._model_vision_name = os.environ.get("MODEL_VISION", model_vision)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        purpose: str = "other",
        request_id: str = "",
    ) -> str:
        """
        Make a text-only generation call.

        Parameters
        ----------
        system_prompt : str
            High-level persona / instruction passed as system_instruction.
        user_prompt : str
            Per-call dynamic content passed as the user turn.
        temperature : float
            0.0 = deterministic (default); raise for creative tasks.
        purpose : str
            'ocr' | 'explanation' | 'other' — logged in usage report.
        request_id : str
            Associated request_id for the usage report ('' if not applicable).

        Returns
        -------
        Response text, or "" on unrecoverable error.
        """
        config_obj = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_prompt if system_prompt else None,
        )

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=self._model_text_name,
                    contents=user_prompt,
                    config=config_obj,
                )
                text = response.text or ""
                self._record(
                    model=self._model_text_name,
                    prompt=user_prompt,
                    response_text=text,
                    response=response,
                    call_type="text",
                    purpose=purpose,
                    request_id=request_id,
                )
                return text
            except Exception as exc:
                if attempt < _MAX_RETRIES - 1 and _is_retryable(exc):
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"  [LLMClient.chat] Retry {attempt + 1} after {delay}s: {exc}")
                    time.sleep(delay)
                else:
                    print(f"  [LLMClient.chat] ERROR: {exc}")
                    return ""
        return ""

    def vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Path,
        temperature: float = 0.0,
        purpose: str = "ocr",
        request_id: str = "",
    ) -> str:
        """
        Make a vision (image + text) call.

        The image is read from *image_path* and passed as a Part to the
        Gemini SDK (which handles encoding internally).

        Returns
        -------
        Response text, or "" on error or if the image file is missing.
        """
        if not image_path.exists():
            print(f"  [LLMClient.vision] Image not found: {image_path}")
            return ""

        image_bytes = image_path.read_bytes()
        mime_type = _mime_for(image_path)

        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        text_part = user_prompt

        config_obj = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_prompt if system_prompt else None,
        )

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=self._model_vision_name,
                    contents=[image_part, text_part],
                    config=config_obj,
                )
                text = response.text or ""
                self._record(
                    model=self._model_vision_name,
                    prompt=user_prompt,
                    response_text=text,
                    response=response,
                    call_type="vision",
                    purpose=purpose,
                    request_id=request_id,
                )
                return text
            except Exception as exc:
                if attempt < _MAX_RETRIES - 1 and _is_retryable(exc):
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    print(
                        f"  [LLMClient.vision] Retry {attempt + 1} after {delay}s "
                        f"for {image_path.name}: {exc}"
                    )
                    time.sleep(delay)
                else:
                    print(f"  [LLMClient.vision] ERROR for {image_path.name}: {exc}")
                    return ""
        return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record(
        self,
        model: str,
        prompt: str,
        response_text: str,
        response=None,
        call_type: str = "text",
        purpose: str = "other",
        request_id: str = "",
    ) -> None:
        """
        Extract token counts from response.usage_metadata if available,
        else estimate from character lengths (1 token ~= 4 chars).
        """
        prompt_tokens = 0
        completion_tokens = 0

        meta = getattr(response, "usage_metadata", None)
        if meta is not None:
            prompt_tokens = getattr(meta, "prompt_token_count", 0) or 0
            completion_tokens = getattr(meta, "candidates_token_count", 0) or 0

        if prompt_tokens == 0:
            prompt_tokens = max(1, len(prompt) // 4)
        if completion_tokens == 0:
            completion_tokens = max(1, len(response_text) // 4)

        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        self._tracker.record_call(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            call_type=call_type,
            purpose=purpose,
            request_id=request_id,
        )


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _mime_for(path: Path) -> str:
    """Return the MIME type for a given image path suffix."""
    return {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif":  "image/gif",
    }.get(path.suffix.lower(), "image/jpeg")


def _is_retryable(exc: Exception) -> bool:
    """Return True for transient API errors worth retrying."""
    msg = str(exc).lower()
    return any(kw in msg for kw in ("429", "503", "quota", "rate limit", "unavailable"))
