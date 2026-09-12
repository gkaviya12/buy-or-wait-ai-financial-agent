"""
llm/client.py — Unified Gemini API client for text and vision calls.
Reads GEMINI_API_KEY from environment. Updates UsageTracker on every call.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

import config
from llm.usage_tracker import UsageTracker

load_dotenv()

# Gemini pricing (USD per 1M tokens) — approximate, update if needed
# gemini-2.0-flash pricing
_PRICE_INPUT_PER_M = 0.10
_PRICE_OUTPUT_PER_M = 0.40


class LLMClient:
    """
    Unified wrapper around Google Gemini for text and vision calls.

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
        genai.configure(api_key=api_key)
        self._tracker = tracker
        self._model_text_name = os.environ.get("MODEL_TEXT", model_text)
        self._model_vision_name = os.environ.get("MODEL_VISION", model_vision)
        self._text_model = genai.GenerativeModel(self._model_text_name)
        self._vision_model = genai.GenerativeModel(self._model_vision_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        """
        Make a text-only generation call.
        Returns the response text, or "" on error.
        """
        full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        try:
            response = self._text_model.generate_content(
                full_prompt,
                generation_config=genai.GenerationConfig(temperature=temperature),
            )
            text = response.text or ""
            self._record(self._model_text_name, full_prompt, text)
            return text
        except Exception as exc:
            print(f"[LLMClient.chat] ERROR: {exc}")
            return ""

    def vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Path,
        temperature: float = 0.0,
    ) -> str:
        """
        Make a vision (image + text) call.
        Returns the response text, or "" on error.
        """
        if not image_path.exists():
            print(f"[LLMClient.vision] Image not found: {image_path}")
            return ""
        try:
            image_data = _load_image_b64(image_path)
            mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
            full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
            parts = [
                {"mime_type": mime_type, "data": image_data},
                full_prompt,
            ]
            response = self._vision_model.generate_content(
                parts,
                generation_config=genai.GenerationConfig(temperature=temperature),
            )
            text = response.text or ""
            self._record(self._model_vision_name, full_prompt, text)
            return text
        except Exception as exc:
            print(f"[LLMClient.vision] ERROR for {image_path.name}: {exc}")
            return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record(self, model: str, prompt: str, response: str) -> None:
        """Estimate token counts and record in UsageTracker."""
        # Approximation: 1 token ≈ 4 characters
        prompt_tokens = len(prompt) // 4
        completion_tokens = len(response) // 4
        cost = (
            prompt_tokens * _PRICE_INPUT_PER_M / 1_000_000
            + completion_tokens * _PRICE_OUTPUT_PER_M / 1_000_000
        )
        self._tracker.record_call(model, prompt_tokens, completion_tokens, cost)


def _load_image_b64(path: Path) -> bytes:
    """Read an image file and return raw bytes (Gemini SDK handles encoding)."""
    return path.read_bytes()
