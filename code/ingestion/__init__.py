"""
ingestion package — Financial state reconstruction from raw CSVs.

Layers:
    message_processor.py  — deterministic regex-based amendment extractor
    event_processor.py    — currency-converting, recurring-pattern-detecting state builder
"""
from ingestion.message_processor import MessageProcessor
from ingestion.event_processor import EventProcessor

__all__ = ["MessageProcessor", "EventProcessor"]
