"""
config.py — Global constants, file paths, and allowed enum values.
All paths are resolved relative to the repository root.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "media" / "images"
OUTPUT_PATH = REPO_ROOT / "output.csv"
USAGE_REPORT_PATH = REPO_ROOT / "code" / "evaluation" / "usage_report.md"

# ---------------------------------------------------------------------------
# Input file paths
# ---------------------------------------------------------------------------

REQUESTS_CSV = DATASET_DIR / "requests.csv"
SAMPLE_REQUESTS_CSV = DATASET_DIR / "sample_requests.csv"
PROFILES_CSV = DATASET_DIR / "financial_profiles.csv"
EVENTS_CSV = DATASET_DIR / "financial_events.csv"
EXCHANGE_RATES_CSV = DATASET_DIR / "exchange_rates.csv"
PAYMENT_OPTIONS_CSV = DATASET_DIR / "request_payment_options.csv"
MESSAGES_CSV = DATASET_DIR / "messages.csv"
IMAGES_CSV = DATASET_DIR / "images.csv"

# ---------------------------------------------------------------------------
# Forecast parameters
# ---------------------------------------------------------------------------

FORECAST_DAYS: int = 90
MAX_SPENDING_CHANGES: int = 3

# Tolerance (days) when matching a recurring interval
RECURRENCE_INTERVAL_TOLERANCE: int = 3

# Minimum occurrences to classify a pattern as recurring
MIN_RECURRENCE_COUNT: int = 2

# Binary-search precision for amount_safe_to_pay (in home currency units)
AMOUNT_SEARCH_PRECISION: float = 0.01

# ---------------------------------------------------------------------------
# Allowed enum values
# ---------------------------------------------------------------------------

AFFORDABILITY_STATUSES = frozenset(
    {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
)

PAYMENT_METHODS = frozenset(
    {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
)

REQUEST_TYPES = frozenset(
    {
        "purchase", "travel", "education", "family_transfer",
        "debt_repayment", "investment", "housing", "emergency_expense", "other",
    }
)

EVENT_STATUSES = frozenset(
    {"settled", "pending", "scheduled", "failed", "cancelled", "unrealized"}
)

FLEXIBILITY_TYPES = frozenset({"fixed", "stoppable", "reducible"})

# Statuses that represent real cash outflows (include in forecast)
CASHFLOW_DEBIT_STATUSES = frozenset({"settled", "scheduled", "pending"})

# Statuses that represent confirmed cash inflows (settled credits only)
CASHFLOW_CREDIT_STATUSES = frozenset({"settled", "scheduled"})

# Statuses to exclude entirely from cash-flow projection
EXCLUDED_STATUSES = frozenset({"failed", "cancelled", "unrealized"})

# ---------------------------------------------------------------------------
# Output CSV columns (exact order required by problem statement)
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

# ---------------------------------------------------------------------------
# Default LLM model names (overridden by env vars)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_TEXT = "gemini-2.0-flash"
DEFAULT_MODEL_VISION = "gemini-2.0-flash"
