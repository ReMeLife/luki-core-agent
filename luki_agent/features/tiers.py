"""Token tier configuration and helpers for LUKi.

These primitives are deliberately backend-only. They define how different
$LUKI holding tiers map to feature expectations (message limits, memory
retention hints, etc.) so the core agent and API gateway can make
consistent decisions and surface accurate copy to the UI.

The concrete UX for each tier (what the user sees) should be controlled
in product and UI layers; this module only encodes the policy.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# High-level feature expectations per tier. These values are aligned with
# the on-chain blueprint but can be tuned without changing external APIs.
FEATURE_TIERS: Dict[str, Dict[str, Any]] = {
    "basic": {
        "daily_messages": 10,
        "memory_retention": "7_days",
        "response_quality": "standard",
        "features": ["basic_chat", "simple_memory"],
    },
    "bronze": {  # ~1,000+ $LUKI
        "daily_messages": 50,
        "memory_retention": "30_days",
        "response_quality": "enhanced",
        "features": ["basic_chat", "extended_memory", "activity_suggestions"],
    },
    "silver": {  # ~10,000+ $LUKI
        "daily_messages": 200,
        "memory_retention": "90_days",
        "response_quality": "premium",
        "features": [
            "all_bronze",
            "personality_modes",
            "voice_responses",
            "care_planning",
        ],
    },
    "gold": {  # ~50,000+ $LUKI
        "daily_messages": 1000,
        "memory_retention": "1_year",
        "response_quality": "premium_plus",
        "features": [
            "all_silver",
            "family_coordination",
            "health_insights",
            "priority_support",
        ],
    },
    "diamond": {  # ~100,000+ $LUKI
        "daily_messages": "unlimited",
        "memory_retention": "permanent",
        "response_quality": "ultimate",
        "features": [
            "all_gold",
            "custom_training",
            "api_access",
            "white_label_options",
        ],
    },
}


def infer_tier_from_balance(balance: Optional[float]) -> str:
    """Infer a coarse-grained tier from an approximate $LUKI balance.

    This is intentionally simple and should not be treated as a
    compliance- or billing-grade check. It is suitable for UX hints and
    soft feature gating only.
    """

    if balance is None or balance <= 0:
        return "basic"
    if balance >= 100_000:
        return "diamond"
    if balance >= 50_000:
        return "gold"
    if balance >= 10_000:
        return "silver"
    if balance >= 1_000:
        return "bronze"
    return "basic"


def get_tier_config(tier: Optional[str]) -> Dict[str, Any]:
    """Return the feature configuration for a given tier.

    Falls back to the "basic" tier when the provided tier is unknown or
    missing. The returned dict is safe to use directly in prompts or for
    light-weight runtime checks.
    """

    key = (tier or "basic").lower()
    return FEATURE_TIERS.get(key, FEATURE_TIERS["basic"])
