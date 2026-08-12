import os
import json
from typing import Dict, Any, Optional

# OpenAI Pricing rates per 1,000,000 tokens (USD)
MODEL_PRICING = {
    "gpt-4o": {"prompt": 2.50 / 1_000_000, "completion": 10.00 / 1_000_000},
    "gpt-5.6": {"prompt": 2.50 / 1_000_000, "completion": 10.00 / 1_000_000},
    "gpt-4o-mini": {"prompt": 0.15 / 1_000_000, "completion": 0.60 / 1_000_000},
    "default": {"prompt": 2.50 / 1_000_000, "completion": 10.00 / 1_000_000}
}

SUMMARY_LOG_PATH = os.path.join("output", "token_costs_summary.json")


def calculate_token_cost(prompt_tokens: int, completion_tokens: int, model: str = "gpt-4o") -> float:
    """Calculates estimated cost in USD based on model pricing."""
    rates = MODEL_PRICING.get(model.lower(), MODEL_PRICING["default"])
    cost = (prompt_tokens * rates["prompt"]) + (completion_tokens * rates["completion"])
    return round(cost, 6)


def track_usage(response_usage: Any, model: str = "gpt-4o", case_name: str = "default") -> Dict[str, Any]:
    """
    Extracts token counts from OpenAI API response.usage object or dict,
    calculates USD cost, updates cumulative summary log, and returns metrics dict.
    """
    if not response_usage:
        return {
            "model": model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "formatted_cost": "$0.000000"
        }

    # Handle pydantic object or dictionary
    if hasattr(response_usage, "prompt_tokens"):
        prompt_tokens = getattr(response_usage, "prompt_tokens", 0)
        completion_tokens = getattr(response_usage, "completion_tokens", 0)
        total_tokens = getattr(response_usage, "total_tokens", 0)
    elif isinstance(response_usage, dict):
        prompt_tokens = response_usage.get("prompt_tokens", 0)
        completion_tokens = response_usage.get("completion_tokens", 0)
        total_tokens = response_usage.get("total_tokens", 0)
    else:
        prompt_tokens, completion_tokens, total_tokens = 0, 0, 0

    cost = calculate_token_cost(prompt_tokens, completion_tokens, model)

    metrics = {
        "case_name": case_name,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": cost,
        "formatted_cost": f"${cost:.6f}"
    }

    # Save to cumulative log
    update_cumulative_log(metrics)
    return metrics


def update_cumulative_log(metrics: Dict[str, Any]):
    """Appends case metrics to cumulative JSON summary log file."""
    os.makedirs("output", exist_ok=True)
    summary_data = {"total_prompt_tokens": 0, "total_completion_tokens": 0, "total_tokens": 0, "total_cost_usd": 0.0, "total_cases": 0, "cases": []}

    if os.path.exists(SUMMARY_LOG_PATH):
        try:
            with open(SUMMARY_LOG_PATH, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception:
            pass

    summary_data["total_prompt_tokens"] = summary_data.get("total_prompt_tokens", 0) + metrics["prompt_tokens"]
    summary_data["total_completion_tokens"] = summary_data.get("total_completion_tokens", 0) + metrics["completion_tokens"]
    summary_data["total_tokens"] = summary_data.get("total_tokens", 0) + metrics["total_tokens"]
    summary_data["total_cost_usd"] = round(summary_data.get("total_cost_usd", 0.0) + metrics["estimated_cost_usd"], 6)
    summary_data["total_cases"] = summary_data.get("total_cases", 0) + 1
    summary_data.setdefault("cases", []).append(metrics)

    with open(SUMMARY_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)


def get_cumulative_metrics() -> Dict[str, Any]:
    """Reads and returns the cumulative token & cost metrics."""
    if os.path.exists(SUMMARY_LOG_PATH):
        try:
            with open(SUMMARY_LOG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["formatted_total_cost"] = f"${data.get('total_cost_usd', 0.0):.6f}"
                return data
        except Exception as e:
            return {"error": str(e)}
    return {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "formatted_total_cost": "$0.000000",
        "total_cases": 0,
        "cases": []
    }
