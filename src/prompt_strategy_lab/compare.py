"""Side-by-side comparison harness.

Runs the same user prompt through multiple strategies and returns a
structured comparison so callers can inspect text, latency, and token
usage uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

from prompt_strategy_lab.runner import OllamaRunner, ModelResponse
from prompt_strategy_lab.strategies import STRATEGIES, Strategy


@dataclass
class StrategyResult:
    strategy: str
    augmented_prompt: str
    response: str
    tokens_used: int
    latency_ms: float


def compare_strategies(
    user_prompt: str,
    strategy_names: Iterable[str],
    runner: OllamaRunner | None = None,
    **gen_kwargs,
) -> list[StrategyResult]:
    runner = runner or OllamaRunner()
    results: list[StrategyResult] = []

    for name in strategy_names:
        if name not in STRATEGIES:
            raise KeyError(
                f"Unknown strategy '{name}'. Available: {sorted(STRATEGIES)}"
            )
        strategy: Strategy = STRATEGIES[name]
        augmented = strategy.augment(user_prompt)
        resp: ModelResponse = runner.run(augmented, **gen_kwargs)
        results.append(
            StrategyResult(
                strategy=name,
                augmented_prompt=augmented,
                response=resp.text,
                tokens_used=resp.tokens_used,
                latency_ms=resp.latency_ms,
            )
        )

    return results


def to_records(results: list[StrategyResult]) -> list[dict]:
    """Convert results to a list of plain dicts (e.g. for `pd.DataFrame`)."""
    return [asdict(r) for r in results]
