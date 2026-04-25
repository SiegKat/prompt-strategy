import pytest

from prompt_strategy_lab.strategies import (
    STRATEGIES,
    ChainOfThought,
    Reflection,
    Template,
    FewShot,
    SelfConsistency,
)


SAMPLE_PROMPT = "Brainstorm features for an AI tutor."


def test_all_strategies_registered():
    expected = {
        "chain_of_thought",
        "reflection",
        "alternative_approaches",
        "template",
        "comparative",
        "fact_check_list",
        "few_shot",
        "cot_with_reflection",
        "self_consistency",
    }
    assert set(STRATEGIES.keys()) == expected


def test_every_strategy_inserts_user_prompt():
    for name, strategy in STRATEGIES.items():
        out = strategy.augment(SAMPLE_PROMPT)
        assert SAMPLE_PROMPT in out, f"{name} did not embed the user prompt"


def test_every_strategy_returns_stripped_string():
    for name, strategy in STRATEGIES.items():
        out = strategy.augment(SAMPLE_PROMPT)
        assert isinstance(out, str)
        assert out == out.strip()
        assert len(out) > len(SAMPLE_PROMPT)


def test_chain_of_thought_mentions_step_by_step():
    out = ChainOfThought().augment(SAMPLE_PROMPT)
    assert "step by step" in out.lower()


def test_reflection_asks_for_critique():
    out = Reflection().augment(SAMPLE_PROMPT)
    assert "review your own output" in out.lower()


def test_template_imposes_structure():
    out = Template().augment(SAMPLE_PROMPT)
    for header in ("Problem Summary:", "Core Features:", "Success Metrics:"):
        assert header in out


def test_few_shot_provides_examples():
    out = FewShot().augment(SAMPLE_PROMPT)
    assert "Example A:" in out
    assert "Example B:" in out


def test_self_consistency_describes_internal_scoring():
    out = SelfConsistency().augment(SAMPLE_PROMPT)
    assert "Score each candidate" in out


def test_unknown_extras_are_ignored_when_template_has_no_placeholder():
    # augment should not crash when called with kwargs the template ignores.
    out = ChainOfThought().augment(SAMPLE_PROMPT)
    assert SAMPLE_PROMPT in out


def test_format_error_when_template_placeholder_missing():
    bad = Template()
    bad.template = "{user_prompt} -- {missing}"
    with pytest.raises(KeyError):
        bad.augment(SAMPLE_PROMPT)
