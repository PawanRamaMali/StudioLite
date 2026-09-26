"""Unit tests for filmmaker/stages.py - the pipeline stage registry,
its dependency helpers, and the QualityReport dataclass."""
from __future__ import annotations

import pytest

from filmmaker import stages


def test_stage_keys_are_unique():
    """The pipeline is index-driven; duplicate keys would break
    downstream_of() and staleness propagation."""
    keys = [s.key for s in stages.STAGES]
    assert len(keys) == len(set(keys)), f"duplicate stage keys: {keys}"


def test_stage_index_roundtrips_every_stage():
    for i, spec in enumerate(stages.STAGES):
        assert stages.stage_index(spec.key) == i


def test_stage_index_raises_on_unknown():
    with pytest.raises(KeyError):
        stages.stage_index("not-a-real-stage")


def test_downstream_of_returns_later_stages_only():
    """downstream_of(producer) must include screenwriter and every stage
    after it, but never producer itself."""
    down = stages.downstream_of("producer")
    assert "producer" not in down
    assert "screenwriter" in down
    assert down[0] == "screenwriter"


def test_downstream_of_last_stage_is_empty():
    last = stages.STAGES[-1].key
    assert stages.downstream_of(last) == []


def test_stage_keys_match_stage_keys_literal_and_order():
    """STAGE_KEYS is the runtime list; the Literal-typed StageKey is only
    a type hint. Both must reflect the same set."""
    assert stages.STAGE_KEYS == [s.key for s in stages.STAGES]


class TestQualityReport:
    def test_score_above_threshold_accepts(self):
        r = stages.QualityReport(score=0.85)
        assert r.accepted is True

    def test_score_below_threshold_rejects(self):
        r = stages.QualityReport(score=0.4)
        assert r.accepted is False

    def test_custom_threshold_flips_decision(self):
        r = stages.QualityReport(score=0.5, accept_threshold=0.6)
        assert r.accepted is False
        r2 = stages.QualityReport(score=0.5, accept_threshold=0.4)
        assert r2.accepted is True

    def test_hints_default_empty_list_is_not_shared(self):
        """Regression: dataclass default_factory prevents the classic
        mutable-default bug where every instance shares one list."""
        a = stages.QualityReport(score=1.0)
        b = stages.QualityReport(score=1.0)
        a.hints.append("something")
        assert b.hints == []
