from docbench.evaluators.metrics import (
    contains_answer,
    exact_match,
    json_exact,
    numeric_tolerance,
    score,
    token_f1,
)


def test_exact_match_normalized():
    assert exact_match("Austin, Texas!", "austin texas") == 1.0
    assert exact_match("foo", "bar") == 0.0


def test_contains():
    assert contains_answer("The answer is INV-2048 on the form", "INV-2048") == 1.0
    assert contains_answer("nothing here", "INV-2048") == 0.0


def test_token_f1():
    assert token_f1("the cat sat", "the cat sat") == 1.0
    assert 0.0 < token_f1("the cat sat", "the dog sat") < 1.0


def test_json_exact():
    assert json_exact('{"a": 1, "b": 2}', '{"b": 2, "a": 1}') == 1.0
    assert json_exact("not json", '{"a": 1}') == 0.0


def test_numeric_tolerance():
    assert numeric_tolerance("about 100.0", "100") == 1.0
    assert numeric_tolerance("50", "100") == 0.0


def test_score_bundle():
    out = score("East", "East", ["exact_match", "contains"])
    assert out == {"exact_match": 1.0, "contains": 1.0}