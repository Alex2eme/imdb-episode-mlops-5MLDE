import pytest

from imdb_mlops.config import PromotionConfig
from imdb_mlops.training.registry import decide_promotion

RULES = PromotionConfig(must_beat_baseline=True, max_rmse=1.0, min_improvement=0.01)


@pytest.mark.parametrize(
    ("rmse", "baseline", "champion", "expected"),
    [
        (0.70, 0.80, None, True),  # premier modèle conforme
        (1.20, 1.50, None, False),  # au-dessus du plafond absolu
        (0.85, 0.80, None, False),  # ne bat pas la baseline
        (0.70, 0.80, 0.75, True),  # améliore le champion
        (0.745, 0.80, 0.75, False),  # amélioration < min_improvement
        (0.80, 0.90, 0.75, False),  # régression
    ],
)
def test_decide_promotion(rmse, baseline, champion, expected):
    decision = decide_promotion(rmse, baseline, champion, RULES)
    assert decision.promote is expected
    assert decision.reason


def test_baseline_rule_can_be_disabled():
    rules = RULES.model_copy(update={"must_beat_baseline": False})
    assert decide_promotion(0.85, 0.80, None, rules).promote
