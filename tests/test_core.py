"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

# Milestone 1: Model public incident reports as ordered sequences of events.
def test_milestone_1():
    from app.core import milestone_1, get_default_incident_reports, IncidentReport

    reports = get_default_incident_reports()
    assert len(reports) >= 20
    for r in reports:
        assert isinstance(r, IncidentReport)
        assert len(r.ordered_sequence) >= 3
        assert r.report_id.startswith("INC-")

    res = milestone_1()
    assert res["status"] == "completed"
    assert res["reports_count"] >= 20
    assert res["vocabulary_size"] >= 10
    assert "traffic_collision" in res["unique_incident_types"]
    assert "road_blockage" in res["unique_incident_types"]
    assert isinstance(res["sample_sequence"], list)
    assert res["average_sequence_length"] >= 3.0


# Milestone 2: Discover frequent incident sequences using `thilimarketbasket`.
def test_milestone_2():
    from app.core import milestone_2, mine_incident_patterns

    patterns = mine_incident_patterns(min_support=0.15, min_confidence=0.35)
    assert patterns["total_frequent_itemsets"] > 0
    assert patterns["total_sequential_rules"] > 0
    assert any("traffic_collision" in item["items"] for item in patterns["frequent_itemsets"])

    res = milestone_2()
    assert res["status"] == "completed"
    assert "patterns_summary" in res
    assert len(res["top_frequent_pairs"]) > 0
    assert len(res["top_sequential_cascades"]) > 0
    for casc in res["top_sequential_cascades"]:
        assert "antecedent" in casc
        assert "consequent" in casc
        assert casc["confidence"] >= 0.35
        assert casc["lift"] > 0.0


# Milestone 3: Train an LM to predict the next likely incident given a current situation.
def test_milestone_3():
    from app.core import milestone_3, IncidentSequenceLM

    lm = IncidentSequenceLM()
    history = lm.continued_pretraining(
        train_sequences=[
            ["traffic_collision", "fuel_leak", "vehicle_fire"],
            ["weather_hazard", "traffic_collision", "road_blockage"],
        ],
        steps=20,
        eval_every=10,
    )
    assert len(history) > 0
    assert lm.is_trained

    preds = lm.predict_next(["traffic_collision", "fuel_leak"], top_k=3)
    assert len(preds) == 3
    assert all(0.0 <= prob <= 1.0 for _, prob in preds)
    # Total sum of top predictions is <= 1.0
    assert sum(prob for _, prob in preds) <= 1.01

    res = milestone_3(steps=50)
    assert res["status"] == "completed"
    assert "evaluation_metrics" in res
    assert res["evaluation_metrics"]["mrr"] > 0.0
    assert len(res["sample_predictions"]) == 3


# Milestone 4: Implement a simple text-based game where the player makes resource allocation decisions based on predicted events.
def test_milestone_4():
    from app.core import milestone_4, IncidentResponseGame, TurnDecision, get_default_incident_reports

    reports = get_default_incident_reports()
    game = IncidentResponseGame()
    init_state = game.start_scenario(reports[0])
    assert "report_id" in init_state
    assert len(init_state["lm_early_warning"]) > 0

    # Execute a turn
    decision = TurnDecision(
        predicted_event="road_blockage",
        allocation={"police": 2, "traffic": 2, "fire": 0, "ambulance": 0},
    )
    turn_res = game.play_turn(decision)
    assert turn_res.turn_index == 1
    assert turn_res.containment_status in ["Fully Contained", "Partially Contained", "Escalated"]
    assert 0.0 <= turn_res.turn_score <= 100.0

    res = milestone_4()
    assert res["status"] == "completed"
    assert res["turns_played"] >= 3
    assert "game_summary" in res
    assert 0.0 <= res["game_summary"]["containment_rate"] <= 1.0


# Milestone 5: Evaluate player performance based on efficient resource use and accurate predictions.
def test_milestone_5(tmp_path):
    from app.core import milestone_5, evaluate_all_strategies, compute_performance_grade

    assert "S" in compute_performance_grade(95.0)
    assert "D" in compute_performance_grade(40.0)

    out_file = str(tmp_path / "test_eval.json")
    res = milestone_5(output_file=out_file)
    assert res["status"] == "completed"
    assert "benchmark_comparison" in res
    bench = res["benchmark_comparison"]
    assert "random" in bench
    assert "reactive" in bench
    assert "lm_assisted" in bench
    assert "oracle" in bench

    # LM-Assisted should significantly outperform Random Baseline in containment and score
    assert bench["lm_assisted"]["overall_score"] > bench["random"]["overall_score"]
    assert bench["lm_assisted"]["containment_rate"] > bench["random"]["containment_rate"]
    assert bench["oracle"]["containment_rate"] == 1.0

