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
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Implement a simple text-based game where the player makes resource allocation decisions based on predicted events.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Evaluate player performance based on efficient resource use and accurate predictions.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

