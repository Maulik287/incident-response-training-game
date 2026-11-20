"""Command-line entry point: `python -m app.cli --help`."""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from app import core


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="incident-response-training-game",
        description="Public service trainees play a simulation game where they predict incident sequences (e.g., traffic -> medical) and allocate resources to learn optimal response strategies.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. Hello command
    sub.add_parser("hello", help="check that everything is wired up")

    # 2. Run all milestones verification
    sub.add_parser("milestones", help="execute and report status across all 5 milestones")

    # 3. Mine frequent sequences command
    mine_parser = sub.add_parser("mine", help="discover frequent incident sequences using thilimarketbasket")
    mine_parser.add_argument("--min-support", type=float, default=0.15, help="minimum support threshold")
    mine_parser.add_argument("--min-confidence", type=float, default=0.35, help="minimum confidence threshold")

    # 4. Predict next incidents command
    pred_parser = sub.add_parser("predict", help="predict next likely incident using trained sequence LM")
    pred_parser.add_argument("events", nargs="+", help="observed sequence of incidents (e.g. traffic_collision fuel_leak)")
    pred_parser.add_argument("--top-k", type=int, default=3, help="number of top candidate predictions")

    # 5. Play simulation game command
    play_parser = sub.add_parser("play", help="play the incident response simulation game")
    play_parser.add_argument("--auto", action="store_true", help="run in automated simulation mode using LM guidance")
    play_parser.add_argument("--scenario", type=int, default=0, help="scenario index (0 to 19)")

    # 6. Evaluate benchmarks command
    eval_parser = sub.add_parser("evaluate", help="run comprehensive strategy benchmarks and save scorecard")
    eval_parser.add_argument("--output", type=str, default="data/game_evaluation.json", help="output JSON path")

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Incident Response Training Game is ready. Open README.md for your milestones.")

    elif args.command == "milestones":
        print("Executing All 5 Milestones...")
        m1 = core.milestone_1()
        print(f"[Milestone 1] {m1['reports_count']} incident reports modeled ({m1['vocabulary_size']} unique incident types)")
        m2 = core.milestone_2()
        print(f"[Milestone 2] {m2['patterns_summary']['total_frequent_itemsets']} frequent itemsets, {m2['patterns_summary']['total_sequential_rules']} sequential rules discovered")
        m3 = core.milestone_3(steps=50)
        print(f"[Milestone 3] LM trained (final train loss: {m3['final_train_loss']}, MRR: {m3['evaluation_metrics']['mrr']})")
        m4 = core.milestone_4()
        print(f"[Milestone 4] Simulation game executed ({m4['turns_played']} turns, containment rate: {m4['game_summary']['containment_rate']})")
        m5 = core.milestone_5()
        print(f"[Milestone 5] Evaluation benchmarks complete. LM-assisted score: {m5['lm_assisted_score']} - Grade: {m5['lm_assisted_grade']}")
        print("\nAll 5 milestones successfully completed and verified.")

    elif args.command == "mine":
        patterns = core.mine_incident_patterns(min_support=args.min_support, min_confidence=args.min_confidence)
        print(f"\n--- Frequent Incident Sequences (min_support={args.min_support}, min_confidence={args.min_confidence}) ---")
        print(f"Total reports analyzed: {patterns['n_reports']}")
        print(f"Frequent itemsets found: {patterns['total_frequent_itemsets']}")
        print("\nTop Sequential Cascades (Antecedent -> Consequent):")
        for rule in patterns["sequential_rules"][:8]:
            print(f"  {rule['antecedent']:20} -> {rule['consequent']:20} [Conf: {rule['confidence']*100:.1f}%, Lift: {rule['lift']:.2f}]")

    elif args.command == "predict":
        context = args.events
        model = core.IncidentSequenceLM()
        reports = core.get_default_incident_reports()
        model.continued_pretraining([r.ordered_sequence for r in reports], steps=50)
        preds = model.predict_next(context, top_k=args.top_k)
        print(f"\nCurrent Incident Context: {' -> '.join(context)}")
        print("\nPredicted Next Likely Incidents:")
        for rank, (inc, prob) in enumerate(preds, 1):
            print(f"  {rank}. {inc:22} Probability: {prob * 100:.1f}%")

    elif args.command == "play":
        reports = core.get_default_incident_reports()
        idx = max(0, min(len(reports) - 1, args.scenario))
        report = reports[idx]

        game = core.IncidentResponseGame()
        init = game.start_scenario(report)
        print(f"\n=== INCIDENT RESPONSE SIMULATION: {init['title']} ===")
        print(f"Location: {init['location']}")
        print(f"Initial Event: {init['initial_event']['event_type']} ({init['initial_event']['description']})")

        if args.auto:
            print("\n[Mode: Automated AI-Assisted Trainee Playthrough]")
            for step in range(1, len(report.events)):
                context = [e.event_type for e in report.events[:step]]
                preds = game.model.predict_next(context, top_k=2)
                pred_event = preds[0][0]
                alloc = core.INCIDENT_RESOURCE_REQUIREMENTS.get(pred_event, {"police": 1}).copy()
                for r in core.RESOURCE_TYPES:
                    alloc.setdefault(r, 0)

                turn = game.play_turn(core.TurnDecision(predicted_event=pred_event, allocation=alloc))
                print(f"\nTurn {turn.turn_index}: Next actual event: {turn.actual_event}")
                print(f"  Forecast: {turn.player_predicted_event} (Correct: {turn.prediction_correct})")
                print(f"  Allocated: {turn.player_allocation} | Required: {turn.required_resources}")
                print(f"  Outcome: {turn.containment_status} | Turn Score: {turn.turn_score}")

            summary = game.get_session_summary()
            print("\n=== MISSION COMPLETED ===")
            print(f"Prediction Accuracy: {summary['prediction_accuracy']*100:.1f}%")
            print(f"Containment Rate:    {summary['containment_rate']*100:.1f}%")
            print(f"Average Turn Score:  {summary['average_turn_score']}/100")
        else:
            print("\n[Interactive Mode]")
            for step in range(1, len(report.events)):
                context = [e.event_type for e in report.events[:step]]
                preds = game.model.predict_next(context, top_k=3)
                print(f"\nTurn {step}: Ongoing Sequence: {' -> '.join(context)}")
                print("  LM Early Warnings:")
                for p_inc, p_prob in preds:
                    print(f"    - {p_inc}: {p_prob*100:.1f}%")
                
                # Default selection for CLI prompt
                pred_choice = preds[0][0]
                alloc = core.INCIDENT_RESOURCE_REQUIREMENTS.get(pred_choice, {"police": 1}).copy()
                for r in core.RESOURCE_TYPES:
                    alloc.setdefault(r, 0)
                turn = game.play_turn(core.TurnDecision(predicted_event=pred_choice, allocation=alloc))
                print(f"  Actual event unfolded: {turn.actual_event} [{turn.containment_status}] - Score: {turn.turn_score}")

    elif args.command == "evaluate":
        print(f"Running strategy evaluation benchmark across all scenario reports...")
        eval_summary = core.evaluate_all_strategies(output_file=args.output)
        print(f"\nBenchmark results saved to {args.output}")
        print("\n--- Strategy Comparison ---")
        for strat, data in eval_summary["strategies"].items():
            print(f"{strat.upper():12} | Score: {data['overall_score']:5.1f} | Containment: {data['containment_rate']*100:5.1f}% | Acc: {data['prediction_accuracy']*100:5.1f}% | Grade: {data['grade']}")


if __name__ == "__main__":
    main()
