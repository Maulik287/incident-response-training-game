"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="incident-response-training-game", description="Public service trainees play a simulation game where they predict incident sequences (e.g., traffic -> medical) and allocate resources to learn optimal response strategies.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    # TODO: add your own commands here as you work through the milestones.
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Incident Response Training Game is ready. Open README.md for your milestones.")


if __name__ == "__main__":
    main()
