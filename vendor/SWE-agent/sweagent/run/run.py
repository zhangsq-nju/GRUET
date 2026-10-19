import argparse
import sys


def main(args: list[str] | None = None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", choices=["run", "r", "run-batch", "b"])
    parsed, remaining = parser.parse_known_args(sys.argv[1:] if args is None else args)
    if parsed.command in ("run-batch", "b"):
        from sweagent.run.run_batch import run_from_cli
    else:
        from sweagent.run.run_single import run_from_cli
    return run_from_cli(remaining)


if __name__ == "__main__":
    sys.exit(main())
