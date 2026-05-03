"""Allow ``python -m luki.eval`` to run the CLI."""

from luki.eval.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
