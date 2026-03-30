"""Compatibility CLI entry point shim."""

import sys

from babappai.cli import main as _main


def main(argv=None):
    print(
        "[DEPRECATION] 'babappaomega' CLI is a compatibility alias. Use 'babappai'. "
        "BABAPPAΩ remains the canonical frozen inference model used by BABAPPAi.",
        file=sys.stderr,
    )
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
