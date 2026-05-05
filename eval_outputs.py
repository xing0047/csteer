"""
Top-level entrypoint for evaluating output JSON files.

This module is the preferred public path:
  python eval_outputs.py <subcommand> ...
"""

from judge.eval_suite import *  # re-export existing eval APIs
from judge.eval_suite import main


if __name__ == "__main__":
    main()
