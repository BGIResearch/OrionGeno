import logging
import os
import warnings

os.environ.setdefault("NUMEXPR_MAX_THREADS", "16")
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in log",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in log",
    category=RuntimeWarning,
)

from oriongeno.parse_args import parse_cmd
from oriongeno.pipeline import run_oriongeno

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def main():
    """Parse CLI arguments and run the inference pipeline."""
    args = parse_cmd()
    if args.command == "pipeline":
        run_oriongeno(args)
    else:
        raise ValueError("Unknown command {}".format(args.command))


if __name__ == "__main__":
    main()
