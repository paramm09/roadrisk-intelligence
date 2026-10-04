"""Leakage audit -> balancing analysis -> baseline + LR + RF + HGB (tuned) -> selection -> final model."""
from _common import parser, run, settings_from

from ml.train import run_training


def main():
    cfg = settings_from(parser(__doc__).parse_args())
    run_training(cfg)
    return 0


if __name__ == "__main__":
    run(main)
