"""One-time evaluation of the FINAL model on the untouched test split + error analysis + importance."""
from _common import parser, run, settings_from

from ml.evaluate import run_final_evaluation


def main():
    cfg = settings_from(parser(__doc__).parse_args())
    run_final_evaluation(cfg)
    return 0


if __name__ == "__main__":
    run(main)
