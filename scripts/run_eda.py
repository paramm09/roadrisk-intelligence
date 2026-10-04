"""Compute reports/eda_summary.json from the processed data (train+validation rows only)."""
from _common import parser, run, settings_from

from ml.eda import run_eda
from ml.utils import load_frame


def main():
    cfg = settings_from(parser(__doc__).parse_args())
    run_eda(load_frame(cfg.processed_base), cfg)
    return 0


if __name__ == "__main__":
    run(main)
