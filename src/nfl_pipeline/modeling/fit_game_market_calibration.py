"""Forward-only game CDF calibration, never leave-one-week-out fitting.

Missing locked distributions cannot be replaced with today's sigma or quotes.
"""
import argparse
from nfl_pipeline.modeling.validate_distribution_policy import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Unsupported legacy installation flag')
    args = parser.parse_args()
    if args.write:
        parser.error('Use the versioned prospective distribution policy review, not --write')
    run(('spread', 'total', 'game_outcome'))


if __name__ == '__main__':
    main()
