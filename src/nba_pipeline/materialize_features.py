"""
Refresh materialized game feature views after parse_all.

This is run as a separate step in the daily pipeline (after Parse + Load)
so that it doesn't contribute to parse_all's 1800 s timeout.

Typical runtime: ~130 s (90 s for game_training_features_mat,
40 s for game_prediction_features_mat).
"""
import logging

from nba_pipeline.parse_all import _materialize_game_features, _PG_DSN


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _materialize_game_features(_PG_DSN)


if __name__ == "__main__":
    main()
