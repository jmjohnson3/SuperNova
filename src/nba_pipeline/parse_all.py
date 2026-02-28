import logging
import psycopg2

from nba_pipeline.parse_games import main as parse_games
from nba_pipeline.parse_meta import main as parse_meta
from nba_pipeline.parse_player_gamelogs import main as parse_player_gamelogs
from nba_pipeline.parse_lineup import main as parse_lineup
from nba_pipeline.parse_boxscore import main as parse_boxscore
from nba_pipeline.parse_pbp import main as parse_pbp
from nba_pipeline.parse_referees import main as parse_referees
from nba_pipeline.parse_oddsapi import parse_prop_odds

log = logging.getLogger("nba_pipeline.parse_all")

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    log.info("Starting FULL parse pipeline")

    # Order matters a little (dimensions first)
    parse_meta()              # venues, teams, standings, injuries
    parse_games()             # nba_games
    parse_player_gamelogs()   # training backbone
    parse_lineup()            # availability / starters
    parse_boxscore()          # game + player boxscores
    parse_pbp()               # advanced features
    parse_referees()          # referee assignments from boxscore payloads
    parse_prop_odds()         # prop lines from odds API

    log.info("ALL PARSERS COMPLETE")

if __name__ == "__main__":
    main()
