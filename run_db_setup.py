"""One-shot script to apply all new DB changes from the April 2026 audit.

Run from the project root:
    python run_db_setup.py

Steps:
  1. Create raw.mlb_elo table (if not exists)
  2. Create raw.mlb_statcast_batting + raw.mlb_statcast_pitching tables
  3. Apply new performance indexes
  4. Apply MLB016 Elo feature view
  5. Compute Elo ratings for all completed MLB games
  6. Fetch Statcast season leaderboards (backfill 2024 + current)
"""

import psycopg2
from pathlib import Path

DSN = "postgresql://josh:password@localhost:5432/nba"
SQL_DIR = Path(__file__).resolve().parent / "sql"


def main():
    conn = psycopg2.connect(DSN)
    conn.autocommit = True
    cur = conn.cursor()

    # 1. Create raw.mlb_elo table
    print("[1/6] Creating raw.mlb_elo table...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS raw.mlb_elo (
            game_slug    TEXT   NOT NULL,
            team_abbr    TEXT   NOT NULL,
            season       TEXT   NOT NULL,
            game_date_et DATE   NOT NULL,
            elo_pre      FLOAT  NOT NULL,
            elo_post     FLOAT  NOT NULL,
            PRIMARY KEY (game_slug, team_abbr)
        );
    """)
    print("       Done.")

    # 2. Create Statcast tables
    print("[2/6] Creating Statcast tables...")
    from mlb_pipeline.crawler_statcast import _DDL_BATTING, _DDL_PITCHING
    cur.execute(_DDL_BATTING)
    cur.execute(_DDL_PITCHING)
    print("       Done.")

    # 3. Apply new performance indexes
    print("[3/6] Applying performance indexes...")
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_mlb_games_venue_id ON raw.mlb_games (venue_id);",
        "CREATE INDEX IF NOT EXISTS idx_mlb_games_home_sp_id ON raw.mlb_games (home_sp_id) WHERE home_sp_id IS NOT NULL;",
        "CREATE INDEX IF NOT EXISTS idx_mlb_games_away_sp_id ON raw.mlb_games (away_sp_id) WHERE away_sp_id IS NOT NULL;",
        "CREATE INDEX IF NOT EXISTS idx_mlb_games_game_date_et ON raw.mlb_games (game_date_et);",
        "CREATE INDEX IF NOT EXISTS idx_mlb_games_status ON raw.mlb_games (status);",
        "CREATE INDEX IF NOT EXISTS idx_mlb_bs_player_stats_team ON raw.mlb_boxscore_player_stats (team_abbr);",
        "CREATE INDEX IF NOT EXISTS idx_mlb_player_gamelogs_team ON raw.mlb_player_gamelogs (team_abbr);",
        "CREATE INDEX IF NOT EXISTS idx_mlb_player_gamelogs_game_date ON raw.mlb_player_gamelogs (game_date_et);",
    ]
    for idx_sql in indexes:
        try:
            cur.execute(idx_sql)
        except Exception as e:
            print(f"       WARN: {e}")
    print(f"       Applied {len(indexes)} indexes.")

    # 4. Apply MLB016 Elo feature view
    print("[4/6] Applying MLB016 Elo feature view...")
    sql_path = SQL_DIR / "MLB016_mlb_elo_features.sql"
    try:
        cur.execute(sql_path.read_text(encoding="utf-8"))
        print("       Done.")
    except Exception as e:
        print(f"       ERROR: {e}")

    cur.close()
    conn.close()

    # 5. Compute Elo ratings
    print("[5/6] Computing MLB Elo ratings...")
    from mlb_pipeline.compute_elo import main as compute_elo_main
    compute_elo_main()

    # 6. Fetch Statcast data (backfill 2024 + current year)
    print("[6/6] Fetching Statcast leaderboards from Baseball Savant...")
    from mlb_pipeline.crawler_statcast import main as statcast_main
    import sys
    sys.argv = ["crawler_statcast", "--backfill"]
    statcast_main()

    print("\nAll done! You can now run: python -m mlb_pipeline.run_daily")


if __name__ == "__main__":
    main()
