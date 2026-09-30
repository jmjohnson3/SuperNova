import logging
from nba_pipeline.fetcher import MySportsFeedsClient
from supernovabets_config import mysportsfeeds_api_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

API_KEY = mysportsfeeds_api_key()
if not API_KEY:
    raise RuntimeError("Missing MYSPORTSFEEDS_API_KEY or MSF_API_KEY")

client = MySportsFeedsClient(api_key=API_KEY)

urls = [
    "https://api.mysportsfeeds.com/v2.1/pull/nba/2025-2026-regular/games.json",
    "https://api.mysportsfeeds.com/v2.1/pull/nba/2025-2026-regular/date/20251022/games.json",
    "https://api.mysportsfeeds.com/v2.1/pull/nba/2025-2026-regular/games/20251022-WAS-MIL/boxscore.json",
    "https://api.mysportsfeeds.com/v2.1/pull/nba/2025-2026-regular/games/20251022-WAS-MIL/playbyplay.json",
    "https://api.mysportsfeeds.com/v2.1/pull/nba/2025-2026-regular/date/20251022/player_gamelogs.json?team=det",
]

for url in urls:
    data = client.fetch_json(url)
    print(url, "keys:", list(data.keys()))
