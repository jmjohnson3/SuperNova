import logging
import json
from nba_pipeline.fetcher import MySportsFeedsClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

API_KEY = "4359aa1b-cc29-4647-a3e5-7314e2"

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
