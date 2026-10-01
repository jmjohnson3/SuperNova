from datetime import date
from pathlib import Path
import re

from nfl_pipeline.integrity import nfl_season


def test_playoff_dates_belong_to_the_previous_calendar_season():
    assert nfl_season(date(2026, 9, 10)) == 2026
    assert nfl_season(date(2027, 1, 17)) == 2026
    assert nfl_season(date(2027, 2, 14)) == 2026
    assert nfl_season(date(2027, 3, 1)) == 2027


def test_no_live_code_derives_the_season_from_the_calendar_year():
    root = Path(__file__).resolve().parent
    offenders = []
    for path in root.rglob("*.py"):
        if "models" in path.parts or path.name.startswith("test_"):
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"season.*\.year\b|\.year\b.*season|now\(\)\.year", line) and "nfl_season" not in line:
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, "\n".join(offenders)
