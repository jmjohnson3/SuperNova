"""Sport-agnostic sharp price-gap research: NBA props and game lines, NCAAF game lines.

The NFL cash strategy (nfl_pipeline.sharp_watch) showed that a soft book's price beating Pinnacle's
no-vig price is the edge that holds at the close. This package applies the same test to other sports
as research: it never stakes, and a sport only earns $1 stakes after its own decisions hold their
edge at the close. It shares no tables or registration with NFL, so it can never stop the NFL scanner.
"""
