-- ============================================================================
-- V005: Odds Juice / Pricing Features
-- ============================================================================
-- Extracts odds pricing (juice/vig) data from odds.nba_game_lines that is
-- currently stored but unused by any feature view.
--
-- The raw table stores American odds prices alongside point values:
--   spread_home_price, spread_away_price   (e.g. -110, -115, +105)
--   total_over_price,  total_under_price
--
-- These prices encode bookmaker margin (vig) and, more importantly, reveal
-- WHERE the money is flowing. Key signals:
--   1. Juice asymmetry   -- When one side is -115 and the other -105, the
--      book is shading toward the -115 side (more liability there).
--   2. Juice movement    -- If the spread price moves from -110 to -120
--      without the point spread changing, sharp money hit that side.
--   3. Cross-book juice  -- Books that disagree on pricing reveal where
--      different sharp/retail flows are landing.
--
-- This view queries odds.nba_game_lines directly (not the open_close view)
-- to extract open/close prices per bookmaker, then aggregates to game level.
-- ============================================================================


CREATE OR REPLACE VIEW features.odds_juice_features AS
WITH
-- ---------------------------------------------------------------------------
-- Step 1: Find the OPEN (earliest) and CLOSE (latest) snapshot per game-book
-- ---------------------------------------------------------------------------
-- Each row in odds.nba_game_lines represents one snapshot from one bookmaker
-- for one game on one date. We rank by fetched_at_utc to find the first
-- (open) and last (close) snapshot.
-- ---------------------------------------------------------------------------
ranked AS (
    SELECT
        gl.as_of_date,
        gl.bookmaker_key,
        gl.home_team,
        gl.away_team,
        gl.spread_home_points,
        gl.spread_away_points,
        gl.total_points,
        gl.spread_home_price,
        gl.spread_away_price,
        gl.total_over_price,
        gl.total_under_price,
        ROW_NUMBER() OVER (
            PARTITION BY gl.as_of_date, gl.bookmaker_key, gl.home_team, gl.away_team
            ORDER BY gl.fetched_at_utc ASC
        ) AS rn_open,
        ROW_NUMBER() OVER (
            PARTITION BY gl.as_of_date, gl.bookmaker_key, gl.home_team, gl.away_team
            ORDER BY gl.fetched_at_utc DESC
        ) AS rn_close
    FROM odds.nba_game_lines gl
    WHERE gl.spread_home_price IS NOT NULL
       OR gl.total_over_price IS NOT NULL
),

-- Closing prices per book-game (latest snapshot)
close_prices AS (
    SELECT
        as_of_date,
        bookmaker_key,
        home_team,
        away_team,
        spread_home_points,
        spread_away_points,
        total_points,
        spread_home_price,
        spread_away_price,
        total_over_price,
        total_under_price
    FROM ranked
    WHERE rn_close = 1
),

-- Opening prices per book-game (earliest snapshot)
open_prices AS (
    SELECT
        as_of_date,
        bookmaker_key,
        home_team,
        away_team,
        spread_home_price   AS open_spread_home_price,
        spread_away_price   AS open_spread_away_price,
        total_over_price    AS open_total_over_price,
        total_under_price   AS open_total_under_price
    FROM ranked
    WHERE rn_open = 1
),

-- ---------------------------------------------------------------------------
-- Step 2: Join open and close prices per book-game
-- ---------------------------------------------------------------------------
book_game AS (
    SELECT
        c.as_of_date,
        c.bookmaker_key,
        c.home_team,
        c.away_team,
        -- Closing prices
        c.spread_home_price,
        c.spread_away_price,
        c.total_over_price,
        c.total_under_price,
        -- Opening prices
        o.open_spread_home_price,
        o.open_spread_away_price,
        o.open_total_over_price,
        o.open_total_under_price,
        -- Juice movement (close - open); a move to more negative = sharps hit that side
        c.spread_home_price - o.open_spread_home_price   AS spread_home_price_move,
        c.spread_away_price - o.open_spread_away_price   AS spread_away_price_move,
        c.total_over_price  - o.open_total_over_price    AS total_over_price_move,
        c.total_under_price - o.open_total_under_price   AS total_under_price_move,
        -- Convert American odds to implied probability for consensus calculations
        -- Negative odds: implied_prob = |price| / (|price| + 100)
        -- Positive odds: implied_prob = 100 / (price + 100)
        CASE
            WHEN c.spread_home_price < 0
                THEN ABS(c.spread_home_price)::numeric / (ABS(c.spread_home_price)::numeric + 100)
            WHEN c.spread_home_price > 0
                THEN 100.0 / (c.spread_home_price::numeric + 100)
            ELSE NULL
        END AS spread_home_implied_prob,
        CASE
            WHEN c.spread_away_price < 0
                THEN ABS(c.spread_away_price)::numeric / (ABS(c.spread_away_price)::numeric + 100)
            WHEN c.spread_away_price > 0
                THEN 100.0 / (c.spread_away_price::numeric + 100)
            ELSE NULL
        END AS spread_away_implied_prob,
        CASE
            WHEN c.total_over_price < 0
                THEN ABS(c.total_over_price)::numeric / (ABS(c.total_over_price)::numeric + 100)
            WHEN c.total_over_price > 0
                THEN 100.0 / (c.total_over_price::numeric + 100)
            ELSE NULL
        END AS total_over_implied_prob,
        CASE
            WHEN c.total_under_price < 0
                THEN ABS(c.total_under_price)::numeric / (ABS(c.total_under_price)::numeric + 100)
            WHEN c.total_under_price > 0
                THEN 100.0 / (c.total_under_price::numeric + 100)
            ELSE NULL
        END AS total_under_implied_prob
    FROM close_prices c
    LEFT JOIN open_prices o
      ON  o.as_of_date    = c.as_of_date
      AND o.bookmaker_key = c.bookmaker_key
      AND o.home_team     = c.home_team
      AND o.away_team     = c.away_team
),

-- ---------------------------------------------------------------------------
-- Step 3: DraftKings-specific juice features
-- ---------------------------------------------------------------------------
dk AS (
    SELECT
        as_of_date,
        home_team,
        away_team,
        spread_home_price           AS dk_spread_home_juice,
        spread_away_price           AS dk_spread_away_juice,
        total_over_price            AS dk_total_over_juice,
        total_under_price           AS dk_total_under_juice,
        spread_home_price_move      AS dk_spread_juice_move,
        total_over_price_move       AS dk_total_over_juice_move,
        spread_home_implied_prob    AS dk_spread_home_implied_prob,
        total_over_implied_prob     AS dk_total_over_implied_prob
    FROM book_game
    WHERE bookmaker_key = 'draftkings'
),

-- ---------------------------------------------------------------------------
-- Step 4: Cross-book consensus on juice (all books)
-- ---------------------------------------------------------------------------
-- Average the raw American odds AND the implied probabilities across books.
-- Std dev of raw prices shows how much books disagree on pricing.
-- ---------------------------------------------------------------------------
consensus AS (
    SELECT
        as_of_date,
        home_team,
        away_team,
        -- Raw American odds averages
        AVG(spread_home_price)::numeric          AS avg_spread_home_juice,
        AVG(spread_away_price)::numeric          AS avg_spread_away_juice,
        AVG(total_over_price)::numeric           AS avg_total_over_juice,
        AVG(total_under_price)::numeric          AS avg_total_under_juice,
        -- Juice skew: difference between sides indicates where action is landing
        -- More negative home juice vs away juice => more money on home spread
        AVG(spread_home_price)::numeric - AVG(spread_away_price)::numeric
            AS spread_juice_skew,
        AVG(total_over_price)::numeric  - AVG(total_under_price)::numeric
            AS total_juice_skew,
        -- Implied probability consensus (averaged across books, removes vig impact)
        AVG(spread_home_implied_prob)            AS spread_home_implied_prob,
        AVG(spread_away_implied_prob)            AS spread_away_implied_prob,
        AVG(total_over_implied_prob)             AS total_over_implied_prob,
        AVG(total_under_implied_prob)            AS total_under_implied_prob,
        -- Std dev of raw prices shows book disagreement on pricing
        STDDEV_SAMP(spread_home_price::numeric)  AS spread_home_juice_std,
        STDDEV_SAMP(total_over_price::numeric)   AS total_over_juice_std,
        -- How many books have price data
        COUNT(DISTINCT bookmaker_key)            AS juice_book_count
    FROM book_game
    WHERE spread_home_price IS NOT NULL
       OR total_over_price IS NOT NULL
    GROUP BY as_of_date, home_team, away_team
)

-- ---------------------------------------------------------------------------
-- Step 5: Final output - one row per game-date
-- ---------------------------------------------------------------------------
SELECT
    -- Game identifiers
    COALESCE(dk.as_of_date, c.as_of_date)     AS as_of_date,
    COALESCE(dk.home_team, c.home_team)        AS home_team_abbr,
    COALESCE(dk.away_team, c.away_team)        AS away_team_abbr,

    -- DraftKings closing juice (raw American odds)
    dk.dk_spread_home_juice,
    dk.dk_spread_away_juice,
    dk.dk_total_over_juice,
    dk.dk_total_under_juice,

    -- DraftKings juice movement (close - open price)
    -- Negative movement = price got more negative = sharps hit that side
    -- even when the point spread itself did not move
    dk.dk_spread_juice_move,
    dk.dk_total_over_juice_move,

    -- Cross-book average juice (raw American odds)
    ROUND(c.avg_spread_home_juice, 1)          AS avg_spread_home_juice,
    ROUND(c.avg_total_over_juice, 1)           AS avg_total_over_juice,

    -- Juice skew: negative => more action on home spread / over
    -- (home price is more negative than away price on average)
    ROUND(c.spread_juice_skew, 1)              AS spread_juice_skew,
    ROUND(c.total_juice_skew, 1)               AS total_juice_skew,

    -- Cross-book disagreement on pricing
    ROUND(c.spread_home_juice_std, 2)          AS spread_home_juice_std,
    ROUND(c.total_over_juice_std, 2)           AS total_over_juice_std,

    -- How many books contributed price data for this game
    c.juice_book_count,

    -- Consensus implied probabilities (averaged across all books)
    -- These remove the vig component and give a cleaner market signal
    ROUND(c.spread_home_implied_prob, 4)       AS spread_home_implied_prob,
    ROUND(c.total_over_implied_prob, 4)        AS total_over_implied_prob

FROM consensus c
LEFT JOIN dk
  ON  dk.as_of_date = c.as_of_date
  AND dk.home_team  = c.home_team
  AND dk.away_team  = c.away_team;
