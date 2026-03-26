-- ============================================================================
-- V009: Standings Detail Features
-- ============================================================================
-- Provides POINT-IN-TIME standings data for each team-game.
--
-- Key fixes (2026-03-25):
--   • streak_signed / streak_length: computed from raw.nba_games results via
--     gap-and-island technique (API JSON never contained this data)
--   • last10_win_pct / last10_wins / last10_losses: computed from last 10
--     completed games before each game (same source)
--   • conference_rank: COALESCEd from snapshot column → JSON fallback →
--     dynamic ranking by running win% within conference (covers all dates,
--     not just the Jan/Feb 2026 snapshot window)
-- ============================================================================

CREATE OR REPLACE VIEW features.team_standings_detail AS
WITH spine AS (
    SELECT season, team_abbr, game_slug, game_date_et, start_ts_utc
    FROM features.team_game_spine
),

-- ── Team-to-conference mapping (stable; derived from standings snapshots) ──
team_conf AS (
    SELECT DISTINCT UPPER(team_abbr) AS team_abbr,
           COALESCE(
               stats->'conferenceRank'->>'conferenceName',
               conference_name
           ) AS conference_name
    FROM raw.nba_standings
    WHERE COALESCE(stats->'conferenceRank'->>'conferenceName', conference_name) IS NOT NULL
),

-- ── All completed team-games, numbered chronologically per team ────────────
completed_games AS (
    SELECT
        team_abbr,
        game_slug,
        game_date_et,
        start_ts_utc,
        (points_for > points_against) AS won,
        ROW_NUMBER() OVER (
            PARTITION BY team_abbr
            ORDER BY game_date_et, start_ts_utc
        ) AS game_num
    FROM features.team_game_spine
    WHERE points_for IS NOT NULL
),

-- ── Gap-and-island streak identification ──────────────────────────────────
streak_islands AS (
    SELECT
        team_abbr,
        game_num,
        won,
        game_num - ROW_NUMBER() OVER (
            PARTITION BY team_abbr, won
            ORDER BY game_num
        ) AS island_id
    FROM completed_games
),

island_summary AS (
    SELECT
        team_abbr,
        island_id,
        won,
        MIN(game_num) AS island_start,
        MAX(game_num) AS island_end,
        COUNT(*)      AS island_len
    FROM streak_islands
    GROUP BY team_abbr, island_id, won
),

-- ── Per-spine-row context: game_num of the last completed game before it ───
spine_context AS (
    SELECT
        sp.team_abbr,
        sp.game_slug,
        MAX(cg.game_num) AS prev_game_num
    FROM spine sp
    LEFT JOIN completed_games cg
      ON cg.team_abbr    = sp.team_abbr
     AND cg.game_date_et < sp.game_date_et
    GROUP BY sp.team_abbr, sp.game_slug
),

-- ── Streak signal going into each game ────────────────────────────────────
streak_data AS (
    SELECT
        sc.game_slug,
        sc.team_abbr,
        -- Streak = games in same run UP TO prev_game_num (not full island length)
        CASE
            WHEN sc.prev_game_num IS NULL THEN 0
            WHEN il.won                   THEN  (sc.prev_game_num - il.island_start + 1)::int
            WHEN NOT il.won               THEN -(sc.prev_game_num - il.island_start + 1)::int
            ELSE 0
        END AS streak_signed_comp,
        COALESCE((sc.prev_game_num - il.island_start + 1)::int, 0) AS streak_length_comp
    FROM spine_context sc
    LEFT JOIN island_summary il
      ON il.team_abbr            = sc.team_abbr
     AND sc.prev_game_num BETWEEN il.island_start AND il.island_end
),

-- ── Last-10 win% / record ─────────────────────────────────────────────────
last10_data AS (
    SELECT
        sc.game_slug,
        sc.team_abbr,
        CASE WHEN COUNT(cg.game_num) >= 1
             THEN COUNT(*) FILTER (WHERE cg.won)::numeric / COUNT(cg.game_num)
             ELSE NULL
        END                                            AS last10_win_pct_comp,
        COUNT(*) FILTER (WHERE     cg.won)::int        AS last10_wins_comp,
        COUNT(*) FILTER (WHERE NOT cg.won)::int        AS last10_losses_comp
    FROM spine_context sc
    LEFT JOIN completed_games cg
      ON cg.team_abbr  = sc.team_abbr
     AND cg.game_num BETWEEN sc.prev_game_num - 9 AND sc.prev_game_num
    GROUP BY sc.game_slug, sc.team_abbr
),

-- ── Dynamic conference rank (set-based, no LATERAL) ───────────────────────
-- Step 1: Running win% for each team at each game they play
all_team_records AS (
    SELECT
        team_abbr,
        game_date_et,
        CASE
            WHEN COALESCE(cnt, 0) > 0
            THEN COALESCE(wins_before, 0)::numeric / cnt
            ELSE 0
        END AS wp_before
    FROM (
        SELECT
            team_abbr,
            game_date_et,
            COALESCE(SUM(CASE WHEN points_for > points_against THEN 1 ELSE 0 END)
                OVER (PARTITION BY team_abbr
                      ORDER BY game_date_et, start_ts_utc
                      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS wins_before,
            COALESCE(COUNT(points_for)
                OVER (PARTITION BY team_abbr
                      ORDER BY game_date_et, start_ts_utc
                      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS cnt
        FROM features.team_game_spine
    ) sub
),

-- Step 2: All unique game dates (to rank at)
all_dates AS (
    SELECT DISTINCT game_date_et FROM features.team_game_spine
),

-- Step 3: For each (date, team), get the most recent win% on or before that date
-- DISTINCT ON carries forward the last known win% to dates when a team doesn't play
team_records_for_ranking AS (
    SELECT DISTINCT ON (d.game_date_et, tc.team_abbr)
        d.game_date_et,
        tc.team_abbr,
        tc.conference_name,
        COALESCE(atr.wp_before, 0) AS wp
    FROM all_dates d
    CROSS JOIN team_conf tc
    LEFT JOIN all_team_records atr
      ON atr.team_abbr    = tc.team_abbr
     AND atr.game_date_et <= d.game_date_et
    ORDER BY d.game_date_et, tc.team_abbr, atr.game_date_et DESC NULLS LAST
),

-- Step 4: Rank within conference at each date
conf_rank_at_date AS (
    SELECT
        team_abbr,
        game_date_et,
        DENSE_RANK() OVER (
            PARTITION BY game_date_et, conference_name
            ORDER BY wp DESC
        )::int AS dynamic_conf_rank
    FROM team_records_for_ranking
),

-- ── Standings point-in-time (snapshot join — covers Feb 2026+ only) ───────
standings_pit AS (
    SELECT DISTINCT ON (sp.season, sp.team_abbr, sp.game_slug)
        sp.season,
        sp.team_abbr,
        sp.game_slug,
        sp.game_date_et,

        s.wins,
        s.losses,
        s.win_pct,
        COALESCE(s.conference_rank,
                 (s.stats->'conferenceRank'->>'rank')::int)       AS conference_rank,
        s.division_rank,
        s.playoff_rank,
        s.games_back,
        s.overall_rank,
        COALESCE(s.conference_name,
                 s.stats->'conferenceRank'->>'conferenceName')    AS conference_name,
        s.division_name,

        COALESCE(
            NULLIF(s.stats->'standings'->>'homeWins', '')::int,
            NULLIF(s.stats->'stats'->'standings'->>'homeWins', '')::int,
            NULLIF(s.stats->>'homeWins', '')::int
        ) AS home_wins,
        COALESCE(
            NULLIF(s.stats->'standings'->>'homeLosses', '')::int,
            NULLIF(s.stats->'stats'->'standings'->>'homeLosses', '')::int,
            NULLIF(s.stats->>'homeLosses', '')::int
        ) AS home_losses,
        COALESCE(
            NULLIF(s.stats->'standings'->>'awayWins', '')::int,
            NULLIF(s.stats->'stats'->'standings'->>'awayWins', '')::int,
            NULLIF(s.stats->>'awayWins', '')::int
        ) AS away_wins,
        COALESCE(
            NULLIF(s.stats->'standings'->>'awayLosses', '')::int,
            NULLIF(s.stats->'stats'->'standings'->>'awayLosses', '')::int,
            NULLIF(s.stats->>'awayLosses', '')::int
        ) AS away_losses

    FROM spine sp
    JOIN raw.nba_standings s
      ON s.season = sp.season
     AND UPPER(s.team_abbr) = sp.team_abbr
     AND s.source_fetched_at_utc::date <= sp.game_date_et
    ORDER BY sp.season, sp.team_abbr, sp.game_slug,
             s.source_fetched_at_utc DESC
)

SELECT
    sp.season,
    sp.team_abbr,
    sp.game_slug,
    sp.game_date_et,

    -- Core standings from snapshot (point-in-time; NULL for pre-snapshot dates)
    spit.win_pct,

    -- Conference rank: snapshot → JSON fix → dynamic running-win%-rank
    COALESCE(
        spit.conference_rank,
        crd.dynamic_conf_rank
    ) AS conference_rank,

    spit.division_rank,
    spit.playoff_rank,
    spit.games_back,
    spit.overall_rank,

    -- Conference flag
    CASE WHEN UPPER(COALESCE(spit.conference_name, tc.conference_name)) LIKE '%EAST%'
         THEN TRUE ELSE FALSE
    END AS is_eastern_conf,

    spit.division_name,

    -- Home / away record (from snapshot; usually NULL — no homeWins in MSF JSON)
    spit.home_wins,
    spit.home_losses,
    CASE WHEN COALESCE(spit.home_wins, 0) + COALESCE(spit.home_losses, 0) > 0
         THEN spit.home_wins::numeric / (spit.home_wins + spit.home_losses)
         ELSE NULL
    END AS home_win_pct,

    spit.away_wins,
    spit.away_losses,
    CASE WHEN COALESCE(spit.away_wins, 0) + COALESCE(spit.away_losses, 0) > 0
         THEN spit.away_wins::numeric / (spit.away_wins + spit.away_losses)
         ELSE NULL
    END AS away_win_pct,

    CASE WHEN COALESCE(spit.home_wins, 0) + COALESCE(spit.home_losses, 0) > 0
          AND COALESCE(spit.away_wins, 0) + COALESCE(spit.away_losses, 0) > 0
         THEN spit.home_wins::numeric / (spit.home_wins + spit.home_losses)
            - spit.away_wins::numeric / (spit.away_wins + spit.away_losses)
         ELSE NULL
    END AS home_away_split,

    -- Streak computed from game results (was always 0 before; API JSON has no streak data)
    COALESCE(sd.streak_signed_comp, 0) AS streak_signed,
    COALESCE(sd.streak_length_comp, 0) AS streak_length,

    -- Last-10 computed from game results (was always NULL before)
    COALESCE(l10.last10_wins_comp,   0) AS last10_wins,
    COALESCE(l10.last10_losses_comp, 0) AS last10_losses,
    l10.last10_win_pct_comp              AS last10_win_pct

FROM spine sp
LEFT JOIN standings_pit spit
    ON spit.season    = sp.season
   AND spit.team_abbr = sp.team_abbr
   AND spit.game_slug = sp.game_slug
LEFT JOIN streak_data sd
    ON sd.game_slug  = sp.game_slug
   AND sd.team_abbr  = sp.team_abbr
LEFT JOIN last10_data l10
    ON l10.game_slug = sp.game_slug
   AND l10.team_abbr = sp.team_abbr
LEFT JOIN team_conf tc
    ON tc.team_abbr = sp.team_abbr
LEFT JOIN conf_rank_at_date crd
    ON crd.team_abbr    = sp.team_abbr
   AND crd.game_date_et = sp.game_date_et;
