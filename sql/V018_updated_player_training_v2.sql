-- ============================================================================
-- V018: Player Training Features V2
-- ============================================================================
-- Replaces V015's player_training_features by adding:
--   1. teammate_pts_out / teammate_out_count — pts lost to teammate injuries
--      (joined from features.game_training_features which already computes this)
--   2. opp_pts/reb/ast_allowed_role_10 — opponent position defense
--      (joined from features.opp_position_defense, V016)
-- ============================================================================

CREATE OR REPLACE VIEW features.player_training_features AS
WITH base AS (
    SELECT
        pg.season,
        pg.game_slug,
        pg.player_id,
        UPPER(pg.team_abbr)     AS team_abbr,
        UPPER(pg.opponent_abbr) AS opponent_abbr,
        pg.is_home,
        pg.start_ts_utc,
        pg.minutes,
        pg.points::numeric   AS points,
        pg.rebounds::numeric AS rebounds,
        pg.assists::numeric  AS assists,
        pg.threes_made::numeric AS threes_made,
        pg.fga::numeric AS fga,
        pg.fta::numeric AS fta
    FROM raw.nba_player_gamelogs pg
    WHERE pg.minutes IS NOT NULL AND pg.minutes > 0
),
final_games AS (
    SELECT
        g.season,
        g.game_slug,
        g.game_date_et,
        g.start_ts_utc,
        g.home_team_abbr,
        g.away_team_abbr
    FROM raw.nba_games g
    WHERE g.status = 'final'
),
joined AS (
    SELECT
        b.season, b.game_slug, b.player_id,
        b.team_abbr, b.opponent_abbr, b.is_home, b.start_ts_utc,
        b.minutes, b.points, b.rebounds, b.assists,
        b.threes_made, b.fga, b.fta,
        g.game_date_et,
        g.start_ts_utc AS game_start_ts_utc,
        g.home_team_abbr,
        g.away_team_abbr
    FROM base b
    JOIN final_games g
      ON g.season = b.season AND g.game_slug = b.game_slug
),
roll AS (
    SELECT
        j.*,
        AVG(j.minutes)  OVER w10 AS min_avg_10,
        AVG(j.minutes)  OVER w5  AS min_avg_5,
        COUNT(*)        OVER w10 AS n_games_prev_10,

        AVG(j.points)   OVER w10 AS pts_avg_10,
        AVG(j.points)   OVER w5  AS pts_avg_5,
        AVG(j.rebounds) OVER w10 AS reb_avg_10,
        AVG(j.rebounds) OVER w5  AS reb_avg_5,
        AVG(j.assists)  OVER w10 AS ast_avg_10,
        AVG(j.assists)  OVER w5  AS ast_avg_5,

        STDDEV_SAMP(j.points)   OVER w10 AS pts_sd_10,
        STDDEV_SAMP(j.rebounds) OVER w10 AS reb_sd_10,
        STDDEV_SAMP(j.assists)  OVER w10 AS ast_sd_10,

        AVG(j.fga)         OVER w10 AS fga_avg_10,
        AVG(j.fta)         OVER w10 AS fta_avg_10,
        AVG(j.threes_made) OVER w10 AS threes_avg_10,

        EXTRACT(EPOCH FROM (
            j.game_start_ts_utc -
            LAG(j.game_start_ts_utc) OVER (PARTITION BY j.player_id ORDER BY j.game_start_ts_utc)
        )) / 86400.0 AS player_rest_days,

        CASE WHEN j.minutes > 0
             THEN (j.fga + 0.44 * j.fta) / j.minutes
             ELSE NULL
        END AS usage_proxy_per_min

    FROM joined j
    WINDOW
        w10 AS (PARTITION BY j.player_id ORDER BY j.game_start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
        w5  AS (PARTITION BY j.player_id ORDER BY j.game_start_ts_utc ROWS BETWEEN  5 PRECEDING AND 1 PRECEDING)
),
enriched AS (
    SELECT
        r.*,
        AVG(r.usage_proxy_per_min) OVER (
            PARTITION BY r.player_id ORDER BY r.game_start_ts_utc
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
        ) AS usage_proxy_avg_10,

        oar.def_rtg_avg_10  AS opp_def_rtg_10,
        oar.off_rtg_avg_10  AS opp_off_rtg_10,

        opace.pace_avg_5    AS opp_pace_avg_5,
        opace.pace_avg_10   AS opp_pace_avg_10,

        CASE
            WHEN mk.market_total IS NULL OR mk.market_spread_home IS NULL THEN NULL
            WHEN r.team_abbr = r.home_team_abbr
                THEN (mk.market_total / 2.0) - (mk.market_spread_home / 2.0)
            WHEN r.team_abbr = r.away_team_abbr
                THEN (mk.market_total / 2.0) + (mk.market_spread_home / 2.0)
            ELSE NULL
        END AS team_implied_total,
        mk.market_total       AS game_market_total,
        mk.market_spread_home AS game_market_spread,

        (COALESCE(hpace.pace_avg_5, 0) + COALESCE(apace.pace_avg_5, 0)) / 2.0 AS game_pace_est_5,

        -- V007 player expanded stats
        pex.stl_avg_5, pex.stl_avg_10,
        pex.blk_avg_5, pex.blk_avg_10,
        pex.tov_avg_5, pex.tov_avg_10,
        pex.stl_plus_blk_avg_10,
        pex.off_reb_avg_10, pex.def_reb_avg_10,
        pex.fg_pct_avg_10, pex.fg3_pct_avg_10, pex.ft_pct_avg_10,
        pex.plus_minus_avg_5, pex.plus_minus_avg_10,
        pex.fouls_avg_5, pex.fouls_avg_10,
        pex.ast_to_tov_ratio_avg_10,

        -- V013 referee foul risk
        rfr.avg_foul_uplift_crew,
        rfr.avg_foul_per_36_uplift_crew,
        rfr.max_foul_uplift_crew,
        rfr.total_games_with_crew AS ref_games_with_crew,

        -- V006 opponent team style
        ostyle.fouls_avg_10     AS opp_fouls_avg_10,
        ostyle.stl_avg_10       AS opp_stl_avg_10,
        ostyle.blk_avg_10       AS opp_blk_avg_10,
        ostyle.pts_paint_avg_10 AS opp_pts_paint_avg_10,
        ostyle.fast_break_pct_avg_10 AS opp_fast_break_pct_10,

        -- ── NEW (V018): teammate injury impact ───────────────────────────────
        CASE WHEN r.team_abbr = UPPER(gtf.home_team_abbr)
             THEN COALESCE(gtf.home_injured_pts_lost, 0)
             ELSE COALESCE(gtf.away_injured_pts_lost, 0)
        END AS teammate_pts_out,
        CASE WHEN r.team_abbr = UPPER(gtf.home_team_abbr)
             THEN COALESCE(gtf.home_injured_out_count, 0)
             ELSE COALESCE(gtf.away_injured_out_count, 0)
        END AS teammate_out_count,

        -- ── NEW (V018): opponent position defense (V016) ─────────────────────
        opd.opp_pts_allowed_role_10,
        opd.opp_reb_allowed_role_10,
        opd.opp_ast_allowed_role_10

    FROM roll r

    LEFT JOIN features.team_opp_adjusted_roll oar
      ON oar.season = r.season AND oar.team_abbr = r.opponent_abbr AND oar.game_slug = r.game_slug
    LEFT JOIN features.team_pregame_rolling_boxscore opace
      ON opace.season = r.season AND opace.team_abbr = r.opponent_abbr AND opace.game_slug = r.game_slug
    LEFT JOIN features.team_pregame_rolling_boxscore hpace
      ON hpace.season = r.season AND hpace.team_abbr = r.home_team_abbr AND hpace.game_slug = r.game_slug
    LEFT JOIN features.team_pregame_rolling_boxscore apace
      ON apace.season = r.season AND apace.team_abbr = r.away_team_abbr AND apace.game_slug = r.game_slug

    LEFT JOIN LATERAL (
        SELECT
            oc.close_total AS market_total,
            oc.close_spread_home_points AS market_spread_home
        FROM odds.nba_game_lines_open_close oc
        WHERE oc.bookmaker_key = 'draftkings'
          AND oc.as_of_date = r.game_date_et
          AND oc.home_team_abbr = r.home_team_abbr
          AND oc.away_team_abbr = r.away_team_abbr
        LIMIT 1
    ) mk ON TRUE

    LEFT JOIN features.player_expanded_rolling pex
      ON pex.season = r.season AND pex.game_slug = r.game_slug AND pex.player_id = r.player_id
    LEFT JOIN features.player_game_referee_foul_risk rfr
      ON rfr.season = r.season AND rfr.game_slug = r.game_slug AND rfr.player_id = r.player_id
    LEFT JOIN features.team_style_profile ostyle
      ON ostyle.season = r.season AND ostyle.game_slug = r.game_slug AND ostyle.team_abbr = r.opponent_abbr

    -- V018: Join game training features for teammate injury data
    LEFT JOIN features.game_training_features gtf
      ON gtf.season = r.season AND gtf.game_slug = r.game_slug

    -- V018: Join opponent position defense (V016)
    LEFT JOIN features.opp_position_defense opd
      ON opd.season = r.season AND opd.game_slug = r.game_slug AND opd.player_id = r.player_id
)
SELECT
    season, game_slug, game_date_et,
    game_start_ts_utc AS start_ts_utc,
    player_id, team_abbr, opponent_abbr, is_home,

    -- Targets
    points, rebounds, assists,

    -- Player rolling features
    n_games_prev_10, min_avg_5, min_avg_10,
    pts_avg_5, pts_avg_10, pts_sd_10,
    reb_avg_5, reb_avg_10, reb_sd_10,
    ast_avg_5, ast_avg_10, ast_sd_10,
    fga_avg_10, fta_avg_10, threes_avg_10,
    player_rest_days, usage_proxy_avg_10,

    -- Matchup context
    opp_def_rtg_10, opp_off_rtg_10, opp_pace_avg_5, opp_pace_avg_10,
    team_implied_total, game_market_total, game_market_spread, game_pace_est_5,

    -- V007 player expanded stats
    stl_avg_5, stl_avg_10, blk_avg_5, blk_avg_10,
    tov_avg_5, tov_avg_10, stl_plus_blk_avg_10,
    off_reb_avg_10, def_reb_avg_10,
    fg_pct_avg_10, fg3_pct_avg_10, ft_pct_avg_10,
    plus_minus_avg_5, plus_minus_avg_10,
    fouls_avg_5, fouls_avg_10, ast_to_tov_ratio_avg_10,

    -- V013 referee foul risk
    avg_foul_uplift_crew, avg_foul_per_36_uplift_crew,
    max_foul_uplift_crew, ref_games_with_crew,

    -- V006 opponent style matchup
    opp_fouls_avg_10, opp_stl_avg_10, opp_blk_avg_10,
    opp_pts_paint_avg_10, opp_fast_break_pct_10,

    -- NEW (V018): teammate injury impact
    teammate_pts_out,
    teammate_out_count,

    -- NEW (V018): opponent position defense
    opp_pts_allowed_role_10,
    opp_reb_allowed_role_10,
    opp_ast_allowed_role_10

FROM enriched;
