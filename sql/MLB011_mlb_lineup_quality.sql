-- MLB011: Lineup quality view
-- Uses actual batting orders from raw.mlb_boxscore_player_stats (completed games)
-- joined to mlb_player_batting_rolling_mat for pre-game rolling stats.
--
-- Depends on: features.mlb_player_batting_rolling_mat (must exist before applying)
-- Applied in _apply_post_matview_views() AFTER _refresh_matviews().
--
-- For prediction (today's upcoming games), lineup columns will be NULL
-- and will be median-imputed by the model.

CREATE OR REPLACE VIEW features.mlb_lineup_quality AS
WITH
-- Actual batting orders from completed boxscores (starters only: positions 1-9)
hist_orders AS (
    SELECT
        bps.game_slug,
        bps.player_id,
        bps.team_abbr,
        bps.batting_order,
        bps.is_home
    FROM raw.mlb_boxscore_player_stats bps
    WHERE bps.batting_order BETWEEN 1 AND 9
),
-- Join to rolling stats at game time (same game_slug = pre-game rolling window)
lineup_with_stats AS (
    SELECT
        o.game_slug,
        o.team_abbr,
        o.is_home,
        o.batting_order,
        br.avg_avg_10,
        br.iso_avg_10,
        br.tb_avg_10,
        br.ab_avg_10,
        br.hits_avg_10,
        br.k_rate_avg_10,
        sb.xwoba,
        sb.xslg,
        sb.barrel_batted_rate,
        sb.hard_hit_percent
    FROM hist_orders o
    LEFT JOIN features.mlb_player_batting_rolling_mat br
        ON br.player_id = o.player_id
       AND br.game_slug = o.game_slug
    LEFT JOIN raw.mlb_statcast_batting sb
        ON sb.player_id   = o.player_id
       AND sb.season_year = CAST(LEFT(o.game_slug, 4) AS INT)
)
SELECT
    game_slug,
    team_abbr,
    is_home,
    -- Batting average rolling (proxy for contact ability)
    AVG(avg_avg_10)                                             AS lineup_avg_avg_10,
    -- SLG proxy: avg total bases / avg at bats per game
    AVG(tb_avg_10 / NULLIF(ab_avg_10, 0))                      AS lineup_slg_avg_10,
    -- Isolated power
    AVG(iso_avg_10)                                             AS lineup_iso_avg_10,
    -- Top-4 SLG (heart of the order)
    AVG(CASE WHEN batting_order BETWEEN 1 AND 4
             THEN tb_avg_10 / NULLIF(ab_avg_10, 0) END)        AS top4_slg_avg_10,
    -- Data completeness: fraction of 9 batters with rolling stats
    COUNT(CASE WHEN avg_avg_10 IS NOT NULL THEN 1 END)::FLOAT / 9
                                                                AS lineup_data_completeness,
    -- Statcast quality (season-level; NULLs where player has no Statcast data)
    AVG(xwoba)              AS lineup_xwoba_avg,
    AVG(xslg)               AS lineup_xslg_avg,
    AVG(barrel_batted_rate) AS lineup_barrel_avg,
    AVG(hard_hit_percent)   AS lineup_hard_hit_avg,
    STDDEV_SAMP(k_rate_avg_10)                            AS lineup_k_pct_std,
    CASE WHEN AVG(k_rate_avg_10) > 0
         THEN STDDEV_SAMP(k_rate_avg_10)
              / NULLIF(AVG(k_rate_avg_10), 0)
         ELSE NULL END                                   AS lineup_k_pct_cv
FROM lineup_with_stats
GROUP BY game_slug, team_abbr, is_home
;
