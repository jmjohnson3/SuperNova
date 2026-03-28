-- MLB004: Static ballpark factors (approximate Statcast 3-year park factors)
-- run_factor > 1.0 = hitter-friendly, < 1.0 = pitcher-friendly
-- hr_factor  > 1.0 = HR-friendly, < 1.0 = suppresses HRs
-- Source: Statcast/Baseball Reference 3-year park factors, approximate values
CREATE TABLE IF NOT EXISTS features.mlb_ballpark_factors (
    venue_id      TEXT        NOT NULL,
    park_name     TEXT        NOT NULL,
    team_abbr     TEXT        NOT NULL,
    run_factor    NUMERIC(5,3) NOT NULL DEFAULT 1.000,
    hr_factor     NUMERIC(5,3) NOT NULL DEFAULT 1.000,
    updated_date  DATE        NOT NULL DEFAULT CURRENT_DATE,
    PRIMARY KEY (venue_id)
);

INSERT INTO features.mlb_ballpark_factors
    (venue_id, park_name, team_abbr, run_factor, hr_factor)
VALUES
    -- American League East
    ('fenway',        'Fenway Park',                    'BOS', 1.047, 1.021),
    ('yankee',        'Yankee Stadium',                 'NYY', 1.023, 1.138),
    ('tropicana',     'Tropicana Field',                'TBR', 0.943, 0.882),
    ('oriole',        'Oriole Park at Camden Yards',    'BAL', 1.012, 1.045),
    ('rogers',        'Rogers Centre',                  'TOR', 1.001, 1.003),
    -- American League Central
    ('guaranteed',    'Guaranteed Rate Field',          'CWS', 1.019, 1.094),
    ('progressive',   'Progressive Field',              'CLE', 0.978, 0.941),
    ('comerica',      'Comerica Park',                  'DET', 0.969, 0.896),
    ('kauffman',      'Kauffman Stadium',               'KCR', 0.952, 0.899),
    ('target',        'Target Field',                   'MIN', 0.981, 0.957),
    -- American League West
    ('minute_maid',   'Minute Maid Park',               'HOU', 1.012, 1.034),
    ('angel',         'Angel Stadium',                  'LAA', 0.981, 0.962),
    ('oakland',       'Oakland Coliseum',               'OAK', 0.921, 0.810),
    ('t_mobile',      'T-Mobile Park',                  'SEA', 0.947, 0.888),
    ('globe_life',    'Globe Life Field',               'TEX', 1.041, 1.068),
    -- National League East
    ('truist',        'Truist Park',                    'ATL', 1.006, 1.012),
    ('marlins',       'loanDepot park',                 'MIA', 0.899, 0.764),
    ('citi',          'Citi Field',                     'NYM', 0.936, 0.891),
    ('citizens',      'Citizens Bank Park',             'PHI', 1.053, 1.102),
    ('nationals',     'Nationals Park',                 'WSN', 0.994, 0.988),
    -- National League Central
    ('wrigley',       'Wrigley Field',                  'CHC', 1.024, 1.052),
    ('great_american','Great American Ball Park',       'CIN', 1.102, 1.221),
    ('american_family','American Family Field',         'MIL', 1.018, 1.076),
    ('pnc',           'PNC Park',                       'PIT', 0.951, 0.918),
    ('busch',         'Busch Stadium',                  'STL', 0.968, 0.924),
    -- National League West
    ('chase',         'Chase Field',                    'ARI', 1.038, 1.063),
    ('coors',         'Coors Field',                    'COL', 1.211, 1.226),
    ('dodger',        'Dodger Stadium',                 'LAD', 0.949, 0.938),
    ('petco',         'Petco Park',                     'SDP', 0.901, 0.847),
    ('oracle',        'Oracle Park',                    'SFG', 0.922, 0.861)
ON CONFLICT (venue_id) DO UPDATE SET
    park_name    = EXCLUDED.park_name,
    team_abbr    = EXCLUDED.team_abbr,
    run_factor   = EXCLUDED.run_factor,
    hr_factor    = EXCLUDED.hr_factor,
    updated_date = EXCLUDED.updated_date
;
