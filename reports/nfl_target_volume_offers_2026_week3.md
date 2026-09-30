# Target Volume: Exact-Lock Validation

retrospective_diagnostic

Production, fixed selections, scoring adjustments and ranking rules were not changed.

| Cohort / pool | Settled | Production Brier | Control Brier | Challenger Brier | Market Brier |
|---|---:|---:|---:|---:|---:|
| ('nfl-20260918T143342Z', '44f8fbfa4bd9b3417ee88fe4e9f821a8c0f937493a574b6dae5b94dc97355219') / all_offers | 16 | 0.2547 | 0.2545 | 0.2536 | 0.2500 |
| ('nfl-20260918T143342Z', '44f8fbfa4bd9b3417ee88fe4e9f821a8c0f937493a574b6dae5b94dc97355219') / fixed_research | 2 | 0.2610 | 0.2547 | 0.2677 | 0.2500 |
| ('nfl-20260918T143342Z', '44f8fbfa4bd9b3417ee88fe4e9f821a8c0f937493a574b6dae5b94dc97355219') / current_ranked | 2 | 0.2610 | 0.2547 | 0.2677 | 0.2500 |
| ('nfl-20260918T143342Z', '44f8fbfa4bd9b3417ee88fe4e9f821a8c0f937493a574b6dae5b94dc97355219') / challenger_ranked | 2 | 0.2610 | 0.2547 | 0.2677 | 0.2500 |
| ('nfl-20260918T143342Z', 'e29cbbca864ea8e37e57d7ba178ee30ff3dd371a412726d569d8f89b063c1afd') / all_offers | 565 | 0.2511 | 0.2506 | 0.2489 | 0.2498 |
| ('nfl-20260918T143342Z', 'e29cbbca864ea8e37e57d7ba178ee30ff3dd371a412726d569d8f89b063c1afd') / fixed_research | 5 | 0.2616 | 0.2606 | 0.2599 | 0.2500 |
| ('nfl-20260918T143342Z', 'e29cbbca864ea8e37e57d7ba178ee30ff3dd371a412726d569d8f89b063c1afd') / current_ranked | 15 | 0.2164 | 0.2417 | 0.2285 | 0.2504 |
| ('nfl-20260918T143342Z', 'e29cbbca864ea8e37e57d7ba178ee30ff3dd371a412726d569d8f89b063c1afd') / challenger_ranked | 12 | 0.2229 | 0.2392 | 0.2270 | 0.2520 |

Exclusions: {"outside_market": 428, "before_registration": 4, "stale_original_quote": 1}

Raw interval coverage is separate from final offered-line probability accuracy.
No deployment approval. Week 3 is development evidence, not an untouched acceptance set.
