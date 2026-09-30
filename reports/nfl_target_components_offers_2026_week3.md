# Target Volume: Exact-Lock Validation

retrospective_diagnostic

Production, fixed selections, scoring adjustments and ranking rules were not changed.

| Cohort / pool | Settled | Production Brier | reference | targets_only | uncertainty_only | combined | Market Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| ('nfl-20260918T143342Z', '44f8fbfa4bd9b3417ee88fe4e9f821a8c0f937493a574b6dae5b94dc97355219') / all_offers | 16 | 0.2547 | 0.2547 | 0.2537 | 0.2554 | 0.2481 | 0.2500 |
| ('nfl-20260918T143342Z', '44f8fbfa4bd9b3417ee88fe4e9f821a8c0f937493a574b6dae5b94dc97355219') / fixed_research | 2 | 0.2610 | 0.2610 | 0.2768 | 0.2657 | 0.2457 | 0.2500 |
| ('nfl-20260918T143342Z', '44f8fbfa4bd9b3417ee88fe4e9f821a8c0f937493a574b6dae5b94dc97355219') / current_ranked | 2 | 0.2610 | 0.2610 | 0.2768 | 0.2657 | 0.2457 | 0.2500 |
| ('nfl-20260918T143342Z', '44f8fbfa4bd9b3417ee88fe4e9f821a8c0f937493a574b6dae5b94dc97355219') / reference_ranked | 2 | 0.2610 | 0.2610 | 0.2768 | 0.2657 | 0.2457 | 0.2500 |
| ('nfl-20260918T143342Z', '44f8fbfa4bd9b3417ee88fe4e9f821a8c0f937493a574b6dae5b94dc97355219') / targets_only_ranked | 1 | 0.3184 | 0.3184 | 0.2862 | 0.2865 | 0.2704 | 0.2500 |
| ('nfl-20260918T143342Z', '44f8fbfa4bd9b3417ee88fe4e9f821a8c0f937493a574b6dae5b94dc97355219') / uncertainty_only_ranked | 1 | 0.3184 | 0.3184 | 0.2862 | 0.2865 | 0.2704 | 0.2500 |
| ('nfl-20260918T143342Z', '44f8fbfa4bd9b3417ee88fe4e9f821a8c0f937493a574b6dae5b94dc97355219') / combined_ranked | 0 | - | - | - | - | - | - |
| ('nfl-20260918T143342Z', 'e29cbbca864ea8e37e57d7ba178ee30ff3dd371a412726d569d8f89b063c1afd') / all_offers | 565 | 0.2511 | 0.2511 | 0.2494 | 0.2518 | 0.2502 | 0.2498 |
| ('nfl-20260918T143342Z', 'e29cbbca864ea8e37e57d7ba178ee30ff3dd371a412726d569d8f89b063c1afd') / fixed_research | 5 | 0.2616 | 0.2616 | 0.2743 | 0.2606 | 0.2611 | 0.2500 |
| ('nfl-20260918T143342Z', 'e29cbbca864ea8e37e57d7ba178ee30ff3dd371a412726d569d8f89b063c1afd') / current_ranked | 15 | 0.2164 | 0.2164 | 0.2364 | 0.2382 | 0.2281 | 0.2504 |
| ('nfl-20260918T143342Z', 'e29cbbca864ea8e37e57d7ba178ee30ff3dd371a412726d569d8f89b063c1afd') / reference_ranked | 15 | 0.2164 | 0.2164 | 0.2364 | 0.2382 | 0.2281 | 0.2504 |
| ('nfl-20260918T143342Z', 'e29cbbca864ea8e37e57d7ba178ee30ff3dd371a412726d569d8f89b063c1afd') / targets_only_ranked | 13 | 0.2376 | 0.2376 | 0.2353 | 0.2493 | 0.2433 | 0.2519 |
| ('nfl-20260918T143342Z', 'e29cbbca864ea8e37e57d7ba178ee30ff3dd371a412726d569d8f89b063c1afd') / uncertainty_only_ranked | 10 | 0.2543 | 0.2543 | 0.2541 | 0.2501 | 0.2412 | 0.2500 |
| ('nfl-20260918T143342Z', 'e29cbbca864ea8e37e57d7ba178ee30ff3dd371a412726d569d8f89b063c1afd') / combined_ranked | 13 | 0.2295 | 0.2295 | 0.2439 | 0.2413 | 0.2409 | 0.2519 |

Exclusions: {"outside_market": 428, "before_registration": 4, "stale_original_quote": 1}

Raw interval coverage is separate from final offered-line probability accuracy.
No deployment approval. Week 3 is development evidence, not an untouched acceptance set.
