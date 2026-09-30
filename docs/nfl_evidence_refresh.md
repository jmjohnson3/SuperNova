# NFL Evidence Refresh and Prospective Trial

Production and bankroll approval are independent of offline training.

## Settlement and Training

`run_close_and_grade` grades all finalized prediction dates after importing the
current season. Existing ledger results refresh without inserting new bets.
Unchanged result/CLV labels retain their observation timestamps. Schema setup
does not run in grading or evidence training.

The exact-line evidence audit scans every prediction, checks pre-lock identity,
prices, true pairing, finalized participation, and model version, then keeps the
first eligible executable decision. Each exclusion has a prediction ID/reason in
`nfl_prop_exact_line_models.json`. Missing labels, pushes, and missing CLV are not
losses. Event identity can use a same-provider/event game quote already known at
the forecast cutoff when the prop payload omitted team names.

Training requires independent player-game samples and chronological NFL weeks.
Training labels must have been observed before the earliest holdout lock. Books
and lines share inverse player-game weights. Fewer than three weeks cannot make
a training/holdout split. Models write to `exact_line_challengers/latest.joblib`,
never the live `nfl_prop_exact_line_models.joblib`; a historical acceptance metric
does not install the model or grant any betting tier.

## Stable Prospective Trial

`python -m nfl_pipeline.modeling.benchmark_offers --pin-current` pins the latest
checksum-verified benchmark in `active_trial.json`. It refuses to replace a
different active trial. The normal daily job scores this pinned trial after the
original ledger locks. Retraining a new benchmark does not move that pointer.

Each shadow preserves the exact forecast ID, book, side, price, line and model
release. Artifact creation must precede context cutoff and scoring must finish
before kickoff. Captures are immutable and repeated runs are idempotent. Failure
to record any required variant for an otherwise eligible lock returns nonzero.

Validation includes raw, heuristic, post-exact and final probabilities, all real
offers, and the exact original micro ledger IDs separately. Hypothetical side
changes never alter the scored outcome. Grouped uncertainty uses NFL weeks.

Component review requires at least three independent prospective weeks, a
positive lower confidence bound for Brier improvement, no calibration regression,
and reasonable 80% interval coverage without a material regression. Both offered
lines and exact micro matches must support the result. Passing means ready for
review, not automatic deployment and never betting approval. Simulated ledger
entries are not proof that a wager was placed.

## Close Collection

While games are inside the close window the scheduled runner performs only odds
capture/parse, CLV resolution and close health reports. It defers training and
result downloads so they cannot occupy the next ten-minute capture slot.
Fresh normalized offers are checked per expected game/book; an empty capture gets
one bounded retry. Persistent failure returns nonzero. Quotes must be after lock,
before kickoff and within 120 minutes. Invalid prices cannot establish CLV.

`exact_line_unavailable_in_captured_feed` means valid-time snapshots contain the
player market at other lines. This is not a timing failure and does not prove
that the sportsbook removed the locked line: feeds may expose only main lines.
Neither those alternative prices nor stale exact prices are substituted into CLV.
Actual bookability and exact-price CLV remain unknown. A true flat valid close
remains zero. The 90% exact-close target and denominator are not relaxed.

Dated close diagnostics preserve old slates even when today's slate is empty.
Missing historical exact quotes cannot be manufactured by rerunning capture.
