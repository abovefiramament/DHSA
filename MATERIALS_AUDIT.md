# Frozen-materials coverage check (2026-09-27)

The existing snapshot contains 154 indexed entries. All 78,270 unique indexed paths exist in the source package and are included in the repository or the nine evidence archives. No duplicate archive paths, archive-member size mismatches, or omitted evidence files were found. The nine archives match the existing release manifest.

## IMDb SFT-start single-bank controls

Eight conditions are included: SV/ReFT, all/prefill, and high_effect/low_effect. Each condition contains 20,480 predictions, 20,480 trajectory records, and 20,480 per-sample scores. Prediction and score identifiers match without duplicates. Each of the ten alpha values has 2,048 predictions. The corresponding parent controller payloads and indexed inputs are present.

## Recovered IMDb DPO-start single-bank supplement

The original 154-entry snapshot omitted the four IMDb DPO-start support-only SV ablations (transfer/nontransfer, each with all/prefill timing). Their completed original results were recovered from the 2026-09-26 run and added as `evidence-010-imdb-dpo-single-bank.zip`; the index now has 158 entries.

Each added condition contains 20,480 predictions, trajectory records, and per-sample scores, plus 81,920 component records. Prediction, trajectory, and score identifiers match without duplicates. Each alpha has 2,048 predictions and scores. The saved reward and KL means match independently aggregated per-sample scores within 1e-8. The included summary matches the completion record. The parent controllers, data/configuration records, matched two-bank evaluations, and execution-source snapshot are already in the original assets and are explicitly linked by the new index entries.

Execution scripts, scientific case settings, controller views, completion records, and submission provenance are included. Account names, home/scratch paths, private source revisions, and job identifiers are sanitized using the existing publication conventions. A separate check confirms that every measured JSONL record and the numerical scientific fields are unchanged. Redundant generation restart caches and Python bytecode are excluded; predictions, token IDs, trajectories, component records, per-sample scores, and summaries are included.

These checks verify packaging and row coverage; they do not rerun training/evaluation or establish numerical reproducibility of every experiment.
