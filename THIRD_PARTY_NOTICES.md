# Licensing scope and upstream sources

Copyright 2026 the DHSA contributors.

Original DHSA code, configurations, and repository documentation are licensed under the [Apache License, Version 2.0](LICENSE). This grant covers the project's original portions, including those in archived source snapshots. [NOTICE](NOTICE) records the project's attribution information; retain applicable notices when redistributing as required by Apache-2.0. Any separately identified third-party material retains its existing terms and notices.

The root code license does not grant new rights over upstream model weights, tokenizers, datasets, source text, external software, or third-party content preserved in predictions and evaluation traces. Their applicable upstream terms continue to govern reuse. Research artifacts in the Release are not blanket-relicensed as Apache-2.0 or CC BY 4.0 datasets or model weights.

The accompanying paper and its original figures are licensed under [CC BY 4.0](LICENSE-PAPER.md), separately from the software. Third-party material is excluded from this grant. The README's academic citation request does not add a restriction to Apache-2.0.

## Baselines and supporting software

The frozen condition records identify the exact sources and revisions used. Upstream projects referenced by those records include:

- [Context-DPO and ConFiQA](https://github.com/byronBBL/Context-DPO)
- [BiPO](https://github.com/CaoYuanpu/BiPO)
- [LoReFT / pyreft](https://github.com/stanfordnlp/pyreft)
- [ITI / honest_llama](https://github.com/likenneth/honest_llama)
- [Contrastive Activation Addition](https://github.com/nrimsky/CAA)
- [AxBench](https://github.com/stanfordnlp/axbench)

When fetching or redistributing upstream implementations, retain their copyright, license, and notice files at the revision you use. Original DHSA adapters and experiment orchestration do not replace the upstream implementation's license. Inclusion in this list acknowledges a dependency or source; it does not assert endorsement by its maintainers.

## Models, data, and evaluators

Use the `public_urls`, `public_source_records`, and `data_construction` fields in [EVIDENCE_INDEX.json](EVIDENCE_INDEX.json) to find each condition's public model/dataset source, revision, and construction. Model and dataset families have different terms; consult the applicable source record and upstream license for your selected condition.

These records cover the ConFiQA, IMDb, and TL;DR sources; starting and DPO policy checkpoints; sentiment scorers; and the other evaluators used in the paper. Credentials and local model installations are supplied by the user. Cite the relevant original methods, models, and datasets alongside the DHSA paper where appropriate.
