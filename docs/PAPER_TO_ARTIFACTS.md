# Paper-to-materials map

Paper: **Direct Hidden-State Alignment: Mapping and Controlling Preference Expression in LLMs**. Numbering follows the September 27, 2026 manuscript. Caption titles and LaTeX labels are included to keep the mapping usable if layout changes.

Repository: [abovefiramament/DHSA](https://github.com/abovefiramament/DHSA). Download [frozen-20260927](https://github.com/abovefiramament/DHSA/releases/tag/frozen-20260927) and extract all ten archives. Paths below are relative to that extracted repository. [EVIDENCE_INDEX.json](../EVIDENCE_INDEX.json) links each condition's configuration, code, data, controllers, predictions, and scores. This page identifies the source material; it is not a set of figure-rendering scripts.

## Main paper

| Paper item | Material entrypoint | What to inspect |
| --- | --- | --- |
| Figure 1: overview of RCM-guided DHSA | [Architecture and extension examples](EXTENDING.md) | Conceptual schematic; DHSA is the framework, RCM provides preference-relative measurements, and CAST implements local control. |
| Figure 2: residual competition in ConFiQA (`fig:rcm_results`) | `mechanism/confiqa_llama_analysis/` and `mechanism/registered/` | `coverage.csv`, `layer_changes.csv`, `statistics.json`; registered ConFiQA SFT/DPO scans and their correspondence records. |
| Table 1: Site results (`tab:site_main`) | Indexed `site/` conditions for ConFiQA and TL;DR | Each entry's `test_scores`, configuration, and selector records. ConFiQA reports macro-average PC across QA/MR/MC; TL;DR uses the paper's order-balanced preference definition. |
| Figure 3(a): IMDb Site curves | Indexed `site/selection/imdb__sentiment__ma921_gpt2_large_sft__*` conditions | Per-strength scores for the six selector conditions. Preserve the Site score and distribution-shift definitions. |
| Figure 3(b): IMDb Performance curves | `performance/imdb/main/`, `performance/imdb/external_bipo/`, `performance/imdb/external_loreft/` | The corresponding indexed `test_scores`, controller settings, and alpha grid. |
| Supporting controls for the IMDb comparison in Figure 3(b) | `bank_controls/cells/imdb_gpt2_large/sft_cast_*/` | Eight SV/ReFT × all/prefill × high/low-effect conditions; use each index entry's linked two-bank comparator. Figure 3(b)'s SV-all operating curve uses the supporting-only bank. |
| Table 2: preference-alignment performance | `performance/confiqa/`, `performance/tldr/`, `performance/extensions/confiqa_llama/`, `performance/extensions/confiqa_qwen_transfer/`, `performance/extensions/tldr_dpo_cast/` | Starting policy, DPO, native baselines, SFT+CAST, and DPO+CAST entries as applicable to each table row. Use the archived selected operating point and held-out scores. |

Useful Site families include `site/selection/` for ConFiQA Llama-3-8B and IMDb GPT-2-large, `site/confiqa_qwen_qa_mr/`, `site/confiqa_qwen_mc/`, `site/final/tldr_gptj_final_complete/`, and `site/tldr_qwen25_14b/`. Their different directory layouts are normalized by the condition index.

## Appendix

| Paper item | Material entrypoint | What to inspect |
| --- | --- | --- |
| Figure 4 and Table 3: cross-domain RCM-Zero effects (`fig:rcm_cross_domain`, `tab:rcm_cross_domain`) | ConFiQA analysis above; the IMDb/TL;DR RCM-Zero entries' `position_records`; `mechanism/registered/correspondence.json` | Recorded per-input intervention effects and aggregate scans for the paper's specified model and scanned set. Keep task score units and component granularity separate. The correspondence file marks exact matches and sources that were not imported. |
| Target outputs and singleton decoded changes (`app:target_outputs`, `app:singleton_cases`) | `mechanism/confiqa_llama_native_outputs/` and indexed `rcm_input_text` / `position_records` | Native predictions, `target_output_coverage.csv`, retained intervention traces and example records. |
| Tables 4–5: CAST optimization and deployment | Each relevant Performance entry's `configuration` | `resolved_config.json`, method plan, training settings, timing definitions, and controller manifests. |
| Table 6: active controller parameter counts | Each relevant entry's `controller_payload` | Retained tensor payloads and their bank/component composition; count the deployed configuration described by the paper. |
| Tables 7–8: candidate opportunities, selection, and audit budgets | `configuration`, `position_records`, `audit_text`, `audit_scores` | Candidate schedules, selected positions, training records, audit decisions, and final bank composition. |
| Table 9: final-generation settings | Indexed generation manifests and resolved configurations | Token limits, decoding parameters, timing, batching, and seed rules for the stated task/model. |
| Table 10: paired Performance intervals (`tab:paired_main_effects`) | The Table 2 entries' `test_scores` and `test_text` | Per-sample inputs for recomputing the paired intervals under the resampling procedure in the paper. |
| Figure 5: IMDb Qwen2.5-14B Site curves (`fig:imdb_qwen_site`) | `site/imdb_qwen/artifacts/` | Six indexed selector conditions and their final per-strength test scores. |
| Figure 6: IMDb DPO+CAST curves (`fig:imdb_dpo_results`) | `performance/imdb/main/performance__imdb__gpt2_large__dpo*` | DPO reference and transfer/nontransfer SV/ReFT all/prefill conditions. |
| Figure 7: IMDb DPO-start bank ablation (`fig:imdb_dpo_bank_ablation`) | `bank_controls/cells/imdb_gpt2_large/dpo_cast_*/high_effect/` | Four support-only SV conditions from `evidence-010-imdb-dpo-single-bank.zip`, plus their original two-bank comparators linked by the index. |
| Theory (`app:theory`) | Manuscript proofs | Mathematical assumptions and derivations are in the paper; they are not empirical archive entries. |

## Locate the exact files for a figure

For example, list the four Figure 7 conditions and their saved scores after extraction:

```bash
python - <<'PY'
import json
from pathlib import Path
entries = json.loads(Path('EVIDENCE_INDEX.json').read_text())['cells']
for entry in entries:
    if entry['cell'].startswith('bank_controls/cells/imdb_gpt2_large/dpo_cast_'):
        print('\n' + entry['cell'])
        for group in ('test_scores', 'configuration', 'execution_source'):
            print(group + ':')
            print('\n'.join(entry['files'].get(group, [])))
PY
```

For another result, select its task/model and method from the table, then inspect that exact entry. An entry can link both a control and its parent comparator; use the file belonging to the requested condition when reading its result.

## Measurement and retention notes

`mechanism/MEASUREMENT_NOTES.md` is the measurement-specific reference. Signed RCM effects are the recorded intervention contrasts. Generation-stage `component_scores` contain activation-norm measurements and must not be substituted for signed effects. Per-sample RCM-Patch effects are not retained for some historical Site conditions; retained aggregates and explicitly reconstructed operands are identified there. Use the original input/protocol records for provenance rather than infer data construction from sample identifiers.

[MATERIALS_AUDIT.md](../MATERIALS_AUDIT.md) covers publication-file completeness and the IMDb supplement. File coverage, retained measurement detail, and a new numerical replay are distinct checks.
