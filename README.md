# Direct Hidden-State Alignment

Code and supplementary materials for **Direct Hidden-State Alignment: Mapping and Controlling Preference Expression in LLMs**.


This repository and its release assets contain the selected formal IMDb, ConFiQA, and TL;DR results,
component-effect measurements, Site comparisons, single-bank controls, execution
code, configurations, test predictions, scores, and evaluation traces.

`EVIDENCE_INDEX.json` is the entry point. Its 158 entries comprise 64 Performance
conditions, 60 Site conditions, 24 single-bank conditions, and 10 source
comparators; shared comparators are not independent experiments. Each entry
links the applicable configuration, data construction, component or position
records, controller, evaluation outputs, source code, and software environment.
The evidence directories are read-only inputs for reproduction.

## Evidence assets

Download all ten `evidence-*.zip` attachments from the matching GitHub Release
and extract each archive into the repository root. The archives preserve the
relative paths in `EVIDENCE_INDEX.json`; `RELEASE_ASSETS.json` lists their sizes
and SHA-256 checksums. The index paths resolve after all ten are extracted.

## Layout

- `performance/`: preference-performance evaluations and their inputs and outputs.
- `site/`: position-selection comparisons and held-out evaluations.
- `bank_controls/`: single-bank evaluations and matched dual-bank comparators.
- `mechanism/`: component-effect statistics and native-output measurements.
- `source/`: code snapshots used by the selected conditions.
- `reproduction/`: path-binding helpers, source records, and execution entrypoints.

## Reproduction

Select a condition in `EVIDENCE_INDEX.json` and use its saved configuration,
data construction, execution source, and environment profile. Bind local model,
dataset, and runtime paths in a working directory outside this package. Do not
change the packaged evidence files.

For a Performance condition, `reproduction/prepare_cell.py` creates a portable
copy of its saved configuration and flow:

```sh
python reproduction/prepare_cell.py --cell performance/tldr/gptj_6b/sft \
  --output /absolute/new/replay-cell
```

The command lists required path bindings. Supply them in a JSON object with
`--bindings /path/bindings.json` and rerun it. It then prints the execution
command. Site conditions use their indexed Site entrypoint; single-bank controls
use `bank_controls/execution/run_case.py` with the indexed `cases.json` entry.
API credentials are supplied by the user at runtime and are not included here.

The published model, dataset, method, and scorer source URLs are included in the
condition index or its linked configuration. The measured numerical outputs and
text are preserved; machine and account identifiers have been anonymized.

## Snapshot coverage

See [MATERIALS_AUDIT.md](MATERIALS_AUDIT.md) for the verified file inventory, IMDb SFT single-bank row coverage, and the recovered DPO-start single-bank supplement.

The four IMDb DPO-start support-only SV conditions are in `evidence-010-imdb-dpo-single-bank.zip`. Their parent controllers and matched two-bank results are in the original nine assets. The added conditions reference the same included `source/code_01` snapshot. See each new index entry for the archived runner and required path/revision-alias bindings; work in a separate copy and preserve the evidence files.
