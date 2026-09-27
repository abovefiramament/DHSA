# Reproducing a frozen condition

Repository: [abovefiramament/DHSA](https://github.com/abovefiramament/DHSA). Evidence: [frozen-20260927](https://github.com/abovefiramament/DHSA/releases/tag/frozen-20260927). Download and extract the ten archives as described in the root [README](../README.md). Shell examples use Bash on Linux and start in the repository root.

## 1. Inspect a condition

The README's CPU-only example reads the archived IMDb SFT summary and reports 2,048 rows. This example lists its inputs, code, and environment references without importing a model:

```bash
python - <<'PY'
import json
from pathlib import Path
cell = 'performance/imdb/main/performance__imdb__gpt2_large__sft'
entry = next(x for x in json.loads(Path('EVIDENCE_INDEX.json').read_text())['cells']
             if x['cell'] == cell)
for group in ('data_construction', 'public_source_records', 'public_urls',
              'environment_registration', 'execution_source', 'reproduction_scripts'):
    print('\n' + group)
    print('\n'.join(entry['files'].get(group, [])))
PY
```

Use cell names exactly as indexed. Some Site keys contain `#`; they identify a condition rather than a literal directory. Files under `performance/`, `site/`, `bank_controls/`, and `mechanism/` become available after extraction.

## 2. Environment and inputs

Archive inspection uses Python 3.11 and the standard library. Re-execution uses Linux, NVIDIA CUDA, Git, and the selected source snapshot's dependencies. Performance measurements used 80-GB H100 devices; the Site freeze records A100 devices. These are recorded platforms, not measured minimum-memory requirements.

Follow the condition's `environment_registration` entries and [runtime profiles](../reproduction/registered/configs/runtime/evidence_runtime_profiles_20260907_v1.json). Additional original launch records are in [reproduction/environments](../reproduction/environments/). The [Site freeze-time record](../reproduction/site/environment_at_freeze.json) includes its package inventory. Each record states whether it describes a run, a freeze-time observation, or partial historical information.

| Recorded profile | Python | PyTorch | Transformers | CUDA runtime |
| --- | --- | --- | --- | --- |
| `tldr_gptj_h100_fresh_20260913` | 3.11.3 | 2.5.1+cu124 | 4.46.3 | 12.4 |
| `site_freeze_time_screscomp_20260723` | 3.11.15 | 2.11.0+cu130 | 4.57.6 | 13.0 |

Use the profile bound to your condition. The IMDb SFT example has a partial historical environment record; neither complete profile above is assigned to it by this guide. Native LoReFT/BiPO use their registered dependencies and official implementations. Install dependencies in a separate environment for the selected condition.

For the IMDb example, download the following exact snapshots in that environment:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='ma921/gpt2-large-sft-imdb',
    revision='f480190690d5abfc0e003ccb4f7e650626019bb9',
    local_dir='/srv/dhsa-inputs/models/gpt2-large-sft-imdb',
)
snapshot_download(
    repo_id='siebert/sentiment-roberta-large-english',
    revision='74cea614e245b0832c770ec9aa51bd58df965b9c',
    local_dir='/srv/dhsa-inputs/models/siebert-sentiment',
)
```

`/srv/dhsa-inputs` is an example location; choose writable local paths. Public model pages are [ma921/gpt2-large-sft-imdb](https://huggingface.co/ma921/gpt2-large-sft-imdb) and [the Siebert scorer](https://huggingface.co/siebert/sentiment-roberta-large-english). Model downloads are additional to the evidence assets. Supply access credentials through the environment or model client's credential store if required.

IMDb data construction is specified in the archived cell's `config/protocol_freeze.json`, `config/resolved_config.json`, and [the data protocol](../source/code_01/configs/performance/imdb_ma921_cleanfirst1024_data_20260923_v3.json). Sources include [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb), [ma921/imdb-generated](https://huggingface.co/datasets/ma921/imdb-generated), and [ma921/imdb-tokenized_noise0](https://huggingface.co/datasets/ma921/imdb-tokenized_noise0). Follow the recorded revisions, splits, filtering, ordering, and seeds. Selected role data and manifests are retained; reconstruction from upstream sources must follow the recorded construction rather than use sample IDs as a selection rule.

The cleaned training token file is included at `reproduction/imdb_registration/imdb_shared_public_pairs_cleanfirst_20260923/training_original_token_ids.jsonl` after extraction. The evidence is not a preinstalled model/data environment: full upstream mirrors and machine-local intermediate paths must be prepared or bound to equivalent recorded inputs. The following command enumerates the exact paths expected by the selected flow.

## 3. Prepare a fresh replay

Keep outputs outside the frozen materials. Request the bindings and write a template:

```bash
export DHSA_REPLAY_DIR="$(pwd)/../DHSA-runs/imdb-sft"
mkdir -p ../DHSA-runs
python reproduction/prepare_cell.py \
  --cell performance/imdb/main/performance__imdb__gpt2_large__sft \
  --output "$DHSA_REPLAY_DIR" > ../DHSA-runs/imdb-sft.required.json
python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path('../DHSA-runs/imdb-sft.required.json').read_text())
Path('../DHSA-runs/imdb-sft.bindings.json').write_text(
    json.dumps(report['required_bindings'], indent=2) + '\n'
)
PY
```

Fill every value in `../DHSA-runs/imdb-sft.bindings.json` with an absolute local path. This archived example emits 23 keys. Several are aliases for the same input:

| Input | Aliases to bind consistently |
| --- | --- |
| Policy model | `models/ma921_gpt2_large_sft_imdb` and emitted `LOCAL_HOME/models/gpt2-large-sft-imdb-f4801906` |
| Scorer model | `models/siebert_sentiment` and emitted `LOCAL_HOME/models/siebert-sentiment-roberta-large-english-74cea614` |
| IMDb source train/test JSONL | `datasets/imdb/train_source` / `test_source`, their `data_sources/stanfordnlp_imdb_*_mirror` aliases, and emitted legacy paths |
| Selected original-comment files | `datasets/imdb/public_selector_comments`, `public_test_comments`, `public_validation_comments`; corresponding `data_sources/imdb_public_disjoint_*` aliases and emitted legacy paths |
| Cleaned training tokens | `datasets/imdb/public_training_cleanfirst`, `data_sources/ma921_cleanfirst1024_original_token_training`, and the emitted training-token legacy path |
| New output root | `outputs/experiments`, outside the evidence package |

The `LOCAL_HOME/...` keys are literal anonymized placeholders. Keep the keys and replace their values. Input roles retain their original construction and separation. The helper traverses all stored configuration fields, including fields not active in a particular stage, so the direct-policy example can list training-related inputs.

Verify that input files/directories exist, then prepare:

```bash
python reproduction/prepare_cell.py \
  --cell performance/imdb/main/performance__imdb__gpt2_large__sft \
  --output "$DHSA_REPLAY_DIR" \
  --bindings ../DHSA-runs/imdb-sft.bindings.json \
  > ../DHSA-runs/imdb-sft.prepared.json
```

Success returns `source`, `runtime_registry`, `cell_root`, and `command`, and creates the new replay configuration. A `required_bindings` response means bindings remain unresolved. Preparation does not run generation or evaluation.

## 4. Run the prepared flow

Execute the returned command from its source snapshot directory, inside the selected Linux/CUDA environment:

```bash
python - <<'PY'
import json, subprocess
from pathlib import Path
prepared = json.loads(Path('../DHSA-runs/imdb-sft.prepared.json').read_text())
if 'command' not in prepared:
    raise SystemExit('Resolve all required bindings before executing the flow.')
print('Source:', prepared['source'])
print('Command:', prepared['command'])
subprocess.run(prepared['command'], cwd=prepared['source'], check=True)
PY
```

The generic resume module is named `experiments.site.run_site_reproduction`; it invokes the shared executor on the saved **Performance** configuration. It does not select a new Site experiment. Run from a Git clone so the executor can record its source revision.

Expected files beneath the new cell root include `config/runtime_config.local.json`, `config/runtime_flow.local.json`, `run_manifest.json`, `test/predictions.jsonl`, `test/trajectory.jsonl`, `test/per_sample_scores.jsonl`, and `test/summary_metrics.json`. The archived direct-policy reference contains 2,048 predictions. Interpret score fields using the task's metric definitions and recorded generation settings.

This documentation update checks archive inspection and configuration preparation. It does not report a newly completed GPU replay or bitwise reproducibility under a reconstructed historical environment.

## 5. CAST, Site, and single-bank examples

For IMDb SFT+CAST-SV with the recorded `all` timing, repeat the same steps with this cell and a new output directory:

```bash
python reproduction/prepare_cell.py \
  --cell performance/imdb/main/performance__imdb__gpt2_large__sft_cast_sv_all \
  --output "$(pwd)/../DHSA-runs/imdb-cast-sv-all"
```

Fill this cell's emitted bindings, prepare, and execute its returned command. Its saved flow includes localization and controller training, followed by evaluation across ten alpha values. The archived result contains 20,480 predictions: 2,048 prompts per alpha. Training and inference timing definitions come from the saved configuration, including their distinct masks and decision states.

For **Site**, follow the selected entry's `reproduction_scripts`, `execution_source`, configuration, and environment records. Historical Site runners and newer shared-flow runners have different command-line interfaces.

For **single-bank controls**, extraction provides `bank_controls/execution/run_case.py` and, for the added IMDb DPO-start conditions, `bank_controls/execution_dpo_sv_20260926/run_case.py`. Their `cases.json` files specify the parent controller, source snapshot, alpha grid, and row count. In a separate working copy, bind the anonymized paths and revision aliases to the indexed parent artifacts, then invoke the runner with its case index. The archived runners retain an H100 device check. See the [extension example](EXTENDING.md#example-2-a-single-bank-control) for their shared component calls.

For **TL;DR**, the recorded evaluator may use an external judge API. Preserve its judge settings and provide credentials at runtime; generation and API evaluation incur their normal resource costs.
