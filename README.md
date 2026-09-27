# Direct Hidden-State Alignment (DHSA)

**A preference-alignment framework that operates directly on inference-time hidden states, designed for a small adaptation footprint and resource-efficient alignment.**

DHSA specifies the adaptation space: internal states used during inference. It leaves the learning objective open to reinforcement learning, distillation, and other forms of preference optimization. This release studies **RCM-guided CAST**, using the pairwise objective described in the paper.

- **DHSA** is the broader framework for aligning preference expression through direct hidden-state adaptation.
- **Residual Competition Maps (RCM)** map an externally specified preference onto inference-time Transformer components. Signed causal effects describe how each measured component supports or opposes that preference under the specified inputs and intervention.
- **Causal Activation State Transition (CAST)** is one instantiation of DHSA. It uses RCM to locate preference-relevant interfaces and learns local vector or low-rank interventions while keeping the base model frozen.

The reported CAST controllers use **256–16,384 trainable parameters** across the studied settings and can be enabled or removed at inference time. Task-specific training budgets, active parameter counts, and runtime measurements are reported in the paper and frozen configurations.

## Paper and authors

**Direct Hidden-State Alignment: Mapping and Controlling Preference Expression in LLMs**

Fansheng Zhang<sup>1,*</sup>, Shengran Guo<sup>2,†</sup>, Zexiao Wang<sup>3,†</sup>, Liang Yuan<sup>4,*</sup>, Jiyuan Chen<sup>1</sup>, Ruikun Luo<sup>5</sup>

<sup>1</sup> Chengdu University · <sup>2</sup> North Carolina State University · <sup>3</sup> Fudan University · <sup>4</sup> Australian Catholic University · <sup>5</sup> University of Macau

<sup>†</sup> Shengran Guo and Zexiao Wang contributed equally and share second authorship.

<sup>*</sup> Corresponding authors: **Liang Yuan** ([liang.yuan@acu.edu.au](mailto:liang.yuan@acu.edu.au)) and **Fansheng Zhang** ([above1firmament@gmail.com](mailto:above1firmament@gmail.com)).

The arXiv link will be added when an identifier is available. See [Citation](#citation) for the current manuscript citation.

[Frozen release](https://github.com/abovefiramament/DHSA/releases/tag/frozen-20260927) · [Reproduction guide](docs/REPRODUCTION.md) · [Paper-to-materials map](docs/PAPER_TO_ARTIFACTS.md) · [Extension examples](docs/EXTENDING.md)

## Modular experiment framework

Reusable scientific components are separated from experiment configuration and workflow orchestration. Dataset preparation, position selection, controller training, generation, and evaluation communicate through typed request/result interfaces. Experiments select components and compose stages while reusing their implementations.

For a new baseline that fits the existing interfaces, integration centers on its method implementation, component and runtime bindings, and method-specific configuration and flow validation. Controls built from available components are expressed through configuration, component selection, and stage orchestration. Shared data and evaluation implementations keep comparison conditions consistent.

| Responsibility | Implementation and extension points |
| --- | --- |
| Data construction | Dataset adapters in [`data/`](source/code_01/data/) |
| Baseline algorithms | [`baseline/implementations/`](source/code_01/baseline/implementations/) |
| Interfaces and registration | [`contracts.py`](source/code_01/experiments/shared/contracts.py), [`component_registry.py`](source/code_01/experiments/shared/component_registry.py), [`component_bindings.py`](source/code_01/baseline/implementations/component_bindings.py) |
| Experimental settings | Shared bundles and method plans in [`config_bundle.py`](source/code_01/experiments/shared/config_bundle.py) |
| Stage execution | [`flow_executor.py`](source/code_01/experiments/shared/flow_executor.py) |

The [Performance controller](source/code_01/experiments/performance/performance_cell_controller.py) composes different sequences for direct policies, CAST, LoReFT, and BiPO with shared generation and evaluation stages. The [Site controller](source/code_01/experiments/site/site_cell_controller.py) reuses shared data, localization, bank, generation, and evaluation operations.

The [extension guide](docs/EXTENDING.md) follows the existing BiPO integration and single-bank control as concrete examples. Architecture links use `source/code_01`; reproduce each published condition with the source snapshot specified in its own index entry.

## Get the materials

The [frozen release](https://github.com/abovefiramament/DHSA/releases/tag/frozen-20260927) contains ten `evidence-*.zip` attachments: **3.06 GB compressed / 13.19 GB extracted** (decimal units), excluding the repository, model checkpoints, and new run outputs. Reserve at least 20 GB for the repository and evidence download/extraction, plus space for models and new experiments.

```bash
git clone https://github.com/abovefiramament/DHSA.git
cd DHSA
gh release download frozen-20260927 --repo abovefiramament/DHSA \
  --pattern 'evidence-*.zip' --dir ../DHSA-assets
python - <<'PY'
import json, zipfile
from pathlib import Path
root = Path.cwd()
for asset in json.loads((root / 'RELEASE_ASSETS.json').read_text())['assets']:
    with zipfile.ZipFile(root.parent / 'DHSA-assets' / asset['file']) as archive:
        archive.extractall(root)
print('Extracted all ten evidence archives into', root)
PY
```

Alternatively, download the ten attachments in a browser and extract each into the repository root. [RELEASE_ASSETS.json](RELEASE_ASSETS.json) records archive sizes and download checksums. Base-model and scorer checkpoints are obtained separately from the pinned public sources linked in the condition index.

## Start with one archived result

After extraction, this CPU-only example reads the frozen IMDb GPT-2-large SFT result:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path.cwd()
cell = 'performance/imdb/main/performance__imdb__gpt2_large__sft'
entries = json.loads((root / 'EVIDENCE_INDEX.json').read_text())['cells']
entry = next(item for item in entries if item['cell'] == cell)
summary_path = next(p for p in entry['files']['test_scores']
                    if p.endswith('/summary_metrics.json'))
summary = json.loads((root / summary_path).read_text())
print('Cell:', cell)
print('Source:', entry['files']['execution_source'])
print('Rows:', summary['rows'])
print(json.dumps(summary['metrics'], indent=2))
PY
```

The saved summary contains **2,048 rows**. The [reproduction guide](docs/REPRODUCTION.md) continues through environment selection, pinned model/data preparation, path bindings, the actual execution command, and expected output files. It also covers the CAST variant and the separate Site and single-bank entrypoints.

## Included experiments and evidence

[EVIDENCE_INDEX.json](EVIDENCE_INDEX.json) contains **158 entries**: 64 Performance conditions, 60 Site conditions, 24 single-bank conditions, and 10 source comparators. Shared comparators are references to shared evidence, not additional independent experiments.

| Task | Performance models | Site models |
| --- | --- | --- |
| ConFiQA | Llama-3-8B, Qwen2-7B | Llama-3-8B, Qwen2.5-14B |
| IMDb | GPT-2-large | GPT-2-large, Qwen2.5-14B |
| TL;DR | GPT-J-6B, LION-Llama-3-8B | GPT-J-6B, Qwen2.5-14B |

Performance includes starting policies, released DPO policies, CAST, and native BiPO/LoReFT comparisons, with additional conditions enumerated in the index. Site compares RCM-Zero, RCM-Patch, ITI, and three fixed random selectors. IMDb single-bank evidence includes eight SFT-start conditions and four DPO-start support-only conditions.

| Path after extraction | Contents |
| --- | --- |
| `performance/` | Performance evaluations, configurations, controllers, predictions, and scores |
| `site/` | Localization comparisons and their recorded inputs and outputs |
| `bank_controls/` | Single-bank controls and matched two-bank source conditions |
| `mechanism/` | RCM measurements, native-output analyses, and measurement notes |
| `source/` | Archived execution-source snapshots, selected per condition |
| `reproduction/` | Path-binding helpers, input-construction records, entrypoints, and environment records |

Use the [paper-to-materials map](docs/PAPER_TO_ARTIFACTS.md) to locate a figure or table. [MATERIALS_AUDIT.md](MATERIALS_AUDIT.md) describes file coverage and the IMDb supplement. After extraction, `mechanism/MEASUREMENT_NOTES.md` records measurement definitions, reconstructed operands, and gaps in retained selector evidence. Machine and account identifiers have been sanitized; recorded scientific values and text are preserved.

## Citation

Please cite the paper when using DHSA, RCM, CAST, or the research materials. The current citation is also available in [CITATION.bib](CITATION.bib) and [CITATION.cff](CITATION.cff); it will be updated with the arXiv identifier.

```bibtex
@unpublished{zhang2026dhsa,
  title  = {Direct Hidden-State Alignment: Mapping and Controlling Preference Expression in {LLMs}},
  author = {Zhang, Fansheng and Guo, Shengran and Wang, Zexiao and Yuan, Liang and Chen, Jiyuan and Luo, Ruikun},
  year   = {2026},
  note   = {Research manuscript. Shengran Guo and Zexiao Wang contributed equally and share second authorship},
  url    = {https://github.com/abovefiramament/DHSA}
}
```

## Acknowledgments

We acknowledge the A100 Computing Center at Chengdu University for providing computational resources. We also acknowledge the Research Computing Centre at The University of Queensland for access to the Bunya supercomputer ([DOI: 10.48610/wf6c-qy55](https://doi.org/10.48610/wf6c-qy55)).

We thank the developers and maintainers of the models, datasets, evaluators, and baseline implementations used in this work. See [third-party sources and licensing scope](THIRD_PARTY_NOTICES.md) and each condition's public source records.

## License and contact

Original DHSA code, configuration, and documentation are licensed under [Apache-2.0](LICENSE). Third-party code, datasets, model weights, and text embedded in research artifacts retain their applicable upstream terms; the code license does not relicense those materials. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

For reproducibility questions, [open an issue](https://github.com/abovefiramament/DHSA/issues) with the evidence cell, source snapshot, command, environment, and relevant error. For research correspondence, contact Liang Yuan or Fansheng Zhang at the addresses above.
