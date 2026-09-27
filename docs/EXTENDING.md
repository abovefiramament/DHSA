# Extending the experiment framework

Start from [DHSA](https://github.com/abovefiramament/DHSA) and the source snapshot linked by your condition in [EVIDENCE_INDEX.json](../EVIDENCE_INDEX.json). The examples below use `source/code_01` to show concrete extension points. Develop new experiments in a separate working copy and retain the frozen publication configuration.

DHSA defines the hidden-state adaptation space. RCM measures components relative to an external preference, and CAST implements one controller and training recipe. A different learning objective, including reinforcement learning or distillation, requires its corresponding backend and scientific configuration. The released CAST experiments use the paper's pairwise objective.

## Example 1: adding a native baseline

The existing BiPO integration illustrates how method-specific training connects to shared data, generation, and evaluation. Its actual integration points are:

| Step | Working reference | Responsibility |
| --- | --- | --- |
| Implement the method | [`bipo.py`](../source/code_01/baseline/implementations/bipo.py) | `BiPOBackend.train_candidate` invokes the registered official trainer and returns `NativeBaselineTrainResult`. |
| Register capabilities | [`component_bindings.py`](../source/code_01/baseline/implementations/component_bindings.py) | `register_bipo_components` registers its scanner, training backend, and generation backend. |
| Bind runtime resources | [`runtime_bindings.py`](../source/code_01/baseline/implementations/runtime_bindings.py) | `register_bipo_runtime_components` supplies the model provider and runtime callbacks. |
| Resolve the method plan | [`parameter_controller.py`](../source/code_01/baseline/controllers/parameter_controller.py) | Method identifiers, kinds, and method-specific validation. |
| Define the experiment sequence | [`performance_cell_controller.py`](../source/code_01/experiments/performance/performance_cell_controller.py) and [`performance_bundle_generator.py`](../source/code_01/experiments/performance/performance_bundle_generator.py) | Candidate schedule, component bindings, explicit settings, and permitted stage sequence. |

The training registration has this form, using the actual BiPO backend:

```python
from baseline.implementations.bipo import BiPOBackend
from baseline.implementations.model_runtime import HuggingFaceModelProvider
from experiments.shared.component_registry import ComponentRegistry

registry = ComponentRegistry()
provider = HuggingFaceModelProvider(device='cuda')
registry.register(
    kind='native_baseline_backend',
    component_id='bipo',
    version=1,
    implementation=BiPOBackend(provider),
    operations=('train_candidate',),
    input_contract='native_bipo/v1',
    output_contract='native_bipo_evidence/v1',
)
```

This shows the registration boundary; it does not train a model. A new native baseline implements the [request/result contract](../source/code_01/experiments/shared/contracts.py), supplies its own registrations, and adds its identifier and validation to the method plan and experiment flow. Preserve the official method's algorithm and objective in its backend. Bind a method-specific generation implementation where its intervention requires one.

The [native baseline controller](../source/code_01/experiments/shared/native_baseline_controller.py) owns candidate execution and validation-based selection. Shared dataset and evaluator components continue to receive the same task definitions and role manifests. For methods compatible with the existing stage interfaces, this confines integration changes to method and configuration boundaries. New operations require an explicit interface/stage extension.

## Example 2: a single-bank control

The archived control changes which trained bank is active, then calls the existing generation and evaluation components. It does not introduce another trainer or scorer.

After extracting the [release](https://github.com/abovefiramament/DHSA/releases/tag/frozen-20260927), inspect `bank_controls/execution/run_case.py` and `cases.json`. The added IMDb DPO-start cases use `bank_controls/execution_dpo_sv_20260926/`. Each case identifies its parent controller, source snapshot, alpha grid, and expected prediction count.

The original runner's controller-view construction can be illustrated without GPU execution:

```python
import copy

def select_single_bank(original, bank_id):
    selected = [p for p in original['ordered_pairs'] if p['bank_id'] == bank_id]
    assert len(original['ordered_pairs']) == 2 and len(selected) == 1
    view = copy.deepcopy(original)
    view['ordered_pairs'] = selected
    view['composition']['bank_order'] = [bank_id]
    view['composition']['mode'] = 'single_bank_ablation'
    view['artifact_closure'] = [selected[0]['vector_payload']]
    return view
```

The complete archived runner also records the original and inactive banks, binds artifact paths, and checks its scientific settings. It passes the resulting controller view through `GenerationRequest` to `CallbackGenerationBackend.generate_rows`, then sends the predictions through `EvaluationRequest` to the existing task evaluator. Test inputs, generation settings, selected alpha values, and scoring definitions come from the original frozen condition.

To inspect the available cases before running, use this command from the repository root:

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path('bank_controls/execution_dpo_sv_20260926/cases.json')
for i, case in enumerate(json.loads(path.read_text())):
    print(i, case['cell'], case['bank_id'], case['expected_prediction_rows'])
PY
```

For execution, copy the runner and case specification to a new working directory, bind the anonymized paths and code-revision aliases to the indexed source and parent artifacts, and invoke `python run_case.py CASE_INDEX`. The frozen runners retain their H100 check. Outputs include the controller view, predictions, trajectories, per-sample scores, summary metrics, and a completion record. The [reproduction guide](REPRODUCTION.md) describes the environment and output requirements.

## Controls through composition

For available components, express the experimental variable in the method/configuration and stage sequence. For example, Site chooses a registered position method while reusing the bank, generation, and evaluation machinery; single-bank controls choose a controller view while reusing generation and scoring. Keep each comparison's input construction, selection roles, timing definitions, and evaluation settings explicit in its configuration. Changing a component's implementation or objective is a method change and belongs in that backend.
