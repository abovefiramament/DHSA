"""Recover score operands algebraically from saved Site scores and signed effects.

No model, scorer, sampling, or API is invoked. Frozen source files are read-only.
Zero: delta = native - changed. Patch: delta = changed - native.
"""
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(encoding="utf-8") as handle:
        yield from (json.loads(line) for line in handle if line.strip())


def main():
    for family in ("zero", "patch"):
        base = ROOT / f"site/imdb_qwen/artifacts/shared_selector/imdb/sentiment/qwen25_14b/rcm_{family}_signed_staged_v1"
        baseline = list(rows(base / "scores/baseline_scores.jsonl"))
        sources = sorted(base.glob("head_scores/*/*/candidate_result.jsonl"))
        sources += sorted(base.glob("layer_scores/candidates/*/candidate_result.jsonl"))
        output = ROOT / "mechanism/imdb_qwen_site_operands" / family
        output.mkdir(parents=True, exist_ok=True)
        count = 0
        with gzip.open(output / "derived_component_scores.jsonl.gz", "wt", encoding="utf-8") as handle:
            for source in sources:
                candidate, = list(rows(source))
                if source.parent.parent.parent.name == "head_scores":
                    generation = base / "heads" / source.parent.parent.name / source.parent.name / "generations.jsonl"
                else:
                    generation = base / "layers/layers" / source.parent.name / "generations.jsonl"
                generated = list(rows(generation))
                assert len(generated) == len(baseline) == len(candidate["deltas"]) == 300
                for i, (native, changed_text, delta) in enumerate(zip(baseline, generated, candidate["deltas"])):
                    assert native["sample_id"] == changed_text["sample_id"]
                    value = native["competition_score"]
                    changed = value - delta if family == "zero" else value + delta
                    assert -1.00000001 <= changed <= 1.00000001
                    row = {"sample_id": native["sample_id"], "sample_index": i,
                           "component_id": candidate["component_id"],
                           "native_score": value, "delta": delta, "changed_score": changed,
                           "changed_score_origin": "algebraic_recovery_from_saved_native_and_delta",
                           "effect_file": source.relative_to(ROOT).as_posix(),
                           "changed_text_file": generation.relative_to(ROOT).as_posix()}
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
        summary = {"family": family, "samples": len(baseline), "components": len(sources),
                   "component_sample_rows": count,
                   "formula": "changed = native - delta" if family == "zero" else "changed = native + delta",
                   "native_scores": (base / "scores/baseline_scores.jsonl").relative_to(ROOT).as_posix(),
                   "execution_source": "site/imdb_qwen/reproduction/code/src/screscomp/cli/site_select_imdb_rcm_staged.py",
                   "execution_function": "_score_candidate",
                   "scope": "Derived operands, not separately archived scorer outputs; floating-point inversion is not a fresh scoring pass. Original effects and text are unchanged."}
        (output / "derivation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"{family}: {count} derived rows; source text and sample order verified.")


if __name__ == "__main__":
    main()
