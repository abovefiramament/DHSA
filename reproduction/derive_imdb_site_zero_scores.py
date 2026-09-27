"""Reconstruct IMDb GPT-2 Large Site Zero scores from archived paired deltas."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "site/selection"
JOB = "imdb__sentiment__ma921_gpt2_large_sft__rcm_zero_signed__primary"
NATIVE = SELECTOR / JOB / "selector/native_baseline.jsonl"
SCAN = SELECTOR / "shared_selector/imdb/sentiment/ma921_gpt2_large_sft/rcm_zero"
OUTPUT = SELECTOR / JOB / "selector/reconstructed_paired_scores.jsonl"
INDEX = ROOT / "EVIDENCE_INDEX.json"


def rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    native = rows(NATIVE)
    sample_ids = [str(row["sample_id"]) for row in native]
    if len(native) != 300 or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Expected 300 distinct archived native selector samples")

    sources = (
        ("layer", SCAN / "layer_results.jsonl"),
        ("head", SCAN / "positive_head_results.jsonl"),
        ("head", SCAN / "negative_head_results.jsonl"),
    )
    output_rows = []
    components = set()
    for level, path in sources:
        for component in rows(path):
            component_id = str(component["component_id"])
            if component_id in components:
                raise ValueError(f"Duplicate component: {component_id}")
            components.add(component_id)
            deltas = [float(value) for value in component["deltas"]]
            if len(deltas) != len(native) or int(component["n"]) != len(native):
                raise ValueError(f"Sample count mismatch: {component_id}")
            if abs(sum(deltas) / len(deltas) - float(component["mean_score_delta"])) > 1e-9:
                raise ValueError(f"Mean delta mismatch: {component_id}")
            for source, delta in zip(native, deltas, strict=True):
                native_score = float(source["competition_score"])
                intervened_score = native_score - delta  # Zero: delta = native - zero.
                if not -1.000001 <= intervened_score <= 1.000001:
                    raise ValueError(f"Reconstructed score outside [-1,1]: {component_id}")
                output_rows.append({
                    "sample_id": str(source["sample_id"]),
                    "component_id": component_id,
                    "component_level": level,
                    "native_score": native_score,
                    "intervened_score_reconstructed": intervened_score,
                    "delta_native_minus_zero": delta,
                })

    with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    index = json.loads(INDEX.read_text(encoding="utf-8"))
    cell = "site/selection/" + JOB
    target = next(row for row in index["cells"] if row["cell"] == cell)
    rel = OUTPUT.relative_to(ROOT).as_posix()
    target["files"]["rcm_reconstructed_scores"] = [rel]
    script = Path(__file__).relative_to(ROOT).as_posix()
    if script not in target["files"]["reproduction_scripts"]:
        target["files"]["reproduction_scripts"].append(script)
    INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Reconstructed {len(output_rows)} paired rows for {len(components)} components")


if __name__ == "__main__":
    main()
