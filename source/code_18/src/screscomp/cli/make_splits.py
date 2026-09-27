from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from screscomp.data import dump_json, load_jsonl
from screscomp.prompts import SUPPORTED_TEMPLATE_FAMILIES


def _parse_family_csv(raw: str) -> list[str]:
    parts = [x.strip() for x in raw.split(",")]
    families = [x for x in parts if x]
    if not families:
        raise ValueError("At least one template family is required.")
    unknown = sorted(set(families) - SUPPORTED_TEMPLATE_FAMILIES)
    if unknown:
        raise ValueError(f"Unsupported template families: {unknown}")
    return families


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create group-based data splits.")
    p.add_argument("--retained_jsonl", type=Path, required=True)
    p.add_argument("--out_manifest", type=Path, required=True)
    p.add_argument(
        "--group_key",
        type=str,
        default="subject_id",
        help="Field used to group samples into non-overlapping splits, e.g. subject_id, fact_id, split_group_id.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--discovery_ratio", type=float, default=0.6)
    p.add_argument("--validation_ratio", type=float, default=0.2)
    p.add_argument(
        "--stratify_by_relation",
        action="store_true",
        help="Allocate groups within each relation bucket to keep relation coverage balanced across splits.",
    )
    p.add_argument(
        "--discovery_template_families",
        type=str,
        default="main_v1",
        help="Comma-separated template families used by discovery split.",
    )
    p.add_argument(
        "--heldout_template_families",
        type=str,
        default="heldout_v1",
        help="Comma-separated template families reserved for held-out evaluation.",
    )
    p.add_argument(
        "--validation_template_mode",
        type=str,
        default="mixed",
        choices=["discovery_only", "mixed", "heldout_only"],
        help="Template pool for validation split.",
    )
    return p.parse_args()


def _choose_template_family(sample_id: str, families: list[str], seed: int) -> str:
    if len(families) == 1:
        return families[0]
    rng = random.Random(f"{seed}:{sample_id}")
    return families[rng.randrange(len(families))]


def _write_id_files(out_manifest: Path, sample_to_split: dict[str, str]) -> None:
    out_dir = out_manifest.parent
    split_to_ids: dict[str, list[str]] = {"discovery": [], "validation": [], "test": []}
    for sample_id, split in sample_to_split.items():
        split_to_ids[split].append(sample_id)
    for split, ids in split_to_ids.items():
        path = out_dir / f"{split}_ids.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(sorted(ids)), encoding="utf-8")


def _relation_signature(rows: list[dict]) -> str:
    relations = sorted({str(row["relation"]) for row in rows})
    return "|".join(relations)


def _assign_groups(
    grouped_ids: list[str],
    discovery_ratio: float,
    validation_ratio: float,
    rng: random.Random,
) -> tuple[set[str], set[str], set[str]]:
    shuffled = list(grouped_ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_disc = int(n * discovery_ratio)
    n_val = int(n * validation_ratio)
    disc_groups = set(shuffled[:n_disc])
    val_groups = set(shuffled[n_disc : n_disc + n_val])
    test_groups = set(shuffled[n_disc + n_val :])
    return disc_groups, val_groups, test_groups


def main() -> None:
    args = parse_args()
    if args.discovery_ratio <= 0 or args.validation_ratio < 0:
        raise ValueError("Split ratios must satisfy discovery_ratio > 0 and validation_ratio >= 0.")
    if args.discovery_ratio + args.validation_ratio >= 1.0:
        raise ValueError("Split ratios must satisfy discovery_ratio + validation_ratio < 1.0.")

    rows = load_jsonl(args.retained_jsonl)
    if not rows:
        raise ValueError("Retained JSONL is empty.")
    if args.group_key not in rows[0]:
        raise ValueError(f"Group key {args.group_key!r} is not present in retained rows.")
    discovery_families = _parse_family_csv(args.discovery_template_families)
    heldout_families = _parse_family_csv(args.heldout_template_families)

    overlap = set(discovery_families) & set(heldout_families)
    if overlap:
        raise ValueError(
            "Discovery and held-out template families must be disjoint. "
            f"Overlapping families: {sorted(overlap)}"
        )

    validation_families: list[str]
    if args.validation_template_mode == "discovery_only":
        validation_families = discovery_families
    elif args.validation_template_mode == "heldout_only":
        validation_families = heldout_families
    else:
        validation_families = [*discovery_families, *heldout_families]

    groups: dict[str, list[dict]] = defaultdict(list)
    sample_to_subject: dict[str, str] = {}
    for row in rows:
        groups[str(row[args.group_key])].append(row)
        sample_to_subject[row["sample_id"]] = row["subject_id"]

    rng = random.Random(args.seed)
    if args.stratify_by_relation:
        relation_to_group_ids: dict[str, list[str]] = defaultdict(list)
        for gid, group_rows in groups.items():
            relation_to_group_ids[_relation_signature(group_rows)].append(gid)
        disc_groups: set[str] = set()
        val_groups: set[str] = set()
        test_groups: set[str] = set()
        for relation_key in sorted(relation_to_group_ids):
            d, v, t = _assign_groups(
                grouped_ids=relation_to_group_ids[relation_key],
                discovery_ratio=args.discovery_ratio,
                validation_ratio=args.validation_ratio,
                rng=rng,
            )
            disc_groups.update(d)
            val_groups.update(v)
            test_groups.update(t)
    else:
        disc_groups, val_groups, test_groups = _assign_groups(
            grouped_ids=list(groups.keys()),
            discovery_ratio=args.discovery_ratio,
            validation_ratio=args.validation_ratio,
            rng=rng,
        )

    sample_to_split: dict[str, str] = {}
    group_to_split: dict[str, str] = {}
    group_to_relation_signature: dict[str, str] = {}
    for gid, group_rows in groups.items():
        split = "test"
        if gid in disc_groups:
            split = "discovery"
        elif gid in val_groups:
            split = "validation"
        group_to_split[gid] = split
        group_to_relation_signature[gid] = _relation_signature(group_rows)
        for row in group_rows:
            sample_to_split[row["sample_id"]] = split

    sample_to_template_family: dict[str, str] = {}
    sample_to_template_families: dict[str, list[str]] = {}
    split_families = {
        "discovery": discovery_families,
        "validation": validation_families,
        "test": heldout_families,
    }
    for sample_id, split in sample_to_split.items():
        families = split_families[split]
        sample_to_template_families[sample_id] = families
        sample_to_template_family[sample_id] = _choose_template_family(sample_id=sample_id, families=families, seed=args.seed)

    subject_to_splits: dict[str, set[str]] = defaultdict(set)
    for sample_id, split in sample_to_split.items():
        subject_to_splits[sample_to_subject[sample_id]].add(split)
    cross_split_subjects = sorted(subject for subject, splits in subject_to_splits.items() if len(splits) > 1)

    manifest = {
        "group_key": args.group_key,
        "stratify_by_relation": args.stratify_by_relation,
        "seed": args.seed,
        "ratios": {
            "discovery": args.discovery_ratio,
            "validation": args.validation_ratio,
            "test": 1.0 - args.discovery_ratio - args.validation_ratio,
        },
        "template_families_by_split": {
            "discovery": discovery_families,
            "validation": validation_families,
            "test": heldout_families,
        },
        "template_bucket_by_family": {
            **{f: "discovery" for f in discovery_families},
            **{f: "heldout" for f in heldout_families},
        },
        "group_to_split": group_to_split,
        "group_to_relation_signature": group_to_relation_signature,
        "sample_to_split": sample_to_split,
        "sample_to_template_family": sample_to_template_family,
        "sample_to_template_families": sample_to_template_families,
        "subject_to_split_consistent": len(cross_split_subjects) == 0,
        "cross_split_subjects": cross_split_subjects,
    }

    dump_json(args.out_manifest, manifest)
    _write_id_files(args.out_manifest, sample_to_split)
    print(
        "[split] groups="
        f"{len(groups)} discovery={len(disc_groups)} validation={len(val_groups)} test={len(test_groups)} "
        f"cross_subjects={len(cross_split_subjects)}"
    )


if __name__ == "__main__":
    main()
