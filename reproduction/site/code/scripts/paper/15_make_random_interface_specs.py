from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


HEAD_RE = re.compile(r"^L(\d+)\.attn\.h(\d+)$")
MLP_RE = re.compile(r"^L(\d+)\.mlp$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create pre-registered random-interface specs for the site challenge. "
            "The selected sites are treated as one shared pool; random controls sample "
            "the same number of sites and then split the pool into role_a/role_b groups."
        )
    )
    p.add_argument("--family", choices=("mlp", "head"), required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--selected-role-a", required=True, help="Comma-separated selected sites for role A.")
    p.add_argument("--selected-role-b", required=True, help="Comma-separated selected sites for role B.")
    p.add_argument("--role-a-name", default="role_a")
    p.add_argument("--role-b-name", default="role_b")
    p.add_argument("--n-layers", type=int, required=True)
    p.add_argument("--n-heads", type=int, default=0, help="Required for --family=head.")
    p.add_argument("--neighbor-radius", type=int, default=3)
    p.add_argument("--front-max-layer", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260703)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def stable_seed(seed: int, *parts: object) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(seed) + int(h[:8], 16)


def layer_of(site: str) -> int:
    m = HEAD_RE.fullmatch(site) or MLP_RE.fullmatch(site)
    if not m:
        raise ValueError(f"Bad site id: {site}")
    return int(m.group(1))


def head_idx_of(site: str) -> int:
    m = HEAD_RE.fullmatch(site)
    if not m:
        raise ValueError(f"Bad head id: {site}")
    return int(m.group(2))


def all_sites(family: str, n_layers: int, n_heads: int) -> list[str]:
    if family == "mlp":
        return [f"L{layer}.mlp" for layer in range(n_layers)]
    if n_heads <= 0:
        raise ValueError("--n-heads must be positive for --family=head")
    return [f"L{layer}.attn.h{head}" for layer in range(n_layers) for head in range(n_heads)]


def component_row(site: str) -> dict[str, str]:
    return {
        "component_id": site,
        "layer_idx": str(layer_of(site)),
        "component_type": "mlp",
    }


def head_row(site: str) -> dict[str, str]:
    return {
        "head_id": site,
        "layer_idx": str(layer_of(site)),
        "head_idx": str(head_idx_of(site)),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        fp.write(",".join(fields) + "\n")
        for row in rows:
            fp.write(",".join(str(row.get(field, "")) for field in fields) + "\n")


def write_site_files(
    *,
    out_dir: Path,
    family: str,
    control: str,
    pool: list[str],
    role_a_size: int,
) -> dict[str, object]:
    control_dir = out_dir / control
    control_dir.mkdir(parents=True, exist_ok=True)
    role_a = pool[:role_a_size]
    role_b = pool[role_a_size:]
    (control_dir / "pool.txt").write_text(",".join(pool) + "\n", encoding="utf-8")
    (control_dir / "role_a.txt").write_text(",".join(role_a) + "\n", encoding="utf-8")
    (control_dir / "role_b.txt").write_text(",".join(role_b) + "\n", encoding="utf-8")
    if family == "mlp":
        write_csv(control_dir / "pool_components.csv", [component_row(site) for site in pool])
        write_csv(control_dir / "role_a_components.csv", [component_row(site) for site in role_a])
        write_csv(control_dir / "role_b_components.csv", [component_row(site) for site in role_b])
    else:
        write_csv(control_dir / "pool_heads.csv", [head_row(site) for site in pool])
        write_csv(control_dir / "role_a_heads.csv", [head_row(site) for site in role_a])
        write_csv(control_dir / "role_b_heads.csv", [head_row(site) for site in role_b])
        (control_dir / "role_a_heads.txt").write_text(",".join(role_a) + "\n", encoding="utf-8")
        (control_dir / "role_b_heads.txt").write_text(",".join(role_b) + "\n", encoding="utf-8")
    return {"pool": pool, "role_a": role_a, "role_b": role_b}


def sample_pool(
    *,
    rng: random.Random,
    candidates: list[str],
    k: int,
    control: str,
) -> list[str]:
    unique = sorted(dict.fromkeys(candidates), key=lambda item: (layer_of(item), item))
    if len(unique) < k:
        raise SystemExit(f"{control}: need {k} candidates, found {len(unique)}")
    return rng.sample(unique, k)


def main() -> None:
    args = parse_args()
    role_a = parse_csv(args.selected_role_a)
    role_b = parse_csv(args.selected_role_b)
    selected_pool = [*role_a, *role_b]
    if not role_a or not role_b:
        raise SystemExit("Both selected roles must be non-empty.")
    if len(set(selected_pool)) != len(selected_pool):
        raise SystemExit(f"Selected pool contains duplicates: {selected_pool}")

    n_total = len(selected_pool)
    selected_set = set(selected_pool)
    selected_layers = {layer_of(site) for site in selected_pool}
    neighbor_layers = {
        layer + offset
        for layer in selected_layers
        for offset in range(-args.neighbor_radius, args.neighbor_radius + 1)
        if 0 <= layer + offset < args.n_layers
    }
    front_layers = {layer for layer in range(args.n_layers) if layer <= args.front_max_layer}
    nonfront_layers = {layer for layer in range(args.n_layers) if layer > args.front_max_layer}
    universe = all_sites(args.family, args.n_layers, args.n_heads)

    def not_selected(site: str) -> bool:
        return site not in selected_set

    neighborhood_candidates = [
        site for site in universe if not_selected(site) and layer_of(site) in neighbor_layers
    ]
    nonlocal_nonfront_candidates = [
        site
        for site in universe
        if not_selected(site) and layer_of(site) in nonfront_layers and layer_of(site) not in neighbor_layers
    ]
    front_candidates = [
        site for site in universe if not_selected(site) and layer_of(site) in front_layers
    ]

    controls: dict[str, dict[str, object]] = {}
    rng = random.Random(stable_seed(args.seed, args.task, args.family, "rcm_neighborhood_random"))
    controls["rcm_neighborhood_random"] = write_site_files(
        out_dir=args.out_dir,
        family=args.family,
        control="rcm_neighborhood_random",
        pool=sample_pool(
            rng=rng,
            candidates=neighborhood_candidates,
            k=n_total,
            control="rcm_neighborhood_random",
        ),
        role_a_size=len(role_a),
    )

    rng = random.Random(stable_seed(args.seed, args.task, args.family, "nonlocal_nonfront_random"))
    controls["nonlocal_nonfront_random"] = write_site_files(
        out_dir=args.out_dir,
        family=args.family,
        control="nonlocal_nonfront_random",
        pool=sample_pool(
            rng=rng,
            candidates=nonlocal_nonfront_candidates,
            k=n_total,
            control="nonlocal_nonfront_random",
        ),
        role_a_size=len(role_a),
    )

    rng = random.Random(stable_seed(args.seed, args.task, args.family, "front_forced_random"))
    front_one = sample_pool(
        rng=rng,
        candidates=front_candidates,
        k=1,
        control="front_forced_random_front_slot",
    )
    remaining = sample_pool(
        rng=rng,
        candidates=[site for site in nonlocal_nonfront_candidates if site not in set(front_one)],
        k=n_total - 1,
        control="front_forced_random_nonfront_slots",
    )
    forced_pool = [*front_one, *remaining]
    rng.shuffle(forced_pool)
    controls["front_forced_random"] = write_site_files(
        out_dir=args.out_dir,
        family=args.family,
        control="front_forced_random",
        pool=forced_pool,
        role_a_size=len(role_a),
    )

    if args.family == "head":
        rng = random.Random(stable_seed(args.seed, args.task, args.family, "same_layer_random_head"))
        same_layer_pool: list[str] = []
        used = set(selected_pool)
        for source in selected_pool:
            layer = layer_of(source)
            candidates = [
                f"L{layer}.attn.h{head}"
                for head in range(args.n_heads)
                if f"L{layer}.attn.h{head}" not in used
            ]
            if not candidates:
                raise SystemExit(f"same_layer_random_head: no candidate for {source}")
            choice = rng.choice(sorted(candidates))
            same_layer_pool.append(choice)
            used.add(choice)
        controls["same_layer_random_head"] = write_site_files(
            out_dir=args.out_dir,
            family=args.family,
            control="same_layer_random_head",
            pool=same_layer_pool,
            role_a_size=len(role_a),
        )

    write_site_files(
        out_dir=args.out_dir,
        family=args.family,
        control="selected",
        pool=selected_pool,
        role_a_size=len(role_a),
    )
    manifest = {
        "task": args.task,
        "family": args.family,
        "role_a_name": args.role_a_name,
        "role_b_name": args.role_b_name,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads if args.family == "head" else "",
        "seed": args.seed,
        "neighbor_radius": args.neighbor_radius,
        "front_max_layer": args.front_max_layer,
        "selected_layers": sorted(selected_layers),
        "neighbor_layers": sorted(neighbor_layers),
        "front_layers": sorted(front_layers),
        "nonfront_layers": sorted(nonfront_layers),
        "candidate_counts": {
            "rcm_neighborhood_random": len(set(neighborhood_candidates)),
            "nonlocal_nonfront_random": len(set(nonlocal_nonfront_candidates)),
            "front_forced_random_front_slot": len(set(front_candidates)),
        },
        "selected": {"pool": selected_pool, "role_a": role_a, "role_b": role_b},
        "controls": controls,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "random_interface_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
