from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import replace

from screscomp.core.schemas import ExcludedFact, FactPairCandidate, FactTriple
from screscomp.prompts.templates import render_prior_only


def _norm(s: str) -> str:
    return " ".join(s.strip().split())


def _truthy(value: str) -> bool:
    return _norm(value).lower() in {"1", "true", "yes", "y", "t"}


def _same_answer(a: str, b: str) -> bool:
    return _norm(a).casefold() == _norm(b).casefold()


def _split_multi(value: str) -> list[str]:
    raw = _norm(value)
    if not raw:
        return []
    return [_norm(part) for part in raw.split("|") if _norm(part)]


def _align_counter_metadata(values: list[str], target_len: int) -> list[str]:
    if target_len <= 0:
        return []
    if not values:
        return [""] * target_len
    if len(values) == 1:
        return values * target_len
    if len(values) != target_len:
        raise ValueError(
            "Explicit counter metadata must have either one value or the same number of values as counter_answer. "
            f"Got {len(values)} values for {target_len} counters."
        )
    return values


def _merge_pipe_fields(left: str, right: str) -> str:
    merged: list[str] = []
    seen: set[str] = set()
    for value in [*_split_multi(left), *_split_multi(right)]:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append(value)
    return "|".join(merged)


def parse_seed_rows(rows: list[dict[str, str]]) -> list[FactTriple]:
    triples: list[FactTriple] = []
    for i, row in enumerate(rows):
        fact_id = row.get("fact_id") or f"fact_{i:06d}"
        subject = _norm(row["subject"])
        relation = _norm(row["relation"])
        question = _norm(row.get("question") or f"What is {subject}'s {relation}?")
        true_answer = _norm(row["true_answer"])
        answer_type = _norm(row.get("answer_type") or "text")
        domain = _norm(row.get("domain") or "general")
        source = _norm(row.get("source") or "seed")
        source_date = _norm(row.get("source_date") or "")
        true_answer_id = _norm(row.get("true_answer_id") or true_answer)
        counter_answer = _norm(row.get("counter_answer") or "")
        counter_answer_id = _norm(row.get("counter_answer_id") or "")
        counter_difficulty = _norm(row.get("counter_difficulty") or "")
        counter_rationale = _norm(row.get("counter_rationale") or "")
        subject_id = _norm(row.get("subject_id") or subject)
        split_group_id = _norm(row.get("split_group_id") or subject_id)
        single_answer_certainty = _norm(row.get("single_answer_certainty") or "")
        time_sensitive = _truthy(row.get("time_sensitive") or "")
        multi_answer = _truthy(row.get("multi_answer") or "")
        ambiguous = _truthy(row.get("ambiguous") or "")
        triples.append(
            FactTriple(
                fact_id=fact_id,
                subject_id=subject_id,
                subject=subject,
                relation=relation,
                question=question,
                true_answer=true_answer,
                answer_type=answer_type,
                domain=domain,
                source=source,
                source_date=source_date,
                true_answer_id=true_answer_id,
                counter_answer=counter_answer,
                counter_answer_id=counter_answer_id,
                counter_difficulty=counter_difficulty,
                counter_rationale=counter_rationale,
                split_group_id=split_group_id,
                single_answer_certainty=single_answer_certainty,
                time_sensitive=time_sensitive,
                multi_answer=multi_answer,
                ambiguous=ambiguous,
            )
        )
    return triples


def normalize_triples(triples: list[FactTriple]) -> list[FactTriple]:
    dedup: dict[tuple[str, str], FactTriple] = {}
    for t in triples:
        key = (t.subject_id.casefold(), t.relation.casefold())
        existing = dedup.get(key)
        if existing is None:
            dedup[key] = t
            continue
        dedup[key] = replace(
            existing,
            counter_answer=_merge_pipe_fields(existing.counter_answer, t.counter_answer),
            counter_answer_id=_merge_pipe_fields(existing.counter_answer_id, t.counter_answer_id),
            counter_difficulty=_merge_pipe_fields(existing.counter_difficulty, t.counter_difficulty),
            counter_rationale=_merge_pipe_fields(existing.counter_rationale, t.counter_rationale),
        )
    return list(dedup.values())


def filter_stable_triples(triples: list[FactTriple]) -> tuple[list[FactTriple], list[ExcludedFact]]:
    retained: list[FactTriple] = []
    excluded: list[ExcludedFact] = []
    for t in triples:
        if t.time_sensitive:
            excluded.append(
                ExcludedFact(fact_id=t.fact_id, subject_id=t.subject_id, exclusion_reason="time_sensitive_fact")
            )
            continue
        if t.multi_answer:
            excluded.append(ExcludedFact(fact_id=t.fact_id, subject_id=t.subject_id, exclusion_reason="multi_answer_fact"))
            continue
        if t.ambiguous:
            excluded.append(ExcludedFact(fact_id=t.fact_id, subject_id=t.subject_id, exclusion_reason="ambiguous_question"))
            continue
        retained.append(t)
    return retained, excluded


def generate_fact_pair_candidates(
    triples: list[FactTriple],
    num_counters: int = 1,
    seed: int = 42,
) -> list[FactPairCandidate]:
    rng = random.Random(seed)
    pool_by_relation: dict[tuple[str, str], list[FactTriple]] = defaultdict(list)
    for t in triples:
        pool_by_relation[(t.relation.casefold(), t.answer_type.casefold())].append(t)

    pairs: list[FactPairCandidate] = []
    for t in triples:
        explicit_counters = _split_multi(t.counter_answer)
        if explicit_counters:
            counter_ids = _align_counter_metadata(_split_multi(t.counter_answer_id), len(explicit_counters))
            counter_difficulties = _align_counter_metadata(_split_multi(t.counter_difficulty), len(explicit_counters))
            counter_rationales = _align_counter_metadata(_split_multi(t.counter_rationale), len(explicit_counters))
            seen_counter: set[str] = set()
            for j, counter in enumerate(explicit_counters):
                counter_key = counter.casefold()
                if counter_key in seen_counter or _same_answer(counter, t.true_answer):
                    continue
                seen_counter.add(counter_key)
                sample_id = f"{t.fact_id}_{j:02d}"
                pairs.append(
                    FactPairCandidate(
                        sample_id=sample_id,
                        fact_id=t.fact_id,
                        subject_id=t.subject_id,
                        subject=t.subject,
                        relation=t.relation,
                        question=t.question,
                        answer_type=t.answer_type,
                        domain=t.domain,
                        true_answer_id=t.true_answer_id,
                        split_group_id=t.split_group_id,
                        candidate_1=t.true_answer,
                        candidate_2=counter,
                        true_answer=t.true_answer,
                        prior_only_prompt="",
                        swapped_prior_only_prompt="",
                        counter_answer_id=counter_ids[j],
                        counter_difficulty=counter_difficulties[j],
                        counter_rationale=counter_rationales[j],
                    )
                )
            continue

        pool = [
            x
            for x in pool_by_relation[(t.relation.casefold(), t.answer_type.casefold())]
            if not _same_answer(x.true_answer, t.true_answer)
        ]
        if not pool:
            continue
        rng.shuffle(pool)
        seen_counter: set[str] = set()
        for j, c in enumerate(pool[:num_counters]):
            counter = _norm(c.true_answer)
            counter_key = counter.casefold()
            if counter_key in seen_counter:
                continue
            seen_counter.add(counter_key)

            sample_id = f"{t.fact_id}_{j:02d}"
            pairs.append(
                FactPairCandidate(
                    sample_id=sample_id,
                    fact_id=t.fact_id,
                    subject_id=t.subject_id,
                    subject=t.subject,
                    relation=t.relation,
                    question=t.question,
                    answer_type=t.answer_type,
                    domain=t.domain,
                    true_answer_id=t.true_answer_id,
                    split_group_id=t.split_group_id,
                    candidate_1=t.true_answer,
                    candidate_2=counter,
                    true_answer=t.true_answer,
                    prior_only_prompt="",
                    swapped_prior_only_prompt="",
                    counter_answer_id=_norm(c.true_answer_id or c.true_answer),
                    counter_difficulty="random_same_type",
                    counter_rationale="sampled from the same relation and answer_type pool",
                )
            )
    return pairs


def build_prior_ready_pairs(
    pairs: list[FactPairCandidate],
    template_family: str = "main_v1",
) -> list[FactPairCandidate]:
    ready: list[FactPairCandidate] = []
    for p in pairs:
        prior = render_prior_only(
            question=p.question,
            option_a=p.candidate_1,
            option_b=p.candidate_2,
            family=template_family,
        )
        swapped = render_prior_only(
            question=p.question,
            option_a=p.candidate_2,
            option_b=p.candidate_1,
            family=template_family,
        )
        ready.append(
            FactPairCandidate(
                sample_id=p.sample_id,
                fact_id=p.fact_id,
                subject_id=p.subject_id,
                subject=p.subject,
                relation=p.relation,
                question=p.question,
                answer_type=p.answer_type,
                domain=p.domain,
                true_answer_id=p.true_answer_id,
                split_group_id=p.split_group_id,
                candidate_1=p.candidate_1,
                candidate_2=p.candidate_2,
                true_answer=p.true_answer,
                prior_only_prompt=prior,
                swapped_prior_only_prompt=swapped,
                counter_answer_id=p.counter_answer_id,
                counter_difficulty=p.counter_difficulty,
                counter_rationale=p.counter_rationale,
            )
        )
    return ready
