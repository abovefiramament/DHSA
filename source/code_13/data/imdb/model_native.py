"""Model-native IMDb source projection for the formal shared data controller."""

from __future__ import annotations

import random
from typing import Any, Callable, Mapping, Sequence

import torch

from data._shared.controller_api import ControllerError
from evaluators.imdb import IMDbSentimentEvaluator


PromptRenderer = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ControllerError(f"IMDb {field} must be a positive integer")
    return value


def _window(spec: Mapping[str, Any], key: str) -> tuple[int, int]:
    value = spec.get(key)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] < 0
        or value[1] <= value[0]
    ):
        raise ControllerError(f"IMDb {key} must be a zero-based half-open window")
    return int(value[0]), int(value[1])


def _prefix_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    adapter: Any,
    construction: Mapping[str, Any],
) -> list[dict[str, Any]]:
    order = construction.get("source_order")
    prefix_spec = construction.get("prefix")
    if not isinstance(order, Mapping) or not isinstance(prefix_spec, Mapping):
        raise ControllerError("IMDb source_order and prefix settings must be registered")
    if order.get("algorithm") != "CPython_random_Random_shuffle" or order.get("shuffle") is not True:
        raise ControllerError("IMDb requires the registered CPython shuffle")
    limit = _positive_int(order.get("max_rows"), field="source_order.max_rows")
    seed = order.get("seed")
    minimum = _positive_int(prefix_spec.get("min_tokens"), field="prefix.min_tokens")
    maximum = _positive_int(prefix_spec.get("max_tokens"), field="prefix.max_tokens")
    if not isinstance(seed, int) or isinstance(seed, bool) or maximum < minimum:
        raise ControllerError("IMDb prefix/shuffle seed or token range is invalid")
    if len(rows) < limit:
        raise ControllerError(f"IMDb source has {len(rows)} rows; requires {limit}")

    rng = random.Random(seed)
    prepared: list[dict[str, Any]] = []
    for source_row_index, source in enumerate(rows[:limit]):
        text = source.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        target_length = rng.randint(minimum, maximum)
        encoded = adapter.tokenizer(
            text.strip(),
            add_special_tokens=False,
            truncation=True,
            max_length=target_length,
        )["input_ids"]
        if len(encoded) < minimum:
            continue
        prefix = adapter.tokenizer.decode(encoded, skip_special_tokens=True).strip()
        if not prefix:
            continue
        prepared.append(
            {
                "source_row_index": source_row_index,
                "prefix": prefix,
                "prefix_token_count": len(encoded),
                "source_label": source.get("label"),
            }
        )
    rng.shuffle(prepared)
    for position, row in enumerate(prepared):
        row["source_order_position"] = position
    return prepared


def _trim_generated(ids: Sequence[int], eos_token_id: int | None) -> list[int]:
    values = [int(value) for value in ids]
    if eos_token_id is not None and eos_token_id in values:
        values = values[: values.index(eos_token_id) + 1]
    return values


def _generate_batches(
    adapter: Any,
    prompts: Sequence[str],
    *,
    generation: Mapping[str, Any],
    completion_index: int,
    completion_seed_stride: int,
) -> list[dict[str, Any]]:
    batch_size = _positive_int(
        generation.get("generation_batch_size"), field="generation_batch_size"
    )
    batch_stride = _positive_int(
        generation.get("batch_seed_stride"), field="batch_seed_stride"
    )
    base_seed = generation.get("seed")
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise ControllerError("IMDb generation seed must be an integer")
    min_new_tokens = generation.get("min_new_tokens")
    if min_new_tokens is not None:
        min_new_tokens = _positive_int(min_new_tokens, field="min_new_tokens")
    output: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(prompts), batch_size)):
        batch_prompts = list(prompts[start : start + batch_size])
        encoded = adapter.encode_prompts(batch_prompts)
        seed = base_seed + completion_index * completion_seed_stride + batch_index * batch_stride
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        kwargs: dict[str, Any] = {
            "max_new_tokens": _positive_int(generation.get("max_new_tokens"), field="max_new_tokens"),
            "do_sample": bool(generation.get("do_sample")),
            "pad_token_id": adapter.tokenizer.pad_token_id,
            "attention_mask": encoded["attention_mask"],
        }
        if min_new_tokens is not None:
            kwargs["min_new_tokens"] = min_new_tokens
        if kwargs["do_sample"]:
            kwargs.update(
                {
                    "temperature": float(generation.get("temperature", 1.0)),
                    "top_p": float(generation.get("top_p", 1.0)),
                    "top_k": int(generation.get("top_k", 50)),
                }
            )
        with torch.inference_mode():
            generated = adapter.model.generate(input_ids=encoded["input_ids"], **kwargs)
        prompt_width = int(encoded["input_ids"].shape[1])
        for row in generated:
            token_ids = _trim_generated(
                row[prompt_width:].tolist(), adapter.tokenizer.eos_token_id
            )
            generated_text = adapter.tokenizer.decode(
                token_ids, skip_special_tokens=True
            ).strip()
            if min_new_tokens is not None and not generated_text:
                raise ControllerError(
                    "IMDb model-native generation produced empty text despite "
                    "the registered min_new_tokens"
                )
            output.append(
                {
                    "generated_token_ids": token_ids,
                    "generated_text": generated_text,
                    "batch_seed": seed,
                }
            )
    if len(output) != len(prompts):
        raise ControllerError("IMDb model generation returned the wrong row count")
    return output


def _role_count(data_spec: Mapping[str, Any], role: str) -> int:
    scaled = data_spec.get("scaled_counts")
    if isinstance(scaled, Mapping) and role in scaled:
        return _positive_int(scaled[role], field=f"scaled_counts.{role}")
    roles = data_spec["construction"]["roles"]
    value = roles[role].get("target_count", roles[role].get("max_count"))
    return _positive_int(value, field=f"roles.{role}")


def prepare_roles(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    *,
    data_spec: Mapping[str, Any],
    model_provider: Any,
    render_prompts: PromptRenderer,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Construct every IMDb role from raw pinned train/test mirrors."""

    construction = data_spec.get("construction")
    runtime = data_spec.get("runtime")
    if not isinstance(construction, Mapping) or not isinstance(runtime, Mapping):
        raise ControllerError("IMDb construction and runtime settings must be registered")
    model = runtime.get("model")
    scorer = runtime.get("sentiment_scorer")
    if not isinstance(model, Mapping) or not isinstance(scorer, Mapping):
        raise ControllerError("IMDb model and sentiment scorer runtime settings are required")
    model_id = model.get("checkpoint")
    if not isinstance(model_id, str) or not model_id:
        raise ControllerError("IMDb runtime model checkpoint is missing")
    adapter = model_provider.load(model_id, mode="inference")
    adapter.configure_input(use_chat_template=bool(model.get("use_chat_template", False)))

    train_order = _prefix_records(train_rows, adapter=adapter, construction=construction)
    test_order = _prefix_records(test_rows, adapter=adapter, construction=construction)
    selector_spec = construction.get("selector_state_generation")
    pair_spec = construction.get("pair_preparation")
    test_spec = construction.get("test")
    if not all(isinstance(value, Mapping) for value in (selector_spec, pair_spec, test_spec)):
        raise ControllerError("IMDb selector, pair, and test construction settings are incomplete")
    selector_window = _window(selector_spec, "source_prompt_window")
    train_window = _window(pair_spec, "train_source_prompt_window")
    validation_window = _window(pair_spec, "validation_source_prompt_window")
    test_window = _window(test_spec, "post_shuffle_window")
    if selector_window[1] != train_window[0] or train_window[1] != validation_window[0]:
        raise ControllerError("IMDb train-role windows must be contiguous and disjoint")

    selector_count = _role_count(data_spec, "selector")
    selector_sources = train_order[selector_window[0] : selector_window[0] + selector_count]
    if len(selector_sources) != selector_count:
        raise ControllerError("IMDb selector source window is too small")
    rendered_selector = [render_prompts(str(row["prefix"]), data_spec["prompt_registry"]) for row in selector_sources]
    selector: list[dict[str, Any]] = []
    condition_rows: dict[str, list[dict[str, Any]]] = {}
    for role_name, prompt_key in (("good", "good_state"), ("base", "base_state")):
        prompts = [str(item["prompts"][prompt_key]) for item in rendered_selector]
        condition_rows[role_name] = _generate_batches(
            adapter,
            prompts,
            generation=selector_spec,
            completion_index=0,
            completion_seed_stride=1,
        )
    for index, (source, rendered) in enumerate(zip(selector_sources, rendered_selector, strict=True)):
        good = condition_rows["good"][index]
        base = condition_rows["base"][index]
        if good["batch_seed"] != base["batch_seed"]:
            raise ControllerError("IMDb selector conditions lost common random numbers")
        selector.append(
            {
                "sample_id": f"imdb-selector-{index:06d}",
                **dict(source),
                **dict(rendered),
                "good_state": {
                    "prompt": rendered["prompts"]["good_state"],
                    **good,
                },
                "base_state": {
                    "prompt": rendered["prompts"]["base_state"],
                    **base,
                },
                "batch_seed": good["batch_seed"],
                "selector_semantics": "shared_model_native_good_base_complete_states",
            }
        )

    evaluator = IMDbSentimentEvaluator(device=str(next(adapter.model.parameters()).device))
    scorer_path = scorer.get("local_path")
    scorer_revision = scorer.get("revision")
    reward_batch_size = scorer.get("reward_batch_size")
    if not isinstance(scorer_path, str) or not isinstance(scorer_revision, str):
        raise ControllerError("IMDb sentiment scorer path/revision must be resolved")
    reward_batch_size = _positive_int(reward_batch_size, field="reward_batch_size")
    completions_per_prompt = _positive_int(
        pair_spec.get("completions_per_prompt"), field="completions_per_prompt"
    )
    completion_stride = _positive_int(
        pair_spec.get("completion_seed_stride"), field="completion_seed_stride"
    )
    pair_generation = {**dict(selector_spec), **dict(pair_spec)}
    pair_generation.pop("min_new_tokens", None)

    def make_pairs(role: str, window: tuple[int, int]) -> list[dict[str, Any]]:
        target = _role_count(data_spec, role)
        candidate_count = (window[1] - window[0]) if data_spec.get("execution_profile") != "formal_scaled_gpu" else target
        sources = train_order[window[0] : window[0] + candidate_count]
        rendered = [render_prompts(str(row["prefix"]), data_spec["prompt_registry"]) for row in sources]
        prompts = [str(item["prompts"]["preference_prefix"]) for item in rendered]
        generated_by_completion = [
            _generate_batches(
                adapter,
                prompts,
                generation=pair_generation,
                completion_index=completion_index,
                completion_seed_stride=completion_stride,
            )
            for completion_index in range(completions_per_prompt)
        ]
        all_texts = [
            generated_by_completion[completion_index][prompt_index]["generated_text"]
            for prompt_index in range(len(prompts))
            for completion_index in range(completions_per_prompt)
        ]
        rewards = evaluator.score_completions(
            all_texts,
            path=scorer_path,
            revision=scorer_revision,
            batch_size=reward_batch_size,
        )
        pairs: list[dict[str, Any]] = []
        cursor = 0
        for index, (source, prompt_info) in enumerate(zip(sources, rendered, strict=True)):
            candidates = []
            for completion_index in range(completions_per_prompt):
                generated = generated_by_completion[completion_index][index]
                candidates.append((float(rewards[cursor]), completion_index, generated))
                cursor += 1
            candidates.sort(key=lambda item: (-item[0], item[1]))
            high, low = candidates[0], candidates[-1]
            if high[0] <= low[0]:
                continue
            pairs.append(
                {
                    "sample_id": f"imdb-{role}-{len(pairs):06d}",
                    **dict(source),
                    **dict(prompt_info),
                    "split": "train" if role == "training" else "validation",
                    "prompt": prompt_info["prompts"]["preference_prefix"],
                    "chosen": high[2]["generated_text"],
                    "rejected": low[2]["generated_text"],
                    "chosen_token_ids": high[2]["generated_token_ids"],
                    "rejected_token_ids": low[2]["generated_token_ids"],
                    "chosen_reward": high[0],
                    "rejected_reward": low[0],
                    "reward_margin": high[0] - low[0],
                }
            )
            if len(pairs) == target:
                break
        if len(pairs) != target:
            raise ControllerError(
                f"IMDb fixed {role} candidate window produced {len(pairs)} positive-margin pairs; requires {target}"
            )
        return pairs

    training = make_pairs("training", train_window)
    validation = make_pairs("validation", validation_window)
    requested_roles = data_spec.get("roles")
    include_audit = isinstance(requested_roles, Mapping) and "head_audit" in requested_roles
    head_audit: list[dict[str, Any]] = []
    audit_window: tuple[int, int] | None = None
    if include_audit:
        audit_spec = construction.get("head_audit")
        if not isinstance(audit_spec, Mapping):
            raise ControllerError("IMDb head-audit construction is missing")
        audit_window = _window(audit_spec, "post_shuffle_window")
        if validation_window[1] != audit_window[0]:
            raise ControllerError("IMDb audit window must follow validation without overlap")
        audit_count = _role_count(data_spec, "head_audit")
        audit_sources = train_order[audit_window[0] : audit_window[0] + audit_count]
        for index, source in enumerate(audit_sources):
            rendered = render_prompts(str(source["prefix"]), data_spec["prompt_registry"])
            head_audit.append(
                {
                    "sample_id": f"imdb-head-audit-{index:06d}",
                    **dict(source),
                    **dict(rendered),
                    "prompt": rendered["prompts"]["preference_prefix"],
                }
            )
        if len(head_audit) != audit_count:
            raise ControllerError("IMDb head-audit source window is too small")
    test_count = _role_count(data_spec, "test")
    test_sources = test_order[test_window[0] : test_window[0] + test_count]
    test = []
    for index, source in enumerate(test_sources):
        rendered = render_prompts(str(source["prefix"]), data_spec["prompt_registry"])
        test.append(
            {
                "sample_id": f"imdb-test-{index:06d}",
                **dict(source),
                **dict(rendered),
                "prompt": rendered["prompts"]["preference_prefix"],
            }
        )
    if len(test) != test_count:
        raise ControllerError("IMDb test source window is too small")
    roles = {
        "selector": selector,
        "training": training,
        "validation": validation,
        "test": test,
    }
    if include_audit:
        roles["head_audit"] = head_audit
    return (
        roles,
        {
            "selector_window": list(selector_window),
            "training_candidate_window": list(train_window),
            "validation_candidate_window": list(validation_window),
            **({"head_audit_window": list(audit_window)} if audit_window else {}),
            "test_window": list(test_window),
            "train_source_rows": len(train_rows),
            "test_source_rows": len(test_rows),
        },
    )
