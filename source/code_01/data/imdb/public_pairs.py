"""IMDb public-token pair data adapter with the registered disjoint source roles."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
from evaluators.imdb import IMDbSentimentEvaluator
from data._shared.controller_api import ControllerError
from .controller import render_registered_prompts
from .model_native import _prefix_records, _generate_batches

SOURCES = {"imdb_train", "imdb_test", "public_training", "selector_comments", "validation_comments", "test_comments"}

def prepare_public_roles(source_rows: Mapping[str, Sequence[Mapping[str, Any]]], *, data_spec: Mapping[str, Any], model_provider: Any):
    if set(source_rows) != SOURCES:
        raise ControllerError(f"IMDb public pair sources must be {sorted(SOURCES)}")
    construction = data_spec["construction"]
    model_spec = data_spec["runtime"]["model"]
    adapter = model_provider.load(model_spec["checkpoint"], mode="inference")
    adapter.configure_input(use_chat_template=bool(model_spec.get("use_chat_template", False)))
    train_prefixes = {row["source_row_index"]: row for row in _prefix_records(source_rows["imdb_train"], adapter=adapter, construction=construction)}
    test_prefixes = {row["source_row_index"]: row for row in _prefix_records(source_rows["imdb_test"], adapter=adapter, construction=construction)}
    def prepared(name: str, projected: Mapping[int, Mapping[str, Any]]):
        result = []
        for source in source_rows[name]:
            index = source.get("source_row_index")
            if index not in projected:
                raise ControllerError(f"{name} source row {index} has no registered prefix")
            result.append({**dict(source), **dict(projected[index])})
        return result
    selector_sources = prepared("selector_comments", train_prefixes)
    validation_sources = prepared("validation_comments", train_prefixes)
    test_sources = prepared("test_comments", test_prefixes)
    if (len(selector_sources), len(validation_sources), len(test_sources)) != (300,384,2048):
        raise ControllerError("IMDb registered public-pair selector/validation/test count drift")
    source_keys = {
        name: {(row["source_split"], row["source_row_index"]) for row in rows}
        for name, rows in (("selector",selector_sources),("validation",validation_sources),("test",test_sources))
    }
    for left,right in (("selector","validation"),("selector","test"),("validation","test")):
        if source_keys[left] & source_keys[right]:
            raise ControllerError(f"IMDb {left}/{right} source overlap")
    training = [dict(row) for row in source_rows["public_training"]]
    if len(training) != construction["roles"]["training"]["target_count"]:
        raise ControllerError("IMDb cleaned public train count drift")
    public_matches = {
        tuple(match)
        for row in training
        for match in row.get("source_comment_matches", [])
    }
    if public_matches & (source_keys["selector"] | source_keys["validation"] | source_keys["test"]):
        raise ControllerError("IMDb public training source overlaps another role")
    for row in training:
        n = row.get("prompt_token_count")
        if not isinstance(n,int) or isinstance(n,bool) or n < 1:
            raise ControllerError("IMDb token pair prompt count is invalid")
        for field in ("pos_input_ids","neg_input_ids"):
            values = row.get(field)
            if not isinstance(values,list) or len(values) <= n:
                raise ControllerError(f"IMDb token pair {field} has no completion")
    generation = construction["selector_state_generation"]
    rendered = [render_registered_prompts(row["prefix"], data_spec["prompt_registry"]) for row in selector_sources]
    conditions = {}
    for role, key in (("good","good_state"),("base","base_state")):
        prompts = [item["prompts"][key] for item in rendered]
        conditions[role] = _generate_batches(adapter,prompts,generation=generation,completion_index=0,completion_seed_stride=1)
    selector = []
    for i,(source,prompt) in enumerate(zip(selector_sources,rendered,strict=True)):
        good,base=conditions["good"][i],conditions["base"][i]
        if good["batch_seed"] != base["batch_seed"]:
            raise ControllerError("IMDb good/base lost common random numbers")
        selector.append({
            "sample_id":f"imdb-selector-{i:06d}",**source,**prompt,
            "good_state":{"prompt":prompt["prompts"]["good_state"],**good},
            "base_state":{"prompt":prompt["prompts"]["base_state"],**base},
            "batch_seed":good["batch_seed"],
            "selector_semantics":"shared_model_native_good_base_complete_states",
        })
    pair_spec = construction["pair_preparation"]
    completion_stride = int(pair_spec["completion_seed_stride"])
    pair_generation={**generation,**pair_spec}
    pair_generation.pop("min_new_tokens",None)
    val_rendered=[render_registered_prompts(row["prefix"],data_spec["prompt_registry"]) for row in validation_sources]
    val_prompts=[item["prompts"]["preference_prefix"] for item in val_rendered]
    generated=[
        _generate_batches(adapter,val_prompts,generation=pair_generation,completion_index=i,completion_seed_stride=completion_stride)
        for i in range(int(pair_spec["completions_per_prompt"]))
    ]
    texts=[generated[i][j]["generated_text"] for j in range(len(val_prompts)) for i in range(len(generated))]
    scorer=data_spec["runtime"]["sentiment_scorer"]
    evaluator=IMDbSentimentEvaluator(device=str(next(adapter.model.parameters()).device))
    rewards=evaluator.score_completions(texts,path=scorer["local_path"],revision=scorer["revision"],batch_size=int(scorer["reward_batch_size"]))
    validation=[]
    for j,(source,prompt) in enumerate(zip(validation_sources,val_rendered,strict=True)):
        candidates=sorted(
            [(float(rewards[j*len(generated)+i]),i,generated[i][j]) for i in range(len(generated))],
            key=lambda item:(-item[0],item[1])
        )
        hi,lo=candidates[0],candidates[-1]
        if hi[0] <= lo[0]:continue
        validation.append({
            "sample_id":f"imdb-validation-{len(validation):06d}",**source,**prompt,
            "split":"validation","prompt":prompt["prompts"]["preference_prefix"],
            "chosen":hi[2]["generated_text"],"rejected":lo[2]["generated_text"],
            "chosen_token_ids":hi[2]["generated_token_ids"],"rejected_token_ids":lo[2]["generated_token_ids"],
            "chosen_reward":hi[0],"rejected_reward":lo[0],"reward_margin":hi[0]-lo[0],
        })
        if len(validation)==construction["roles"]["validation"]["target_count"]:break
    if len(validation)!=construction["roles"]["validation"]["target_count"]:
        raise ControllerError("IMDb validation candidate pool did not yield registered 256 pairs")
    test=[]
    for i,source in enumerate(test_sources):
        prompt=render_registered_prompts(source["prefix"],data_spec["prompt_registry"])
        test.append({"sample_id":f"imdb-test-{i:06d}",**source,**prompt,"prompt":prompt["prompts"]["preference_prefix"]})
    roles={"selector":selector,"training":training,"validation":validation,"test":test}
    return roles,{"source_mode":"ma921_cleaned_token_pairs","role_counts":{k:len(v) for k,v in roles.items()},"source_overlap_policy":"disjoint original comments across public pair source and selected IMDb roles"}

