"""DeepSeek V4 Pro implementation of the formal TL;DR human-reference judge.

This module is a standalone typed evaluator backend. It neither imports nor
executes a legacy runner. Credentials are read only when an approved formal
evaluation calls the judge; they are not experiment settings or path entries.
"""

from __future__ import annotations

import http.client
import hashlib
import json
import os
import random
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


JUDGE_PROTOCOL = "tldr_dpo_appendix_c2_concise_pairwise_v1"
SYSTEM_PROMPT = (
    "You are an impartial evaluator. Follow the requested comparison criteria "
    "exactly. The summaries are anonymous. Return only a valid JSON object."
)


@dataclass(frozen=True, slots=True)
class _JudgeSettings:
    base_url: str
    api_key_env: str
    model: str
    thinking_mode: str
    reasoning_effort: str
    timeout_seconds: float
    max_retries: int
    max_output_tokens: int
    random_seed: int


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _settings(evaluation_config: Mapping[str, Any]) -> _JudgeSettings:
    profile = _mapping(evaluation_config.get("profile"), field="evaluation.profile")
    judge = _mapping(profile.get("judge"), field="evaluation.profile.judge")
    if judge.get("provider") != "deepseek_openai_compatible":
        raise ValueError("TLDR judge provider must be deepseek_openai_compatible")
    if judge.get("model") != "deepseek-v4-pro":
        raise ValueError("TLDR formal judge model must be deepseek-v4-pro")
    if judge.get("protocol") != JUDGE_PROTOCOL:
        raise ValueError(f"TLDR judge protocol must be {JUDGE_PROTOCOL}")
    if judge.get("thinking_mode") != "disabled":
        raise ValueError("TLDR formal judge must disable thinking mode")
    if judge.get("reasoning_effort") != "high":
        raise ValueError("TLDR formal judge reasoning_effort must be high")
    if judge.get("full_order_swap") is not True:
        raise ValueError("TLDR formal judge requires full order swap")
    if judge.get("human_only_final_amendment") is not True:
        raise ValueError("TLDR formal judge requires human-only comparisons")
    if judge.get("first_order") != "deterministic_randomized_per_sample":
        raise ValueError("TLDR first order must be deterministic_randomized_per_sample")
    random_seed = judge.get("random_seed")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ValueError("TLDR judge random_seed must be an integer")
    request = _mapping(judge.get("request"), field="judge.request")
    base_url = _string(request.get("base_url"), field="judge.request.base_url").rstrip("/")
    api_key_env = _string(request.get("api_key_env"), field="judge.request.api_key_env")
    timeout_seconds = request.get("timeout_seconds")
    max_retries = request.get("max_retries")
    max_output_tokens = request.get("max_output_tokens")
    if not isinstance(timeout_seconds, (int, float)) or float(timeout_seconds) <= 0:
        raise ValueError("judge.request.timeout_seconds must be positive")
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 1:
        raise ValueError("judge.request.max_retries must be a positive integer")
    if (
        not isinstance(max_output_tokens, int)
        or isinstance(max_output_tokens, bool)
        or max_output_tokens < 1
    ):
        raise ValueError("judge.request.max_output_tokens must be a positive integer")
    return _JudgeSettings(
        base_url=base_url,
        api_key_env=api_key_env,
        model="deepseek-v4-pro",
        thinking_mode="disabled",
        reasoning_effort="high",
        timeout_seconds=float(timeout_seconds),
        max_retries=max_retries,
        max_output_tokens=max_output_tokens,
        random_seed=random_seed,
    )


def _prompt(*, source: str, summary_a: str, summary_b: str) -> str:
    newline = chr(10)
    return (
        "Which of the following summaries does a better job of summarizing the "
        "most important points in the given forum post, without including "
        "unimportant or irrelevant details? A good summary is both precise and "
        "concise."
        f"{newline}{newline}Post:{newline}{source}"
        f"{newline}{newline}Summary A:{newline}{summary_a}"
        f"{newline}{newline}Summary B:{newline}{summary_b}"
        f"{newline}{newline}First provide a one-sentence comparison of the two summaries, "
        f"explaining which you prefer and why. Then choose only A or B.{newline}"
        f'Return JSON exactly as:{newline}'
        '{"comparison": "one-sentence comparison and '
        'explanation", "preferred": "A or B"}'
    )


def _seed_for(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _first_mapping(*, sample_id: str, seed: int) -> dict[str, str]:
    systems = ["candidate", "human_reference"]
    random.Random(_seed_for(seed, "first", sample_id, *systems)).shuffle(systems)
    return dict(zip(("A", "B"), systems, strict=True))


class DeepSeekV4ProHumanReferenceJudge:
    """Concrete remote judge with no text-overlap or local-model fallback."""

    @staticmethod
    def _response_fields(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("judge.response.choices must be a non-empty list")
        choice = _mapping(choices[0], field="judge.response.choice")
        message = _mapping(choice.get("message"), field="judge.response.message")
        content = _string(
            message.get("content"),
            field="judge.response.message.content",
        )
        judgment = _mapping(
            json.loads(content),
            field="judge.response.message.json",
        )
        preferred = _string(judgment.get("preferred"), field="judge.preferred").upper()
        comparison = _string(judgment.get("comparison"), field="judge.comparison")
        if preferred not in {"A", "B"}:
            raise ValueError("TLDR judge preferred must be A or B")
        usage = payload.get("usage", {})
        return {
            "preferred": preferred,
            "comparison": comparison,
            "model": str(payload.get("model", "")),
            "usage": dict(_mapping(usage, field="judge.usage")),
        }

    def _call(
        self,
        *,
        settings: _JudgeSettings,
        source: str,
        summary_a: str,
        summary_b: str,
        checkpoint_path: Path | None = None,
    ) -> Mapping[str, Any]:
        request_identity = {
            "model": settings.model,
            "protocol": JUDGE_PROTOCOL,
            "thinking_mode": settings.thinking_mode,
            "reasoning_effort": settings.reasoning_effort,
            "source": source,
            "summary_a": summary_a,
            "summary_b": summary_b,
        }
        if checkpoint_path is not None and checkpoint_path.is_file():
            cached = _mapping(
                json.loads(checkpoint_path.read_text(encoding="utf-8")),
                field="judge.checkpoint",
            )
            if cached.get("request") != request_identity:
                raise ValueError("TLDR judge checkpoint request differs from current row")
            return _mapping(cached.get("result"), field="judge.checkpoint.result")
        api_key = os.environ.get(settings.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"TLDR judge requires {settings.api_key_env} in the execution environment"
            )
        body = {
            "model": settings.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _prompt(
                        source=source,
                        summary_a=summary_a,
                        summary_b=summary_b,
                    ),
                },
            ],
            "thinking": {"type": settings.thinking_mode},
            "response_format": {"type": "json_object"},
            "max_tokens": settings.max_output_tokens,
            "stream": False,
        }
        last_error: Exception | None = None
        for attempt in range(settings.max_retries):
            try:
                request = urllib.request.Request(
                    f"{settings.base_url}/chat/completions",
                    data=json.dumps(body).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(
                    request,
                    timeout=settings.timeout_seconds,
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                result = self._response_fields(
                    _mapping(payload, field="judge.response.payload")
                )
                if checkpoint_path is not None:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary: str | None = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            mode="w",
                            encoding="utf-8",
                            dir=checkpoint_path.parent,
                            prefix=f".{checkpoint_path.name}.",
                            suffix=".tmp",
                            delete=False,
                        ) as handle:
                            temporary = handle.name
                            json.dump(
                                {"request": request_identity, "result": dict(result)},
                                handle,
                                ensure_ascii=False,
                                indent=2,
                                sort_keys=True,
                            )
                            handle.write("\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                        os.replace(temporary, checkpoint_path)
                    finally:
                        if temporary is not None and os.path.exists(temporary):
                            os.unlink(temporary)
                return result
            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                urllib.error.URLError,
                http.client.IncompleteRead,
                TimeoutError,
            ) as exc:
                last_error = exc
                if attempt + 1 < settings.max_retries:
                    time.sleep(min(2**attempt, 16))
        assert last_error is not None
        raise RuntimeError(f"TLDR DeepSeek judge failed after retries: {last_error}")

    def evaluate_pair(
        self,
        *,
        prediction: Mapping[str, Any],
        human_reference: Mapping[str, Any],
        evaluation_config: Mapping[str, Any],
        checkpoint_dir: Path | None = None,
        checkpoint_key: str | None = None,
    ) -> Mapping[str, Any]:
        settings = _settings(evaluation_config)
        source = _string(human_reference.get("prompt"), field="TLDR reference.prompt")
        candidate = _string(
            prediction.get("generated_text"),
            field="TLDR prediction.generated_text",
        )
        human = _string(
            human_reference.get("reference_summary"),
            field="TLDR reference.reference_summary",
        )
        if (checkpoint_dir is None) != (checkpoint_key is None):
            raise ValueError("TLDR judge checkpoint_dir and checkpoint_key must be set together")
        sample_id = str(
            prediction.get("source_sample_id", prediction.get("sample_id", ""))
        )
        first_mapping = _first_mapping(sample_id=sample_id, seed=settings.random_seed)
        first_summaries = {
            "candidate": candidate,
            "human_reference": human,
        }
        first_a = first_summaries[first_mapping["A"]]
        first_b = first_summaries[first_mapping["B"]]
        swapped_mapping = {"A": first_mapping["B"], "B": first_mapping["A"]}
        if checkpoint_key is not None and (
            not checkpoint_key
            or any(not (char.isalnum() or char in "._-") for char in checkpoint_key)
        ):
            raise ValueError("TLDR judge checkpoint_key contains unsafe characters")
        first = self._call(
            settings=settings,
            source=source,
            summary_a=first_a,
            summary_b=first_b,
            checkpoint_path=(
                checkpoint_dir / f"{checkpoint_key}.first.json"
                if checkpoint_dir is not None and checkpoint_key is not None
                else None
            ),
        )
        swapped = self._call(
            settings=settings,
            source=source,
            summary_a=first_b,
            summary_b=first_a,
            checkpoint_path=(
                checkpoint_dir / f"{checkpoint_key}.swap.json"
                if checkpoint_dir is not None and checkpoint_key is not None
                else None
            ),
        )
        first_win = float(first_mapping[str(first["preferred"])] == "candidate")
        swapped_win = float(swapped_mapping[str(swapped["preferred"])] == "candidate")
        trace = {
            "provider": "deepseek_openai_compatible",
            "model": settings.model,
            "protocol": JUDGE_PROTOCOL,
            "thinking_mode": settings.thinking_mode,
            "reasoning_effort": settings.reasoning_effort,
            "first_pass": {
                "label_to_system": first_mapping,
                **first,
            },
            "order_swap": {
                "label_to_system": swapped_mapping,
                **swapped,
            },
        }
        return {
            "first_pass_win": first_win,
            "order_swap_win": swapped_win,
            "order_swap_agreement": float(first_win == swapped_win),
            "trace": trace,
        }
