"""DeepSeek V4 Pro implementation of the formal TL;DR human-reference judge.

This module is a standalone typed evaluator backend. It neither imports nor
executes a legacy runner. Credentials are read only when an approved formal
evaluation calls the judge; they are not experiment settings or path entries.
"""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
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
    if _mapping(judge.get("first_order"), field="judge.first_order") != {
        "candidate": "A",
        "human_reference": "B",
    }:
        raise ValueError("TLDR first order must be candidate=A, human_reference=B")
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
        'Return JSON exactly as: {"comparison": "one-sentence comparison and '
        'explanation", "preferred": "A or B"}'
    )


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
    ) -> Mapping[str, Any]:
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
                return self._response_fields(
                    _mapping(payload, field="judge.response.payload")
                )
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
        first = self._call(
            settings=settings,
            source=source,
            summary_a=candidate,
            summary_b=human,
        )
        swapped = self._call(
            settings=settings,
            source=source,
            summary_a=human,
            summary_b=candidate,
        )
        first_win = float(first["preferred"] == "A")
        swapped_win = float(swapped["preferred"] == "B")
        trace = {
            "provider": "deepseek_openai_compatible",
            "model": settings.model,
            "protocol": JUDGE_PROTOCOL,
            "thinking_mode": settings.thinking_mode,
            "reasoning_effort": settings.reasoning_effort,
            "first_pass": {
                "label_to_system": {"A": "candidate", "B": "human_reference"},
                **first,
            },
            "order_swap": {
                "label_to_system": {"A": "human_reference", "B": "candidate"},
                **swapped,
            },
        }
        return {
            "first_pass_win": first_win,
            "order_swap_win": swapped_win,
            "order_swap_agreement": float(first_win == swapped_win),
            "trace": trace,
        }
