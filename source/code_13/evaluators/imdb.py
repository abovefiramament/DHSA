"""Pinned Siebert reward evaluator for IMDb Site runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.shared.contracts import EvaluationRequest, EvaluationResult

from .callback import align_prediction_references, write_evaluation


class IMDbSentimentEvaluator:
    """Evaluate completions with the registered frozen Siebert classifier.

    The scorer path is supplied only by the compiled machine runtime config.
    The method never discovers a local model path or falls back to keywords.
    """

    def __init__(self, *, device: str) -> None:
        self.device = str(device)
        self._loaded_path: str | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._positive_id: int | None = None
        self._negative_id: int | None = None

    @staticmethod
    def _runtime_settings(request: EvaluationRequest) -> tuple[str, str, int]:
        config = request.evaluation_config
        profile = config.get("profile")
        runtime = config.get("runtime")
        if not isinstance(profile, Mapping) or not isinstance(runtime, Mapping):
            raise ValueError("IMDb evaluator requires profile and runtime settings")
        scorer = profile.get("scorer")
        if not isinstance(scorer, Mapping):
            raise ValueError("IMDb evaluator requires a scorer profile")
        revision = scorer.get("revision")
        batch_size = scorer.get("reward_batch_size")
        path = runtime.get("sentiment_model_path")
        if not isinstance(revision, str) or not revision:
            raise ValueError("IMDb scorer revision must be explicit")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("IMDb scorer reward_batch_size must be a positive integer")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("IMDb scorer needs an absolute runtime sentiment_model_path")
        return path, revision, batch_size

    def _load(self, path: str, revision: str) -> None:
        if self._loaded_path == path:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("IMDb formal evaluation requires transformers") from exc
        tokenizer = AutoTokenizer.from_pretrained(path, revision=revision, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            path,
            revision=revision,
            local_files_only=True,
        )
        model.to(self.device)
        model.eval()
        label_to_id = {
            str(label).upper(): int(index)
            for label, index in dict(getattr(model.config, "label2id", {})).items()
        }
        if "POSITIVE" not in label_to_id or "NEGATIVE" not in label_to_id:
            raise ValueError("Siebert scorer must expose POSITIVE and NEGATIVE labels")
        self._loaded_path = path
        self._tokenizer = tokenizer
        self._model = model
        self._positive_id = label_to_id["POSITIVE"]
        self._negative_id = label_to_id["NEGATIVE"]

    def score_completions(
        self,
        texts: Sequence[str],
        *,
        path: str,
        revision: str,
        batch_size: int,
    ) -> list[float]:
        """Return P(POSITIVE)-P(NEGATIVE) in registered fixed-size batches."""

        self._load(path, revision)
        assert self._tokenizer is not None and self._model is not None
        assert self._positive_id is not None and self._negative_id is not None
        import torch

        values: list[float] = []
        device = next(self._model.parameters()).device
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                probabilities = torch.softmax(self._model(**encoded).logits.float(), dim=-1)
            values.extend(
                float(row[self._positive_id].item() - row[self._negative_id].item())
                for row in probabilities
            )
        return values

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        aligned = align_prediction_references(request)
        path, revision, batch_size = self._runtime_settings(request)
        texts = [str(prediction.get("generated_text", "") or "") for prediction, _ in aligned]
        sentiment = self.score_completions(
            texts,
            path=path,
            revision=revision,
            batch_size=batch_size,
        )
        if len(sentiment) != len(aligned):
            raise ValueError("IMDb scorer returned the wrong number of rewards")
        scored = []
        for (prediction, _reference), reward, text in zip(aligned, sentiment, texts):
            ratio = prediction.get("sampled_sequence_logprob_ratio")
            token_kl = prediction.get("token_kl_audit")
            token_count = prediction.get("generated_token_count")
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
                raise ValueError("IMDb predictions require a real sampled_sequence_logprob_ratio")
            if isinstance(token_kl, bool) or not isinstance(token_kl, (int, float)):
                raise ValueError("IMDb predictions require a real token_kl_audit")
            if isinstance(token_count, bool) or not isinstance(token_count, (int, float)):
                raise ValueError("IMDb predictions require generated_token_count")
            metrics = {
                "mean_positive_sentiment_score": reward,
                "positive_rate": float(reward > 0.0),
                "sampled_sequence_logprob_ratio": float(ratio),
                "token_kl_audit": float(token_kl),
                "mean_completion_tokens": float(token_count),
                "health": float(bool(text.strip())),
                "actual_n": 1.0,
            }
            for name in (
                "incremental_sampled_sequence_logprob_ratio",
                "incremental_token_kl_audit",
            ):
                value = prediction.get(name)
                if value is not None:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ValueError(f"IMDb prediction field {name} must be numeric")
                    metrics[name] = float(value)
            scored.append((prediction, metrics))
        return write_evaluation(request, scored)
