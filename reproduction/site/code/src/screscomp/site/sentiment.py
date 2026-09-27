from __future__ import annotations

from typing import Any


class FrozenSentimentScorer:
    def __init__(self, score_spec: dict[str, Any], *, device: str) -> None:
        from transformers import pipeline

        pipeline_device = -1
        if str(device).startswith("cuda"):
            pipeline_device = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
        self.score_spec = score_spec
        self.classifier = pipeline(
            "text-classification",
            model=score_spec["scorer_model"],
            revision=score_spec["scorer_revision"],
            device=pipeline_device,
        )

    def score(self, texts: list[str]) -> list[dict[str, float]]:
        outputs = self.classifier(
            texts,
            batch_size=int(self.score_spec["reward_batch_size"]),
            truncation=True,
            max_length=512,
            top_k=None,
        )
        rows = []
        for output in outputs:
            by_label = {str(row["label"]).upper(): float(row["score"]) for row in output}
            positive = by_label.get(str(self.score_spec["positive_label"]).upper(), 0.0)
            negative = by_label.get("NEGATIVE", 0.0)
            rows.append(
                {
                    "positive_sentiment_score": positive,
                    "negative_sentiment_score": negative,
                    "competition_score": positive - negative,
                }
            )
        return rows
