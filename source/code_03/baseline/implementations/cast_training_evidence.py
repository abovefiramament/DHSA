"""Check executed CAST training facts before reusing a bank."""
from math import ceil
from typing import Any, Mapping


def validate_cast_training_evidence(record: Mapping[str, Any], training: Mapping[str, Any]) -> None:
    batch = int(training["train_batch_size"])
    epochs = int(training["epochs"])
    rows = int(record["training_rows"])
    if min(batch, epochs, rows) <= 0:
        raise ValueError("CAST training evidence has invalid counts")
    expected_steps = epochs * ceil(rows / batch)
    if int(record["optimizer_steps"]) != expected_steps:
        raise ValueError(f"CAST executed optimizer_steps={record['optimizer_steps']} differs from registered {expected_steps}")
    if "train_rows" in training and rows != int(training["train_rows"]):
        raise ValueError("CAST executed training_rows differs from registered count")
    # Manifest names/defaults mirror CAST.train_bank; no new training policy.
    expected = dict(training, learning_rate=training["lr"], shuffle_train=training.get("shuffle_train", True))
    for key in ("train_batch_size", "epochs", "score_mode", "learning_rate", "weight_decay", "shuffle_train", "epoch_shuffle_seed"):
        if key not in expected or key not in record or record[key] != expected[key]:
            raise ValueError(f"CAST executed training field differs or is unrecorded: {key}")
