"""
Staleness detection for checkpoints.

Editing generate_dataset.py, schema.py, dataset.py or model.py does NOTHING
to a checkpoint that's already trained -- you have to re-run
generate_dataset.py and train.py for the change to count. It's easy to
forget, and then you spend an hour asking why your fix doesn't work.

train.py stores a fingerprint of the dataset CSV, those source files and the terrain / city
choices in the checkpoint. evaluate.py recomputes it and warns loudly if they differ.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

SOURCE_FILES = ["schema.py", "generate_dataset.py", "dataset.py", "model.py"]


def fingerprint_sources(dataset_path: str) -> str:
    hasher = hashlib.sha256()
    dataset_file = Path(dataset_path)
    hasher.update(dataset_file.read_bytes() if dataset_file.exists() else b"MISSING_DATASET")
    here = Path(__file__).resolve().parent
    for name in SOURCE_FILES:
        path = here / name
        if path.exists():
            hasher.update(path.read_bytes())
    # The terrain and city choices come from YOUR citygen, not from text2city.
    # Add a terrain to terrain.py and the model's terrain head no longer
    # matches -- so the choices are part of the fingerprint too.
    from text2city.schema import PARAM_SCHEMA
    choices = [(p.name, list(p.choices)) for p in PARAM_SCHEMA if p.choices]
    hasher.update(repr(choices).encode())
    return hasher.hexdigest()[:16]


def is_stale(checkpoint_fingerprint, dataset_path: str) -> bool:
    """True if the checkpoint was trained on different data, code or terrains."""
    return checkpoint_fingerprint is None or fingerprint_sources(dataset_path) != checkpoint_fingerprint


def check_staleness(checkpoint_fingerprint, dataset_path: str) -> None:
    """Prints a loud warning (never raises) if the checkpoint looks stale."""
    if checkpoint_fingerprint is None:
        print("WARNING: checkpoint has no fingerprint (trained by an older train.py) -- consider retraining.")
        return
    if is_stale(checkpoint_fingerprint, dataset_path):
        print("=" * 70)
        print("WARNING: this checkpoint appears STALE.")
        print("The dataset CSV, schema.py / generate_dataset.py / dataset.py / model.py,")
        print("or your citygen terrain list changed since it was trained. Results below")
        print("reflect the OLD model.")
        print("Fix: re-run generate_dataset.py, then train.py.")
        print("=" * 70)
