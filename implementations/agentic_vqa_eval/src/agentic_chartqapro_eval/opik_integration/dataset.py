"""Register ChartQAPro samples as an Opik Dataset.

Usage:
    python -m agentic_chartqapro_eval.opik_integration.dataset \
        --split test --n 25
"""

import argparse
from typing import Optional

from .client import get_client


def register_dataset(
    samples,
    dataset_name: str = "ChartQAPro",
    split: str = "test",
) -> Optional[object]:
    """Insert PerceivedSamples into an Opik Dataset named ``{dataset_name}_{split}``.

    Returns the Dataset object, or None if Opik is unavailable.
    """
    client = get_client()
    if client is None:
        return None

    name = f"{dataset_name}_{split}"
    try:
        dataset = client.get_or_create_dataset(name=name)
        items = [
            {
                "source_id": s.sample_id,  # stored as data field; Opik auto-generates UUID v7 id
                "question": s.question,
                "expected_output": s.expected_output,
                "question_type": s.question_type.value,
                "image_path": s.image_path or "",
                "choices": s.choices or [],
            }
            for s in samples
        ]
        dataset.insert(items)
        print(f"[opik] Registered {len(items)} samples → dataset '{name}'")
        return dataset
    except Exception as exc:
        print(f"[opik] Dataset registration failed: {exc}")
        return None


def main() -> None:
    """Register ChartQAPro dataset samples in Opik."""
    parser = argparse.ArgumentParser(
        description="Register ChartQAPro samples as Opik dataset"
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--image_dir", default="data/chartqapro_images")
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()

    from ..datasets.chartqapro_loader import load_chartqapro

    samples = load_chartqapro(
        split=args.split, n=args.n, image_dir=args.image_dir, cache_dir=args.cache_dir
    )
    register_dataset(samples, split=args.split)


if __name__ == "__main__":
    main()
