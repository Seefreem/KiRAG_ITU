"""Calculate perfect-retriever upper bounds for the retrieval metrics."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate macro-averaged Precision@k, Recall@k, F1@k, and "
            "NumDoc@k for a perfect retriever."
        )
    )
    parser.add_argument(
        "--data_file",
        required=True,
        help="QA-pair JSON file containing a supporting_facts field",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="number of documents returned for every question",
    )
    return parser.parse_args()


def load_examples(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def get_relevant_document_ids(example):
    """Return the unique document IDs appearing in supporting_facts."""
    return {str(document_id) for document_id, _ in example["supporting_facts"]}


def perfect_retriever_metrics(num_relevant, k):
    """Calculate one query's metrics when all relevant documents rank first."""
    if num_relevant == 0:
        return 0.0, 0.0, 0.0, float(k)

    true_positives = min(num_relevant, k)
    precision = true_positives / k
    recall = true_positives / num_relevant
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1, float(k)


def calculate_upper_bounds(examples, k):
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if not examples:
        raise ValueError("data_file contains no examples")

    per_query_metrics = [
        perfect_retriever_metrics(
            num_relevant=len(get_relevant_document_ids(example)),
            k=k,
        )
        for example in examples
    ]

    metric_names = [f"Precision@{k}", f"Recall@{k}", f"F1@{k}", f"NumDoc@{k}"]
    return {
        name: sum(values) / len(values)
        for name, values in zip(metric_names, zip(*per_query_metrics))
    }


def main():
    args = parse_args()
    examples = load_examples(args.data_file)
    metrics = calculate_upper_bounds(examples, args.k)

    print(f"File: {args.data_file}")
    print(f"Questions: {len(examples)}")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
