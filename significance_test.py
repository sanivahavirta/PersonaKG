import argparse
import json
import os
from datetime import datetime

from rouge_score import rouge_scorer
from scipy.stats import wilcoxon
import evaluate  # for BLEU


def load_bleu_scores(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["completed_samples"]
    bleu_metric = evaluate.load("bleu")
    scores = []

    for sample in samples:
        target = sample["target"].strip()
        prediction = sample["prediction"].strip()
        if not target or not prediction:
            continue  # Skip empty cases
        result = bleu_metric.compute(predictions=[prediction], references=[[target]])
        scores.append(result["bleu"])

    return scores


def load_rouge_scores(file_path: str, metric: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["completed_samples"]
    scorer = rouge_scorer.RougeScorer([metric], use_stemmer=True)
    scores = []

    for sample in samples:
        target = sample["target"].strip()
        prediction = sample["prediction"].strip()
        if not target or not prediction:
            continue
        score = scorer.score(target, prediction)[metric].fmeasure
        scores.append(score)

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_a", help="First checkpoint file")
    parser.add_argument("file_b", help="Second checkpoint file")
    parser.add_argument(
        "--metric",
        default="rougeL",
        choices=["bleu", "rouge1", "rouge2", "rougeL", "rougeLsum"],
        help="Metric to use"
    )
    parser.add_argument("--output", default=None, help="Path to save the Wilcoxon test results text file")
    args = parser.parse_args()

    # Load scores
    if args.metric == "bleu":
        scores_a = load_bleu_scores(args.file_a)
        scores_b = load_bleu_scores(args.file_b)
    else:
        scores_a = load_rouge_scores(args.file_a, args.metric)
        scores_b = load_rouge_scores(args.file_b, args.metric)

    if len(scores_a) != len(scores_b):
        raise ValueError("The two checkpoint files contain different numbers of usable (non-empty) samples.")

    # Wilcoxon signed-rank test
    w_stat, p_val = wilcoxon(scores_a, scores_b)
    mean_a = sum(scores_a) / len(scores_a)
    mean_b = sum(scores_b) / len(scores_b)

    # Output file setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or os.path.join("results", f"wilcoxon_{args.metric}_{timestamp}.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Prepare output
    lines = [
        f"Checkpoint A: {os.path.basename(args.file_a)}",
        f"Checkpoint B: {os.path.basename(args.file_b)}",
        f"Metric: {args.metric}",
        f"Samples: {len(scores_a)}",
        f"Mean {args.metric} A: {mean_a:.6f}",
        f"Mean {args.metric} B: {mean_b:.6f}",
        f"Wilcoxon statistic: {w_stat}",
        f"p-value: {p_val:.6f}",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wilcoxon test results saved to {output_path}")


if __name__ == "__main__":
    main()
