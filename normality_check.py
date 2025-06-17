import json
import matplotlib.pyplot as plt
import scipy.stats as stats

def extract_scores(file_path, metric="bleu"):
    from evaluate import load
    from rouge_score import rouge_scorer

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data["completed_samples"]
    scores = []

    if metric == "bleu":
        bleu = load("bleu")
        for s in samples:
            pred = s["prediction"].strip()
            ref = s["target"].strip()
            if pred and ref:
                score = bleu.compute(predictions=[pred], references=[[ref]])["bleu"]
                scores.append(score)
    else:
        scorer = rouge_scorer.RougeScorer([metric], use_stemmer=True)
        for s in samples:
            pred = s["prediction"].strip()
            ref = s["target"].strip()
            if pred and ref:
                score = scorer.score(ref, pred)[metric].fmeasure
                scores.append(score)

    return scores

# Example: compare two conditions
scores_a = extract_scores(r"C:\Users\vahav\OneDrive - HvA\Courses\Master's project\KG-repo\PersonaKG\results\checkpoint_experiment28_with_KG_PeaCoK.json", metric="rougeLsum")
scores_b = extract_scores(r"C:\Users\vahav\OneDrive - HvA\Courses\Master's project\KG-repo\PersonaKG\results\checkpoint_experiment26_without_KG.json", metric="rougeLsum")

# Paired differences
differences = [a - b for a, b in zip(scores_a, scores_b)]

# Shapiro-Wilk Test
stat, p = stats.shapiro(differences)
print(f"Shapiro-Wilk test: W = {stat:.4f}, p = {p:.6f}")

# Histogram
plt.hist(differences, bins=30, edgecolor="black")
plt.title("Histogram of Score Differences")
plt.xlabel("Difference")
plt.ylabel("Frequency")
plt.show()

# Q-Q Plot
stats.probplot(differences, dist="norm", plot=plt)
plt.title("Q-Q Plot of Score Differences")
plt.show()