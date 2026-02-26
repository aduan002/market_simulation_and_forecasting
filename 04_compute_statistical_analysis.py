import pandas as pd
import numpy as np
from itertools import combinations
import argparse
from scipy.stats import t

def main(file_path: str, output_path: str, metric: str = "mean_absolute_error", alpha: float = 0.05):
    df = pd.read_csv(file_path)

    summary = (
        df.groupby("algorithm")[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    algorithms = summary["algorithm"].tolist()
    pivot = df.pivot(index="fold", columns="algorithm", values=metric).sort_index()

    dominance = {algo: set() for algo in algorithms}
    n_folds = pivot.shape[0]

    # t critical value for two-sided CI
    t_crit = t.ppf(1 - alpha/2, df=n_folds - 1)

    for a, b in combinations(algorithms, 2):
        diff = pivot[a] - pivot[b]  # A minus B
        mean_diff = diff.mean()
        std_diff = diff.std(ddof=1)
        se_diff = std_diff / np.sqrt(n_folds)

        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff

        # If CI entirely below zero, then A better
        if ci_upper < 0:
            dominance[a].add(b)

        # If CI entirely above zero, then B better
        elif ci_lower > 0:
            dominance[b].add(a)

        # else: inconclusive, then no edge

    wins = {algo: len(dominance[algo]) for algo in algorithms}

    summary["wins"] = summary["algorithm"].map(wins)
    summary["beats"] = summary["algorithm"].map(lambda a: ",".join(sorted(dominance[a])))

    summary = summary.sort_values(["wins", "mean"], ascending=[False, True]).reset_index(drop=True)
    summary["rank"] = summary["wins"].rank(method="dense", ascending=False).astype(int)

    summary.to_csv(output_path, index=False)
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--metric", default="mean_absolute_error")
    parser.add_argument("--alpha", type=float, default=0.05)

    args = parser.parse_args()
    main(args.input, args.output, metric=args.metric, alpha=args.alpha)