from algorithms import ConstantStatsNUC
from utils import rnu


def find_optimal_cssbnuc_params(
    frames,
    alpha_candidates=(0.90, 0.95, 0.97, 0.99),
    T_candidates=(0.001, 0.002, 0.005, 0.01),
    eps=1e-6,
):
    if not frames:
        raise RuntimeError("Please provide frames")

    trials = []

    for alpha in alpha_candidates:
        for T in T_candidates:
            nuc = ConstantStatsNUC(alpha=alpha, T=T, eps=eps)
            corrected = nuc.run(frames=frames, normalize=True)
            rnu_vals = [rnu(f) for f in corrected]
            mean_rnu = float(sum(rnu_vals) / len(rnu_vals))

            trials.append(
                {
                    "alpha": float(alpha),
                    "T": float(T),
                    "mean_rnu": mean_rnu,
                }
            )

    trials.sort(key=lambda x: x["mean_rnu"])
    best = trials[0]

    return {
        "best_alpha": best["alpha"],
        "best_T": best["T"],
        "best_mean_rnu": best["mean_rnu"],
        "all_trials": trials,
    }
