from algorithms import SceneBasedTwoPointNUC
from utils import rnu


def find_optimal_training_frames(
    frames,
    lower_percentile,
    upper_percentile,
    min_valid_ratio,
    num_regions=(5, 10, 20),
    num_training_frames=(1.0, 0.5, 0.25, 0.1),
):
    if not frames:
        raise RuntimeError("Please provide frames")
    trials = []

    for ratio in num_training_frames:
        for regions in num_regions:
            nuc = SceneBasedTwoPointNUC(
                num_regions=regions,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
                min_valid_ratio=min_valid_ratio,
            )
            corrected = nuc.run(frames, frame_ratio=ratio)
            rnu_vals = [rnu(f) for f in corrected]
            mean_rnu = float(sum(rnu_vals) / len(rnu_vals))

            N = len(frames)
            k = min(max(2, int(round(ratio * N))), N)
            trials.append(
                {
                    "ratio": float(ratio),
                    "k_frames": int(k),
                    "num_regions": int(regions),
                    "mean_rnu": mean_rnu,
                }
            )
    trials.sort(key=lambda x: x["mean_rnu"])
    best = trials[0]

    return {
        "best_ratio": best["ratio"],
        "best_k_frames": best["k_frames"],
        "best_num_regions": best["num_regions"],
        "best_mean_rnu": best["mean_rnu"],
        "all_trials": trials,
    }
