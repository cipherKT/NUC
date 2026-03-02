from algorithms import MultiSegmentNUC
from utils import rnu


def find_optimal_multisegment_num_regions(
    frames,
    lower_percentile,
    upper_percentile,
    min_valid_ratio,
    dv_tolerance,
    num_region_candidates=(5, 10, 15, 20, 25, 30),
):
    if not frames:
        raise RuntimeError("Please provide frames")

    trials = []

    for num_regions in num_region_candidates:
        nuc = MultiSegmentNUC(
            num_regions=num_regions,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            min_valid_ratio=min_valid_ratio,
            dv_tolerance=dv_tolerance,
        )
        corrected = nuc.run(frames=frames)
        rnu_vals = [rnu(f) for f in corrected]
        mean_rnu = float(sum(rnu_vals) / len(rnu_vals))

        trials.append(
            {
                "num_regions": int(num_regions),
                "mean_rnu": mean_rnu,
            }
        )

    trials.sort(key=lambda x: x["mean_rnu"])
    best = trials[0]

    return {
        "best_num_regions": best["num_regions"],
        "best_mean_rnu": best["mean_rnu"],
        "all_trials": trials,
    }
