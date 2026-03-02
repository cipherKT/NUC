from algorithms import AdaptiveLMSSBNUC, GatedAdaptiveLMSSBNUC, StandardLMSSBNUC
from utils import rnu


def find_optimal_lmssbnuc_params(
    frames,
    variant,
    sigma=5.0,
    kernel_size=21,
    local_var_window=5,
    epsilon_candidates=None,
    K_candidates=None,
    M_scale_candidates=None,
    T_candidates=None,
):
    if not frames:
        raise RuntimeError("Please provide frames")

    trials = []

    if variant == "standard":
        if epsilon_candidates is None:
            epsilon_candidates = (0.01, 0.03, 0.05, 0.08)

        for epsilon in epsilon_candidates:
            nuc = StandardLMSSBNUC(
                epsilon=epsilon,
                sigma=sigma,
                kernel_size=kernel_size,
            )
            corrected = nuc.run(frames=frames)
            mean_rnu = float(sum(rnu(f) for f in corrected) / len(corrected))
            trials.append(
                {
                    "variant": variant,
                    "epsilon": float(epsilon),
                    "mean_rnu": mean_rnu,
                }
            )

    elif variant == "adaptive":
        if K_candidates is None:
            K_candidates = (0.02, 0.05, 0.08)
        if M_scale_candidates is None:
            M_scale_candidates = (0.25, 0.5, 0.75, 1.0)

        for K in K_candidates:
            for M_scale in M_scale_candidates:
                nuc = AdaptiveLMSSBNUC(
                    K=K,
                    M_scale=M_scale,
                    sigma=sigma,
                    kernel_size=kernel_size,
                    local_var_window=local_var_window,
                )
                corrected = nuc.run(frames=frames)
                mean_rnu = float(sum(rnu(f) for f in corrected) / len(corrected))
                trials.append(
                    {
                        "variant": variant,
                        "K": float(K),
                        "M_scale": float(M_scale),
                        "mean_rnu": mean_rnu,
                    }
                )

    elif variant == "gated":
        if K_candidates is None:
            K_candidates = (0.02, 0.05, 0.08)
        if M_scale_candidates is None:
            M_scale_candidates = (0.25, 0.5, 0.75)
        if T_candidates is None:
            T_candidates = (0.001, 0.002, 0.005)

        for K in K_candidates:
            for M_scale in M_scale_candidates:
                for T in T_candidates:
                    nuc = GatedAdaptiveLMSSBNUC(
                        K=K,
                        M_scale=M_scale,
                        sigma=sigma,
                        kernel_size=kernel_size,
                        local_var_window=local_var_window,
                        T=T,
                    )
                    corrected = nuc.run(frames=frames)
                    mean_rnu = float(sum(rnu(f) for f in corrected) / len(corrected))
                    trials.append(
                        {
                            "variant": variant,
                            "K": float(K),
                            "M_scale": float(M_scale),
                            "T": float(T),
                            "mean_rnu": mean_rnu,
                        }
                    )
    else:
        raise ValueError(f"Unknown lmssbnuc variant: {variant}")

    trials.sort(key=lambda x: x["mean_rnu"])
    best = trials[0]

    out = {
        "variant": variant,
        "best_mean_rnu": best["mean_rnu"],
        "all_trials": trials,
    }
    if variant == "standard":
        out["best_epsilon"] = best["epsilon"]
    elif variant == "adaptive":
        out["best_K"] = best["K"]
        out["best_M_scale"] = best["M_scale"]
    else:
        out["best_K"] = best["K"]
        out["best_M_scale"] = best["M_scale"]
        out["best_T"] = best["T"]

    return out
