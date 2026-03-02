import argparse
import sys

from algorithms import (
    AdaptiveLMSSBNUC,
    ConstantStatsNUC,
    GatedAdaptiveLMSSBNUC,
    MultiSegmentNUC,
    SceneBasedTwoPointNUC,
    StandardLMSSBNUC,
)
from data import DataSetHandler
from optimal.cssbnuc import find_optimal_cssbnuc_params
from optimal.lmssbnuc import find_optimal_lmssbnuc_params
from optimal.multisegmentNUC import find_optimal_multisegment_num_regions
from optimal.twopointNUC import find_optimal_training_frames
from utils import Visualizer, col_mad, row_mad


def _parse_csv_floats(raw, flag_name):
    try:
        values = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid float in {flag_name}: {raw}") from exc
    if not values:
        raise ValueError(f"{flag_name} must contain at least one value")
    return values


def _parse_csv_ints(raw, flag_name):
    try:
        values = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid int in {flag_name}: {raw}") from exc
    if not values:
        raise ValueError(f"{flag_name} must contain at least one value")
    return values


def run_twopointnuc(data_path, output_path, args, show_frames=False):
    loader = DataSetHandler(data_path, ext="png")
    frames = loader.load()

    num_frames, h, w = loader.get_shape()
    print(f"[*] Loaded {num_frames} frames ({h} x {w})")
    if args.optimize:
        ratio_candidates = (
            args.opt_twopoint_ratios
            if args.opt_twopoint_ratios is not None
            else (1.0, 0.5, 0.25, 0.1)
        )
        region_candidates = (
            args.opt_twopoint_regions
            if args.opt_twopoint_regions is not None
            else ((5, 10, 20) if args.optimize_fast else (5, 10, 15, 20, 25))
        )
        optimal_hp = find_optimal_training_frames(
            frames=frames,
            lower_percentile=args.twopointnuc_lower_percentile,
            upper_percentile=args.twopointnuc_upper_percentile,
            min_valid_ratio=args.twopointnuc_min_valid_ratio,
            num_regions=region_candidates,
            num_training_frames=ratio_candidates,
        )
        print(
            "[*] Optimal TwoPointNUC hyperparameters: "
            f"ratio={optimal_hp['best_ratio']:.2f} | "
            f"k_frames={optimal_hp['best_k_frames']} | "
            f"num_regions={optimal_hp['best_num_regions']} | "
            f"mean_rnu={optimal_hp['best_mean_rnu']:.6f}"
        )
        nuc = SceneBasedTwoPointNUC(
            num_regions=optimal_hp["best_num_regions"],
            lower_percentile=args.twopointnuc_lower_percentile,
            upper_percentile=args.twopointnuc_upper_percentile,
            min_valid_ratio=args.twopointnuc_min_valid_ratio,
        )
        c_frames = nuc.run(frames=frames, frame_ratio=optimal_hp["best_ratio"])
    else:
        print("[*] Optimization disabled. Using CLI/default TwoPointNUC hyperparameters.")
        nuc = SceneBasedTwoPointNUC(
            num_regions=args.twopointnuc_num_regions,
            lower_percentile=args.twopointnuc_lower_percentile,
            upper_percentile=args.twopointnuc_upper_percentile,
            min_valid_ratio=args.twopointnuc_min_valid_ratio,
        )
        c_frames = nuc.run(frames=frames, frame_ratio=1.0)

    print("[*] Saving corrected frames...")
    loader.save_corrected_frames(
        corrected_frames=c_frames,
        output_dir=output_path,
        lower_percentile=1,
        upper_percentile=99,
    )
    print(f"[-] Output saved to: {output_path}")

    if show_frames:
        print("[*] Displaying raw vs corrected frames...")
        Visualizer.show_raw_vs_corrected(
            raw_frames=frames, corrected_dir=output_path, pause=0.03
        )


def run_multisegmentnuc(data_path, output_path, args, show_frames=False):
    loader = DataSetHandler(data_path, ext="png")
    frames = loader.load()

    num_frames, h, w = loader.get_shape()
    print(f"[*] Loaded {num_frames} frames ({h} x {w})")

    if args.optimize:
        region_candidates = (
            args.opt_multisegment_regions
            if args.opt_multisegment_regions is not None
            else ((10, 20, 30) if args.optimize_fast else (5, 10, 15, 20, 25, 30))
        )
        optimal_hp = find_optimal_multisegment_num_regions(
            frames=frames,
            lower_percentile=args.multisegmentnuc_lower_percentile,
            upper_percentile=args.multisegmentnuc_upper_percentile,
            min_valid_ratio=args.multisegmentnuc_min_valid_ratio,
            dv_tolerance=args.multisegmentnuc_dv_tolerance,
            num_region_candidates=region_candidates,
        )
        print(
            "[*] Optimal MultiSegmentNUC hyperparameters: "
            f"num_regions={optimal_hp['best_num_regions']} | "
            f"mean_rnu={optimal_hp['best_mean_rnu']:.6f}"
        )
        nuc = MultiSegmentNUC(
            num_regions=optimal_hp["best_num_regions"],
            lower_percentile=args.multisegmentnuc_lower_percentile,
            upper_percentile=args.multisegmentnuc_upper_percentile,
            min_valid_ratio=args.multisegmentnuc_min_valid_ratio,
            dv_tolerance=args.multisegmentnuc_dv_tolerance,
        )
    else:
        print("[*] Optimization disabled. Using CLI/default MultiSegmentNUC hyperparameters.")
        nuc = MultiSegmentNUC(
            num_regions=args.multisegmentnuc_num_regions,
            lower_percentile=args.multisegmentnuc_lower_percentile,
            upper_percentile=args.multisegmentnuc_upper_percentile,
            min_valid_ratio=args.multisegmentnuc_min_valid_ratio,
            dv_tolerance=args.multisegmentnuc_dv_tolerance,
        )
    c_frames = nuc.run(frames=frames)

    print("[*] Saving corrected frames...")
    loader.save_corrected_frames(
        corrected_frames=c_frames,
        output_dir=output_path,
        lower_percentile=1,
        upper_percentile=99,
    )
    print(f"[-] Output saved to: {output_path}")

    if show_frames:
        print("[*] Displaying raw vs corrected frames...")
        Visualizer.show_raw_vs_corrected(
            raw_frames=frames, corrected_dir=output_path, pause=0.03
        )


def run_cssbnuc(data_path, output_path, args, show_frames=False):
    loader = DataSetHandler(data_path, ext="png")
    frames = loader.load()

    num_frames, h, w = loader.get_shape()
    print(f"[*] Loaded {num_frames} frames ({h} x {w})")

    if args.optimize:
        alpha_candidates = (
            args.opt_css_alpha
            if args.opt_css_alpha is not None
            else ((0.95, 0.97, 0.99) if args.optimize_fast else (0.90, 0.95, 0.97, 0.99))
        )
        T_candidates = (
            args.opt_css_T
            if args.opt_css_T is not None
            else ((0.002, 0.005, 0.01) if args.optimize_fast else (0.001, 0.002, 0.005, 0.01))
        )
        optimal_hp = find_optimal_cssbnuc_params(
            frames=frames,
            alpha_candidates=alpha_candidates,
            T_candidates=T_candidates,
            eps=args.cssbnuc_eps,
        )
        print(
            "[*] Optimal ConstantStatsNUC hyperparameters: "
            f"alpha={optimal_hp['best_alpha']:.4f} | "
            f"T={optimal_hp['best_T']:.6f} | "
            f"mean_rnu={optimal_hp['best_mean_rnu']:.6f}"
        )
        nuc = ConstantStatsNUC(
            alpha=optimal_hp["best_alpha"],
            T=optimal_hp["best_T"],
            eps=args.cssbnuc_eps,
        )
    else:
        print("[*] Optimization disabled. Using CLI/default ConstantStatsNUC hyperparameters.")
        nuc = ConstantStatsNUC(
            alpha=args.cssbnuc_alpha,
            T=args.cssbnuc_T,
            eps=args.cssbnuc_eps,
        )

    # DataSetHandler returns raw float32 (0–65535), so normalize inside the algo
    c_frames = nuc.run(frames=frames, normalize=True)

    # Report column MAD improvement on last frame as a quick quality check
    last_raw = frames[-1] / 65535.0
    last_corr = c_frames[-1]
    print(
        f"[-] Last frame column MAD — before: {col_mad(last_raw):.6f} | "
        f"after: {col_mad(last_corr):.6f}"
    )
    print(
        f"[-] Last frame row MAD    — before: {row_mad(last_raw):.6f} | "
        f"after: {row_mad(last_corr):.6f}"
    )

    print("[*] Saving corrected frames...")
    loader.save_corrected_frames(
        corrected_frames=c_frames,
        output_dir=output_path,
        lower_percentile=1,
        upper_percentile=99,
    )
    print(f"[-] Output saved to: {output_path}")

    if show_frames:
        print("[*] Displaying raw vs corrected frames...")
        Visualizer.show_raw_vs_corrected(
            raw_frames=frames, corrected_dir=output_path, pause=0.03
        )


def run_lmssbnuc(data_path, output_path, args, show_frames=False):
    loader = DataSetHandler(data_path, ext="png")
    frames = loader.load()

    num_frames, h, w = loader.get_shape()
    print(f"[*] Loaded {num_frames} frames ({h} x {w})")

    # Normalise from raw uint16 range to [0, 1]
    frames = [f / 65535.0 for f in frames]

    variant = args.lmssbnuc_variant
    if args.optimize:
        lms_epsilon_candidates = (
            args.opt_lms_epsilon
            if args.opt_lms_epsilon is not None
            else ((0.03, 0.05) if args.optimize_fast else None)
        )
        lms_K_candidates = (
            args.opt_lms_K
            if args.opt_lms_K is not None
            else ((0.02, 0.05) if args.optimize_fast else None)
        )
        lms_M_scale_candidates = (
            args.opt_lms_M_scale
            if args.opt_lms_M_scale is not None
            else ((0.5, 0.75) if args.optimize_fast else None)
        )
        lms_T_candidates = (
            args.opt_lms_T
            if args.opt_lms_T is not None
            else ((0.002, 0.005) if args.optimize_fast else None)
        )
        optimal_hp = find_optimal_lmssbnuc_params(
            frames=frames,
            variant=variant,
            sigma=args.lmssbnuc_sigma,
            kernel_size=args.lmssbnuc_kernel_size,
            local_var_window=args.lmssbnuc_local_var_window,
            epsilon_candidates=lms_epsilon_candidates,
            K_candidates=lms_K_candidates,
            M_scale_candidates=lms_M_scale_candidates,
            T_candidates=lms_T_candidates,
        )

        if variant == "standard":
            print(
                "[*] Optimal LMS-SBNUC (standard): "
                f"epsilon={optimal_hp['best_epsilon']:.6f} | "
                f"mean_rnu={optimal_hp['best_mean_rnu']:.6f}"
            )
            nuc = StandardLMSSBNUC(
                epsilon=optimal_hp["best_epsilon"],
                sigma=args.lmssbnuc_sigma,
                kernel_size=args.lmssbnuc_kernel_size,
            )
        elif variant == "adaptive":
            print(
                "[*] Optimal LMS-SBNUC (adaptive): "
                f"K={optimal_hp['best_K']:.6f} | "
                f"M_scale={optimal_hp['best_M_scale']:.6f} | "
                f"mean_rnu={optimal_hp['best_mean_rnu']:.6f}"
            )
            nuc = AdaptiveLMSSBNUC(
                K=optimal_hp["best_K"],
                M_scale=optimal_hp["best_M_scale"],
                sigma=args.lmssbnuc_sigma,
                kernel_size=args.lmssbnuc_kernel_size,
                local_var_window=args.lmssbnuc_local_var_window,
            )
        elif variant == "gated":
            print(
                "[*] Optimal LMS-SBNUC (gated): "
                f"K={optimal_hp['best_K']:.6f} | "
                f"M_scale={optimal_hp['best_M_scale']:.6f} | "
                f"T={optimal_hp['best_T']:.6f} | "
                f"mean_rnu={optimal_hp['best_mean_rnu']:.6f}"
            )
            nuc = GatedAdaptiveLMSSBNUC(
                K=optimal_hp["best_K"],
                M_scale=optimal_hp["best_M_scale"],
                sigma=args.lmssbnuc_sigma,
                kernel_size=args.lmssbnuc_kernel_size,
                local_var_window=args.lmssbnuc_local_var_window,
                T=optimal_hp["best_T"],
            )
        else:
            raise ValueError(f"Unknown lmssbnuc variant: {variant}")
    else:
        print("[*] Optimization disabled. Using CLI/default LMS-SBNUC hyperparameters.")
        if variant == "standard":
            nuc = StandardLMSSBNUC(
                epsilon=args.lmssbnuc_epsilon,
                sigma=args.lmssbnuc_sigma,
                kernel_size=args.lmssbnuc_kernel_size,
            )
        elif variant == "adaptive":
            nuc = AdaptiveLMSSBNUC(
                K=args.lmssbnuc_K,
                M_scale=args.lmssbnuc_M_scale,
                sigma=args.lmssbnuc_sigma,
                kernel_size=args.lmssbnuc_kernel_size,
                local_var_window=args.lmssbnuc_local_var_window,
            )
        elif variant == "gated":
            nuc = GatedAdaptiveLMSSBNUC(
                K=args.lmssbnuc_K,
                M_scale=args.lmssbnuc_M_scale,
                sigma=args.lmssbnuc_sigma,
                kernel_size=args.lmssbnuc_kernel_size,
                local_var_window=args.lmssbnuc_local_var_window,
                T=args.lmssbnuc_T,
            )
        else:
            raise ValueError(f"Unknown lmssbnuc variant: {variant}")

    print(f"[*] Running LMS-SBNUC ({variant})...")
    c_frames = nuc.run(frames=frames)

    last_raw = frames[-1]
    last_corr = c_frames[-1]
    print(
        f"[-] Last frame column MAD — before: {col_mad(last_raw):.6f} | "
        f"after: {col_mad(last_corr):.6f}"
    )

    print("[*] Saving corrected frames...")
    loader.save_corrected_frames(
        corrected_frames=c_frames,
        output_dir=output_path,
        lower_percentile=1,
        upper_percentile=99,
    )
    print(f"[-] Output saved to: {output_path}")

    if show_frames:
        print("[*] Displaying raw vs corrected frames...")
        Visualizer.show_raw_vs_corrected(
            raw_frames=[f * 65535.0 for f in frames],
            corrected_dir=output_path,
            pause=0.03,
        )


# Real defaults for every method's hyperparameters.
# Also used to detect which params belong to which method.
METHOD_PARAMS = {
    "cssbnuc": {
        "cssbnuc_alpha": 0.97,
        "cssbnuc_T": 0.005,
        "cssbnuc_eps": 1e-6,
    },
    "twopointnuc": {
        "twopointnuc_num_regions": 20,
        "twopointnuc_lower_percentile": 5.0,
        "twopointnuc_upper_percentile": 95.0,
        "twopointnuc_min_valid_ratio": 0.8,
    },
    "multisegmentnuc": {
        "multisegmentnuc_num_regions": 20,
        "multisegmentnuc_lower_percentile": 5.0,
        "multisegmentnuc_upper_percentile": 95.0,
        "multisegmentnuc_min_valid_ratio": 0.8,
        "multisegmentnuc_dv_tolerance": None,
    },
    "lmssbnuc": {
        "lmssbnuc_variant": "standard",
        "lmssbnuc_epsilon": 0.05,
        "lmssbnuc_K": 0.05,
        "lmssbnuc_M_scale": 0.5,
        "lmssbnuc_sigma": 5.0,
        "lmssbnuc_kernel_size": 21,
        "lmssbnuc_local_var_window": 5,
        "lmssbnuc_T": 0.002,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="NUC Correction Algorithms",
    )

    # ── Required arguments ───────────────────────────────────────────────────
    parser.add_argument(
        "-d", "--data", required=True, help="Path to raw frame directory"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Directory to save corrected frames"
    )
    parser.add_argument(
        "-m",
        "--method",
        required=True,
        choices=["twopointnuc", "multisegmentnuc", "cssbnuc", "lmssbnuc"],
        help="NUC method to use",
    )
    parser.add_argument(
        "-s",
        "--show_frames",
        action="store_true",
        help="Display raw vs corrected frames using matplotlib",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_false",
        dest="optimize",
        help="Disable hyperparameter optimization and use provided/default parameters.",
    )
    parser.add_argument(
        "--optimize-fast",
        action="store_true",
        help="Use smaller candidate grids for quicker optimization.",
    )
    parser.add_argument(
        "--opt-twopoint-ratios",
        type=str,
        default=None,
        help="Comma-separated frame-ratio candidates for TwoPoint optimization (e.g. 1.0,0.5,0.25,0.1).",
    )
    parser.add_argument(
        "--opt-twopoint-regions",
        type=str,
        default=None,
        help="Comma-separated num_regions candidates for TwoPoint optimization (e.g. 5,10,20,25).",
    )
    parser.add_argument(
        "--opt-multisegment-regions",
        type=str,
        default=None,
        help="Comma-separated num_regions candidates for MultiSegment optimization (e.g. 5,10,15,20,25,30).",
    )
    parser.add_argument(
        "--opt-css-alpha",
        type=str,
        default=None,
        help="Comma-separated alpha candidates for ConstantStats optimization.",
    )
    parser.add_argument(
        "--opt-css-T",
        type=str,
        default=None,
        help="Comma-separated T candidates for ConstantStats optimization.",
    )
    parser.add_argument(
        "--opt-lms-epsilon",
        type=str,
        default=None,
        help="Comma-separated epsilon candidates for LMS standard variant optimization.",
    )
    parser.add_argument(
        "--opt-lms-K",
        type=str,
        default=None,
        help="Comma-separated K candidates for LMS adaptive/gated optimization.",
    )
    parser.add_argument(
        "--opt-lms-M-scale",
        type=str,
        default=None,
        help="Comma-separated M_scale candidates for LMS adaptive/gated optimization.",
    )
    parser.add_argument(
        "--opt-lms-T",
        type=str,
        default=None,
        help="Comma-separated T candidates for LMS gated optimization.",
    )
    parser.set_defaults(optimize=True, optimize_fast=False)

    # ── CS-SBNUC hyperparameters ─────────────────────────────────────────────
    cs = parser.add_argument_group("cssbnuc hyperparameters")
    cs.add_argument(
        "--cssbnuc-alpha",
        type=float,
        default=None,
        dest="cssbnuc_alpha",
        help="Exponential smoothing factor for running mean and spread (default: 0.97)",
    )
    cs.add_argument(
        "--cssbnuc-T",
        type=float,
        default=None,
        dest="cssbnuc_T",
        help="Change-detection threshold; pixels update only if |Y - Y_prev| > T (default: 0.005)",
    )
    cs.add_argument(
        "--cssbnuc-eps",
        type=float,
        default=None,
        dest="cssbnuc_eps",
        help="Stability epsilon added to spread S to avoid division by zero (default: 1e-6)",
    )

    # ── Two-Point NUC hyperparameters ────────────────────────────────────────
    tp = parser.add_argument_group("twopointnuc hyperparameters")
    tp.add_argument(
        "--twopointnuc-num-regions",
        type=int,
        default=None,
        dest="twopointnuc_num_regions",
        help="Number of temporal bins for region selection (default: 20)",
    )
    tp.add_argument(
        "--twopointnuc-lower-percentile",
        type=float,
        default=None,
        dest="twopointnuc_lower_percentile",
        help="Lower percentile for outlier trimming during prior estimation (default: 5.0)",
    )
    tp.add_argument(
        "--twopointnuc-upper-percentile",
        type=float,
        default=None,
        dest="twopointnuc_upper_percentile",
        help="Upper percentile for outlier trimming during prior estimation (default: 95.0)",
    )
    tp.add_argument(
        "--twopointnuc-min-valid-ratio",
        type=float,
        default=None,
        dest="twopointnuc_min_valid_ratio",
        help="Minimum fraction of frames that must be valid for prior estimation (default: 0.8)",
    )

    # ── Multi-Segment NUC hyperparameters ────────────────────────────────────
    ms = parser.add_argument_group("multisegmentnuc hyperparameters")
    ms.add_argument(
        "--multisegmentnuc-num-regions",
        type=int,
        default=None,
        dest="multisegmentnuc_num_regions",
        help="Number of temporal bins for region selection (default: 20)",
    )
    ms.add_argument(
        "--multisegmentnuc-lower-percentile",
        type=float,
        default=None,
        dest="multisegmentnuc_lower_percentile",
        help="Lower percentile for outlier trimming (default: 5.0)",
    )
    ms.add_argument(
        "--multisegmentnuc-upper-percentile",
        type=float,
        default=None,
        dest="multisegmentnuc_upper_percentile",
        help="Upper percentile for outlier trimming (default: 95.0)",
    )
    ms.add_argument(
        "--multisegmentnuc-min-valid-ratio",
        type=float,
        default=None,
        dest="multisegmentnuc_min_valid_ratio",
        help="Minimum fraction of frames that must be valid (default: 0.8)",
    )
    ms.add_argument(
        "--multisegmentnuc-dv-tolerance",
        type=float,
        default=None,
        dest="multisegmentnuc_dv_tolerance",
        help="Segmentation tolerance for D_V steps (default: auto-computed from data)",
    )

    # ── LMS-SBNUC hyperparameters ─────────────────────────────────────────────
    lms = parser.add_argument_group("lmssbnuc hyperparameters")
    lms.add_argument(
        "--lmssbnuc-variant",
        type=str,
        default=None,
        dest="lmssbnuc_variant",
        choices=["standard", "adaptive", "gated"],
        help="LMS-SBNUC variant to run (default: standard)",
    )
    lms.add_argument(
        "--lmssbnuc-epsilon",
        type=float,
        default=None,
        dest="lmssbnuc_epsilon",
        help="Fixed LMS learning rate used by the standard variant (default: 0.05)",
    )
    lms.add_argument(
        "--lmssbnuc-K",
        type=float,
        default=None,
        dest="lmssbnuc_K",
        help="Base learning rate numerator for adaptive/gated variants (default: 0.05)",
    )
    lms.add_argument(
        "--lmssbnuc-M-scale",
        type=float,
        default=None,
        dest="lmssbnuc_M_scale",
        help="Local variance scale factor for adaptive/gated variants (default: 0.5)",
    )
    lms.add_argument(
        "--lmssbnuc-sigma",
        type=float,
        default=None,
        dest="lmssbnuc_sigma",
        help="Gaussian blur sigma for the reference image (default: 5.0)",
    )
    lms.add_argument(
        "--lmssbnuc-kernel-size",
        type=int,
        default=None,
        dest="lmssbnuc_kernel_size",
        help="Informational kernel extent (default: 21)",
    )
    lms.add_argument(
        "--lmssbnuc-local-var-window",
        type=int,
        default=None,
        dest="lmssbnuc_local_var_window",
        help="Window size for local variance estimation in adaptive/gated variants (default: 5)",
    )
    lms.add_argument(
        "--lmssbnuc-T",
        type=float,
        default=None,
        dest="lmssbnuc_T",
        help="Change-detection threshold for the gated variant (default: 0.002)",
    )

    args = parser.parse_args()
    try:
        args.opt_twopoint_ratios = (
            _parse_csv_floats(args.opt_twopoint_ratios, "--opt-twopoint-ratios")
            if args.opt_twopoint_ratios is not None
            else None
        )
        args.opt_twopoint_regions = (
            _parse_csv_ints(args.opt_twopoint_regions, "--opt-twopoint-regions")
            if args.opt_twopoint_regions is not None
            else None
        )
        args.opt_multisegment_regions = (
            _parse_csv_ints(args.opt_multisegment_regions, "--opt-multisegment-regions")
            if args.opt_multisegment_regions is not None
            else None
        )
        args.opt_css_alpha = (
            _parse_csv_floats(args.opt_css_alpha, "--opt-css-alpha")
            if args.opt_css_alpha is not None
            else None
        )
        args.opt_css_T = (
            _parse_csv_floats(args.opt_css_T, "--opt-css-T")
            if args.opt_css_T is not None
            else None
        )
        args.opt_lms_epsilon = (
            _parse_csv_floats(args.opt_lms_epsilon, "--opt-lms-epsilon")
            if args.opt_lms_epsilon is not None
            else None
        )
        args.opt_lms_K = (
            _parse_csv_floats(args.opt_lms_K, "--opt-lms-K")
            if args.opt_lms_K is not None
            else None
        )
        args.opt_lms_M_scale = (
            _parse_csv_floats(args.opt_lms_M_scale, "--opt-lms-M-scale")
            if args.opt_lms_M_scale is not None
            else None
        )
        args.opt_lms_T = (
            _parse_csv_floats(args.opt_lms_T, "--opt-lms-T")
            if args.opt_lms_T is not None
            else None
        )
    except ValueError as exc:
        parser.error(str(exc))

    # ── Cross-method validation ──────────────────────────────────────────────
    # All hyperparameter defaults are None so we can tell if the user actually
    # passed a flag. Error if any flag from a non-selected method was given.
    foreign_flags = []
    for method, params in METHOD_PARAMS.items():
        if method == args.method:
            continue
        for dest, real_default in params.items():
            value = getattr(args, dest)
            # A value is "user-supplied" when it differs from the real default
            # AND is not None (None is our sentinel for "not passed").
            # Special case: real_default is None (dv_tolerance) — can't
            # distinguish, but that's harmless since None is the actual default.
            if value is not None and value != real_default:
                flag = "--" + dest.replace("_", "-")
                foreign_flags.append((flag, method))

    if foreign_flags:
        lines = "\n".join(f"  {flag}  (belongs to '{m}')" for flag, m in foreign_flags)
        parser.error(
            f"You selected method '{args.method}' but passed flags for other methods:\n"
            + lines
        )

    # ── Apply real defaults for the chosen method ────────────────────────────
    for dest, real_default in METHOD_PARAMS[args.method].items():
        if getattr(args, dest) is None:
            setattr(args, dest, real_default)

    dispatch = {
        "twopointnuc": run_twopointnuc,
        "multisegmentnuc": run_multisegmentnuc,
        "cssbnuc": run_cssbnuc,
        "lmssbnuc": run_lmssbnuc,
    }

    runner = dispatch.get(args.method)
    if runner is None:
        print(f"[X] Unknown method: {args.method}")
        sys.exit(1)

    runner(args.data, args.output, args, args.show_frames)


if __name__ == "__main__":
    main()
