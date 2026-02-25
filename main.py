import argparse
import sys

from algorithms import ConstantStatsNUC, MultiSegmentNUC, SceneBasedTwoPointNUC
from data import DataSetHandler
from utils import Visualizer, col_mad, row_mad


def run_twopointnuc(data_path, output_path, args, show_frames=False):
    loader = DataSetHandler(data_path, ext="png")
    frames = loader.load()

    num_frames, h, w = loader.get_shape()
    print(f"[*] Loaded {num_frames} frames ({h} x {w})")

    nuc = SceneBasedTwoPointNUC(
        num_regions=args.twopointnuc_num_regions,
        lower_percentile=args.twopointnuc_lower_percentile,
        upper_percentile=args.twopointnuc_upper_percentile,
        min_valid_ratio=args.twopointnuc_min_valid_ratio,
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


def run_multisegmentnuc(data_path, output_path, args, show_frames=False):
    loader = DataSetHandler(data_path, ext="png")
    frames = loader.load()

    num_frames, h, w = loader.get_shape()
    print(f"[*] Loaded {num_frames} frames ({h} x {w})")

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
        choices=["twopointnuc", "multisegmentnuc", "cssbnuc"],
        help="NUC method to use",
    )
    parser.add_argument(
        "-s",
        "--show_frames",
        action="store_true",
        help="Display raw vs corrected frames using matplotlib",
    )

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

    args = parser.parse_args()

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
        lines = "\n".join(
            f"  {flag}  (belongs to '{m}')" for flag, m in foreign_flags
        )
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
    }

    runner = dispatch.get(args.method)
    if runner is None:
        print(f"[X] Unknown method: {args.method}")
        sys.exit(1)

    runner(args.data, args.output, args, args.show_frames)


if __name__ == "__main__":
    main()
