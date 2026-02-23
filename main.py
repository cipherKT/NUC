import argparse
import sys

from datasetHandler import DataSetHandler
from multiSegmentNUC import MultiSegmentNUC
from twoPointNUC import SceneBasedTwoPointNUC


def run_twopointnuc(data_path, output_path, show_frames=False):
    loader = DataSetHandler(data_path, ext="png")
    frames = loader.load()

    num_frames, h, w = loader.get_shape()
    print(f"[*] Loaded {num_frames} frames ({h} x {w})")

    nuc = SceneBasedTwoPointNUC(
        num_regions=20, lower_percentile=5, upper_percentile=95, min_valid_ratio=0.8
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
        loader.show_raw_vs_corrected(
            raw_frames=frames, corrected_dir=output_path, pause=0.03
        )


def run_multisegmentnuc(data_path, output_path, show_frames=False):
    loader = DataSetHandler(data_path, ext="png")

    frames = loader.load()
    num_frame, h, w = loader.get_shape()
    print(f"[*] Loaded {num_frame} frames ({h}x{w})")

    nuc = MultiSegmentNUC(
        num_regions=20,
        lower_percentile=5.0,
        upper_percentile=95.0,
        min_valid_ratio=0.8,
        dv_tolerance=None,
    )
    c_frames = nuc.run(frames=frames)
    print("[*] Saving Corrected frames....")
    loader.save_corrected_frames(
        corrected_frames=c_frames,
        output_dir=output_path,
        lower_percentile=1,
        upper_percentile=99,
    )
    print(f"[-] Output saved to {output_path}")

    if show_frames:
        print("[*] Displaying output")
        loader.show_raw_vs_corrected(
            raw_frames=frames, corrected_dir=output_path, pause=0.03
        )


def main():
    parser = argparse.ArgumentParser(description="NUC Correction Algorithms")

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
        choices=["twopointnuc", "multisegmentnuc"],
        help="NUC method to use",
    )

    parser.add_argument(
        "-s",
        "--show_frames",
        action="store_true",
        help="Display corrected PNG frames using matplotlib",
    )
    args = parser.parse_args()

    if args.method == "twopointnuc":
        run_twopointnuc(args.data, args.output, args.show_frames)
    elif args.method == "multisegmentnuc":
        run_multisegmentnuc(args.data, args.output, args.show_frames)

    else:
        print(f"[X] unknown method: {args.method}")
        sys.exit(1)


if __name__ == "__main__":
    main()
