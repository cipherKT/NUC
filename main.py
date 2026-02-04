import argparse
import sys

from datasetHandler import DataSetHandler
from twoPointNUC import SceneBasedTwoPointNUC


def run_twopointnuc(data_path, output_path, show_frames=False):
    loader = DataSetHandler(data_path, ext="png")
    frames = loader.load()

    num_frames, h, w = loader.get_shape()
    print(f"[*] Loaded {num_frames} frames ({h} x {w})")

    nuc = SceneBasedTwoPointNUC(num_regions=10)

    print("[*] Estimating pseudo-prior...")
    prior = nuc.estimate_prior(frames)
    print(f"[-] Prior slope = {prior:.6f}")

    print("[*] Building temporal matrix...")
    matrix = nuc.build_temporal_matrix(frames)

    print("[*] Sorting temporal matrix...")
    sorted_matrix = nuc.sort_matrix(matrix)

    print("[*] Partitioning regions...")
    regions = nuc.partition_regions(sorted_matrix)

    print("[*] Selecting optimal regions...")
    idx_l, idx_h = nuc.select_regions(sorted_matrix, regions)
    print(f"[-] Selected regions: {idx_l} -> {idx_h}")

    print("[*] Estimating gain and offset...")
    gain, offset = nuc.estimate_gain_offset(sorted_matrix, regions, idx_l, idx_h)

    print("[*] Applying correction...")
    corrected_frames = nuc.apply_correction_and_store(frames, gain, offset, h, w)

    print("[*] Saving corrected frames...")
    loader.save_corrected_frames(
        corrected_frames=corrected_frames,
        output_dir=output_path,
        lower_percentile=1,
        upper_percentile=99,
    )

    print(f"[-] Output saved to: {output_path}")

    if show_frames:
        print("[*] Displaying raw vs corrected frames...")
        loader.show_raw_vs_corrected(
            raw_frames=frames,
            corrected_dir=output_path,
            pause=0.03,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Scene-Based Two-Point Non-Uniformity Correction (SB-TPNUC)"
    )

    parser.add_argument(
        "-d",
        "--data",
        required=True,
        help="Path to directory containing raw infrared frames (PNG, uint16)",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Directory to save corrected frames",
    )

    parser.add_argument(
        "-m",
        "--method",
        required=True,
        choices=["twopointnuc"],
        help="NUC method to use",
    )

    parser.add_argument(
        "-s",
        "--show_frames",
        action="store_true",
        help="Display raw vs corrected frames using matplotlib",
    )

    args = parser.parse_args()

    if args.method == "twopointnuc":
        run_twopointnuc(args.data, args.output, args.show_frames)
    else:
        print(f"[X] Unknown method: {args.method}")
        sys.exit(1)


if __name__ == "__main__":
    main()
