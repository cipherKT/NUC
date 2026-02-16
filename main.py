import argparse
import sys

from datasetHandler import DataSetHandler
from twoPointNUC import SceneBasedTwoPointNUC


def run_twopointnuc(data_path, output_path, show_frames=False):
    loader = DataSetHandler(data_path, ext="png")
    frames = loader.load()

    num_frames, h, w = loader.get_shape()

    print(f"[*] Loaded {num_frames} frames ({h} x {w})")

    nuc = SceneBasedTwoPointNUC(
        num_regions=20, lower_percentile=5, upper_percentile=95, min_valid_ratio=0.8
    )

    print("[*] Estimating pseudo-prior....")
    prior = nuc.estimate_prior(frames=frames)
    print(f"[-] Prior Slope = {prior:.6f}")

    print("[*] Building Temporal Matrix...")
    matrix = nuc.build_temporal_matrix(frames=frames)
    print(f"[-] Matrix shape: {matrix.shape}")

    print("[*] Sorting Temporal Matrix...")
    sorted_matrix = nuc.sort_matrix(matrix=matrix)

    print("[*] Partitioning Regions...")
    regions = nuc.partition_regions(sorted_matrix=sorted_matrix)
    print(f"[-] Created {len(regions)} regions")

    print("[*] Selecting Optimal Regions....")
    idx_l, idx_h = nuc.select_regions(
        sorted_matrix=sorted_matrix, regions=regions, min_mean_diff=100
    )
    print(f"[-] Selected regions: {idx_l},{idx_h}")

    print("[*] Estimating gain and offset....")
    gain, offset = nuc.estimate_gain_offset(
        sorted_matrix=sorted_matrix, regions=regions, idx_low=idx_l, idx_high=idx_h
    )
    print(f"[-] Gain shape: {gain.shape}, Offset shape: {offset.shape}")
    print(f"[-] Gain range: [{gain.min():.4f}, {gain.max():.4f}]")
    print(f"[-] Offset range: [{offset.min():.4f}, {offset.max():.4f}]")

    print("[*] Applying corrections....")
    c_frames = nuc.apply_correction_and_store(
        frames=frames, gain=gain, offset=offset, h=h, w=w
    )
    print(f"[-] Corrected {len(c_frames)} frames")

    print("[*] Saving Corrected Frames....")
    loader.save_corrected_frames(
        corrected_frames=c_frames,
        output_dir=output_path,
        lower_percentile=1,
        upper_percentile=99,
    )

    print(f"[-] Output is saved to: {output_path}")

    if show_frames:
        print("[*] Displaying raw vs corrected frames...")
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
        choices=["twopointnuc"],
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
    else:
        print(f"[X] unknown method: {args.method}")
        sys.exit(1)


if __name__ == "__main__":
    main()
