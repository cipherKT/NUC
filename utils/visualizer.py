import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    @staticmethod
    def normalize_for_display(frame, percentile_low=1, percentile_high=99):

        values = frame.flatten()
        lo = np.percentile(values, percentile_low)
        hi = np.percentile(values, percentile_high)

        if hi <= lo:
            return np.zeros_like(frame, dtype=np.float32)

        normalized = np.clip((frame - lo) / (hi - lo), 0.0, 1.0)
        return normalized.astype(np.float32)

    @staticmethod
    def show_raw_vs_corrected(raw_frames, corrected_dir, pause=0.03):

        files = sorted(
            f for f in os.listdir(corrected_dir) if f.lower().endswith(".png")
        )

        if len(files) == 0:
            raise RuntimeError(f"No corrected PNG files found in: {corrected_dir}")

        num_frames = min(len(files), len(raw_frames))
        if len(files) != len(raw_frames):
            print(
                f"[!] Warning: raw frames ({len(raw_frames)}) and corrected files "
                f"({len(files)}) count mismatch — showing {num_frames} frames"
            )

        plt.figure(figsize=(12, 5))

        for idx in range(num_frames):
            corr_path = os.path.join(corrected_dir, files[idx])
            corr = cv2.imread(corr_path, cv2.IMREAD_UNCHANGED)

            if corr is None:
                print(f"[!] Warning: Could not load corrected frame: {corr_path}")
                continue

            raw_display = Visualizer.normalize_for_display(raw_frames[idx])
            corr_display = corr.astype(np.float32) / 65535.0

            plt.clf()

            plt.subplot(1, 2, 1)
            plt.imshow(raw_display, cmap="winter", vmin=0, vmax=1)
            plt.title("Raw")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(corr_display, cmap="winter", vmin=0, vmax=1)
            plt.title("Corrected")
            plt.axis("off")

            plt.suptitle(f"Frame {idx + 1}/{num_frames}")
            plt.tight_layout()
            plt.pause(pause)

        plt.close()
