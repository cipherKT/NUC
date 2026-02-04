import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


class DataSetHandler:
    def __init__(self, directory, ext="png"):
        self.directory = directory
        self.ext = ext
        self.frames = []

    def load(self):
        pattern = os.path.join(self.directory, f"*.{self.ext}")
        files = sorted(glob.glob(pattern))
        if len(files) == 0:
            raise RuntimeError("No images found")
        frames = []

        for f in files:
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)

            if img is None:
                raise RuntimeError(f"Failed to read image: {f}")
            if img.dtype != np.uint16:
                raise ValueError(f"Expected 16-bit image got {img.dtype} for {f}")

            img = img.astype(np.float32)
            frames.append(img)
        self.frames = frames
        return frames

    def get_shape(self):
        if not self.frames:
            raise RuntimeError("Frames not loaded yet")
        h, w = self.frames[0].shape
        return len(self.frames), h, w

    def save_corrected_frames(
        self, corrected_frames, output_dir, lower_percentile=1, upper_percentile=99
    ):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for idx, frame in enumerate(corrected_frames):
            values = frame.flatten()

            lo = np.percentile(values, lower_percentile)
            hi = np.percentile(values, upper_percentile)

            if hi <= lo:
                # Handle edge case where all values are the same
                scaled_uint16 = np.zeros_like(frame, dtype=np.uint16)
            else:
                scaled = (frame - lo) / (hi - lo)
                scaled = np.clip(scaled, 0.0, 1.0)
                scaled_uint16 = (scaled * 65535).astype(np.uint16)

            filename = os.path.join(output_dir, f"corrected_{idx:04d}.png")

            cv2.imwrite(filename, scaled_uint16)

    def show_raw_vs_corrected(self, raw_frames, corrected_dir, pause=0.03):
        """
        Display raw vs corrected frames side by side.

        Args:
            raw_frames: List of raw frame arrays
            corrected_dir: Directory containing corrected PNG files
            pause: Pause duration between frames
        """
        files = sorted(
            [f for f in os.listdir(corrected_dir) if f.lower().endswith(".png")]
        )

        if len(files) == 0:
            raise RuntimeError("No corrected PNG files found")

        if len(files) != len(raw_frames):
            print(
                f"[!] Warning: number of raw ({len(raw_frames)}) and corrected ({len(files)}) frames do not match"
            )
            # Use the minimum to avoid index errors
            num_frames = min(len(files), len(raw_frames))
        else:
            num_frames = len(files)

        plt.figure(figsize=(12, 5))

        for idx in range(num_frames):
            raw = raw_frames[idx]
            fname = files[idx]

            corr_path = os.path.join(corrected_dir, fname)
            corr = cv2.imread(corr_path, cv2.IMREAD_UNCHANGED)

            if corr is None:
                print(f"[!] Warning: Could not load corrected frame: {corr_path}")
                continue

            corr = corr.astype(np.float32)

            plt.clf()

            # Normalize raw frame for display (1st and 99th percentile)
            raw_display = self._normalize_for_display(
                raw, percentile_low=1, percentile_high=99
            )

            # Corrected frame is already normalized when saved
            # But we normalize again for consistent display
            corr_display = corr / 65535.0  # Convert back to 0-1 range

            # Raw frame
            plt.subplot(1, 2, 1)
            plt.imshow(raw_display, cmap="winter", vmin=0, vmax=1)
            plt.title("Raw")
            plt.axis("off")

            # Corrected frame
            plt.subplot(1, 2, 2)
            plt.imshow(corr_display, cmap="winter", vmin=0, vmax=1)
            plt.title("Corrected")
            plt.axis("off")

            plt.suptitle(f"Frame {idx + 1}/{num_frames}")
            plt.tight_layout()
            plt.pause(pause)

        plt.close()

    def _normalize_for_display(self, frame, percentile_low=1, percentile_high=99):
        """
        Normalize frame to 0-1 range using percentile clipping.

        Args:
            frame: Input frame array
            percentile_low: Lower percentile for clipping
            percentile_high: Upper percentile for clipping

        Returns:
            Normalized frame in 0-1 range
        """
        values = frame.flatten()
        lo = np.percentile(values, percentile_low)
        hi = np.percentile(values, percentile_high)

        if hi <= lo:
            return np.zeros_like(frame, dtype=np.float32)

        normalized = (frame - lo) / (hi - lo)
        normalized = np.clip(normalized, 0.0, 1.0)

        return normalized
