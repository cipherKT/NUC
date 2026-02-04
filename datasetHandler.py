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
                raise ValueError(f"Expected uint16 image, got {img.dtype}")

            frames.append(img.astype(np.float32))

        self.frames = frames
        return frames

    def get_shape(self):
        if not self.frames:
            raise RuntimeError("Frames not loaded")
        h, w = self.frames[0].shape
        return len(self.frames), h, w

    def save_corrected_frames(
        self, corrected_frames, output_dir, lower_percentile=1, upper_percentile=99
    ):
        os.makedirs(output_dir, exist_ok=True)

        for idx, frame in enumerate(corrected_frames):
            lo = np.percentile(frame, lower_percentile)
            hi = np.percentile(frame, upper_percentile)
            scaled = np.clip((frame - lo) / (hi - lo), 0, 1)
            out = (scaled * 65535).astype(np.uint16)

            cv2.imwrite(os.path.join(output_dir, f"corrected_{idx:04d}.png"), out)

    def show_raw_vs_corrected(self, raw_frames, corrected_dir, pause=0.03):
        files = sorted(f for f in os.listdir(corrected_dir) if f.endswith(".png"))

        plt.figure(figsize=(10, 4))
        plt.set_cmap("winter")

        for idx, fname in enumerate(files):
            raw = raw_frames[idx]
            corr = cv2.imread(
                os.path.join(corrected_dir, fname), cv2.IMREAD_UNCHANGED
            ).astype(np.float32)

            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(raw)
            plt.title("Raw")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(corr)
            plt.title("Corrected")
            plt.axis("off")

            plt.suptitle(f"Frame {idx + 1}/{len(files)}")
            plt.pause(pause)

        plt.close()
