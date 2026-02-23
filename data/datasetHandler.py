import glob
import os

import cv2
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
                raise ValueError(f"Expected 16-bit image, got {img.dtype} for {f}")
            frames.append(img.astype(np.float32))

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
                scaled_uint16 = np.zeros_like(frame, dtype=np.uint16)
            else:
                scaled = np.clip((frame - lo) / (hi - lo), 0.0, 1.0)
                scaled_uint16 = (scaled * 65535).astype(np.uint16)

            filename = os.path.join(output_dir, f"corrected_{idx:04d}.png")
            cv2.imwrite(filename, scaled_uint16)
