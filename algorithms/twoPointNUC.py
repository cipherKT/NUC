import numpy as np
from sklearn.linear_model import HuberRegressor


class SceneBasedTwoPointNUC:
    def __init__(
        self,
        num_regions,
        lower_percentile,
        upper_percentile,
        min_valid_ratio,
    ):
        self.num_regions = num_regions
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.min_valid_ratio = min_valid_ratio
        self.prior_slope = None
        self.frames = None
        self.corrected_frames = None

    def _validate_ratio(self, frame_ratio: float) -> float:
        if frame_ratio <= 0 or frame_ratio > 1:
            raise ValueError(f"frame_ratio must be in (0,1], got {frame_ratio}")
        return frame_ratio

    def _select_training_frames(self, frames: list, frame_ratio: float = 1.0) -> list:
        if not frames:
            raise RuntimeError("Please provide valid frames")
        ratio = self._validate_ratio(frame_ratio)
        total = len(frames)
        k = max(2, int(round(ratio * total)))
        k = min(k, total)

        return frames[:k]

    def estimate_prior(self, frames):
        medians = []
        stds = []

        for frame in frames:
            values = frame.flatten()
            lo = np.percentile(values, self.lower_percentile)
            hi = np.percentile(values, self.upper_percentile)

            mask = (values >= lo) & (values <= hi)
            trimmed = values[mask]

            if trimmed.size < values.size * 0.5:
                continue

            med = np.median(trimmed)
            mad = np.median(np.abs(trimmed - med))
            std = 1.4826 * mad

            medians.append(med)
            stds.append(std)

        medians = np.array(medians)
        stds = np.array(stds)

        if len(medians) < self.min_valid_ratio * len(frames):
            raise RuntimeError("Please use more number of frames")

        X = medians.reshape(-1, 1)
        y = stds

        model = HuberRegressor(fit_intercept=False)
        model.fit(X, y)

        self.prior_slope = model.coef_[0]
        return self.prior_slope

    def build_temporal_matrix(self, frames, frame_ratio: float = 1.0):
        if frames is None or len(frames) == 0:
            raise RuntimeError("Please provide valid frames")
        training_frames = self._select_training_frames(frames, frame_ratio=frame_ratio)
        self.frames = training_frames

        first_frame = training_frames[0]
        h, w = first_frame.shape

        num_frames = len(training_frames)
        num_pixels = h * w

        # Create matrix more efficiently
        matrix = np.zeros((num_pixels, num_frames), dtype=np.float32)

        for t in range(num_frames):
            frame = training_frames[t]

            if frame.shape != (h, w):
                raise ValueError("Frame size is not matching")

            # Flatten in row-major order (default for numpy)
            matrix[:, t] = frame.flatten()

        return matrix

    def sort_matrix(self, matrix):
        if matrix.ndim != 2:
            raise ValueError("Input matrix is not 2D")

        # Sort each row (pixel's temporal values) efficiently
        sorted_matrix = np.sort(matrix, axis=1)

        return sorted_matrix

    def partition_regions(self, sorted_matrix):
        if sorted_matrix.ndim != 2:
            raise ValueError("Input matrix is not 2D")

        num_pixels, num_frames = sorted_matrix.shape
        n = self.num_regions

        if n <= 1:
            raise ValueError("Number of regions must be greater than 1")

        region_width = num_frames // n
        regions = []

        start = 0
        for r in range(n):
            if r == n - 1:
                end = num_frames
            else:
                end = start + region_width

            regions.append((start, end))
            start = end
        return regions

    def select_regions(self, sorted_matrix, regions, min_mean_diff=None):
        if self.prior_slope is None:
            raise RuntimeError("Estimate prior first")

        num_regions = len(regions)

        # Stats calculation - vectorized
        region_med = []
        region_stds = []

        for start, end in regions:
            # Get all values in this region
            values = sorted_matrix[:, start:end].flatten()

            med = np.median(values)
            mad = np.median(np.abs(values - med))
            std = 1.4826 * mad

            region_med.append(med)
            region_stds.append(std)

        region_med = np.array(region_med)
        region_stds = np.array(region_stds)

        # Deviation calculation
        predicted_stds = self.prior_slope * region_med
        deviations = np.abs(region_stds - predicted_stds)

        if min_mean_diff is None:
            dynamic_range = np.max(region_med) - np.min(region_med)
            min_mean_diff = 0.2 * dynamic_range
            print(min_mean_diff)

        best_score = None
        idx_low = None
        idx_high = None

        for i in range(num_regions):
            for j in range(i + 1, num_regions):  # j > i
                mean_diff = abs(region_med[j] - region_med[i])
                if mean_diff < min_mean_diff:
                    continue

                score = deviations[i] + deviations[j]

                if best_score is None or score < best_score:
                    best_score = score
                    idx_low = i
                    idx_high = j

        if idx_low is None:
            idx_low = int(np.argmin(region_med))
            idx_high = int(np.argmax(region_med))

        return idx_low, idx_high

    def estimate_gain_offset(self, sorted_matrix, regions, idx_low, idx_high, eps=1e-6):
        start_i, end_i = regions[idx_low]
        start_j, end_j = regions[idx_high]

        num_pixels = sorted_matrix.shape[0]

        # Calculate global mean values for the two regions
        region_i_values = sorted_matrix[:, start_i:end_i]
        region_j_values = sorted_matrix[:, start_j:end_j]

        p_i = np.mean(region_i_values)
        p_j = np.mean(region_j_values)

        # Calculate per-pixel mean values
        q_i = np.mean(region_i_values, axis=1)  # Shape: (num_pixels,)
        q_j = np.mean(region_j_values, axis=1)  # Shape: (num_pixels,)

        # Calculate gain and offset
        denominator = q_j - q_i

        # Handle division by zero
        gain = np.ones(num_pixels, dtype=np.float32)
        offset = np.zeros(num_pixels, dtype=np.float32)

        valid_mask = np.abs(denominator) >= eps

        gain[valid_mask] = (p_j - p_i) / denominator[valid_mask]
        offset[valid_mask] = p_i - gain[valid_mask] * q_i[valid_mask]

        return gain, offset

    def apply_correction_and_store(self, frames, gain, offset, h, w):
        c_frames = []

        # Reshape gain and offset to 2D for broadcasting
        gain_2d = gain.reshape(h, w)
        offset_2d = offset.reshape(h, w)

        for frame in frames:
            # Apply correction: corrected = gain * raw + offset
            corrected = gain_2d * frame + offset_2d
            c_frames.append(corrected)

        self.corrected_frames = c_frames
        return c_frames

    def run(self, frames, min_mean_diff=None, frame_ratio: float = 1.0):
        if not frames or len(frames) == 0:
            raise RuntimeError("Please provide valid frames")

        h, w = frames[0].shape
        training_frames = self._select_training_frames(frames, frame_ratio=frame_ratio)

        # Step 1 — Estimate prior slope
        print("[*] Estimating pseudo-prior...")
        prior = self.estimate_prior(frames=training_frames)
        print(f"[-] Prior slope = {prior:.6f}")

        # Step 2 — Build temporal matrix
        print("[*] Building temporal matrix...")
        matrix = self.build_temporal_matrix(frames=training_frames, frame_ratio=1.0)
        print(f"[-] Matrix shape: {matrix.shape}")

        # Step 3 — Sort temporal matrix
        print("[*] Sorting temporal matrix...")
        sorted_matrix = self.sort_matrix(matrix=matrix)
        print(f"[-] Sorted matrix shape: {sorted_matrix.shape}")

        # Step 4 — Partition regions
        print("[*] Partitioning regions...")
        regions = self.partition_regions(sorted_matrix=sorted_matrix)
        print(f"[-] Created {len(regions)} regions")

        # Step 5 — Select optimal region pair
        print("[*] Selecting optimal regions...")
        idx_l, idx_h = self.select_regions(
            sorted_matrix=sorted_matrix, regions=regions, min_mean_diff=min_mean_diff
        )
        print(f"[-] Selected regions: low={idx_l}, high={idx_h}")

        # Step 6 — Estimate gain and offset
        print("[*] Estimating gain and offset...")
        gain, offset = self.estimate_gain_offset(
            sorted_matrix=sorted_matrix, regions=regions, idx_low=idx_l, idx_high=idx_h
        )
        print(f"[-] Gain shape: {gain.shape}, Offset shape: {offset.shape}")
        print(f"[-] Gain range:   [{gain.min():.4f}, {gain.max():.4f}]")
        print(f"[-] Offset range: [{offset.min():.4f}, {offset.max():.4f}]")

        # Step 7 — Apply correction
        print("[*] Applying corrections...")
        corrected_frames = self.apply_correction_and_store(
            frames=frames, gain=gain, offset=offset, h=h, w=w
        )
        print(f"[-] Corrected {len(corrected_frames)} frames")

        return corrected_frames
