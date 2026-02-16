import numpy as np
from sklearn.linear_model import HuberRegressor


class SceneBasedTwoPointNUC:
    def __init__(
        self, num_regions, lower_percentile, upper_percentile, min_valid_ratio
    ):
        self.num_regions = num_regions
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.min_valid_ratio = min_valid_ratio
        self.prior_slope = None
        self.frames = None
        self.corrected_frames = None

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

    def build_temporal_matrix(self, frames):
        if frames is None or len(frames) == 0:
            raise RuntimeError("Please provide valid frames")
        self.frames = frames

        first_frame = frames[0]
        h, w = first_frame.shape

        num_frames = len(frames)
        num_pixels = h * w

        # Create matrix more efficiently
        matrix = np.zeros((num_pixels, num_frames), dtype=np.float32)

        for t in range(num_frames):
            frame = frames[t]

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
        """
        Apply correction to frames using gain and offset vectors.

        Args:
            frames: List of frames to correct
            gain: 1D array of gain coefficients (length = h*w)
            offset: 1D array of offset coefficients (length = h*w)
            h: Height of frames
            w: Width of frames
        """
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
