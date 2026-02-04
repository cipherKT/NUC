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

            trimmed = []
            for v in values:
                if v >= lo and v <= hi:
                    trimmed.append(v)

            trimmed = np.array(trimmed)
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

        matrix = np.zeros((num_pixels, num_frames), dtype=np.float32)

        for t in range(num_frames):
            frame = frames[t]

            if frame.shape != (h, w):
                raise ValueError("Frame size is not matching")

            pixel_index = 0
            for i in range(h):
                for j in range(w):
                    matrix[pixel_index, t] = frame[i, j]
                    pixel_index += 1
        return matrix

    def sort_matrix(self, matrix):
        if matrix.ndim != 2:
            raise ValueError("Input matrix is not 2D")
        num_pixels, num_frames = matrix.shape

        sorted_matrix = np.zeros_like(matrix)
        for pixel_idx in range(num_pixels):
            values = []

            for t in range(num_frames):
                values.append(matrix[pixel_idx, t])

            values.sort()
            for t in range(num_frames):
                sorted_matrix[pixel_idx, t] = values[t]

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
        # Stats calc
        region_med = []
        region_stds = []

        for start, end in regions:
            values = []

            for p in range(sorted_matrix.shape[0]):
                for t in range(start, end):
                    values.append(sorted_matrix[p, t])
            values = np.array(values)

            med = np.median(values)
            mad = np.median(np.abs(values - med))
            std = 1.4826 * mad

            region_med.append(med)
            region_stds.append(std)
        region_med = np.array(region_med)
        region_stds = np.array(region_stds)

        # Deviation
        deviations = []

        for i in range(num_regions):
            predicted_std = self.prior_slope * region_med[i]
            deviation = abs(region_stds[i] - predicted_std)
            deviations.append(deviation)

        deviations = np.array(deviations)

        if min_mean_diff is None:
            dynamic_range = np.max(region_med) - np.min(region_med)
            min_mean_diff = 0.2 * dynamic_range

        best_score = None
        idx_low = None
        idx_high = None

        for i in range(num_regions):
            for j in range(num_regions):
                if i >= j:
                    continue

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

        values_i = []
        values_j = []

        for p in range(num_pixels):
            for t in range(start_i, end_i):
                values_i.append(sorted_matrix[p, t])
            for t in range(start_j, end_j):
                values_j.append(sorted_matrix[p, t])
        values_i = np.array(values_i)
        values_j = np.array(values_j)

        p_i = np.mean(values_i)
        p_j = np.mean(values_j)

        gain = np.zeros(num_pixels, dtype=np.float32)
        offset = np.zeros(num_pixels, dtype=np.float32)

        for p in range(num_pixels):
            sum_i = 0.0
            count_i = 0

            for t in range(start_i, end_i):
                sum_i += sorted_matrix[p, t]
                count_i += 1
            sum_j = 0.0
            count_j = 0

            for t in range(start_j, end_j):
                sum_j += sorted_matrix[p, t]
                count_j += 1

            q_i = sum_i / count_i
            q_j = sum_j / count_j

            denominator = q_j - q_i

            if abs(denominator) < eps:
                gain[p] = 1.0
                offset[p] = 0.0
            else:
                gain[p] = (p_j - p_i) / denominator
                offset[p] = p_i - gain[p] * q_i
        return gain, offset

    def apply_correction_and_store(self, frames, gain, offset, h, w):

        c_frames = []
        num_pixels = h * w

        for f in frames:
            corrected = np.zeros((h, w), dtype=np.float32)

            p_idx = 0
            for i in range(h):
                for j in range(w):
                    if p_idx >= num_pixels:
                        continue

                    corrected[i, j] = gain[p_idx] * f[i, j] + offset[p_idx]
                    p_idx += 1

            c_frames.append(corrected)

        self.corrected_frames = c_frames
        return c_frames
