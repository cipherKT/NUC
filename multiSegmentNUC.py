import numpy as np


class MultiSegmentNUC:
    def __init__(
        self,
        num_regions: int = 20,
        lower_percentile: float = 5.0,
        upper_percentile: float = 95.0,
        min_valid_ratio: float = 0.8,
        dv_tolerance: float = None,
    ):
        """
        Scene-based multi-segment NUC correction.
        Based on: Li Dandan et al., IEEE Photonics Journal, 2024.

        Args:
            num_regions:      Number of uniform regions to partition sorted data into
            lower_percentile: Lower percentile for trimming in prior estimation
            upper_percentile: Upper percentile for trimming in prior estimation
            min_valid_ratio:  Minimum ratio of valid frames required
            dv_tolerance:     Tolerance for D_V difference when grouping regions
                              into linear segments. If None, auto-computed as
                              0.25% of the dynamic range of region means.
        """
        self.num_regions = num_regions
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.min_valid_ratio = min_valid_ratio
        self.dv_tolerance = dv_tolerance

        # State populated during pipeline — not stored as class state,
        # passed between methods as return values
        self.corrected_frames = None

    def build_temporal_matrix(self, frames: list) -> np.ndarray:
        """
        Build a temporal matrix from frames.

        Args:
            frames: List of 2D float32 arrays of shape (H, W)

        Returns:
            matrix: np.ndarray of shape (H, W, num_frames)
                    matrix[r] gives (W, num_frames) for detector row r
        """
        if not frames or len(frames) == 0:
            raise RuntimeError("Please provide valid frames")

        h, w = frames[0].shape
        num_frames = len(frames)

        matrix = np.zeros((h, w, num_frames), dtype=np.float32)

        for t, frame in enumerate(frames):
            if frame.shape != (h, w):
                raise ValueError(
                    f"Frame {t} shape mismatch: expected {(h, w)}, got {frame.shape}"
                )
            matrix[:, :, t] = frame

        return matrix

    def sort_matrix(self, matrix: np.ndarray, trim_cols: int = 10) -> np.ndarray:
        """
        Sort each pixel's temporal values in ascending order and trim
        saturated/bad pixel columns from both ends.

        Args:
            matrix:    np.ndarray of shape (H, W, num_frames)
            trim_cols: Number of columns to discard from each end after
                       sorting (default 10, as per paper). Set 0 to disable.

        Returns:
            sorted_matrix: np.ndarray of shape (H, W, num_frames - 2*trim_cols)
        """
        if matrix.ndim != 3:
            raise ValueError(
                f"Expected 3D matrix (H, W, num_frames), got shape {matrix.shape}"
            )

        num_frames = matrix.shape[2]

        if trim_cols > 0:
            if 2 * trim_cols >= num_frames:
                raise ValueError(
                    f"trim_cols={trim_cols} too large for num_frames={num_frames}. "
                    f"Must be less than num_frames/2={num_frames // 2}"
                )

        # Sort along the frames axis (axis=2) for every pixel
        sorted_matrix = np.sort(matrix, axis=2)

        # Trim top and bottom columns to remove saturation and bad pixels
        if trim_cols > 0:
            sorted_matrix = sorted_matrix[:, :, trim_cols:-trim_cols]

        return sorted_matrix

    def partition_regions(self, sorted_matrix: np.ndarray) -> list:
        """
        Partition the sorted temporal axis into N uniform regions.

        Args:
            sorted_matrix: np.ndarray of shape (H, W, num_frames)

        Returns:
            regions: List of (start, end) index tuples along the frames axis.
                     Length = num_regions
        """
        if sorted_matrix.ndim != 3:
            raise ValueError(
                f"Expected 3D matrix (H, W, num_frames), got shape {sorted_matrix.shape}"
            )

        num_frames = sorted_matrix.shape[2]
        n = self.num_regions

        if n <= 1:
            raise ValueError("num_regions must be greater than 1")

        if n > num_frames:
            raise ValueError(
                f"num_regions={n} cannot exceed num_frames={num_frames} after trimming"
            )

        region_width = num_frames // n
        regions = []
        start = 0

        for r in range(n):
            if r == n - 1:
                end = num_frames  # last region absorbs any remainder
            else:
                end = start + region_width

            regions.append((start, end))
            start = end

        return regions

    def compute_region_means(self, sorted_matrix: np.ndarray, regions: list) -> tuple:
        """
        Compute global and per-row region means.

        Args:
            sorted_matrix: np.ndarray of shape (H, W, num_frames)
            regions: List of (start, end) index tuples, length = num_regions

        Returns:
            T: np.ndarray of shape (num_regions,)
               Global mean of each region across all rows and pixels.
               Corresponds to Tn in the paper.

            Q: np.ndarray of shape (H, num_regions)
               Per-detector-row mean of each region.
               Q[r, n] = mean of all pixels in row r within region n.
               Corresponds to Qn(c) in the paper.
        """
        if sorted_matrix.ndim != 3:
            raise ValueError(
                f"Expected 3D matrix (H, W, num_frames), got shape {sorted_matrix.shape}"
            )

        H, W, _ = sorted_matrix.shape
        num_regions = len(regions)

        T = np.zeros(num_regions, dtype=np.float32)
        Q = np.zeros((H, num_regions), dtype=np.float32)

        for n, (start, end) in enumerate(regions):
            region_slice = sorted_matrix[:, :, start:end]  # (H, W, region_width)

            # Global mean — scalar for this region
            T[n] = np.mean(region_slice)

            # Per-row mean — average over W pixels and region_width frames
            Q[:, n] = np.mean(region_slice, axis=(1, 2))  # (H,)

        return T, Q

    def identify_linear_segments(self, T: np.ndarray) -> list:
        """
        Identify linear segments of the S-curve by analyzing consecutive
        differences between region means (D_V values).

        Args:
            T: np.ndarray of shape (num_regions,)
               Global mean per region (Tn from paper)

        Returns:
            segments: List of lists, each inner list contains region indices
                      belonging to the same linear segment.
                      e.g. [[0,1,2], [3,4,5,6], [7,8,...,19]]
        """
        if len(T) < 2:
            raise ValueError("Need at least 2 regions to identify segments")

        # Auto-compute tolerance if not provided
        # Scaled as 0.25% of dynamic range — equivalent to paper's 10/4095 for 12-bit
        D_V = np.diff(T)  # shape (num_regions - 1,)

        if self.dv_tolerance is None:
            median_dv = float(np.median(np.abs(D_V)))

            if median_dv == 0:
                print("[-] Data is perfectly linear — using single segment")
                return [list(range(len(T)))]

            # Coefficient of variation of D_V — measures how nonlinear the data is
            # Low CV means D_V is roughly constant → data is essentially linear
            cv = float(np.std(D_V) / median_dv)
            print(f"[-] D_V coefficient of variation = {cv:.4f}")

            if cv < 0.5:
                print("[-] Data appears linear (CV < 0.5) — using single segment")
                return [list(range(len(T)))]

            # Genuine nonlinearity detected — segment with 50% of median D_V
            dv_tolerance = 0.5 * median_dv
            print(
                f"[-] Nonlinear data detected (CV={cv:.4f}), "
                f"dv_tolerance = {dv_tolerance:.4f}"
            )
        else:
            dv_tolerance = self.dv_tolerance

        # Compute D_V[n] = T[n] - T[n-1], indices 1..num_regions-1
        D_V = np.diff(T)  # shape (num_regions - 1,)

        # Group regions into segments
        # A new segment starts when |D_V[n] - D_V[n-1]| > tolerance
        segments = []
        current_segment = [0]  # always start with region 0

        for n in range(1, len(T)):
            if n == 1:
                # No D_V[n-1] to compare against yet — extend current segment
                current_segment.append(n)
            else:
                dv_diff = abs(D_V[n - 1] - D_V[n - 2])
                if dv_diff <= dv_tolerance:
                    current_segment.append(n)
                else:
                    segments.append(current_segment)
                    current_segment = [n]

        # Don't forget the last segment
        segments.append(current_segment)

        # Sanity check — every region must appear exactly once
        all_indices = [idx for seg in segments for idx in seg]
        assert len(all_indices) == len(T), "Region indices lost during segmentation"

        return segments

    def select_region_pair_per_segment(self, segments: list) -> list:
        """
        For each linear segment, select dark and bright region indices
        using Equation 6 from the paper.

        Args:
            segments: List of lists of region indices per segment
                      e.g. [[0,1,2], [3,4,5,6,7], ...]

        Returns:
            pairs: List of (idx_low, idx_high) tuples — one per segment.
                   These are actual region indices (into the regions list),
                   not positions within the segment.
        """
        pairs = []

        for seg in segments:
            m = len(seg)

            if m == 1:
                # Degenerate segment — use same region for both
                pairs.append((seg[0], seg[0]))

            elif 2 <= m <= 3:
                # Equation 6, first case: pick 1st and last
                pairs.append((seg[0], seg[-1]))

            else:
                # Equation 6, second case: find valid a where m = 2a + b, b >= a, a > 1
                selected = None
                for a in range(2, m):
                    b = m - 2 * a
                    if b >= a:
                        # Valid — convert to 0-indexed region indices
                        idx_low = seg[a - 1]  # a-th region (1-indexed → 0-indexed)
                        idx_high = seg[
                            m - a
                        ]  # (m-a+1)-th region (1-indexed → 0-indexed)
                        selected = (idx_low, idx_high)
                        break

                if selected is None:
                    # No valid a found — fallback to first and last
                    selected = (seg[0], seg[-1])

                pairs.append(selected)

        return pairs

    def estimate_gain_offset_per_segment(
        self,
        sorted_matrix: np.ndarray,
        regions: list,
        segments: list,
        pairs: list,
        T: np.ndarray,
        Q: np.ndarray,
        eps: float = 1e-6,
    ) -> list:
        """
        Estimate gain and offset per row for each linear segment
        using Equation 7 from the paper.

        Args:
            sorted_matrix: np.ndarray of shape (H, W, num_frames)
            regions:       List of (start, end) index tuples
            segments:      List of lists of region indices per segment
            pairs:         List of (idx_low, idx_high) per segment
            T:             np.ndarray of shape (num_regions,) — global region means
            Q:             np.ndarray of shape (H, num_regions) — per-row region means
            eps:           Small value to avoid division by zero

        Returns:
            corrections: List of dicts, one per segment, each containing:
                         {
                           'gain':   np.ndarray of shape (H, W)
                           'offset': np.ndarray of shape (H, W)
                           'T_low':  float — global mean of dark reference region
                           'T_high': float — global mean of bright reference region
                         }
        """
        H, W, _ = sorted_matrix.shape

        corrections = []

        for seg, pair in zip(segments, pairs):
            idx_low, idx_high = pair
            correction = {}

            if idx_low == idx_high:
                # Degenerate segment — mark for borrowing
                correction["gain"] = None
                correction["offset"] = None
                correction["T_low"] = float(T[idx_low])
                correction["T_high"] = float(T[idx_high])
                correction["degenerate"] = True

            else:
                # T1, T2 — global means of dark and bright reference regions
                T1 = float(T[idx_low])  # scalar
                T2 = float(T[idx_high])  # scalar

                # Q1, Q2 — per-row means of dark and bright reference regions
                Q1 = Q[:, idx_low]  # shape (H,)
                Q2 = Q[:, idx_high]  # shape (H,)

                # Solve Equation 7 per row:
                # k * Q1[r] + b = T1
                # k * Q2[r] + b = T2
                # Subtracting: k = (T2 - T1) / (Q2[r] - Q1[r])
                # Then:        b = T1 - k * Q1[r]
                denominator = Q2 - Q1  # shape (H,)

                gain_row = np.ones(H, dtype=np.float32)
                offset_row = np.zeros(H, dtype=np.float32)
                row_dynamic_range = (
                    Q[:, -1] - Q[:, 0]
                )  # max region mean - min region mean per row
                print(
                    f"[-] Row dynamic range: min={row_dynamic_range.min():.2f}, "
                    f"max={row_dynamic_range.max():.2f}, "
                    f"median={np.median(row_dynamic_range):.2f}"
                )
                # Rows with dynamic range below 1% of global dynamic range are considered bad
                global_dynamic_range = float(T[-1] - T[0])
                bad_row_threshold = 0.01 * global_dynamic_range
                bad_rows = row_dynamic_range < bad_row_threshold
                valid_rows_mask = ~bad_rows

                print(f"[-] Bad rows detected: {np.where(bad_rows)[0].tolist()}")

                valid_mask = (np.abs(denominator) >= eps) & valid_rows_mask
                gain_row[valid_mask] = (T2 - T1) / denominator[valid_mask]
                offset_row[valid_mask] = T1 - gain_row[valid_mask] * Q1[valid_mask]

                invalid_rows = np.where(~valid_mask)[0]
                valid_rows = np.where(valid_mask)[0]

                if len(valid_rows) > 0 and len(invalid_rows) > 0:
                    for r in invalid_rows:
                        nearest_valid = valid_rows[np.argmin(np.abs(valid_rows - r))]
                        gain_row[r] = gain_row[nearest_valid]
                        offset_row[r] = offset_row[nearest_valid]

                # Broadcast to (H, W) — same gain/offset for all pixels in a row
                gain_2d = np.tile(gain_row[:, np.newaxis], (1, W))  # (H, W)
                offset_2d = np.tile(offset_row[:, np.newaxis], (1, W))  # (H, W)

                correction["gain"] = gain_2d
                correction["offset"] = offset_2d
                correction["T_low"] = T1
                correction["T_high"] = T2
                correction["degenerate"] = False

            corrections.append(correction)

        # --- Borrow nearest valid segment coefficients for degenerate segments ---
        # Left to right: borrow from nearest valid left neighbour
        last_valid = None
        for s_idx in range(len(corrections)):
            if not corrections[s_idx]["degenerate"]:
                last_valid = s_idx
            elif last_valid is not None:
                corrections[s_idx]["gain"] = corrections[last_valid]["gain"]
                corrections[s_idx]["offset"] = corrections[last_valid]["offset"]
                corrections[s_idx]["degenerate"] = False

        # Right to left: catches degenerate segments at the start
        last_valid = None
        for s_idx in range(len(corrections) - 1, -1, -1):
            if not corrections[s_idx]["degenerate"]:
                last_valid = s_idx
            elif last_valid is not None:
                corrections[s_idx]["gain"] = corrections[last_valid]["gain"]
                corrections[s_idx]["offset"] = corrections[last_valid]["offset"]
                corrections[s_idx]["degenerate"] = False

        # Final fallback — if all segments were degenerate
        for s_idx in range(len(corrections)):
            if corrections[s_idx]["degenerate"]:
                corrections[s_idx]["gain"] = np.ones((H, W), dtype=np.float32)
                corrections[s_idx]["offset"] = np.zeros((H, W), dtype=np.float32)

        return corrections

    def apply_correction(self, frames: list, corrections: list, h: int, w: int) -> list:
        """
        Apply per-segment gain/offset correction to each frame.
        Interpolates between the two nearest segments for each pixel.

        Args:
            frames:      List of raw 2D float32 arrays of shape (H, W)
            corrections: List of dicts from estimate_gain_offset_per_segment
            h:           Frame height
            w:           Frame width

        Returns:
            corrected_frames: List of corrected 2D float32 arrays of shape (H, W)
        """
        # Build segment anchor arrays
        T_lows = np.array([c["T_low"] for c in corrections], dtype=np.float32)
        T_highs = np.array([c["T_high"] for c in corrections], dtype=np.float32)
        T_mids = (T_lows + T_highs) / 2.0  # (S,)

        # Stack gain and offset maps: (S, H, W)
        gains = np.stack([c["gain"] for c in corrections], axis=0)
        offsets = np.stack([c["offset"] for c in corrections], axis=0)

        # Precompute meshgrid for advanced indexing
        h_idx, w_idx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

        corrected_frames = []

        for frame in frames:
            # Distance from every pixel to every segment midpoint
            pixel_exp = frame[:, :, np.newaxis]  # (H, W, 1)
            T_mids_exp = T_mids[np.newaxis, np.newaxis, :]  # (1, 1, S)
            distances = np.abs(pixel_exp - T_mids_exp)  # (H, W, S)

            # Nearest and second nearest segment indices
            nearest_idx = np.argmin(distances, axis=2)  # (H, W)

            distances_copy = distances.copy()
            distances_copy[h_idx, w_idx, nearest_idx] = np.inf
            second_idx = np.argmin(distances_copy, axis=2)  # (H, W)

            # Inverse distance weights
            d1 = distances[h_idx, w_idx, nearest_idx]  # (H, W)
            d2 = distances[h_idx, w_idx, second_idx]  # (H, W)

            exact_mask = d1 < 1e-6
            total_dist = d1 + d2
            w1 = np.where(exact_mask, 1.0, d2 / total_dist)  # (H, W)
            w2 = np.where(exact_mask, 0.0, d1 / total_dist)  # (H, W)

            # Gather and interpolate gain and offset
            gain_map = (
                w1 * gains[nearest_idx, h_idx, w_idx]
                + w2 * gains[second_idx, h_idx, w_idx]
            )
            offset_map = (
                w1 * offsets[nearest_idx, h_idx, w_idx]
                + w2 * offsets[second_idx, h_idx, w_idx]
            )

            corrected = gain_map * frame + offset_map
            corrected_frames.append(corrected.astype(np.float32))

        return corrected_frames

    def run(self, frames: list) -> list:
        """
        Orchestrate the full multi-segment NUC pipeline.

        Args:
            frames: List of 2D float32 arrays of shape (H, W)

        Returns:
            corrected_frames: List of corrected 2D float32 arrays of shape (H, W)
        """
        if not frames or len(frames) == 0:
            raise RuntimeError("Please provide valid frames")

        h, w = frames[0].shape

        # Step 1 — Build temporal matrix
        print("[*] Building temporal matrix...")
        matrix = self.build_temporal_matrix(frames)
        print(f"[-] Matrix shape: {matrix.shape}")

        # Step 2 — Sort matrix
        print("[*] Sorting temporal matrix...")
        sorted_matrix = self.sort_matrix(matrix)
        print(f"[-] Sorted matrix shape: {sorted_matrix.shape}")

        # Step 3 — Partition regions
        print("[*] Partitioning regions...")
        regions = self.partition_regions(sorted_matrix)
        print(f"[-] Created {len(regions)} regions")

        # Step 4 — Compute region means
        print("[*] Computing region means...")
        T, Q = self.compute_region_means(sorted_matrix, regions)
        print(f"[-] T shape: {T.shape}, Q shape: {Q.shape}")
        print(f"[-] T range: [{T.min():.2f}, {T.max():.2f}]")

        # Step 5 — Identify linear segments
        print("[*] Identifying linear segments...")
        segments = self.identify_linear_segments(T)
        print(f"[-] Found {len(segments)} linear segments")
        for i, seg in enumerate(segments):
            print(f"    Segment {i}: regions {seg[0]}..{seg[-1]} ({len(seg)} regions)")

        # Step 6 — Select region pairs per segment
        print("[*] Selecting region pairs per segment...")
        pairs = self.select_region_pair_per_segment(segments)
        for i, (lo, hi) in enumerate(pairs):
            print(
                f"    Segment {i}: region pair ({lo}, {hi}), "
                f"T_low={T[lo]:.2f}, T_high={T[hi]:.2f}"
            )

        # Step 7 — Estimate gain and offset per segment
        print("[*] Estimating gain and offset per segment...")
        corrections = self.estimate_gain_offset_per_segment(
            sorted_matrix=sorted_matrix,
            regions=regions,
            segments=segments,
            pairs=pairs,
            T=T,
            Q=Q,
        )
        for i, c in enumerate(corrections):
            print(
                f"    Segment {i}: "
                f"gain [{c['gain'].min():.4f}, {c['gain'].max():.4f}], "
                f"offset [{c['offset'].min():.4f}, {c['offset'].max():.4f}]"
            )

        # Step 8 — Apply correction
        print("[*] Applying corrections...")
        corrected_frames = self.apply_correction(
            frames=frames, corrections=corrections, h=h, w=w
        )
        print(f"[-] Corrected {len(corrected_frames)} frames")

        self.corrected_frames = corrected_frames
        return corrected_frames
