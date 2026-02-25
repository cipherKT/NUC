import numpy as np


class ConstantStatsNUC:
    def __init__(self, alpha: float = 0.97, T: float = 0.005, eps: float = 1e-6):
        self.alpha = alpha
        self.T = T
        self.eps = eps

    def init_stats(self, first_frame: np.ndarray):
        global_mean = np.mean(first_frame)
        global_mad = np.mean(np.abs(first_frame - global_mean))

        M = np.full_like(first_frame, global_mean, dtype=np.float32)
        S = np.full_like(first_frame, global_mad, dtype=np.float32)
        Y_prev = np.full_like(first_frame, np.inf, dtype=np.float32)

        return M, S, Y_prev

    def update(self, Y: np.ndarray, Y_prev: np.ndarray, M: np.ndarray, S: np.ndarray):
        change_mask = np.abs(Y - Y_prev) > self.T
        update_fraction = np.mean(change_mask)

        M_new = M.copy()
        S_new = S.copy()

        M_new[change_mask] = (1.0 - self.alpha) * Y[change_mask] + self.alpha * M[
            change_mask
        ]
        S_new[change_mask] = (1.0 - self.alpha) * np.abs(
            Y[change_mask] - M_new[change_mask]
        ) + self.alpha * S[change_mask]
        return M_new, S_new, update_fraction

    def compute_gain_offset(self, M: np.ndarray, S: np.ndarray):
        g = 1.0 / (S + self.eps)
        o = -M / (S + self.eps)
        return g, o

    def apply(self, Y: np.ndarray, g: np.ndarray, o: np.ndarray):
        return g * Y + o

    def run(self, frames: list, normalize: bool = False) -> list:
        if not frames:
            raise RuntimeError("No frames provided")
        if normalize:
            frames = [f / 65535.0 for f in frames]
        M, S, Y_prev = self.init_stats(frames[0])
        corrected_frames = []

        for n, Y in enumerate(frames):
            M, S, update_fraction = self.update(Y, Y_prev, M, S)
            g, o = self.compute_gain_offset(M, S)
            X_hat = self.apply(Y, g, o)

            corrected_frames.append(X_hat.astype(np.float32))
            print(
                f"Frame {n + 1:4d} | "
                f"Update fraction: {update_fraction:.4f} | "
                f"mean(M): {np.mean(M):.4f} | "
                f"mean(S): {np.mean(S):.4f}"
            )
            Y_prev = Y
        return corrected_frames
