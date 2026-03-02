import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from utils import col_mad


class _BaseLMSSBNUC:
    """Shared helpers for LMS-based Scene-Based NUC algorithms."""

    def __init__(self, sigma: float, kernel_size: int):
        self.sigma = sigma
        self.kernel_size = kernel_size

    def _create_desired_image(self, Y: np.ndarray) -> np.ndarray:
        """Gaussian-blur the frame to get the desired (reference) image."""
        return gaussian_filter(Y, sigma=self.sigma)

    def _estimate_local_variance(
        self, Y: np.ndarray, window_size: int
    ) -> np.ndarray:
        """Per-pixel local variance via uniform filter."""
        local_mean = uniform_filter(Y, size=window_size, mode="reflect")
        local_mean_sq = uniform_filter(Y**2, size=window_size, mode="reflect")
        local_var = local_mean_sq - local_mean**2
        return np.maximum(local_var, 0)


class StandardLMSSBNUC(_BaseLMSSBNUC):
    """
    Standard (fixed step-size) LMS Scene-Based NUC.

    Parameters
    ----------
    epsilon : float
        Fixed LMS learning rate applied to both gain and offset.
    sigma : float
        Standard deviation for the Gaussian blur reference image.
    kernel_size : int
        (Informational) kernel extent matching the original script.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        sigma: float = 5.0,
        kernel_size: int = 21,
    ):
        super().__init__(sigma=sigma, kernel_size=kernel_size)
        self.epsilon = epsilon

    def run(self, frames: list) -> list:
        if not frames:
            raise RuntimeError("No frames provided")

        g = np.ones_like(frames[0], dtype=np.float32)
        o = np.zeros_like(frames[0], dtype=np.float32)

        corrected_frames = []

        for n, Y in enumerate(frames):
            B = self._create_desired_image(Y)

            X_hat = g * Y + o
            E = X_hat - B

            g -= self.epsilon * E * Y
            o -= self.epsilon * E

            corrected_frames.append(X_hat.astype(np.float32))
            print(
                f"Frame {n + 1:4d} | "
                f"g={np.mean(g):.4f} | "
                f"o={np.mean(o):.4f} | "
                f"ColMAD: {col_mad(Y):.4f} → {col_mad(X_hat):.4f}"
            )

        return corrected_frames


class AdaptiveLMSSBNUC(_BaseLMSSBNUC):
    """
    Adaptive (spatially-varying step-size) LMS Scene-Based NUC.

    The learning rate is modulated per-pixel by local intensity variance:
        epsilon(x,y) = K / (1 + M_scale^2 * local_var(x,y))

    Parameters
    ----------
    K : float
        Base learning rate numerator.
    M_scale : float
        Scale factor for local variance in the denominator.
    sigma : float
        Standard deviation for the Gaussian blur reference image.
    kernel_size : int
        (Informational) kernel extent matching the original script.
    local_var_window : int
        Window size for local variance estimation.
    """

    def __init__(
        self,
        K: float = 0.05,
        M_scale: float = 0.5,
        sigma: float = 5.0,
        kernel_size: int = 21,
        local_var_window: int = 5,
    ):
        super().__init__(sigma=sigma, kernel_size=kernel_size)
        self.K = K
        self.M_scale = M_scale
        self.local_var_window = local_var_window

    def run(self, frames: list) -> list:
        if not frames:
            raise RuntimeError("No frames provided")

        g = np.ones_like(frames[0], dtype=np.float32)
        o = np.zeros_like(frames[0], dtype=np.float32)

        corrected_frames = []

        for n, Y in enumerate(frames):
            B = self._create_desired_image(Y)
            local_var = self._estimate_local_variance(Y, self.local_var_window)

            epsilon_adaptive = self.K / (1 + self.M_scale**2 * local_var)

            X_hat = g * Y + o
            E = X_hat - B

            g -= epsilon_adaptive * E * Y
            o -= epsilon_adaptive * E

            corrected_frames.append(X_hat.astype(np.float32))
            print(
                f"Frame {n + 1:4d} | "
                f"e={np.mean(epsilon_adaptive):.4f} | "
                f"g={np.mean(g):.4f} | "
                f"ColMAD: {col_mad(Y):.4f} → {col_mad(X_hat):.4f}"
            )

        return corrected_frames


class GatedAdaptiveLMSSBNUC(_BaseLMSSBNUC):
    """
    Gated Adaptive LMS Scene-Based NUC.

    Pixels only update their gain/offset when the reference image changes by
    more than a threshold T relative to the previous frame, preventing drift
    on static scenes.

    Parameters
    ----------
    K : float
        Base learning rate numerator.
    M_scale : float
        Scale factor for local variance in the denominator.
    sigma : float
        Standard deviation for the Gaussian blur reference image.
    kernel_size : int
        (Informational) kernel extent matching the original script.
    local_var_window : int
        Window size for local variance estimation.
    T : float
        Change-detection threshold for the gating mask.
    """

    def __init__(
        self,
        K: float = 0.05,
        M_scale: float = 0.5,
        sigma: float = 5.0,
        kernel_size: int = 21,
        local_var_window: int = 5,
        T: float = 0.002,
    ):
        super().__init__(sigma=sigma, kernel_size=kernel_size)
        self.K = K
        self.M_scale = M_scale
        self.local_var_window = local_var_window
        self.T = T

    def run(self, frames: list) -> list:
        if not frames:
            raise RuntimeError("No frames provided")

        g = np.ones_like(frames[0], dtype=np.float32)
        o = np.zeros_like(frames[0], dtype=np.float32)
        Z = np.full_like(frames[0], np.inf, dtype=np.float32)

        corrected_frames = []

        for n, Y in enumerate(frames):
            B = self._create_desired_image(Y)
            local_var = self._estimate_local_variance(Y, self.local_var_window)

            change_mask = np.abs(B - Z) > self.T
            update_fraction = np.mean(change_mask)

            epsilon_adaptive = np.zeros_like(Y, dtype=np.float32)
            epsilon_adaptive[change_mask] = self.K / (
                1 + self.M_scale**2 * local_var[change_mask]
            )

            X_hat = g * Y + o
            E = X_hat - B

            g -= epsilon_adaptive * E * Y
            o -= epsilon_adaptive * E

            Z[change_mask] = B[change_mask]

            corrected_frames.append(X_hat.astype(np.float32))
            print(
                f"Frame {n + 1:4d} | "
                f"Update={update_fraction:.4f} | "
                f"g={np.mean(g):.4f} | "
                f"ColMAD: {col_mad(Y):.4f} → {col_mad(X_hat):.4f}"
            )

        return corrected_frames
