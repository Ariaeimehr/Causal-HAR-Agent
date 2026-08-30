"""
Statistical and Spectral Feature Extraction for Inertial Time-Series Sensors.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from scipy import signal, stats


class SensorStatisticalFeatureExtractor:
    """
    Extracts comprehensive time-domain, frequency-domain, and inter-axial
    statistical features from windowed multi-channel inertial sensor data.
    """

    def __init__(self, sampling_rate_hz: float = 50.0) -> None:
        """
        Initialize the feature extractor.

        Args:
            sampling_rate_hz: Sampling frequency of inertial measurements.
        """
        self.fs = sampling_rate_hz

    def extract_channel_features(self, x: np.ndarray) -> Dict[str, float]:
        """
        Extracts statistical descriptors from a single 1D sensor time-series signal.

        Args:
            x (np.ndarray): 1D array of temporal sensor readings.

        Returns:
            Dict[str, float]: Extracted statistical and spectral metrics.
        """
        mean_val = float(np.mean(x))
        std_val = float(np.std(x))
        var_val = float(np.var(x))
        min_val = float(np.min(x))
        max_val = float(np.max(x))
        ptp_val = float(max_val - min_val)
        iqr_val = float(stats.iqr(x))
        skew_val = float(stats.skew(x))
        kurt_val = float(stats.kurtosis(x))
        rms_val = float(np.sqrt(np.mean(x**2)))

        # Zero / Mean crossing rates
        zero_crossings = int(np.count_nonzero(np.diff(np.signbit(x))))
        centered = x - mean_val
        mean_crossings = int(np.count_nonzero(np.diff(np.signbit(centered))))

        # Fast Fourier Transform for spectral features
        n = len(x)
        fft_vals = np.fft.rfft(x * np.hanning(n))
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)
        power_spectrum = np.abs(fft_vals) ** 2
        total_energy = float(np.sum(power_spectrum)) / n

        # Dominant frequency
        if len(power_spectrum) > 1:
            dom_freq_idx = np.argmax(power_spectrum[1:]) + 1
            dom_freq = float(fft_freqs[dom_freq_idx])
        else:
            dom_freq = 0.0

        # Spectral Entropy
        ps_norm = power_spectrum / (np.sum(power_spectrum) + 1e-12)
        spectral_entropy = float(
            -np.sum(ps_norm * np.log2(ps_norm + 1e-12))
        )

        return {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "variance": round(var_val, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "peak_to_peak": round(ptp_val, 4),
            "iqr": round(iqr_val, 4),
            "skewness": round(skew_val, 4),
            "kurtosis": round(kurt_val, 4),
            "rms": round(rms_val, 4),
            "zero_crossing_rate": round(zero_crossings / n, 4),
            "mean_crossing_rate": round(mean_crossings / n, 4),
            "dominant_frequency_hz": round(dom_freq, 3),
            "spectral_energy": round(total_energy, 4),
            "spectral_entropy": round(spectral_entropy, 4),
        }

    def extract_multivariate_window(
        self,
        window: np.ndarray,
        channel_names: List[str],
    ) -> Dict[str, Any]:
        """
        Extracts univariate channel features plus inter-axial cross-correlations
        for a multi-dimensional window.

        Args:
            window: Array of shape (T, C) containing temporal sensor signals.
            channel_names: List of C strings with channel identifiers.

        Returns:
            Dict[str, Any]: Comprehensive dictionary containing channel features
                            and pairwise correlation matrices.
        """
        num_timesteps, num_channels = window.shape
        channel_features: Dict[str, Dict[str, float]] = {}

        for c_idx in range(num_channels):
            c_name = channel_names[c_idx] if c_idx < len(channel_names) else f"ch_{c_idx}"
            channel_features[c_name] = self.extract_channel_features(window[:, c_idx])

        # Cross-correlation matrix
        corr_matrix = np.corrcoef(window.T)
        corr_dict: Dict[str, Dict[str, float]] = {}
        for i in range(num_channels):
            name_i = channel_names[i]
            corr_dict[name_i] = {}
            for j in range(num_channels):
                name_j = channel_names[j]
                val = float(corr_matrix[i, j])
                corr_dict[name_i][name_j] = round(val if not np.isnan(val) else 0.0, 3)

        # Jerk (derivative of acceleration) metrics
        jerk_dict: Dict[str, float] = {}
        for c_idx, c_name in enumerate(channel_names):
            if "acc" in c_name:
                jerk_signal = np.diff(window[:, c_idx]) * self.fs
                jerk_dict[f"{c_name}_jerk_std"] = round(float(np.std(jerk_signal)), 4)
                jerk_dict[f"{c_name}_jerk_rms"] = round(
                    float(np.sqrt(np.mean(jerk_signal**2))), 4
                )

        return {
            "num_timesteps": num_timesteps,
            "sampling_rate_hz": self.fs,
            "channel_features": channel_features,
            "cross_correlation_matrix": corr_dict,
            "jerk_features": jerk_dict,
        }
