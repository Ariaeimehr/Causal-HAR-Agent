"""
Dataset and time-series simulation utilities for Human Activity Recognition (HAR).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset


class HARDataset(Dataset):
    """
    PyTorch Dataset for multi-modal wearable inertial sensor streams.

    Attributes:
        data (torch.Tensor): Shape (N, T, C) where N is the number of samples,
                             T is the temporal window length, and C is sensor channels.
        labels (torch.Tensor): Shape (N,) activity category ground truth.
        channel_names (List[str]): Semantic descriptors of each channel.
        activity_names (List[str]): Mapping from class ID to activity name.
    """

    def __init__(
        self,
        data: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        channel_names: Optional[List[str]] = None,
        activity_names: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the HAR Dataset.

        Args:
            data: Input sensor tensor of shape (N, T, C) or (N, C, T).
            labels: Integer label vector of shape (N,).
            channel_names: List of strings naming sensor channels.
            activity_names: Names corresponding to class integer labels.
        """
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).float()
        else:
            self.data = data.float()

        # Ensure shape is (N, T, C)
        if self.data.ndim == 3 and self.data.shape[1] < self.data.shape[2]:
            # Permute if (N, C, T) to (N, T, C)
            if self.data.shape[1] <= 18:
                self.data = self.data.permute(0, 2, 1)

        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = labels.long()

        self.num_samples, self.seq_len, self.num_channels = self.data.shape

        self.channel_names = channel_names or [
            f"sensor_ch_{i}" for i in range(self.num_channels)
        ]
        self.activity_names = activity_names or [
            f"activity_{i}" for i in range(int(self.labels.max().item()) + 1)
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


class SyntheticSensorSimulator:
    """
    Biomechanical kinematics simulator for generating realistic multi-sensor HAR data.
    Simulates tri-axial accelerometer and gyroscope streams across ankle, waist, and wrist
    locations based on forward kinematic coupled oscillatory equations.
    """

    DEFAULT_CHANNELS: List[str] = [
        "ankle_acc_x",
        "ankle_acc_y",
        "ankle_acc_z",
        "ankle_gyro_x",
        "ankle_gyro_y",
        "ankle_gyro_z",
        "waist_acc_x",
        "waist_acc_y",
        "waist_acc_z",
        "waist_gyro_x",
        "waist_gyro_y",
        "waist_gyro_z",
    ]

    ACTIVITIES: List[str] = [
        "Walking",
        "Ascending Stairs",
        "Descending Stairs",
        "Running",
        "Falling",
        "Cycling",
    ]

    def __init__(
        self,
        sampling_rate_hz: float = 50.0,
        window_duration_sec: float = 2.56,
        seed: int = 42,
    ) -> None:
        """
        Initialize the kinematic simulator.

        Args:
            sampling_rate_hz: Frequency of wearable sensor sampling.
            window_duration_sec: Window length in seconds (standard 128 samples at 50Hz).
            seed: Random seed for deterministic reproducibility.
        """
        self.fs = sampling_rate_hz
        self.window_len = int(sampling_rate_hz * window_duration_sec)
        self.rng = np.random.default_rng(seed)

    def generate_window(
        self,
        activity: str,
        subject_id: int = 1,
        noise_level: float = 0.05,
    ) -> np.ndarray:
        """
        Generates a single kinematic time-series window (T, C) obeying physical causal chains.

        Args:
            activity: Name of the activity from ACTIVITIES.
            subject_id: Identifier influencing subject-specific biomechanical parameters.
            noise_level: Standard deviation of additive Gaussian sensor noise.

        Returns:
            np.ndarray: Array of shape (T, 12) with simulated sensor channels.
        """
        t = np.linspace(0, self.window_len / self.fs, self.window_len)
        T = self.window_len

        # Subject cadence variance
        stride_freq = 1.0 + 0.1 * (subject_id % 5)
        phase_offset = (subject_id * 0.3) % (2 * np.pi)

        signals = np.zeros((T, len(self.DEFAULT_CHANNELS)))

        if activity == "Walking":
            # 1. Ankle pitch and thrust creates primary kinetic cycle
            f = stride_freq * 1.8
            ankle_acc_z = 1.2 * np.sin(2 * np.pi * f * t + phase_offset) + 9.81
            ankle_acc_x = 0.8 * np.cos(2 * np.pi * f * t + phase_offset)
            ankle_acc_y = 0.3 * np.sin(4 * np.pi * f * t + phase_offset)
            ankle_gyro_y = 2.5 * np.cos(2 * np.pi * f * t + phase_offset)
            ankle_gyro_x = 0.5 * np.sin(2 * np.pi * f * t)
            ankle_gyro_z = 0.4 * np.cos(4 * np.pi * f * t)

            # 2. Waist follows ankle ground reaction force with a phase delay and dampening
            delay = int(0.04 * self.fs)
            waist_acc_z = np.roll(0.6 * ankle_acc_z + 4.0, delay)
            waist_acc_x = np.roll(0.5 * ankle_acc_x, delay)
            waist_acc_y = 0.2 * np.sin(2 * np.pi * f * t + np.pi / 4)
            waist_gyro_x = 0.4 * np.sin(2 * np.pi * f * t + phase_offset)
            waist_gyro_y = 0.7 * np.cos(2 * np.pi * f * t + phase_offset)
            waist_gyro_z = 0.3 * np.sin(4 * np.pi * f * t)

        elif activity == "Ascending Stairs":
            # Steeper vertical impulse and pitch on ankle
            f = stride_freq * 1.2
            ankle_acc_z = 2.4 * np.sin(2 * np.pi * f * t + phase_offset) ** 2 + 9.81
            ankle_acc_x = 1.1 * np.cos(2 * np.pi * f * t + phase_offset)
            ankle_acc_y = 0.4 * np.sin(2 * np.pi * f * t)
            ankle_gyro_y = 3.8 * np.sin(2 * np.pi * f * t + phase_offset)
            ankle_gyro_x = 0.7 * np.cos(2 * np.pi * f * t)
            ankle_gyro_z = 0.5 * np.sin(4 * np.pi * f * t)

            waist_acc_z = 1.3 * np.roll(ankle_acc_z - 9.81, 4) + 9.81
            waist_acc_x = 0.7 * np.roll(ankle_acc_x, 4)
            waist_acc_y = 0.3 * np.sin(2 * np.pi * f * t)
            waist_gyro_x = 0.6 * np.sin(2 * np.pi * f * t)
            waist_gyro_y = 1.1 * np.cos(2 * np.pi * f * t)
            waist_gyro_z = 0.4 * np.sin(2 * np.pi * f * t)

        elif activity == "Descending Stairs":
            # Sharp impact deceleration shocks transmitted from ankle to waist
            f = stride_freq * 1.4
            impact = np.zeros(T)
            impact_indices = np.arange(0, T, int(self.fs / f))
            for idx in impact_indices:
                if idx < T - 5:
                    impact[idx : idx + 5] = [3.5, 2.1, -1.0, -0.4, 0.0]

            ankle_acc_z = impact + 9.81 + 0.5 * np.sin(2 * np.pi * f * t)
            ankle_acc_x = 1.0 * np.cos(2 * np.pi * f * t)
            ankle_acc_y = 0.5 * np.sin(2 * np.pi * f * t)
            ankle_gyro_y = -3.2 * np.sin(2 * np.pi * f * t)
            ankle_gyro_x = 0.8 * np.cos(2 * np.pi * f * t)
            ankle_gyro_z = 0.6 * np.sin(2 * np.pi * f * t)

            waist_acc_z = np.roll(0.7 * impact, 3) + 9.81
            waist_acc_x = 0.6 * np.cos(2 * np.pi * f * t)
            waist_acc_y = 0.4 * np.sin(2 * np.pi * f * t)
            waist_gyro_x = 0.7 * np.sin(2 * np.pi * f * t)
            waist_gyro_y = -1.2 * np.sin(2 * np.pi * f * t)
            waist_gyro_z = 0.5 * np.cos(2 * np.pi * f * t)

        elif activity == "Running":
            f = stride_freq * 2.8
            ankle_acc_z = 4.5 * np.sin(2 * np.pi * f * t) + 9.81
            ankle_acc_x = 2.8 * np.cos(2 * np.pi * f * t)
            ankle_acc_y = 1.2 * np.sin(4 * np.pi * f * t)
            ankle_gyro_y = 6.2 * np.cos(2 * np.pi * f * t)
            ankle_gyro_x = 1.5 * np.sin(2 * np.pi * f * t)
            ankle_gyro_z = 1.1 * np.cos(4 * np.pi * f * t)

            waist_acc_z = np.roll(2.2 * np.sin(2 * np.pi * f * t), 2) + 9.81
            waist_acc_x = 1.6 * np.cos(2 * np.pi * f * t)
            waist_acc_y = 0.8 * np.sin(2 * np.pi * f * t)
            waist_gyro_x = 1.2 * np.sin(2 * np.pi * f * t)
            waist_gyro_y = 2.1 * np.cos(2 * np.pi * f * t)
            waist_gyro_z = 0.9 * np.sin(4 * np.pi * f * t)

        elif activity == "Falling":
            # Sudden loss of balance: rapid gyro rotation followed by zero-g freefall and impact
            fall_point = int(T * 0.4)
            ankle_acc_z = np.full(T, 9.81)
            ankle_acc_x = np.zeros(T)
            ankle_acc_y = np.zeros(T)
            ankle_gyro_y = np.zeros(T)
            ankle_gyro_x = np.zeros(T)
            ankle_gyro_z = np.zeros(T)

            # High angular velocity before collapse
            ankle_gyro_y[fall_point - 10 : fall_point + 10] = np.linspace(0, 8.5, 20)
            # Freefall near 0g
            ankle_acc_z[fall_point : fall_point + 15] = 1.2
            # Impact spike
            ankle_acc_z[fall_point + 15 : fall_point + 20] = [18.5, 14.2, 5.0, 1.0, 0.0]

            waist_acc_z = np.roll(ankle_acc_z, 2)
            waist_acc_x = np.roll(ankle_acc_x, 2)
            waist_acc_y = np.zeros(T)
            waist_gyro_x = np.roll(ankle_gyro_x, 2)
            waist_gyro_y = np.roll(ankle_gyro_y, 2)
            waist_gyro_z = np.zeros(T)

        elif activity == "Cycling":
            f = stride_freq * 1.5
            # Smooth rotary patterns with high angular velocity and low linear impact
            ankle_acc_z = 1.5 * np.sin(2 * np.pi * f * t) + 9.81
            ankle_acc_x = 1.5 * np.cos(2 * np.pi * f * t)
            ankle_acc_y = 0.2 * np.sin(2 * np.pi * f * t)
            ankle_gyro_y = 4.5 * np.ones(T) + 0.8 * np.sin(2 * np.pi * f * t)
            ankle_gyro_x = 0.3 * np.cos(2 * np.pi * f * t)
            ankle_gyro_z = 0.2 * np.sin(2 * np.pi * f * t)

            waist_acc_z = 0.4 * np.sin(2 * np.pi * f * t) + 9.81
            waist_acc_x = 0.3 * np.cos(2 * np.pi * f * t)
            waist_acc_y = 0.1 * np.sin(2 * np.pi * f * t)
            waist_gyro_x = 0.2 * np.sin(2 * np.pi * f * t)
            waist_gyro_y = 0.5 * np.cos(2 * np.pi * f * t)
            waist_gyro_z = 0.1 * np.sin(2 * np.pi * f * t)

        else:
            raise ValueError(f"Unknown activity: {activity}")

        # Assemble channel array
        raw = np.column_stack([
            ankle_acc_x,
            ankle_acc_y,
            ankle_acc_z,
            ankle_gyro_x,
            ankle_gyro_y,
            ankle_gyro_z,
            waist_acc_x,
            waist_acc_y,
            waist_acc_z,
            waist_gyro_x,
            waist_gyro_y,
            waist_gyro_z,
        ])

        # Add Gaussian sensor noise
        noise = self.rng.normal(0, noise_level, size=raw.shape)
        return raw + noise

    def generate_benchmark_dataset(
        self,
        samples_per_class: int = 100,
        test_ratio: float = 0.2,
    ) -> Tuple[HARDataset, HARDataset]:
        """
        Generates standard train and test splits for all simulated HAR classes.

        Args:
            samples_per_class: Number of window samples per activity class.
            test_ratio: Proportion allocated to testing split.

        Returns:
            Tuple[HARDataset, HARDataset]: (train_dataset, test_dataset)
        """
        all_data: List[np.ndarray] = []
        all_labels: List[int] = []

        num_train = int(samples_per_class * (1.0 - test_ratio))

        train_x, train_y = [], []
        test_x, test_y = [], []

        for class_idx, activity in enumerate(self.ACTIVITIES):
            for sample_i in range(samples_per_class):
                window = self.generate_window(
                    activity=activity,
                    subject_id=sample_i % 10,
                    noise_level=0.08,
                )
                if sample_i < num_train:
                    train_x.append(window)
                    train_y.append(class_idx)
                else:
                    test_x.append(window)
                    test_y.append(class_idx)

        train_dataset = HARDataset(
            data=np.stack(train_x),
            labels=np.array(train_y),
            channel_names=self.DEFAULT_CHANNELS,
            activity_names=self.ACTIVITIES,
        )

        test_dataset = HARDataset(
            data=np.stack(test_x),
            labels=np.array(test_y),
            channel_names=self.DEFAULT_CHANNELS,
            activity_names=self.ACTIVITIES,
        )

        return train_dataset, test_dataset
