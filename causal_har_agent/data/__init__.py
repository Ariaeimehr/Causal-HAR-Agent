"""
Data loading, synthesis, and feature extraction modules for Human Activity Recognition (HAR).
"""

from causal_har_agent.data.dataset import HARDataset, SyntheticSensorSimulator
from causal_har_agent.data.feature_extractor import SensorStatisticalFeatureExtractor

__all__ = [
    "HARDataset",
    "SyntheticSensorSimulator",
    "SensorStatisticalFeatureExtractor",
]
