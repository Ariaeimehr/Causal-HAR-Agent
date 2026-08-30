"""
Causal-HAR-Agent: Academic framework for combining Human Activity Recognition (HAR)
time-series with LLM-powered Causal Discovery and Causal-Guided Transformer Architectures.
"""

from causal_har_agent.causal.gemini_discovery import GeminiCausalDiscoverer
from causal_har_agent.causal.graph_structure import CausalGraph
from causal_har_agent.causal.prompt_builder import BiomechanicalPromptBuilder
from causal_har_agent.data.dataset import HARDataset, SyntheticSensorSimulator
from causal_har_agent.data.feature_extractor import SensorStatisticalFeatureExtractor
from causal_har_agent.evaluation.robustness_metrics import RobustnessEvaluator
from causal_har_agent.models.baseline_transformer import BaselineHARTransformer
from causal_har_agent.models.causal_transformer import CausalGuidedHARTransformer

__version__ = "0.1.0"
__all__ = [
    "HARDataset",
    "SyntheticSensorSimulator",
    "SensorStatisticalFeatureExtractor",
    "BiomechanicalPromptBuilder",
    "GeminiCausalDiscoverer",
    "CausalGraph",
    "BaselineHARTransformer",
    "CausalGuidedHARTransformer",
    "RobustnessEvaluator",
]
