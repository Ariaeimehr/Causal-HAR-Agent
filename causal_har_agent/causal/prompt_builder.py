"""
Biomechanical Prompt Engineering and Statistical Feature Formatting for Gemini Causal Discovery.
"""

from typing import Any, Dict, List, Optional
import json


class BiomechanicalPromptBuilder:
    """
    Constructs rigorous academic prompts that synthesize windowed inertial sensor
    statistical features, anatomical placement semantics, and biomechanical kinematics
    to guide LLM-based causal discovery of Directed Acyclic Graphs (DAGs).
    """

    SYSTEM_INSTRUCTION: str = (
        "You are an expert computational biomechanist and causal inference theorist "
        "specializing in Human Activity Recognition (HAR) and Structural Causal Models (SCMs). "
        "Your task is to analyze extracted time-series statistical features and inter-axial "
        "cross-correlations across multi-channel wearable inertial sensors (accelerometers, "
        "gyroscopes) to identify genuine physical cause-and-effect relationships (Directed Acyclic Graph). "
        "Distinguish true causal kinematic mechanisms (e.g., foot strike impact propagating to pelvis, "
        "joint angular torque driving linear segment acceleration) from spurious statistical correlations."
    )

    def __init__(self, activity_context: str = "General Locomotion") -> None:
        """
        Initialize the prompt builder.

        Args:
            activity_context: Name or category of physical activity being analyzed.
        """
        self.activity_context = activity_context

    def format_feature_summary_table(
        self,
        features_dict: Dict[str, Any],
    ) -> str:
        """
        Formats channel statistics into a markdown table suitable for LLM ingestion.

        Args:
            features_dict: Output dictionary from SensorStatisticalFeatureExtractor.

        Returns:
            str: Markdown formatted table of statistical descriptors.
        """
        channel_feats = features_dict.get("channel_features", {})
        lines = [
            "| Channel Name | Mean (g or rad/s) | Std Dev | Peak-to-Peak | Dominant Freq (Hz) | Spectral Energy | Entropy |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for ch_name, stats in channel_feats.items():
            lines.append(
                f"| `{ch_name}` | {stats.get('mean', 0.0):.3f} | {stats.get('std', 0.0):.3f} | "
                f"{stats.get('peak_to_peak', 0.0):.3f} | {stats.get('dominant_frequency_hz', 0.0):.2f} | "
                f"{stats.get('spectral_energy', 0.0):.3f} | {stats.get('spectral_entropy', 0.0):.3f} |"
            )

        return "\n".join(lines)

    def format_correlation_matrix_summary(
        self,
        features_dict: Dict[str, Any],
        threshold: float = 0.4,
    ) -> str:
        """
        Filters and highlights significant pairwise cross-correlations.

        Args:
            features_dict: Output dictionary from SensorStatisticalFeatureExtractor.
            threshold: Minimum absolute correlation coefficient to include.

        Returns:
            str: Text summary of salient pairwise correlations.
        """
        corr_matrix = features_dict.get("cross_correlation_matrix", {})
        pairs = []

        channel_keys = list(corr_matrix.keys())
        for i, ch_a in enumerate(channel_keys):
            for ch_b in channel_keys[i + 1 :]:
                val = corr_matrix.get(ch_a, {}).get(ch_b, 0.0)
                if abs(val) >= threshold:
                    pairs.append(f"- Corr(`{ch_a}`, `{ch_b}`) = {val:+.3f}")

        if not pairs:
            return "No cross-channel correlations exceeded the threshold."

        return "\n".join(pairs)

    def build_causal_discovery_prompt(
        self,
        features_dict: Dict[str, Any],
        activity_name: Optional[str] = None,
        candidate_channels: Optional[List[str]] = None,
    ) -> str:
        """
        Synthesizes a complete academic prompt for Gemini Causal Graph Discovery.

        Args:
            features_dict: Output from feature extractor containing statistics.
            activity_name: Specific activity label (e.g. 'Ascending Stairs', 'Running').
            candidate_channels: List of sensor channel identifiers.

        Returns:
            str: The complete structured prompt.
        """
        activity = activity_name or self.activity_context
        table_str = self.format_feature_summary_table(features_dict)
        corr_str = self.format_correlation_matrix_summary(features_dict)

        jerk_feats = features_dict.get("jerk_features", {})
        jerk_str = json.dumps(jerk_feats, indent=2)

        prompt = f"""# Scientific Objective: Causal Kinematic Discovery in Wearable HAR

## Context & Physical Scenario
- **Observed Physical Activity:** {activity}
- **Sensor Modalities:** 3-Axis Accelerometers (Linear acceleration in g) and 3-Axis Gyroscopes (Angular velocity in rad/s) placed on lower extremities (Ankle) and core body (Waist/Trunk).
- **Sampling Frequency:** {features_dict.get('sampling_rate_hz', 50.0)} Hz

## Extracted Sensor Statistical Descriptors
{table_str}

## High Salience Inter-Axial Correlations
{corr_str}

## Jerk & Impact Dynamics (Derivative of Acceleration)
```json
{jerk_str}
```

## Biomechanical Causal Principles
When deducing the Directed Acyclic Graph (DAG) $\mathcal{{G}} = (\mathcal{{V}}, \mathcal{{E}})$, strictly adhere to:
1. **Kinetic Chain Hierarchy:** Ground contact forces and distal segment rotations (ankle pitch/yaw) cause inertial reactions at proximal segments (waist/trunk), typically with a forward propagation delay.
2. **Rotational-to-Linear Coupling:** Centripetal and angular accelerations ($\omega \times (\omega \times r)$ and $\dot{{\omega}} \times r$) cause linear acceleration components on the same rigid body segment.
3. **Acyclicity:** The resulting causal structure must be a strict DAG without directed cycles.
4. **Distinction from Correlation:** High correlation does not imply bidirectional causation. Identify which variable acts as the primary mechanical driver (parent node) and which is the dependent consequence (child node).

## Required Output Specification
Output a valid JSON object matching the following structure:
```json
{{
  "activity": "{activity}",
  "reasoning_summary": "High-level biomechanical explanation of motion kinematics",
  "nodes": [
    {{
      "id": "channel_name",
      "anatomical_location": "ankle | waist | wrist",
      "sensor_type": "accelerometer | gyroscope",
      "axis": "x | y | z",
      "description": "Functional mechanical role"
    }}
  ],
  "edges": [
    {{
      "source": "parent_channel_id",
      "target": "child_channel_id",
      "confidence": 0.85,
      "mechanism": "Biomechanical justification (e.g. joint torque creates tangential acceleration)",
      "time_delay_ms": 25.0
    }}
  ],
  "confounders": [
    {{
      "name": "e.g. Gravity / Cadence Pace",
      "affected_nodes": ["ankle_acc_z", "waist_acc_z"]
    }}
  ]
}}
```
"""
        return prompt.strip()
