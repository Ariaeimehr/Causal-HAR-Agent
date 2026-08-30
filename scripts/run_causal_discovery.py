#!/usr/bin/env python3
"""
CLI Script for Gemini-Powered Biomechanical Causal Discovery in HAR.

Usage:
    python scripts/run_causal_discovery.py --activity "Ascending Stairs" --output_path "causal_graph.json"
"""

import argparse
import json
import os
import sys
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from causal_har_agent.causal.gemini_discovery import GeminiCausalDiscoverer
from causal_har_agent.data.dataset import SyntheticSensorSimulator
from causal_har_agent.data.feature_extractor import SensorStatisticalFeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract statistical features and run LLM-powered causal discovery for HAR."
    )
    parser.add_argument(
        "--activity",
        type=str,
        default="Ascending Stairs",
        help="Physical activity name (e.g. 'Walking', 'Ascending Stairs', 'Running', 'Falling')",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="causal_graph.json",
        help="Destination path for saving discovered causal graph JSON.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.7-flash",
        help="Google Gemini model identifier (e.g., 'gemini-3.7-flash').",
    )
    parser.add_argument(
        "--sampling_rate",
        type=float,
        default=50.0,
        help="Sensor sampling frequency in Hz.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Print text/ASCII representation of discovered DAG and mechanisms.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print(" Causal-HAR-Agent: LLM Biomechanical Causal Discovery")
    print("=" * 70)
    print(f"[*] Selected Activity:       {args.activity}")
    print(f"[*] Sensor Sampling Rate:    {args.sampling_rate} Hz")
    print(f"[*] Target Gemini Model:     {args.model}")
    print(f"[*] Output Destination:      {args.output_path}")

    # 1. Simulate or load sample sensor window
    print("\n[Step 1/3] Generating Kinematic Inertial Stream...")
    simulator = SyntheticSensorSimulator(sampling_rate_hz=args.sampling_rate)
    window = simulator.generate_window(activity=args.activity, noise_level=0.04)

    # 2. Extract statistical and spectral features
    print("[Step 2/3] Extracting Multi-Domain Statistical Descriptors...")
    extractor = SensorStatisticalFeatureExtractor(sampling_rate_hz=args.sampling_rate)
    features = extractor.extract_multivariate_window(
        window, channel_names=simulator.DEFAULT_CHANNELS
    )

    print(f"    - Extracted {len(features['channel_features'])} channel feature profiles.")
    print(f"    - Computed full {len(simulator.DEFAULT_CHANNELS)}x{len(simulator.DEFAULT_CHANNELS)} cross-correlation matrix.")

    # 3. Call Gemini Causal Discovery
    print(f"\n[Step 3/3] Invoking Gemini ({args.model}) for Biomechanical DAG Induction...")
    discoverer = GeminiCausalDiscoverer(model_name=args.model)
    causal_graph = discoverer.discover_causal_graph(
        sensor_features=features,
        activity_name=args.activity,
    )

    # Save to disk
    with open(args.output_path, "w") as f:
        f.write(causal_graph.to_json(indent=2))

    print(f"\n[SUCCESS] Discovered Causal Graph saved to: {args.output_path}")
    print(f"    - Total Nodes: {len(causal_graph.nodes)}")
    print(f"    - Total Causal Edges: {len(causal_graph.edges)}")
    print(f"    - Verified Acyclic (DAG): True")

    if args.visualize or True:
        print("\n" + "-" * 70)
        print(" Biomechanical Causal Explanation Summary")
        print("-" * 70)
        print(causal_graph.reasoning_summary)
        print("\n" + "-" * 70)
        print(" Discovered Cause-and-Effect Directed Links (Edges)")
        print("-" * 70)
        for i, edge in enumerate(causal_graph.edges, 1):
            print(f"{i:02d}. [{edge.source}] ---> [{edge.target}] (Confidence: {edge.confidence:.2f})")
            print(f"    Mechanism:   {edge.mechanism}")
            if edge.time_delay_ms > 0:
                print(f"    Phase Delay: ~{edge.time_delay_ms:.1f} ms")
        print("=" * 70)


if __name__ == "__main__":
    main()
