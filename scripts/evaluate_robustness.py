#!/usr/bin/env python3
"""
Academic Robustness Evaluation Script for Causal-HAR-Agent.
Compares Baseline Transformer vs. Causal-Guided Transformer under Out-Of-Distribution (OOD) corruptions.

Usage:
    python scripts/evaluate_robustness.py --epochs 15 --batch_size 32
"""

import argparse
import json
import os
import sys
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from causal_har_agent.causal.gemini_discovery import GeminiCausalDiscoverer
from causal_har_agent.causal.graph_structure import CausalGraph
from causal_har_agent.data.dataset import HARDataset, SyntheticSensorSimulator
from causal_har_agent.data.feature_extractor import SensorStatisticalFeatureExtractor
from causal_har_agent.evaluation.robustness_metrics import RobustnessEvaluator
from causal_har_agent.models.baseline_transformer import BaselineHARTransformer
from causal_har_agent.models.causal_transformer import CausalGuidedHARTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate robustness of Causal-HAR Transformer vs. Baseline Transformer."
    )
    parser.add_argument(
        "--causal_graph",
        type=str,
        default=None,
        help="Optional path to pre-generated causal_graph.json.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Training epochs for quick academic benchmarking.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for DataLoaders.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=80,
        help="Number of simulated time-series windows per class.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="benchmark_results.json",
        help="Output path to save evaluation results.",
    )
    return parser.parse_args()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    name: str = "Model",
) -> None:
    """Trains a model using AdamW and Cross-Entropy Loss."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(range(epochs), desc=f"Training {name}", leave=False)
    for _ in pbar:
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print(" Causal-HAR-Agent: Out-Of-Distribution Robustness Benchmark")
    print("=" * 75)
    print(f"[*] Hardware Device:         {device}")
    print(f"[*] Training Epochs:         {args.epochs}")
    print(f"[*] Batch Size:              {args.batch_size}")
    print(f"[*] Samples per Class:       {args.samples_per_class}")

    # 1. Prepare Dataset
    print("\n[Step 1/4] Generating Benchmark HAR Dataset (6 Activities)...")
    simulator = SyntheticSensorSimulator()
    train_dataset, test_dataset = simulator.generate_benchmark_dataset(
        samples_per_class=args.samples_per_class
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Obtain Causal Graph
    print("\n[Step 2/4] Obtaining Biomechanical Structural Causal Graph...")
    if args.causal_graph and os.path.exists(args.causal_graph):
        print(f"    - Loading pre-computed causal graph from: {args.causal_graph}")
        with open(args.causal_graph, "r") as f:
            causal_graph = CausalGraph.from_json(f.read())
    else:
        print("    - Extracting features and prompting Gemini Causal Discoverer...")
        extractor = SensorStatisticalFeatureExtractor()
        sample_window = simulator.generate_window("Ascending Stairs")
        feats = extractor.extract_multivariate_window(
            sample_window, simulator.DEFAULT_CHANNELS
        )
        discoverer = GeminiCausalDiscoverer()
        causal_graph = discoverer.discover_causal_graph(feats, "Locomotion")

    print(f"    - Causal Graph Active: {len(causal_graph.nodes)} Nodes, {len(causal_graph.edges)} Edges.")

    # 3. Instantiate Models
    print("\n[Step 3/4] Initializing & Training Models...")
    in_channels = len(simulator.DEFAULT_CHANNELS)
    num_classes = len(simulator.ACTIVITIES)

    baseline_model = BaselineHARTransformer(
        in_channels=in_channels,
        num_classes=num_classes,
        d_model=64,
        nhead=4,
        num_layers=2,
    )

    causal_model = CausalGuidedHARTransformer(
        in_channels=in_channels,
        num_classes=num_classes,
        d_model=64,
        nhead=4,
        num_layers=2,
        causal_graph=causal_graph,
        ordered_channels=simulator.DEFAULT_CHANNELS,
    )

    print("    - Training Standard Baseline Transformer...")
    train_model(baseline_model, train_loader, epochs=args.epochs, lr=args.lr, device=device, name="Baseline Transformer")

    print("    - Training Causal-Guided HAR Transformer...")
    train_model(causal_model, train_loader, epochs=args.epochs, lr=args.lr, device=device, name="Causal-HAR Transformer")

    # 4. Comprehensive Robustness Evaluation
    print("\n[Step 4/4] Executing Out-Of-Distribution Robustness Perturbation Sweeps...")
    evaluator = RobustnessEvaluator(device=device)
    results = evaluator.run_comparative_robustness_sweep(
        baseline_model=baseline_model,
        causal_model=causal_model,
        test_loader=test_loader,
        noise_sigmas=[0.1, 0.3, 0.5, 0.8, 1.0],
        dropout_rates=[0.1, 0.2, 0.3, 0.4],
    )

    # Print Table
    table_markdown = evaluator.print_academic_summary_table(results)
    print("\n" + "=" * 75)
    print(" EMPIRICAL ROBUSTNESS BENCHMARK RESULTS")
    print("=" * 75)
    print(table_markdown)
    print("=" * 75)

    # Save to disk
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] Full benchmark metrics successfully saved to: {args.output_json}")


if __name__ == "__main__":
    main()
