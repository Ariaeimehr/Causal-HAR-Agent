"""
Academic Robustness Evaluation Metrics and Out-of-Distribution (OOD) Perturbation Engine.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tabulate import tabulate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class RobustnessEvaluator:
    """
    Evaluates in-distribution clean accuracy and out-of-distribution robustness
    under controlled physical perturbations (sensor noise, channel dropout, subject drift).
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        perturbation_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Evaluates a trained PyTorch model on a given dataloader.

        Args:
            model: PyTorch HAR neural network.
            dataloader: DataLoader with test/validation batches.
            perturbation_fn: Optional transform applying noise or channel dropout.

        Returns:
            Dict[str, float]: accuracy, macro_f1, weighted_f1
        """
        model.eval()
        model.to(self.device)

        all_preds: List[int] = []
        all_targets: List[int] = []

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                if perturbation_fn is not None:
                    x = perturbation_fn(x)

                logits, _ = model(x)
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(y.cpu().numpy().tolist())

        acc = float(accuracy_score(all_targets, all_preds))
        macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))

        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
        }

    @staticmethod
    def apply_gaussian_noise(x: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
        """Injects additive white Gaussian sensor noise N(0, sigma^2)."""
        noise = torch.randn_like(x) * sigma
        return x + noise

    @staticmethod
    def apply_channel_dropout(x: torch.Tensor, drop_rate: float = 0.3) -> torch.Tensor:
        """Simulates sensor hardware blackout by zeroing entire sensor channels."""
        B, T, C = x.shape
        mask = (torch.rand(B, 1, C, device=x.device) > drop_rate).float()
        return x * mask

    @staticmethod
    def apply_sensor_misalignment(x: torch.Tensor, max_angle_deg: float = 25.0) -> torch.Tensor:
        """Simulates wearable sensor rotation drift on body."""
        B, T, C = x.shape
        # Apply 2D rotational perturbation between x and y axes
        rad = np.radians(max_angle_deg)
        theta = (torch.rand(B, 1, 1, device=x.device) * 2 - 1) * rad
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        x_perturbed = x.clone()
        if C >= 3:
            # Perturb ankle acc
            acc_x, acc_y = x[:, :, 0:1], x[:, :, 1:2]
            x_perturbed[:, :, 0:1] = acc_x * cos_t - acc_y * sin_t
            x_perturbed[:, :, 1:2] = acc_x * sin_t + acc_y * cos_t

        return x_perturbed

    def run_comparative_robustness_sweep(
        self,
        baseline_model: nn.Module,
        causal_model: nn.Module,
        test_loader: DataLoader,
        noise_sigmas: Optional[List[float]] = None,
        dropout_rates: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Executes an end-to-end comparative robustness benchmark across multiple
        corruption regimes and prints formatted academic comparison tables.
        """
        sigmas = noise_sigmas or [0.1, 0.3, 0.5, 0.8, 1.0]
        drops = dropout_rates or [0.1, 0.2, 0.3, 0.4]

        # 1. Clean in-distribution performance
        clean_base = self.evaluate_model(baseline_model, test_loader)
        clean_causal = self.evaluate_model(causal_model, test_loader)

        results: Dict[str, Any] = {
            "clean": {
                "baseline": clean_base,
                "causal": clean_causal,
            },
            "gaussian_noise_sweep": [],
            "channel_dropout_sweep": [],
            "sensor_misalignment": {},
        }

        # 2. Gaussian Noise Sweep
        for s in sigmas:
            fn = lambda x, s_val=s: self.apply_gaussian_noise(x, sigma=s_val)
            b_eval = self.evaluate_model(baseline_model, test_loader, perturbation_fn=fn)
            c_eval = self.evaluate_model(causal_model, test_loader, perturbation_fn=fn)
            results["gaussian_noise_sweep"].append({
                "sigma": s,
                "baseline_accuracy": b_eval["accuracy"],
                "causal_accuracy": c_eval["accuracy"],
                "delta_accuracy": round(c_eval["accuracy"] - b_eval["accuracy"], 4),
                "baseline_f1": b_eval["macro_f1"],
                "causal_f1": c_eval["macro_f1"],
            })

        # 3. Channel Dropout Sweep
        for d in drops:
            fn = lambda x, d_val=d: self.apply_channel_dropout(x, drop_rate=d_val)
            b_eval = self.evaluate_model(baseline_model, test_loader, perturbation_fn=fn)
            c_eval = self.evaluate_model(causal_model, test_loader, perturbation_fn=fn)
            results["channel_dropout_sweep"].append({
                "dropout_rate": d,
                "baseline_accuracy": b_eval["accuracy"],
                "causal_accuracy": c_eval["accuracy"],
                "delta_accuracy": round(c_eval["accuracy"] - b_eval["accuracy"], 4),
                "baseline_f1": b_eval["macro_f1"],
                "causal_f1": c_eval["macro_f1"],
            })

        # 4. Sensor Misalignment
        fn_mis = lambda x: self.apply_sensor_misalignment(x, max_angle_deg=30.0)
        b_mis = self.evaluate_model(baseline_model, test_loader, perturbation_fn=fn_mis)
        c_mis = self.evaluate_model(causal_model, test_loader, perturbation_fn=fn_mis)
        results["sensor_misalignment"] = {
            "baseline_accuracy": b_mis["accuracy"],
            "causal_accuracy": c_mis["accuracy"],
            "delta_accuracy": round(c_mis["accuracy"] - b_mis["accuracy"], 4),
        }

        return results

    def print_academic_summary_table(self, benchmark_results: Dict[str, Any]) -> str:
        """Formats evaluation results as a scientific Markdown table."""
        headers = [
            "Test Condition",
            "Baseline Acc (%)",
            "Causal-HAR Acc (%)",
            "Δ Robustness Advantage",
            "Baseline Macro F1",
            "Causal-HAR Macro F1",
        ]
        rows = []

        clean_b = benchmark_results["clean"]["baseline"]
        clean_c = benchmark_results["clean"]["causal"]
        rows.append([
            "Clean (In-Distribution)",
            f"{clean_b['accuracy']*100:.1f}%",
            f"{clean_c['accuracy']*100:.1f}%",
            f"+{(clean_c['accuracy'] - clean_b['accuracy'])*100:+.1f}%",
            f"{clean_b['macro_f1']:.3f}",
            f"{clean_c['macro_f1']:.3f}",
        ])

        for noise_item in benchmark_results["gaussian_noise_sweep"]:
            s = noise_item["sigma"]
            b_acc = noise_item["baseline_accuracy"]
            c_acc = noise_item["causal_accuracy"]
            rows.append([
                f"Gaussian Noise (σ={s:.1f})",
                f"{b_acc*100:.1f}%",
                f"{c_acc*100:.1f}%",
                f"+{(c_acc - b_acc)*100:+.1f}%",
                f"{noise_item['baseline_f1']:.3f}",
                f"{noise_item['causal_f1']:.3f}",
            ])

        for drop_item in benchmark_results["channel_dropout_sweep"]:
            d = drop_item["dropout_rate"]
            b_acc = drop_item["baseline_accuracy"]
            c_acc = drop_item["causal_accuracy"]
            rows.append([
                f"Sensor Dropout (p={d:.1f})",
                f"{b_acc*100:.1f}%",
                f"{c_acc*100:.1f}%",
                f"+{(c_acc - b_acc)*100:+.1f}%",
                f"{drop_item['baseline_f1']:.3f}",
                f"{drop_item['causal_f1']:.3f}",
            ])

        mis = benchmark_results["sensor_misalignment"]
        b_acc = mis["baseline_accuracy"]
        c_acc = mis["causal_accuracy"]
        rows.append([
            "Sensor Angle Misalign (30°)",
            f"{b_acc*100:.1f}%",
            f"{c_acc*100:.1f}%",
            f"+{(c_acc - b_acc)*100:+.1f}%",
            "-",
            "-",
        ])

        table_str = tabulate(rows, headers=headers, tablefmt="github")
        return table_str
