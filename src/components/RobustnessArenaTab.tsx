import React, { useState, useEffect } from "react";
import { RobustnessBenchmarkResult } from "../types";
import { ShieldCheck, Sliders, TrendingUp, AlertTriangle, Cpu, CheckCircle } from "lucide-react";

export const RobustnessArenaTab: React.FC = () => {
  const [noiseSigma, setNoiseSigma] = useState<number>(0.4);
  const [dropoutRate, setDropoutRate] = useState<number>(0.2);
  const [misalignmentDeg, setMisalignmentDeg] = useState<number>(15);
  const [benchmarkData, setBenchmarkData] = useState<RobustnessBenchmarkResult | null>(null);
  const [isEvaluating, setIsEvaluating] = useState<boolean>(false);

  const runEvaluation = async () => {
    setIsEvaluating(true);
    try {
      const res = await fetch("/api/evaluate/robustness", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          noiseSigma,
          dropoutRate,
          misalignmentDeg,
        }),
      });
      const data = await res.json();
      setBenchmarkData(data);
    } catch (e) {
      console.error("Error evaluating robustness:", e);
    } finally {
      setIsEvaluating(false);
    }
  };

  useEffect(() => {
    runEvaluation();
  }, [noiseSigma, dropoutRate, misalignmentDeg]);

  // Synthetic attention heatmaps for visual comparison (10x10 token matrix)
  const renderAttentionHeatmap = (isCausal: boolean) => {
    const size = 10;
    const grid = [];

    for (let i = 0; i < size; i++) {
      const row = [];
      for (let j = 0; j < size; j++) {
        let intensity = 0;
        if (isCausal) {
          // Structured causal attention: strong diagonal and forward kinematic causal dependencies
          if (i === j) intensity = 0.85 + (i % 2) * 0.1;
          else if (j < i && i - j <= 3) intensity = 0.6 - (i - j) * 0.15;
          else if (j === 0 || j === 2) intensity = 0.45; // Ankle driver token
          else intensity = 0.05 + (Math.sin(i * j) * 0.05); // Suppressed non-causal noise
        } else {
          // Unconstrained baseline attention: scattered spurious noisy correlations
          const base = 0.35 + 0.3 * Math.sin(i * 1.5 + j * 0.8);
          const noiseComponent = (noiseSigma * 0.4) * ((i + j) % 3 === 0 ? 0.4 : -0.2);
          intensity = Math.max(0.05, Math.min(0.95, base + noiseComponent));
        }
        row.push(intensity);
      }
      grid.push(row);
    }

    return (
      <div className="grid grid-cols-10 gap-0.5 bg-slate-950 p-2 rounded-lg border border-slate-800">
        {grid.map((row, rIdx) =>
          row.map((val, cIdx) => {
            const hue = isCausal ? "6, 182, 212" : "239, 68, 68"; // Cyan vs Red
            const alpha = Math.max(0.15, val);
            return (
              <div
                key={`${rIdx}-${cIdx}`}
                className="w-full aspect-square rounded-[2px] transition-colors"
                style={{ backgroundColor: `rgba(${hue}, ${alpha})` }}
                title={`Token (${rIdx}, ${cIdx}): Attention = ${val.toFixed(2)}`}
              />
            );
          })
        )}
      </div>
    );
  };

  const curr = benchmarkData?.currentPerturbation;

  return (
    <div className="space-y-6">
      {/* Top Banner: Perturbation Controller */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl">
        <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-800">
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center">
              <Sliders className="w-4 h-4 mr-2 text-cyan-400" />
              Out-of-Distribution (OOD) Stress Test Bench
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Simulate sensor hardware degradation, Gaussian noise, and cross-subject placement shift
            </p>
          </div>
          <span className="text-xs font-semibold px-2.5 py-1 rounded bg-slate-800 text-cyan-300 border border-slate-700">
            Real-time Evaluation Engine
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Slider 1: Gaussian Noise */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="font-semibold text-slate-300">Additive Gaussian Noise ($\sigma$)</span>
              <span className="font-mono font-bold text-cyan-400">{noiseSigma.toFixed(2)} g</span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.5"
              step="0.05"
              value={noiseSigma}
              onChange={(e) => setNoiseSigma(parseFloat(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>0.0 (Clean)</span>
              <span>0.75 (Moderate)</span>
              <span>1.50 (Extreme)</span>
            </div>
          </div>

          {/* Slider 2: Channel Dropout */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="font-semibold text-slate-300">Channel Dropout Rate ($p$)</span>
              <span className="font-mono font-bold text-indigo-400">{(dropoutRate * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.0"
              max="0.6"
              step="0.05"
              value={dropoutRate}
              onChange={(e) => setDropoutRate(parseFloat(e.target.value))}
              className="w-full accent-indigo-400 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>0% (All Channels)</span>
              <span>30% (Drop 3-4 Ch)</span>
              <span>60% (Severe Blackout)</span>
            </div>
          </div>

          {/* Slider 3: Sensor Misalignment Angle */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="font-semibold text-slate-300">Sensor Rotation Angle Drift ($\theta$)</span>
              <span className="font-mono font-bold text-purple-400">{misalignmentDeg}°</span>
            </div>
            <input
              type="range"
              min="0"
              max="45"
              step="1"
              value={misalignmentDeg}
              onChange={(e) => setMisalignmentDeg(parseInt(e.target.value))}
              className="w-full accent-purple-400 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>0° (Aligned)</span>
              <span>20° (Moderate Shift)</span>
              <span>45° (Severe Drift)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Metric Cards Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Baseline Accuracy */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-md">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-rose-400 flex items-center justify-between">
            <span>Standard Transformer</span>
            <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
          </div>
          <div className="text-2xl font-black text-white mt-1 font-mono">
            {curr ? `${curr.baselineAcc}%` : "--"}
          </div>
          <div className="text-xs text-slate-400 mt-1 flex items-center justify-between">
            <span>Clean: 94.4%</span>
            <span className="text-rose-400 font-mono font-semibold">
              {curr ? `-${curr.relativeDropBaseline}% drop` : ""}
            </span>
          </div>
        </div>

        {/* Causal-HAR Accuracy */}
        <div className="bg-slate-900 border border-cyan-900/60 rounded-xl p-4 shadow-md bg-gradient-to-b from-cyan-950/20 to-slate-900">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-cyan-400 flex items-center justify-between">
            <span>Causal-HAR Transformer</span>
            <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
          </div>
          <div className="text-2xl font-black text-cyan-300 mt-1 font-mono">
            {curr ? `${curr.causalAcc}%` : "--"}
          </div>
          <div className="text-xs text-slate-400 mt-1 flex items-center justify-between">
            <span>Clean: 95.2%</span>
            <span className="text-emerald-400 font-mono font-semibold">
              {curr ? `-${curr.relativeDropCausal}% drop` : ""}
            </span>
          </div>
        </div>

        {/* Robustness Advantage Delta */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-md">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-emerald-400 flex items-center justify-between">
            <span>Causal Advantage (Δ)</span>
            <TrendingUp className="w-3.5 h-3.5 text-emerald-400" />
          </div>
          <div className="text-2xl font-black text-emerald-400 mt-1 font-mono">
            {curr ? `+${curr.deltaAdvantage}%` : "--"}
          </div>
          <div className="text-xs text-slate-400 mt-1">
            Absolute accuracy gain under OOD corruption
          </div>
        </div>

        {/* Macro F1 Comparison */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-md">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-300">
            Macro F1 Score (Corrupted)
          </div>
          <div className="flex items-baseline space-x-2 mt-1 font-mono">
            <span className="text-2xl font-black text-cyan-300">{curr ? curr.causalF1 : "--"}</span>
            <span className="text-sm text-slate-500 font-normal">vs {curr ? curr.baselineF1 : "--"} (Base)</span>
          </div>
          <div className="text-xs text-slate-400 mt-1">
            Multiclass activity balance preservation
          </div>
        </div>
      </div>

      {/* Attention Map & Mechanism Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Baseline Attention */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h4 className="text-xs font-bold text-rose-400 uppercase tracking-wider">
                Baseline Self-Attention Heatmap
              </h4>
              <p className="text-xs text-slate-400">
                Unconstrained: Softmax(QKᵀ / √d) conflates sensor noise with spurious links
              </p>
            </div>
            <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-rose-950 text-rose-400 border border-rose-800">
              Noise Vulnerable
            </span>
          </div>

          {renderAttentionHeatmap(false)}

          <div className="mt-3 p-3 bg-slate-950 rounded-lg text-xs text-slate-300 border border-slate-800">
            <p className="leading-relaxed">
              <strong>Spurious Correlation Confounding:</strong> Without a causal prior, attention weights over-fit to transient co-occurrences. Under sensor channel dropout or Gaussian noise, the attention distribution disperses randomly across corrupted tokens.
            </p>
          </div>
        </div>

        {/* Causal Guided Attention */}
        <div className="bg-slate-900 border border-cyan-900/60 rounded-xl p-5 shadow-xl">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h4 className="text-xs font-bold text-cyan-400 uppercase tracking-wider">
                Causal-Guided Prior Attention Heatmap
              </h4>
              <p className="text-xs text-slate-400">
                Regularized: Softmax(QKᵀ / √d + λ·M_causal) restricts flow to physical DAG
              </p>
            </div>
            <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-cyan-950 text-cyan-300 border border-cyan-800">
              Invariant Pathways
            </span>
          </div>

          {renderAttentionHeatmap(true)}

          <div className="mt-3 p-3 bg-slate-950 rounded-lg text-xs text-slate-300 border border-slate-800">
            <p className="leading-relaxed">
              <strong>Kinematic Invariance:</strong> Attention is anchored to true biomechanical drivers (ground reaction thrust and joint torque). Even if extraneous channels are zeroed or corrupted, the core causal path remains intact.
            </p>
          </div>
        </div>
      </div>

      {/* Empirical Noise Sweep Degradation Curves */}
      {benchmarkData && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl">
          <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-3">
            Simulated Accuracy vs. Gaussian Sensor Noise Level ($\sigma$)
          </h4>
          <div className="h-44 bg-slate-950 rounded-lg p-3 border border-slate-800 relative flex items-end">
            <svg className="w-full h-full" viewBox="0 0 100 50" preserveAspectRatio="none">
              {/* Gridlines */}
              <line x1="0" y1="10" x2="100" y2="10" stroke="#334155" strokeDasharray="1 1" strokeWidth="0.3" />
              <line x1="0" y1="25" x2="100" y2="25" stroke="#334155" strokeDasharray="1 1" strokeWidth="0.3" />
              <line x1="0" y1="40" x2="100" y2="40" stroke="#334155" strokeDasharray="1 1" strokeWidth="0.3" />

              {/* Baseline Curve */}
              <polyline
                fill="none"
                stroke="#f43f5e"
                strokeWidth="1.8"
                points={benchmarkData.sweeps.noise
                  .map((pt, idx) => {
                    const x = (idx / (benchmarkData.sweeps.noise.length - 1)) * 100;
                    const y = 50 - (pt.baselineAcc / 100) * 50;
                    return `${x},${y}`;
                  })
                  .join(" ")}
              />

              {/* Causal Curve */}
              <polyline
                fill="none"
                stroke="#06b6d4"
                strokeWidth="2.2"
                points={benchmarkData.sweeps.noise
                  .map((pt, idx) => {
                    const x = (idx / (benchmarkData.sweeps.noise.length - 1)) * 100;
                    const y = 50 - (pt.causalAcc / 100) * 50;
                    return `${x},${y}`;
                  })
                  .join(" ")}
              />
            </svg>
          </div>
          <div className="flex items-center justify-between text-xs text-slate-400 mt-2 font-mono">
            <div className="flex items-center space-x-4">
              <span className="flex items-center text-cyan-400 font-semibold">
                <span className="w-3 h-0.5 bg-cyan-400 mr-1.5 inline-block"></span> Causal-HAR Transformer (High Robustness)
              </span>
              <span className="flex items-center text-rose-400 font-semibold">
                <span className="w-3 h-0.5 bg-rose-500 mr-1.5 inline-block"></span> Standard Baseline Transformer (Rapid Drop)
              </span>
            </div>
            <span>$\sigma = 0.0 \to 1.5$</span>
          </div>
        </div>
      )}
    </div>
  );
};
