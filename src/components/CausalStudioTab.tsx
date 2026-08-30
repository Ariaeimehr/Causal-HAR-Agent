import React, { useState, useEffect } from "react";
import { RawSensorStream, FeaturePayload, CausalGraphData } from "../types";
import { DAGVisualizer } from "./DAGVisualizer";
import { Play, Sparkles, RefreshCw, Layers, Table, Code2, AlertCircle } from "lucide-react";

export const CausalStudioTab: React.FC = () => {
  const [selectedActivity, setSelectedActivity] = useState<string>("Ascending Stairs");
  const [sensorStream, setSensorStream] = useState<RawSensorStream | null>(null);
  const [features, setFeatures] = useState<FeaturePayload | null>(null);
  const [causalGraph, setCausalGraph] = useState<CausalGraphData | null>(null);
  const [activePrompt, setActivePrompt] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isDiscovering, setIsDiscovering] = useState<boolean>(false);
  const [selectedInterventionNode, setSelectedInterventionNode] = useState<string | null>(null);
  const [activeSubTab, setActiveSubTab] = useState<"waveform" | "features" | "prompt">("waveform");

  const activities = [
    "Walking",
    "Ascending Stairs",
    "Descending Stairs",
    "Running",
    "Falling",
    "Cycling",
  ];

  // Fetch or generate sensor data
  const loadSensorData = async (activity: string) => {
    setIsLoading(true);
    try {
      const res = await fetch("/api/sensor/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ activity, samplingRate: 50, durationSec: 2.56 }),
      });
      const json = await res.json();
      setSensorStream(json.raw);
      setFeatures(json.features);
    } catch (e) {
      console.error("Error loading sensor stream:", e);
    } finally {
      setIsLoading(false);
    }
  };

  // Run Gemini Causal Discovery
  const runCausalDiscovery = async () => {
    if (!features) return;
    setIsDiscovering(true);
    try {
      const res = await fetch("/api/causal/discover", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          activity: selectedActivity,
          features,
        }),
      });
      const json = await res.json();
      if (json.causalGraph) {
        setCausalGraph(json.causalGraph);
        setActivePrompt(json.prompt);
      }
    } catch (e) {
      console.error("Error discovering causal graph:", e);
    } finally {
      setIsDiscovering(false);
    }
  };

  useEffect(() => {
    loadSensorData(selectedActivity);
  }, [selectedActivity]);

  useEffect(() => {
    if (features && !causalGraph) {
      runCausalDiscovery();
    }
  }, [features]);

  return (
    <div className="space-y-6">
      {/* Control Toolbar */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center space-x-3">
          <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
            HAR Activity Context:
          </label>
          <div className="flex flex-wrap gap-1.5">
            {activities.map((act) => (
              <button
                key={act}
                onClick={() => {
                  setSelectedActivity(act);
                  setCausalGraph(null);
                }}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  selectedActivity === act
                    ? "bg-cyan-500 text-white shadow-md shadow-cyan-500/30"
                    : "bg-slate-800 text-slate-300 hover:bg-slate-700 hover:text-white"
                }`}
              >
                {act}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => loadSensorData(selectedActivity)}
            disabled={isLoading}
            className="px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center border border-slate-700 disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 mr-1.5 ${isLoading ? "animate-spin" : ""}`} />
            Regenerate Sample Window
          </button>
          <button
            onClick={runCausalDiscovery}
            disabled={isDiscovering || !features}
            className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white flex items-center shadow-lg shadow-cyan-500/25 disabled:opacity-50"
          >
            <Sparkles className={`w-3.5 h-3.5 mr-1.5 ${isDiscovering ? "animate-pulse" : ""}`} />
            {isDiscovering ? "Gemini Reasoning..." : "Discover DAG with Gemini"}
          </button>
        </div>
      </div>

      {/* Main Grid: Data & Features on Left, DAG on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Waveforms & Statistics & Prompt */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-xl">
            {/* Subtab navigation */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
              <span className="text-xs font-bold text-slate-200 uppercase tracking-wider flex items-center">
                <Layers className="w-4 h-4 mr-1.5 text-cyan-400" />
                Kinematic Sensor Stream
              </span>

              <div className="flex space-x-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
                <button
                  onClick={() => setActiveSubTab("waveform")}
                  className={`px-2.5 py-1 text-[11px] font-medium rounded ${
                    activeSubTab === "waveform" ? "bg-slate-800 text-cyan-300" : "text-slate-400 hover:text-white"
                  }`}
                >
                  Signals
                </button>
                <button
                  onClick={() => setActiveSubTab("features")}
                  className={`px-2.5 py-1 text-[11px] font-medium rounded ${
                    activeSubTab === "features" ? "bg-slate-800 text-cyan-300" : "text-slate-400 hover:text-white"
                  }`}
                >
                  Stats Table
                </button>
                <button
                  onClick={() => setActiveSubTab("prompt")}
                  className={`px-2.5 py-1 text-[11px] font-medium rounded ${
                    activeSubTab === "prompt" ? "bg-slate-800 text-cyan-300" : "text-slate-400 hover:text-white"
                  }`}
                >
                  LLM Prompt
                </button>
              </div>
            </div>

            {/* Subtab Content: Waveforms */}
            {activeSubTab === "waveform" && sensorStream && (
              <div className="space-y-4">
                {/* Ankle Acceleration Signal */}
                <div>
                  <div className="flex items-center justify-between text-[11px] text-slate-300 mb-1">
                    <span className="font-semibold text-cyan-400">Ankle 3-Axis Accel (g)</span>
                    <span className="text-slate-500 font-mono">50 Hz • 128 samples</span>
                  </div>
                  <div className="h-20 bg-slate-950 rounded-lg p-2 border border-slate-800/80 flex items-end relative overflow-hidden">
                    <svg className="w-full h-full" viewBox="0 0 128 60" preserveAspectRatio="none">
                      {/* Gridline */}
                      <line x1="0" y1="30" x2="128" y2="30" stroke="#334155" strokeDasharray="2 2" strokeWidth="0.5" />
                      {/* Acc Z */}
                      <polyline
                        fill="none"
                        stroke="#06b6d4"
                        strokeWidth="1.5"
                        points={sensorStream.data["ankle_acc_z"]
                          .map((val, i) => `${i},${30 - (val - 9.81) * 6}`)
                          .join(" ")}
                      />
                      {/* Acc X */}
                      <polyline
                        fill="none"
                        stroke="#38bdf8"
                        strokeWidth="1"
                        strokeDasharray="2 1"
                        points={sensorStream.data["ankle_acc_x"]
                          .map((val, i) => `${i},${30 - val * 8}`)
                          .join(" ")}
                      />
                    </svg>
                  </div>
                  <div className="flex items-center justify-between text-[10px] text-slate-500 mt-1 font-mono">
                    <span className="text-cyan-400">── Acc Z (Vertical Ground Thrust)</span>
                    <span className="text-sky-300">- - Acc X (Forward Thrust)</span>
                  </div>
                </div>

                {/* Waist Acceleration Signal */}
                <div>
                  <div className="flex items-center justify-between text-[11px] text-slate-300 mb-1">
                    <span className="font-semibold text-indigo-400">Waist 3-Axis Accel (Pelvic Center-of-Mass)</span>
                    <span className="text-slate-500 font-mono">Propagated Phase Delay</span>
                  </div>
                  <div className="h-20 bg-slate-950 rounded-lg p-2 border border-slate-800/80 flex items-end relative overflow-hidden">
                    <svg className="w-full h-full" viewBox="0 0 128 60" preserveAspectRatio="none">
                      <line x1="0" y1="30" x2="128" y2="30" stroke="#334155" strokeDasharray="2 2" strokeWidth="0.5" />
                      <polyline
                        fill="none"
                        stroke="#818cf8"
                        strokeWidth="1.5"
                        points={sensorStream.data["waist_acc_z"]
                          .map((val, i) => `${i},${30 - (val - 9.81) * 6}`)
                          .join(" ")}
                      />
                      <polyline
                        fill="none"
                        stroke="#c084fc"
                        strokeWidth="1"
                        points={sensorStream.data["waist_gyro_y"]
                          .map((val, i) => `${i},${30 - val * 8}`)
                          .join(" ")}
                      />
                    </svg>
                  </div>
                  <div className="flex items-center justify-between text-[10px] text-slate-500 mt-1 font-mono">
                    <span className="text-indigo-400">── Waist Acc Z (Axial Impact Load)</span>
                    <span className="text-purple-300">── Waist Gyro Y (Pelvic Pitch)</span>
                  </div>
                </div>
              </div>
            )}

            {/* Subtab Content: Statistical Features Table */}
            {activeSubTab === "features" && features && (
              <div className="max-h-[310px] overflow-y-auto pr-1">
                <table className="w-full text-left text-xs">
                  <thead className="text-[10px] text-slate-400 uppercase bg-slate-950/80 sticky top-0">
                    <tr>
                      <th className="py-2 px-2">Channel</th>
                      <th className="py-2 px-2">Mean</th>
                      <th className="py-2 px-2">Std Dev</th>
                      <th className="py-2 px-2">Dom Freq</th>
                      <th className="py-2 px-2">Energy</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800 text-[11px] font-mono">
                    {Object.entries<any>(features.channel_features).map(([ch, s]) => (
                      <tr key={ch} className="hover:bg-slate-800/40">
                        <td className="py-1.5 px-2 font-medium text-cyan-300">{ch}</td>
                        <td className="py-1.5 px-2 text-slate-300">{s.mean}</td>
                        <td className="py-1.5 px-2 text-slate-300">{s.std}</td>
                        <td className="py-1.5 px-2 text-slate-300">{s.dominant_frequency_hz} Hz</td>
                        <td className="py-1.5 px-2 text-slate-300">{s.spectral_energy}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Subtab Content: LLM Prompt Preview */}
            {activeSubTab === "prompt" && (
              <div className="max-h-[310px] overflow-y-auto bg-slate-950 p-3 rounded-lg border border-slate-800 text-[11px] font-mono text-slate-300 whitespace-pre-wrap leading-relaxed">
                {activePrompt || "Prompt will be constructed automatically when generating Causal Graph."}
              </div>
            )}
          </div>

          {/* Reasoning Summary Card */}
          {causalGraph && (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-xl">
              <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider mb-2 flex items-center">
                <Sparkles className="w-3.5 h-3.5 mr-1.5 text-cyan-400" />
                Gemini Biomechanical SCM Deduction
              </h4>
              <p className="text-xs text-slate-300 leading-relaxed bg-slate-950/80 p-3 rounded-lg border border-slate-800">
                {causalGraph.reasoning_summary}
              </p>
            </div>
          )}
        </div>

        {/* Right Column: Interactive DAG Visualizer */}
        <div className="lg:col-span-7">
          {causalGraph ? (
            <DAGVisualizer
              graph={causalGraph}
              selectedInterventionNode={selectedInterventionNode}
              onInterveneNode={(nodeId) => setSelectedInterventionNode(nodeId)}
            />
          ) : (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-12 text-center text-slate-400 flex flex-col items-center justify-center h-full min-h-[400px]">
              <Sparkles className="w-8 h-8 text-cyan-400 animate-pulse mb-3" />
              <p className="text-sm font-semibold text-slate-200">Generating Structural Causal Graph...</p>
              <p className="text-xs text-slate-400 mt-1">Invoking Gemini 3.7 Flash with extracted sensor descriptors</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
