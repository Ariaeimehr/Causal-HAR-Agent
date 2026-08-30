import React, { useState } from "react";
import { CausalGraphData, SensorNode, SensorEdge } from "../types";
import { Zap, Info, ArrowRight, ShieldAlert, CheckCircle2 } from "lucide-react";

interface DAGVisualizerProps {
  graph: CausalGraphData;
  onInterveneNode?: (nodeId: string | null) => void;
  selectedInterventionNode: string | null;
}

export const DAGVisualizer: React.FC<DAGVisualizerProps> = ({
  graph,
  onInterveneNode,
  selectedInterventionNode,
}) => {
  const [hoveredEdge, setHoveredEdge] = useState<SensorEdge | null>(null);
  const [hoveredNode, setHoveredNode] = useState<SensorNode | null>(null);

  // Compute downstream causal descendants when a node is intervened (do(X))
  const getDescendants = (startNodeId: string): Set<string> => {
    const visited = new Set<string>();
    const queue = [startNodeId];

    while (queue.length > 0) {
      const current = queue.shift()!;
      const children = graph.edges
        .filter((e) => e.source === current)
        .map((e) => e.target);

      for (const child of children) {
        if (!visited.has(child)) {
          visited.add(child);
          queue.push(child);
        }
      }
    }
    return visited;
  };

  const descendants = selectedInterventionNode
    ? getDescendants(selectedInterventionNode)
    : new Set<string>();

  // Layout node positions across 3 anatomical columns (Distal -> Intermediate -> Proximal)
  // Column 1: Ankle Sensors (Distal Ground Contact)
  // Column 2: Gyro Angular Drivers
  // Column 3: Waist / Pelvic Core (Proximal Response)
  const getNodeCoordinates = (node: SensorNode, index: number, total: number) => {
    const isAnkle = node.id.includes("ankle");
    const isGyro = node.id.includes("gyro");

    let col = 1; // Middle
    if (isAnkle && isGyro) col = 0; // Left column: Ankle Gyros (initiator)
    else if (isAnkle && !isGyro) col = 1; // Center-left: Ankle Accel
    else if (!isAnkle && isGyro) col = 2; // Center-right: Waist Gyro
    else col = 3; // Right column: Waist Accel (Core reaction)

    const x = 70 + col * 200;
    const rowIdx = index % 3;
    const y = 60 + rowIdx * 110 + (col % 2 === 1 ? 25 : 0);

    return { x, y };
  };

  const nodeMap = new Map<string, { node: SensorNode; x: number; y: number }>();
  graph.nodes.forEach((node, idx) => {
    const coords = getNodeCoordinates(node, idx, graph.nodes.length);
    nodeMap.set(node.id, { node, ...coords });
  });

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl relative overflow-hidden">
      {/* Visualizer Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4 pb-3 border-b border-slate-800">
        <div>
          <div className="flex items-center space-x-2">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              Discovered Structural Causal Graph (DAG)
            </h3>
            <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-emerald-950 text-emerald-400 border border-emerald-800 flex items-center">
              <CheckCircle2 className="w-3 h-3 mr-1 text-emerald-400" />
              Verified Acyclic
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Click any sensor node to simulate Pearl's Interventional Operator: <code className="text-cyan-300 font-mono text-xs">do(X)</code>
          </p>
        </div>

        {/* Legend */}
        <div className="flex items-center space-x-4 text-xs">
          <div className="flex items-center space-x-1.5">
            <span className="w-3 h-3 rounded-full bg-cyan-500/20 border border-cyan-400"></span>
            <span className="text-slate-300">Ankle (Distal)</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-3 h-3 rounded-full bg-indigo-500/20 border border-indigo-400"></span>
            <span className="text-slate-300">Waist (Proximal)</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-3 h-3 rounded-full bg-amber-500/20 border border-amber-400"></span>
            <span className="text-slate-300">Intervened Target</span>
          </div>
        </div>
      </div>

      {/* SVG Canvas */}
      <div className="relative w-full h-[410px] bg-slate-950/80 rounded-lg border border-slate-800/70 overflow-hidden flex items-center justify-center">
        <svg viewBox="0 0 760 400" className="w-full h-full">
          <defs>
            {/* Standard Arrow Marker */}
            <marker
              id="causal-arrow"
              viewBox="0 0 10 10"
              refX="18"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#06b6d4" />
            </marker>

            {/* Active / Downstream Arrow Marker */}
            <marker
              id="causal-arrow-active"
              viewBox="0 0 10 10"
              refX="18"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#f59e0b" />
            </marker>
          </defs>

          {/* Render Causal Edges */}
          {graph.edges.map((edge, idx) => {
            const src = nodeMap.get(edge.source);
            const tgt = nodeMap.get(edge.target);
            if (!src || !tgt) return null;

            const isDownstream =
              selectedInterventionNode &&
              (edge.source === selectedInterventionNode || descendants.has(edge.source)) &&
              descendants.has(edge.target);

            const isHovered = hoveredEdge === edge;

            // Curved quadratic bezier path
            const dx = tgt.x - src.x;
            const dy = tgt.y - src.y;
            const cx = (src.x + tgt.x) / 2;
            const cy = (src.y + tgt.y) / 2 - 20;

            const strokeColor = isDownstream
              ? "#f59e0b"
              : isHovered
              ? "#38bdf8"
              : "#0284c7";

            const strokeWidth = isDownstream ? 3 : isHovered ? 2.5 : Math.max(1.2, edge.confidence * 2);

            return (
              <g
                key={`edge-${idx}`}
                className="cursor-pointer transition-all"
                onMouseEnter={() => setHoveredEdge(edge)}
                onMouseLeave={() => setHoveredEdge(null)}
              >
                <path
                  d={`M ${src.x} ${src.y} Q ${cx} ${cy} ${tgt.x} ${tgt.y}`}
                  fill="none"
                  stroke={strokeColor}
                  strokeWidth={strokeWidth}
                  strokeDasharray={isDownstream ? "5 3" : "none"}
                  markerEnd={isDownstream ? "url(#causal-arrow-active)" : "url(#causal-arrow)"}
                  opacity={
                    selectedInterventionNode && !isDownstream && edge.source !== selectedInterventionNode
                      ? 0.25
                      : 0.85
                  }
                />
                {/* Edge confidence label on center */}
                <circle cx={cx} cy={cy} r="10" fill="#0f172a" stroke={strokeColor} strokeWidth="1" />
                <text
                  x={cx}
                  y={cy + 3}
                  textAnchor="middle"
                  fill="#e2e8f0"
                  fontSize="8.5"
                  fontWeight="600"
                  className="pointer-events-none select-none font-mono"
                >
                  {edge.confidence.toFixed(2)}
                </text>
              </g>
            );
          })}

          {/* Render Sensor Nodes */}
          {Array.from(nodeMap.values()).map(({ node, x, y }) => {
            const isIntervened = selectedInterventionNode === node.id;
            const isAffected = descendants.has(node.id);
            const isAnkle = node.id.includes("ankle");

            let nodeFill = isAnkle ? "#083344" : "#1e1b4b";
            let strokeColor = isAnkle ? "#06b6d4" : "#6366f1";

            if (isIntervened) {
              nodeFill = "#78350f";
              strokeColor = "#f59e0b";
            } else if (isAffected) {
              nodeFill = "#451a03";
              strokeColor = "#d97706";
            }

            return (
              <g
                key={node.id}
                className="cursor-pointer select-none transition-transform hover:scale-105"
                onClick={() => {
                  if (onInterveneNode) {
                    onInterveneNode(isIntervened ? null : node.id);
                  }
                }}
                onMouseEnter={() => setHoveredNode(node)}
                onMouseLeave={() => setHoveredNode(null)}
              >
                {/* Outer halo */}
                <circle
                  cx={x}
                  cy={y}
                  r="24"
                  fill={nodeFill}
                  stroke={strokeColor}
                  strokeWidth={isIntervened ? "3" : "2"}
                  className="transition-colors duration-200"
                />

                {/* Sensor icon abbreviation */}
                <text
                  x={x}
                  y={y - 3}
                  textAnchor="middle"
                  fill={isIntervened ? "#fbbf24" : "#f8fafc"}
                  fontSize="10"
                  fontWeight="700"
                  className="font-mono"
                >
                  {node.id.includes("gyro") ? "GYR" : "ACC"}
                </text>

                {/* Axis designation */}
                <text
                  x={x}
                  y={y + 10}
                  textAnchor="middle"
                  fill={isIntervened ? "#fde68a" : "#94a3b8"}
                  fontSize="9"
                  fontWeight="600"
                  className="font-mono"
                >
                  {node.id.split("_").slice(-2).join("_").toUpperCase()}
                </text>

                {/* Node Label Below */}
                <text
                  x={x}
                  y={y + 36}
                  textAnchor="middle"
                  fill="#cbd5e1"
                  fontSize="9"
                  fontWeight="500"
                >
                  {node.id}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Selected Intervention Banner */}
        {selectedInterventionNode && (
          <div className="absolute top-3 left-3 bg-amber-950/90 border border-amber-600/80 rounded-lg px-3 py-2 text-xs text-amber-200 backdrop-blur shadow-lg flex items-center space-x-2">
            <Zap className="w-4 h-4 text-amber-400 animate-pulse" />
            <div>
              <span className="font-bold">Intervention Active: </span>
              <code className="bg-amber-900/60 px-1.5 py-0.5 rounded font-mono text-amber-300">
                do({selectedInterventionNode} = δ)
              </code>
              <div className="text-[11px] text-amber-300/80 mt-0.5">
                Downstream affected kinematic targets: {descendants.size} nodes
              </div>
            </div>
            <button
              onClick={() => onInterveneNode && onInterveneNode(null)}
              className="ml-2 text-amber-400 hover:text-white underline text-[11px]"
            >
              Reset
            </button>
          </div>
        )}
      </div>

      {/* Mechanism & Details Footer Box */}
      <div className="mt-4 p-3.5 bg-slate-950 rounded-lg border border-slate-800 text-xs">
        {hoveredEdge ? (
          <div>
            <div className="flex items-center space-x-2 text-cyan-400 font-semibold mb-1">
              <ArrowRight className="w-3.5 h-3.5" />
              <span>
                Causal Link: <code className="text-white">{hoveredEdge.source}</code> →{" "}
                <code className="text-white">{hoveredEdge.target}</code>
              </span>
              <span className="ml-auto text-slate-400">
                Confidence: <strong className="text-cyan-300">{(hoveredEdge.confidence * 100).toFixed(0)}%</strong>
              </span>
            </div>
            <p className="text-slate-300">
              <strong className="text-slate-400">Physical Mechanism: </strong>
              {hoveredEdge.mechanism}
            </p>
            {hoveredEdge.time_delay_ms && (
              <div className="text-[11px] text-slate-400 mt-1">
                Estimated Kinematic Phase Propagation Delay: ~{hoveredEdge.time_delay_ms} ms
              </div>
            )}
          </div>
        ) : hoveredNode ? (
          <div>
            <div className="flex items-center space-x-2 text-cyan-400 font-semibold mb-1">
              <Info className="w-3.5 h-3.5" />
              <span>
                Node: <code className="text-white">{hoveredNode.id}</code> ({hoveredNode.anatomical_location.toUpperCase()} {hoveredNode.sensor_type})
              </span>
            </div>
            <p className="text-slate-300">
              {hoveredNode.description || "Sensor channel stream measuring 3D kinematic motion dynamics."}
            </p>
          </div>
        ) : (
          <div className="flex items-center text-slate-400 space-x-2">
            <Info className="w-4 h-4 text-cyan-400 flex-shrink-0" />
            <span>
              Hover over directed edges to read the deduced biomechanical mechanism, or click a node to perform counterfactual intervention testing.
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
