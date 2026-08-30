import React, { useState, useEffect } from "react";
import { Folder, FileCode, Copy, Check, Download, FileText, ChevronRight, ChevronDown } from "lucide-react";

interface CodebaseExplorerTabProps {
  onDownloadZip: () => void;
  isDownloadingZip?: boolean;
}

export const CodebaseExplorerTab: React.FC<CodebaseExplorerTabProps> = ({
  onDownloadZip,
  isDownloadingZip = false,
}) => {
  const [files, setFiles] = useState<Record<string, string>>({});
  const [selectedFile, setSelectedFile] = useState<string>("causal_har_agent/causal/gemini_discovery.py");
  const [copied, setCopied] = useState<boolean>(false);
  const [openFolders, setOpenFolders] = useState<Record<string, boolean>>({
    causal_har_agent: true,
    "causal_har_agent/causal": true,
    "causal_har_agent/data": true,
    "causal_har_agent/models": true,
    "causal_har_agent/evaluation": true,
    scripts: true,
  });

  useEffect(() => {
    fetch("/api/codebase/tree")
      .then((res) => res.json())
      .then((data) => {
        if (data.files) {
          setFiles(data.files);
        }
      })
      .catch((err) => console.error("Error fetching codebase tree:", err));
  }, []);

  const handleCopy = () => {
    if (files[selectedFile]) {
      navigator.clipboard.writeText(files[selectedFile]);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const toggleFolder = (folder: string) => {
    setOpenFolders((prev) => ({ ...prev, [folder]: !prev[folder] }));
  };

  const treeStructure = [
    {
      name: "causal_har_agent",
      isFolder: true,
      children: [
        { name: "__init__.py", path: "causal_har_agent/__init__.py" },
        {
          name: "causal",
          isFolder: true,
          children: [
            { name: "__init__.py", path: "causal_har_agent/causal/__init__.py" },
            { name: "gemini_discovery.py", path: "causal_har_agent/causal/gemini_discovery.py" },
            { name: "prompt_builder.py", path: "causal_har_agent/causal/prompt_builder.py" },
            { name: "graph_structure.py", path: "causal_har_agent/causal/graph_structure.py" },
          ],
        },
        {
          name: "data",
          isFolder: true,
          children: [
            { name: "__init__.py", path: "causal_har_agent/data/__init__.py" },
            { name: "dataset.py", path: "causal_har_agent/data/dataset.py" },
            { name: "feature_extractor.py", path: "causal_har_agent/data/feature_extractor.py" },
          ],
        },
        {
          name: "models",
          isFolder: true,
          children: [
            { name: "__init__.py", path: "causal_har_agent/models/__init__.py" },
            { name: "baseline_transformer.py", path: "causal_har_agent/models/baseline_transformer.py" },
            { name: "causal_transformer.py", path: "causal_har_agent/models/causal_transformer.py" },
          ],
        },
        {
          name: "evaluation",
          isFolder: true,
          children: [
            { name: "__init__.py", path: "causal_har_agent/evaluation/__init__.py" },
            { name: "robustness_metrics.py", path: "causal_har_agent/evaluation/robustness_metrics.py" },
          ],
        },
      ],
    },
    {
      name: "scripts",
      isFolder: true,
      children: [
        { name: "run_causal_discovery.py", path: "scripts/run_causal_discovery.py" },
        { name: "evaluate_robustness.py", path: "scripts/evaluate_robustness.py" },
      ],
    },
    { name: "pyproject.toml", path: "pyproject.toml" },
    { name: "requirements.txt", path: "requirements.txt" },
    { name: "README.md", path: "README.md" },
  ];

  return (
    <div className="space-y-6">
      {/* Action Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h3 className="text-sm font-bold text-white uppercase tracking-wider">
            Causal-HAR-Agent Academic Python Codebase
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Complete, PEP 8 compliant, type-annotated PyTorch & Google GenAI Python modules
          </p>
        </div>

        <button
          onClick={onDownloadZip}
          disabled={isDownloadingZip}
          className="inline-flex items-center px-4 py-2 rounded-lg text-xs font-semibold bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-lg shadow-cyan-500/20 disabled:opacity-50"
        >
          <Download className="w-4 h-4 mr-2" />
          {isDownloadingZip ? "Packaging ZIP..." : "Download Full Repo (.zip)"}
        </button>
      </div>

      {/* Main File Explorer & Code Viewer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl min-h-[620px]">
        {/* Left Side: Tree View */}
        <div className="lg:col-span-4 border-r border-slate-800 p-3.5 bg-slate-950/60 overflow-y-auto max-h-[620px]">
          <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-3 px-2">
            Repository File Hierarchy
          </div>

          <div className="space-y-1 text-xs">
            {treeStructure.map((item, idx) => {
              if (item.isFolder) {
                const isOpen = openFolders[item.name] ?? true;
                return (
                  <div key={idx} className="space-y-1">
                    <button
                      onClick={() => toggleFolder(item.name)}
                      className="w-full flex items-center px-2 py-1.5 rounded text-slate-300 hover:bg-slate-800/60 font-semibold"
                    >
                      {isOpen ? <ChevronDown className="w-3.5 h-3.5 mr-1 text-slate-500" /> : <ChevronRight className="w-3.5 h-3.5 mr-1 text-slate-500" />}
                      <Folder className="w-4 h-4 mr-1.5 text-cyan-400" />
                      <span>{item.name}/</span>
                    </button>

                    {isOpen && (
                      <div className="pl-4 space-y-1 border-l border-slate-800/80 ml-3">
                        {item.children?.map((child: any, cIdx: number) => {
                          if (child.isFolder) {
                            const childPath = `${item.name}/${child.name}`;
                            const isChildOpen = openFolders[childPath] ?? true;
                            return (
                              <div key={cIdx} className="space-y-1">
                                <button
                                  onClick={() => toggleFolder(childPath)}
                                  className="w-full flex items-center px-2 py-1 rounded text-slate-400 hover:bg-slate-800/60 font-medium"
                                >
                                  {isChildOpen ? <ChevronDown className="w-3 h-3 mr-1 text-slate-600" /> : <ChevronRight className="w-3 h-3 mr-1 text-slate-600" />}
                                  <Folder className="w-3.5 h-3.5 mr-1.5 text-indigo-400" />
                                  <span>{child.name}/</span>
                                </button>
                                {isChildOpen && (
                                  <div className="pl-4 space-y-0.5 border-l border-slate-800/80 ml-3">
                                    {child.children?.map((leaf: any, lIdx: number) => (
                                      <button
                                        key={lIdx}
                                        onClick={() => setSelectedFile(leaf.path)}
                                        className={`w-full flex items-center px-2 py-1 rounded text-left transition-colors ${
                                          selectedFile === leaf.path
                                            ? "bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/40"
                                            : "text-slate-400 hover:bg-slate-800/40 hover:text-slate-200"
                                        }`}
                                      >
                                        <FileCode className="w-3.5 h-3.5 mr-1.5 text-slate-400" />
                                        <span className="truncate">{leaf.name}</span>
                                      </button>
                                    ))}
                                  </div>
                                )}
                              </div>
                            );
                          }
                          return (
                            <button
                              key={cIdx}
                              onClick={() => setSelectedFile(child.path)}
                              className={`w-full flex items-center px-2 py-1 rounded text-left transition-colors ${
                                selectedFile === child.path
                                  ? "bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/40"
                                  : "text-slate-400 hover:bg-slate-800/40 hover:text-slate-200"
                              }`}
                            >
                              <FileCode className="w-3.5 h-3.5 mr-1.5 text-slate-400" />
                              <span className="truncate">{child.name}</span>
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              }

              // Root files
              return (
                <button
                  key={idx}
                  onClick={() => setSelectedFile(item.path)}
                  className={`w-full flex items-center px-2 py-1.5 rounded text-left transition-colors ${
                    selectedFile === item.path
                      ? "bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/40"
                      : "text-slate-400 hover:bg-slate-800/40 hover:text-slate-200"
                  }`}
                >
                  <FileText className="w-3.5 h-3.5 mr-2 text-slate-400" />
                  <span>{item.name}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right Side: Code Viewer */}
        <div className="lg:col-span-8 flex flex-col bg-slate-950">
          {/* File Header Bar */}
          <div className="flex items-center justify-between px-4 py-2.5 bg-slate-900 border-b border-slate-800 text-xs">
            <div className="flex items-center space-x-2 text-slate-300 font-mono">
              <FileCode className="w-4 h-4 text-cyan-400" />
              <span>/{selectedFile}</span>
            </div>

            <button
              onClick={handleCopy}
              className="flex items-center px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 mr-1 text-emerald-400" />
                  <span className="text-emerald-400 font-medium">Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5 mr-1" />
                  <span>Copy Code</span>
                </>
              )}
            </button>
          </div>

          {/* Code Body */}
          <div className="flex-1 p-4 overflow-auto max-h-[560px] font-mono text-xs leading-relaxed text-slate-200 selection:bg-cyan-900 selection:text-cyan-200">
            <pre>
              <code>{files[selectedFile] || "# Loading file content..."}</code>
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};
