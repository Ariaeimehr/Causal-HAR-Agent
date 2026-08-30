import React from "react";
import { Activity, GitBranch, Cpu, BookOpen, Download, Sparkles, ShieldCheck } from "lucide-react";

interface HeaderProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  onDownloadZip: () => void;
  isDownloadingZip?: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  onDownloadZip,
  isDownloadingZip = false,
}) => {
  const tabs = [
    { id: "studio", label: "Causal Discovery & DAG", icon: GitBranch },
    { id: "robustness", label: "Transformer Robustness Arena", icon: ShieldCheck },
    { id: "codebase", label: "Python Repository Explorer", icon: Cpu },
    { id: "methodology", label: "Mathematical Methodology", icon: BookOpen },
  ];

  return (
    <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Title */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-500/20 text-white font-bold text-lg">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-lg font-bold text-white tracking-tight">Causal-HAR-Agent</h1>
                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-cyan-950 text-cyan-400 border border-cyan-800">
                  <Sparkles className="w-3 h-3 mr-1 text-cyan-400" />
                  Gemini 3.7 SCM
                </span>
                <span className="hidden sm:inline-flex px-2 py-0.5 rounded text-xs font-medium bg-indigo-950 text-indigo-300 border border-indigo-800">
                  PyTorch 2.0+
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden md:block">
                LLM-Driven Structural Causal Discovery & Invariant Attention for Sensor Kinematics
              </p>
            </div>
          </div>

          {/* Quick Export Action */}
          <div className="flex items-center space-x-3">
            <button
              onClick={onDownloadZip}
              disabled={isDownloadingZip}
              className="inline-flex items-center px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-gradient-to-r from-cyan-500 to-blue-600 text-white hover:from-cyan-400 hover:to-blue-500 transition-all shadow-md shadow-cyan-500/20 disabled:opacity-50"
            >
              <Download className="w-3.5 h-3.5 mr-1.5" />
              {isDownloadingZip ? "Generating ZIP..." : "Download Python Repo"}
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex space-x-1 overflow-x-auto no-scrollbar border-t border-slate-800/80 pt-1">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center px-4 py-2.5 text-xs font-medium border-b-2 whitespace-nowrap transition-colors ${
                  isActive
                    ? "border-cyan-400 text-cyan-300 bg-slate-800/40"
                    : "border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700"
                }`}
              >
                <Icon className={`w-4 h-4 mr-2 ${isActive ? "text-cyan-400" : "text-slate-400"}`} />
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
};
