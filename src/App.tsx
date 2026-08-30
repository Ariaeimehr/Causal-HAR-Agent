import React, { useState } from "react";
import JSZip from "jszip";
import { Header } from "./components/Header";
import { CausalStudioTab } from "./components/CausalStudioTab";
import { RobustnessArenaTab } from "./components/RobustnessArenaTab";
import { CodebaseExplorerTab } from "./components/CodebaseExplorerTab";
import { MethodologyTab } from "./components/MethodologyTab";

export default function App() {
  const [activeTab, setActiveTab] = useState<string>("studio");
  const [isDownloadingZip, setIsDownloadingZip] = useState<boolean>(false);

  const handleDownloadZip = async () => {
    setIsDownloadingZip(true);
    try {
      const res = await fetch("/api/codebase/tree");
      const { files } = await res.json();

      const zip = new JSZip();

      // Add all Python project files into the zip archive
      for (const [filePath, content] of Object.entries<string>(files)) {
        zip.file(filePath, content);
      }

      const zipBlob = await zip.generateAsync({ type: "blob" });
      const url = URL.createObjectURL(zipBlob);

      const a = document.createElement("a");
      a.href = url;
      a.download = "Causal-HAR-Agent.zip";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error("Error generating codebase ZIP:", e);
    } finally {
      setIsDownloadingZip(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Header & Navigation */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onDownloadZip={handleDownloadZip}
        isDownloadingZip={isDownloadingZip}
      />

      {/* Main Content Arena */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === "studio" && <CausalStudioTab />}
        {activeTab === "robustness" && <RobustnessArenaTab />}
        {activeTab === "codebase" && (
          <CodebaseExplorerTab
            onDownloadZip={handleDownloadZip}
            isDownloadingZip={isDownloadingZip}
          />
        )}
        {activeTab === "methodology" && <MethodologyTab />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-4 text-center text-xs text-slate-500">
        <p>
          Causal-HAR-Agent • Academic Research Framework for Invariant Human Activity Recognition • Powered by Google Gemini 3.7 & PyTorch
        </p>
      </footer>
    </div>
  );
}
