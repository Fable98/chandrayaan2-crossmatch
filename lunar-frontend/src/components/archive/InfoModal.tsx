"use client";

import React from "react";

export interface InfoModalContent {
  tag: string;
  title: string;
  subtitle: string;
  paragraphs: string[];
  specs?: { label: string; value: string }[];
}

interface Props {
  content: InfoModalContent;
  onClose: () => void;
}

export default function InfoModal({ content, onClose }: Props) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-2xl animate-fade-in">
      <div className="relative flex max-h-[90vh] w-full max-w-2xl flex-col overflow-hidden rounded-3xl border border-white/15 bg-[#0c0e24]/90 text-white shadow-[0_30px_90px_rgba(0,0,0,0.9)] ring-1 ring-white/10">
        {/* Header Bar */}
        <div className="flex items-center justify-between border-b border-white/10 bg-white/[0.02] px-6 py-4 backdrop-blur-md">
          <div className="flex items-center gap-2">
            <span className="rounded-full border border-purple-400/30 bg-purple-500/10 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-purple-300">
              {content.tag}
            </span>
            <span className="text-sm font-bold text-white">
              {content.title}
            </span>
          </div>

          <button
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded-full border border-white/10 bg-white/[0.04] text-xs text-slate-300 transition hover:bg-white/[0.08] hover:text-white"
            title="Close"
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
          <div>
            <h3 className="text-lg font-bold text-white">
              {content.subtitle}
            </h3>
          </div>

          <div className="space-y-4 text-xs leading-relaxed text-slate-300">
            {content.paragraphs.map((p, idx) => (
              <p key={idx}>{p}</p>
            ))}
          </div>

          {content.specs && (
            <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
              <span className="text-xs font-semibold uppercase tracking-wider text-purple-300 block mb-3">
                Technical Specifications &amp; Details
              </span>
              <div className="space-y-2 text-xs font-mono">
                {content.specs.map((s, idx) => (
                  <div key={idx} className="flex justify-between border-b border-white/[0.05] pb-2">
                    <span className="text-slate-400 font-sans">{s.label}</span>
                    <span className="text-white font-bold">{s.value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-white/10 bg-white/[0.02] px-6 py-3 text-xs">
          <span className="text-slate-400">ISRO Chandrayaan-2 Cross-Match</span>
          <button
            onClick={onClose}
            className="rounded-full border border-white/10 bg-white/[0.04] px-4 py-1.5 text-xs text-slate-300 hover:bg-white/[0.08] transition"
          >
            Close ✕
          </button>
        </div>
      </div>
    </div>
  );
}
