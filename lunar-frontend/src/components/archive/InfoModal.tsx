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
    <div className="fixed inset-0 z-[2000] flex items-center justify-center bg-slate-900/40 p-4 backdrop-blur-sm animate-fade-in">
      <div className="relative flex max-h-[90vh] w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-slate-200/80 dark:border-[#1b2029] bg-white dark:bg-[#0e1117] text-slate-800 dark:text-slate-200 shadow-2xl">
        {/* Header Bar */}
        <div className="flex items-center justify-between border-b border-slate-100 dark:border-[#1b2029] bg-white dark:bg-[#0e1117] px-6 py-4">
          <div className="flex items-center gap-2.5">
            <span className="rounded-lg border border-indigo-100 dark:border-indigo-900/50 bg-indigo-50 dark:bg-indigo-950/30 px-2.5 py-0.5 text-[11px] font-bold uppercase tracking-wider text-[#4F46E5] dark:text-indigo-300">
              {content.tag}
            </span>
            <span className="text-sm font-bold text-slate-900 dark:text-slate-100">
              {content.title}
            </span>
          </div>

          <button
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded-lg border border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 text-xs text-slate-400 transition hover:bg-slate-50 dark:hover:bg-white/10 hover:text-slate-700 dark:hover:text-slate-200"
            title="Close"
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
          <div>
            <h3 className="text-lg font-bold text-slate-900 dark:text-slate-100">
              {content.subtitle}
            </h3>
          </div>

          <div className="space-y-4 text-xs leading-relaxed text-slate-600 dark:text-slate-300">
            {content.paragraphs.map((p, idx) => (
              <p key={idx}>{p}</p>
            ))}
          </div>

          {content.specs && (
            <div className="rounded-xl border border-slate-200/80 dark:border-[#1b2029] bg-slate-50 dark:bg-white/5 p-4">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 block mb-3">
                Technical Specifications &amp; Details
              </span>
              <div className="space-y-2 text-xs font-mono">
                {content.specs.map((s, idx) => (
                  <div key={idx} className="flex justify-between border-b border-slate-200/50 dark:border-[#1b2029] pb-2">
                    <span className="text-slate-500 dark:text-slate-400 font-sans">{s.label}</span>
                    <span className="text-slate-900 dark:text-slate-100 font-bold">{s.value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-slate-100 dark:border-[#1b2029] bg-slate-50/50 dark:bg-white/5 px-6 py-3 text-xs">
          <span className="text-slate-500 dark:text-slate-400 font-medium">ISRO Chandrayaan-2 Cross-Match</span>
          <button
            onClick={onClose}
            className="rounded-xl bg-[#4F46E5] hover:bg-[#4338CA] px-4 py-1.5 text-xs font-semibold text-white transition shadow-sm"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
