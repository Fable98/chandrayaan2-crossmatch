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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="relative flex max-h-[90vh] w-full max-w-2xl flex-col overflow-hidden rounded-lg border border-[#374151] bg-[#111827] text-[#e5e7eb] shadow-xl">
        {/* Header Bar */}
        <div className="flex items-center justify-between border-b border-[#1f2937] bg-[#1f2937] px-6 py-4">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-[#60a5fa]">
              {content.tag}
            </span>
            <span className="text-sm font-semibold text-white">
              {content.title}
            </span>
          </div>

          <button
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded border border-[#374151] bg-[#111827] text-xs text-[#9ca3af] transition-colors hover:border-[#4b5563] hover:text-white"
            title="Close"
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
          <div>
            <h3 className="text-base font-semibold text-white">
              {content.subtitle}
            </h3>
          </div>

          <div className="space-y-4 text-xs leading-relaxed text-[#9ca3af]">
            {content.paragraphs.map((p, idx) => (
              <p key={idx}>{p}</p>
            ))}
          </div>

          {content.specs && (
            <div className="rounded-md border border-[#374151] bg-[#1f2937] p-4">
              <span className="text-xs font-medium uppercase tracking-wider text-[#d1d5db] block mb-3">
                Technical Specifications &amp; Details
              </span>
              <div className="space-y-2 text-xs">
                {content.specs.map((s, idx) => (
                  <div key={idx} className="flex justify-between border-b border-[#374151] pb-1.5">
                    <span className="text-[#9ca3af]">{s.label}</span>
                    <span className="text-white font-medium">{s.value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-[#1f2937] bg-[#1f2937] px-6 py-3 text-xs">
          <span className="text-[#9ca3af]">ISRO Chandrayaan-2 Cross-Match</span>
          <button
            onClick={onClose}
            className="rounded border border-[#374151] bg-[#111827] px-4 py-1.5 text-xs text-[#d1d5db] hover:bg-[#374151] transition-colors"
          >
            Close ✕
          </button>
        </div>
      </div>
    </div>
  );
}
