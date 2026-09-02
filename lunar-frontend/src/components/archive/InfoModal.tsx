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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-4 backdrop-blur-md">
      <div className="relative flex max-h-[90vh] w-full max-w-2xl flex-col overflow-hidden border border-[#23211d] bg-[#0d0d11] text-[#f0f2f5] shadow-[0_0_50px_rgba(0,0,0,0.8)]">
        {/* Header Bar */}
        <div className="flex items-center justify-between border-b border-[#23211d] bg-[#121217] px-6 py-4">
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs text-[#d4af37]">[ {content.tag} ]</span>
            <span className="font-mono text-xs font-bold uppercase tracking-wider text-white">
              {content.title}
            </span>
          </div>

          <button
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center border border-[#23211d] font-mono text-xs text-[#9a958e] transition-colors hover:border-[#d4af37] hover:text-[#d4af37] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4af37]"
            title="Close Info"
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
          <div>
            <h3 className="font-serif text-2xl italic text-[#e8d5b5]">
              {content.subtitle}
            </h3>
          </div>

          <div className="space-y-4 text-xs leading-relaxed text-[#9a958e]">
            {content.paragraphs.map((p, idx) => (
              <p key={idx}>{p}</p>
            ))}
          </div>

          {content.specs && (
            <div className="border border-[#23211d] bg-[#121217] p-4">
              <span className="font-mono text-[10px] uppercase tracking-wider text-[#d4af37] block mb-2">
                Technical Specifications &amp; Provenance
              </span>
              <div className="space-y-2 font-mono text-xs">
                {content.specs.map((s, idx) => (
                  <div key={idx} className="flex justify-between border-b border-[#1f1d18] pb-1">
                    <span className="text-[#6b665f]">{s.label}</span>
                    <span className="text-white font-medium">{s.value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-[#23211d] bg-[#121217] px-6 py-3 font-mono text-[11px]">
          <span className="text-[#6b665f]">ISRO Chandrayaan-2 Cross-Match</span>
          <button
            onClick={onClose}
            className="border border-[#38342d] bg-[#0d0d11] px-4 py-1.5 text-[#d4af37] hover:bg-[#2c2619] transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
          >
            Close ✕
          </button>
        </div>
      </div>
    </div>
  );
}
