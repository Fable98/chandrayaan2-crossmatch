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
    <div className="fixed inset-0 z-[2000] flex items-center justify-center bg-black/60 p-4 backdrop-blur-xs animate-fade-in font-mono">
      <div className="retro-outset relative flex max-h-[90vh] w-full max-w-2xl flex-col overflow-hidden bg-[#E7E2D6] dark:bg-[#1A201E] text-[#1E2321] dark:text-[#E7E2D6] shadow-2xl">
        {/* Retro Window Titlebar */}
        <div className="flex items-center justify-between bg-[#1F4743] px-3 py-1.5 text-xs font-bold font-mono text-white select-none shrink-0 border-b border-[#143532]">
          <div className="flex items-center gap-2">
            <span className="text-sm">ℹ️</span>
            <span className="tracking-wider uppercase">
              {content.tag} // {content.title}
            </span>
          </div>

          <div className="flex items-center space-x-1">
            <button
              type="button"
              className="window-ctrl-btn"
              title="Minimize"
              tabIndex={-1}
            >
              _
            </button>
            <button
              type="button"
              className="window-ctrl-btn"
              title="Maximize"
              tabIndex={-1}
            >
              □
            </button>
            <button
              type="button"
              onClick={onClose}
              className="window-ctrl-btn hover:bg-rose-700 hover:text-white font-bold"
              title="Close [Esc]"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4">
          <div className="border-b border-[#8B8579] dark:border-[#2D3835] pb-3">
            <h3 className="text-base font-bold text-[#1E2321] dark:text-[#E7E2D6]">
              {content.subtitle}
            </h3>
          </div>

          <div className="space-y-3 text-xs leading-relaxed text-[#4A524E] dark:text-[#A8B2AD]">
            {content.paragraphs.map((p, idx) => (
              <p key={idx}>{p}</p>
            ))}
          </div>

          {content.specs && (
            <div className="retro-inset p-3 bg-[#DED8CB]/50 dark:bg-[#141817]">
              <span className="text-[11px] font-bold uppercase tracking-wider text-[#555C58] dark:text-[#8C9893] block mb-2">
                Technical Specifications &amp; Parameters
              </span>
              <div className="space-y-1.5 text-xs font-mono">
                {content.specs.map((s, idx) => (
                  <div key={idx} className="flex justify-between border-b border-[#B9B2A5]/50 dark:border-[#2D3835] pb-1">
                    <span className="text-[#555C58] dark:text-[#8C9893]">{s.label}</span>
                    <span className="text-[#1E2321] dark:text-[#E7E2D6] font-bold">{s.value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-[#8B8579] dark:border-[#2D3835] bg-[#DED8CB] dark:bg-[#141817] px-4 py-2 text-xs">
          <span className="text-[#555C58] dark:text-[#8C9893] font-medium">ISRO Chandrayaan-2 Cross-Match</span>
          <button
            onClick={onClose}
            className="retro-button-primary px-4 py-1 text-xs font-mono font-bold"
          >
            OK [Enter]
          </button>
        </div>
      </div>
    </div>
  );
}

