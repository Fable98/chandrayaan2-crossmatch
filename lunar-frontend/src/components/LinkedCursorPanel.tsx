"use client";

import { useEffect, useRef, useState } from "react";
import type { MatchPoint } from "@/lib/types";
import { imageUrl } from "@/lib/api";

const TILE_PX = 512;
const NEARBY_PX = 45;

interface Props {
  tripletId: string;
  points: MatchPoint[];
}

interface Selection {
  selectedIndex: number;
  match: MatchPoint;
  source: "ohrc" | "tmc" | "chip";
}

function findNearestMatch(
  px: [number, number],
  points: MatchPoint[],
  sensor: "ohrc" | "tmc"
): { match: MatchPoint; index: number; distance: number } | null {
  if (!points || points.length === 0) return null;
  let best = points[0];
  let bestIdx = 0;
  let bestDist = Infinity;

  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const targetPx = sensor === "ohrc" ? p.ohrc_px : p.tmc_px;
    if (!targetPx || targetPx.length < 2) continue;
    const dx = targetPx[0] - px[0];
    const dy = targetPx[1] - px[1];
    const d = Math.sqrt(dx * dx + dy * dy);
    if (d < bestDist) {
      bestDist = d;
      best = p;
      bestIdx = i;
    }
  }

  return { match: best, index: bestIdx, distance: bestDist };
}

export default function LinkedCursorPanel({ tripletId, points }: Props) {
  const [selection, setSelection] = useState<Selection | null>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [clickNotice, setClickNotice] = useState<string | null>(null);

  const ohrcRef = useRef<HTMLDivElement>(null);
  const tmcRef = useRef<HTMLDivElement>(null);

  // Reset selection when region changes
  useEffect(() => {
    setSelection(null);
    setHoveredIndex(null);
    setClickNotice(null);
  }, [tripletId]);

  const handleCanvasClick = (
    e: React.MouseEvent<HTMLDivElement>,
    sensor: "ohrc" | "tmc"
  ) => {
    const ref = sensor === "ohrc" ? ohrcRef : tmcRef;
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const xFrac = (e.clientX - rect.left) / rect.width;
    const yFrac = (e.clientY - rect.top) / rect.height;
    const px: [number, number] = [xFrac * TILE_PX, yFrac * TILE_PX];

    const nearest = findNearestMatch(px, points, sensor);
    if (nearest && nearest.distance < NEARBY_PX) {
      setSelection({
        selectedIndex: nearest.index,
        match: nearest.match,
        source: sensor,
      });
      setClickNotice(null);
    } else {
      setClickNotice(
        `No match within ${NEARBY_PX}px of clicked location. Click on or near an active correspondence dot.`
      );
      setTimeout(() => setClickNotice(null), 3000);
    }
  };

  const handleSelectIndex = (idx: number) => {
    if (idx >= 0 && idx < points.length) {
      setSelection({
        selectedIndex: idx,
        match: points[idx],
        source: "chip",
      });
      setClickNotice(null);
    }
  };

  const activeIdx = selection?.selectedIndex ?? null;
  const activeMatch = selection?.match ?? null;

  return (
    <div className="flex h-full flex-col">
      {/* Top Status Header */}
      <div className="flex items-center justify-between border-b border-border bg-panel/80 px-5 py-3">
        <div>
          <h2 className="text-xs font-semibold uppercase tracking-wider text-white">
            Linked Cursor ·{" "}
            <span className="font-normal text-ink-dim lowercase">
              Click any match point on OHRC or TMC-2
            </span>
          </h2>
          <p className="text-[10px] font-mono text-ink-faint">
            LoFTR + CFOG sub-pixel verified geometric correspondence across sensor scales
          </p>
        </div>
        <div className="flex items-center gap-2">
          {activeIdx !== null && (
            <span className="rounded-full border border-gold/40 bg-gold/15 px-2.5 py-0.5 font-mono text-2xs font-semibold text-gold animate-pulse">
              Match #{activeIdx + 1} Selected
            </span>
          )}
          <span className="rounded-full border border-teal/30 bg-teal/10 px-2.5 py-0.5 font-mono text-2xs font-semibold text-teal">
            {points.length} verified pairs
          </span>
        </div>
      </div>

      {/* Dual Image Workspace */}
      <div className="flex flex-1 items-center justify-center gap-6 p-4 md:gap-8 md:p-6">
        {/* Left: OHRC Image Pane */}
        <ImagePane
          sensor="ohrc"
          label="OHRC · 0.25–0.32 m/px (Source)"
          innerRef={ohrcRef}
          src={imageUrl(`/images/ohrc/${tripletId}`)}
          onCanvasClick={(e) => handleCanvasClick(e, "ohrc")}
          points={points}
          coordKey="ohrc_px"
          selectedIndex={activeIdx}
          hoveredIndex={hoveredIndex}
          onSelectIndex={handleSelectIndex}
          onHoverIndex={setHoveredIndex}
        />

        {/* Center: Interactive Linkage Indicator */}
        <div className="flex flex-col items-center gap-2 text-ink-faint">
          <div className="flex flex-col items-center">
            <span
              className={`font-mono text-xs font-bold transition-all duration-200 ${
                activeMatch ? "text-teal drop-shadow-[0_0_8px_rgba(63,181,201,0.6)]" : "text-ink-faint"
              }`}
            >
              {activeMatch
                ? `${(activeMatch.confidence * 100).toFixed(0)}% conf`
                : "—"}
            </span>
            <span className="font-mono text-[9px] uppercase tracking-widest text-[#6b665f]">
              {activeMatch ? `Match #${activeIdx! + 1}` : "Select Point"}
            </span>
          </div>

          <div className="relative flex items-center justify-center">
            <div
              className={`h-0.5 w-12 transition-all duration-300 ${
                activeMatch ? "bg-teal shadow-[0_0_8px_rgba(63,181,201,0.8)]" : "bg-teal/30"
              }`}
            />
            <div
              className={`absolute h-2 w-2 rounded-full transition-all duration-300 ${
                activeMatch
                  ? "bg-teal scale-125 shadow-[0_0_10px_rgba(63,181,201,1)]"
                  : "bg-teal/40"
              }`}
            />
          </div>

          <span className="font-mono text-[9px] text-ink-dim text-center">
            {activeMatch ? "↔ Verified Link" : "Click dot"}
          </span>
        </div>

        {/* Right: TMC-2 Image Pane */}
        <ImagePane
          sensor="tmc"
          label="TMC-2 · ~4–5 m/px (Reference)"
          innerRef={tmcRef}
          src={imageUrl(`/images/tmc/${tripletId}`)}
          onCanvasClick={(e) => handleCanvasClick(e, "tmc")}
          points={points}
          coordKey="tmc_px"
          selectedIndex={activeIdx}
          hoveredIndex={hoveredIndex}
          onSelectIndex={handleSelectIndex}
          onHoverIndex={setHoveredIndex}
        />
      </div>

      {/* Point Quick Selector Chips */}
      {points.length > 0 && (
        <div className="flex items-center gap-2 overflow-x-auto border-t border-border/80 bg-panel/40 px-5 py-2">
          <span className="font-mono text-[10px] uppercase tracking-wider text-ink-dim shrink-0">
            Select Pair:
          </span>
          <div className="flex items-center gap-1.5 overflow-x-auto">
            {points.map((p, idx) => {
              const isSelected = idx === activeIdx;
              const isHovered = idx === hoveredIndex;
              return (
                <button
                  key={idx}
                  onClick={() => handleSelectIndex(idx)}
                  onMouseEnter={() => setHoveredIndex(idx)}
                  onMouseLeave={() => setHoveredIndex(null)}
                  className={`flex items-center gap-1.5 rounded-full px-2.5 py-0.5 font-mono text-2xs transition-all duration-150 ${
                    isSelected
                      ? "border border-teal bg-teal/20 text-teal font-bold shadow-[0_0_12px_rgba(63,181,201,0.4)] scale-105"
                      : isHovered
                      ? "border border-teal/50 bg-teal/10 text-teal-light"
                      : "border border-white/10 bg-panel-raised/80 text-ink-dim hover:border-white/20 hover:text-white"
                  }`}
                >
                  <span className="h-1.5 w-1.5 rounded-full bg-teal" />
                  <span>#{idx + 1}</span>
                  <span className="text-[9px] text-ink-faint">
                    {(p.confidence * 100).toFixed(0)}%
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Bottom Readout & Instructions */}
      <div className="border-t border-border bg-panel/60 px-5 py-2.5 text-2xs font-mono text-ink-dim">
        {clickNotice && (
          <span className="text-alert font-medium animate-pulse">
            {clickNotice}
          </span>
        )}

        {!clickNotice && activeMatch === null && (
          <span>
            <span className="text-teal font-semibold">• Teal dots</span> mark verified
            correspondences on both OHRC (left) and TMC-2 (right). Click any dot on either side to
            inspect the cross-sensor alignment and projected coordinates.
          </span>
        )}

        {!clickNotice && activeMatch !== null && (
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span>
              <span className="text-white font-semibold">Match #{activeIdx! + 1}: </span>
              <span className="text-white/80">ohrc_px</span>=(
              <span className="text-white font-bold">{activeMatch.ohrc_px[0].toFixed(1)}</span>,{" "}
              <span className="text-white font-bold">{activeMatch.ohrc_px[1].toFixed(1)}</span>)
              {" → "}
              <span className="text-teal">tmc_px</span>=(
              <span className="text-teal font-bold">{activeMatch.tmc_px[0].toFixed(1)}</span>,{" "}
              <span className="text-teal font-bold">{activeMatch.tmc_px[1].toFixed(1)}</span>)
            </span>
            <span className="text-ink-faint">
              Confidence:{" "}
              <span className="text-teal font-semibold">
                {(activeMatch.confidence * 100).toFixed(1)}%
              </span>
              {activeMatch.ohrc_latlon && (
                <span className="ml-3 text-ink-dim">
                  Lat/Lon: ({activeMatch.ohrc_latlon[0].toFixed(3)}°, {activeMatch.ohrc_latlon[1].toFixed(3)}°)
                </span>
              )}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function ImagePane({
  sensor,
  label,
  src,
  onCanvasClick,
  innerRef,
  points,
  coordKey,
  selectedIndex,
  hoveredIndex,
  onSelectIndex,
  onHoverIndex,
}: {
  sensor: "ohrc" | "tmc";
  label: string;
  src: string;
  onCanvasClick: (e: React.MouseEvent<HTMLDivElement>) => void;
  innerRef: React.RefObject<HTMLDivElement>;
  points: MatchPoint[];
  coordKey: "ohrc_px" | "tmc_px";
  selectedIndex: number | null;
  hoveredIndex: number | null;
  onSelectIndex: (idx: number) => void;
  onHoverIndex: (idx: number | null) => void;
}) {
  return (
    <div className="flex flex-col items-center gap-2.5">
      <div
        ref={innerRef}
        onClick={onCanvasClick}
        className="group relative h-[340px] w-[340px] overflow-hidden rounded-xl border border-border bg-panel-raised shadow-2xl cursor-crosshair select-none"
      >
        {/* Sensor raster tile */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          key={src}
          src={src}
          alt={label}
          className="h-full w-full select-none object-cover lunar-tile-contrast"
          draggable={false}
          onError={(e) => {
            (e.currentTarget as HTMLImageElement).style.display = "none";
          }}
        />

        {/* Sensor Label Tag overlay */}
        <div className="pointer-events-none absolute top-2 left-2 z-10 rounded bg-black/60 px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider text-ink-dim backdrop-blur-sm">
          {sensor.toUpperCase()}
        </div>

        {/* All Verified Correspondence Markers */}
        {points.map((p, idx) => {
          const coords = p[coordKey];
          if (!coords || coords.length < 2) return null;
          const xFrac = coords[0] / TILE_PX;
          const yFrac = coords[1] / TILE_PX;
          const isSelected = idx === selectedIndex;
          const isHovered = idx === hoveredIndex;

          return (
            <div
              key={idx}
              onClick={(e) => {
                e.stopPropagation();
                onSelectIndex(idx);
              }}
              onMouseEnter={() => onHoverIndex(idx)}
              onMouseLeave={() => onHoverIndex(null)}
              className="absolute -translate-x-1/2 -translate-y-1/2 cursor-pointer transition-all duration-150"
              style={{
                left: `${xFrac * 100}%`,
                top: `${yFrac * 100}%`,
                zIndex: isSelected ? 30 : isHovered ? 25 : 15,
              }}
            >
              {/* Selected Concentric Pulsing Rings */}
              {isSelected && (
                <>
                  <span className="absolute -inset-2 rounded-full bg-teal/30 animate-ping" />
                  <span className="absolute -inset-1 rounded-full border border-teal shadow-[0_0_12px_rgba(63,181,201,1)]" />
                </>
              )}

              {/* Hovered Outer Ring */}
              {isHovered && !isSelected && (
                <span className="absolute -inset-1.5 rounded-full border border-white/60 bg-white/10 animate-pulse" />
              )}

              {/* Central Core Marker Dot */}
              <div
                className={`relative flex items-center justify-center rounded-full transition-all duration-150 ${
                  isSelected
                    ? "h-4 w-4 bg-teal ring-2 ring-white shadow-[0_0_16px_rgba(63,181,201,1)] text-[9px] font-bold text-black"
                    : isHovered
                    ? "h-3.5 w-3.5 bg-teal ring-1 ring-white/80 shadow-[0_0_10px_rgba(63,181,201,0.9)] text-[8px] font-bold text-black"
                    : "h-2.5 w-2.5 bg-teal shadow-[0_0_6px_rgba(63,181,201,0.8)] ring-1 ring-teal/60 hover:scale-125"
                }`}
              >
                {(isSelected || isHovered) && (
                  <span className="select-none leading-none font-mono">
                    {idx + 1}
                  </span>
                )}
              </div>

              {/* Mini Tooltip on Hover or Selection */}
              {(isHovered || isSelected) && (
                <div
                  className={`pointer-events-none absolute bottom-full left-1/2 mb-1.5 -translate-x-1/2 whitespace-nowrap rounded border px-1.5 py-0.5 font-mono text-[9px] shadow-lg backdrop-blur-md transition-opacity ${
                    isSelected
                      ? "border-teal/60 bg-[#0e1726]/95 text-teal font-semibold"
                      : "border-white/20 bg-black/90 text-white"
                  }`}
                >
                  #{idx + 1} ({coords[0].toFixed(0)}, {coords[1].toFixed(0)}) · {(p.confidence * 100).toFixed(0)}%
                </div>
              )}
            </div>
          );
        })}
      </div>

      <span className="font-mono text-2xs text-ink-dim tracking-wide">
        {label}
      </span>
    </div>
  );
}
