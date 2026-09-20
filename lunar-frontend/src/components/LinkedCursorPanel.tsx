"use client";

import { useEffect, useRef, useState } from "react";
import type { MatchPoint } from "@/lib/types";
import { imageUrl } from "@/lib/api";
import GeoRefBadge from "./GeoRefBadge";
import { referenceSensorLabel, sourceSensorLabel } from "@/lib/sensors";

const TILE_PX = 512;
const NEARBY_PX = 45;
// Maximum pane size. Panes are fluid (full column width, square aspect) so
// they shrink on narrow viewports instead of overflowing their grid column.
// The raster is letterboxed (contain-fit) inside — never cropped — and the
// coordinate frame is matched exactly to the displayed image rect.
const PANE_MAX_PX = 340;

interface Props {
  tripletId: string;
  points: MatchPoint[];
  referenceMode?: "tmc" | "lro_nac";
  // Optional provenance notice (e.g. why zero dots are shown despite
  // reported inliers). Rendered as an honest banner, never silently empty.
  notice?: string | null;
  // Triplet footprint overlap from the ingest matcher (same number the
  // ingest results table shows). Null when the region manifest never
  // recorded one (older runs) — header then omits the badge.
  overlapPct?: number | null;
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

export default function LinkedCursorPanel({ tripletId, points, referenceMode = "tmc", notice = null, overlapPct = null }: Props) {
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
  }, [tripletId, referenceMode]);

  const activeIdx = selection ? selection.selectedIndex : hoveredIndex;
  const activeMatch = activeIdx !== null && activeIdx !== undefined ? points[activeIdx] ?? null : null;

  const handleSelectIndex = (index: number) => {
    const p = points[index];
    if (!p) return;
    setSelection({ selectedIndex: index, match: p, source: "chip" });
    setClickNotice(null);
  };

  const handleCanvasClick = (
    e: React.MouseEvent<HTMLDivElement>,
    sensor: "ohrc" | "tmc",
    // Natural raster dims of the pane that was clicked (ImagePane measures
    // the loaded file — tiles are not guaranteed 512², e.g. cropped LRO
    // swaths). Click fractions map into the SAME space the dots use
    // (coords / nat dims below), so nearest-matching stays self-consistent
    // at any native resolution. Defaults keep the 512 contract when the
    // image hasn't loaded yet.
    tileW: number = TILE_PX,
    tileH: number = TILE_PX
  ) => {
    const rect = e.currentTarget.getBoundingClientRect();
    // currentTarget is the coordinate frame div, which is exactly the
    // displayed image rect (letterbox bars live outside it via offX/offY),
    // so no letterbox subtraction is needed here — but the fraction is
    // scaled by the pane's own natural dims, never a hardcoded 512.
    const w = tileW > 0 ? tileW : TILE_PX;
    const h = tileH > 0 ? tileH : TILE_PX;
    const clickX = ((e.clientX - rect.left) / rect.width) * w;
    const clickY = ((e.clientY - rect.top) / rect.height) * h;

    const nearest = findNearestMatch([clickX, clickY], points, sensor);
    // Hit radius scales with the tile so a 1024-px raster has the same
    // ~9%-of-tile forgiveness as the 512 contract (45/512).
    const nearbyPx = NEARBY_PX * (Math.max(w, h) / TILE_PX);

    if (nearest && nearest.distance <= nearbyPx) {
      setSelection({
        selectedIndex: nearest.index,
        match: nearest.match,
        source: sensor,
      });
      setClickNotice(null);
    } else {
      setClickNotice(
        `Clicked at (${clickX.toFixed(0)}, ${clickY.toFixed(0)} px) · No correspondence within ${nearbyPx.toFixed(0)}px`
      );
      setTimeout(() => setClickNotice(null), 3000);
    }
  };

  const isLro = referenceMode === "lro_nac";
  const refLabel = referenceSensorLabel(isLro);
  const refSrc = isLro ? imageUrl(`/images/lro_nac/${tripletId}`) : imageUrl(`/images/tmc/${tripletId}`);

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-xl border border-border bg-panel text-ink shadow-panel">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border bg-panel-raised px-5 py-3">
        <div className="flex items-center gap-3">
          <span className="font-mono text-2xs uppercase tracking-widest text-teal">
            Sub-Pixel Match Verification
          </span>
          <span className="text-xs text-ink-dim font-mono">
            {points.length} verified tie points
          </span>
          {typeof overlapPct === "number" && (
            <span
              title="Shared OHRC+TMC-2+IIRS footprint overlap from triplet matching"
              className="rounded-full border border-teal/30 bg-teal/10 px-2 py-0.5 font-mono text-[11px] font-bold text-teal"
            >
              {overlapPct.toFixed(1)}% overlap
            </span>
          )}
        </div>
        <span className="rounded bg-teal/10 border border-teal/30 px-2 py-0.5 font-mono text-3xs uppercase tracking-wider text-teal">
          Stage 4: LK Refinement
        </span>
      </div>

      {notice && points.length === 0 && (
        <div className="mx-5 mt-4 rounded-xl border border-amber-200 dark:border-amber-900/50 bg-amber-50 dark:bg-amber-950/30 px-4 py-2.5 font-mono text-[11px] leading-relaxed text-amber-800 dark:text-amber-200">
          <span className="font-black uppercase tracking-wider">No correspondence dots: </span>
          {notice}
        </div>
      )}

      {/* Dual Canvas Arena */}
      <div className="grid flex-1 grid-cols-1 md:grid-cols-[1fr_auto_1fr] items-center gap-4 p-5">
        {/* Left: OHRC Image Pane */}
        <ImagePane
          sensor="ohrc"
          label={sourceSensorLabel()}
          innerRef={ohrcRef}
          src={imageUrl(`/images/ohrc/${tripletId}`)}
          onCanvasClick={(e, tileW, tileH) => handleCanvasClick(e, "ohrc", tileW, tileH)}
          points={points}
          coordKey="ohrc_px"
          selectedIndex={activeIdx}
          hoveredIndex={hoveredIndex}
          onSelectIndex={handleSelectIndex}
          onHoverIndex={setHoveredIndex}
        />

        {/* Center: Interactive Correlation Bridge */}
        <div className="flex flex-col items-center gap-2 text-ink-faint">
          <div className="flex flex-col items-center">
            <span
              className={`font-mono text-xs font-bold transition-all duration-200 ${
                activeMatch ? "text-teal drop-shadow-[0_0_8px_rgba(63,181,201,0.6)]" : "text-ink-faint"
              }`}
            >
              {activeMatch
                ? `${((activeMatch.confidence ?? 0) * 100).toFixed(0)}% conf`
                : "—"}
            </span>
            <span className="font-mono text-[9px] uppercase tracking-widest text-ink-faint">
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

        {/* Right: Reference Image Pane (TMC-2 or LRO NAC) */}
        <ImagePane
          sensor="tmc"
          label={refLabel}
          innerRef={tmcRef}
          src={refSrc}
          onCanvasClick={(e, tileW, tileH) => handleCanvasClick(e, "tmc", tileW, tileH)}
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
                      ? "border border-teal/50 bg-teal/10 text-teal"
                      : "border border-slate-300/70 dark:border-white/10 bg-panel-raised/80 text-ink-dim hover:border-slate-400 dark:hover:border-white/20 hover:text-slate-900 dark:hover:text-white"
                  }`}
                >
                  <span className="h-1.5 w-1.5 rounded-full bg-teal" />
                  <span>#{idx + 1}</span>
                  <span className="text-[9px] text-ink-faint">
                    {((p.confidence ?? 0) * 100).toFixed(0)}%
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
              <span className="text-slate-900 dark:text-white font-semibold">Match #{activeIdx! + 1}: </span>
              <span className="text-slate-600 dark:text-white/80">ohrc_px</span>=(
              <span className="text-slate-900 dark:text-white font-bold">{activeMatch.ohrc_px[0].toFixed(1)}</span>,{" "}
              <span className="text-slate-900 dark:text-white font-bold">{activeMatch.ohrc_px[1].toFixed(1)}</span>)
              {" → "}
              <span className="text-teal">tmc_px</span>=(
              <span className="text-teal font-bold">{activeMatch.tmc_px[0].toFixed(1)}</span>,{" "}
              <span className="text-teal font-bold">{activeMatch.tmc_px[1].toFixed(1)}</span>)
            </span>
            <span className="text-ink-faint">
              Confidence:{" "}
              <span className="text-teal font-semibold">
                {((activeMatch.confidence ?? 0) * 100).toFixed(1)}%
              </span>
              {activeMatch.sigma_major_px !== undefined && activeMatch.sigma_major_px !== null && (
                <span className="ml-3 text-ink-dim">
                  Uncertainty:{" "}
                  <span className="text-teal font-mono font-bold">
                    σ=({activeMatch.sigma_major_px.toFixed(2)}, {activeMatch.sigma_minor_px?.toFixed(2)})px
                  </span>
                  {activeMatch.ellipse_angle_deg !== undefined && activeMatch.ellipse_angle_deg !== null && (
                    <span className="text-ink-faint"> @ {activeMatch.ellipse_angle_deg.toFixed(1)}°</span>
                  )}
                  {activeMatch.uncertainty_status && (
                    <span className={`ml-2 rounded px-1.5 py-0.5 text-[9px] uppercase font-bold tracking-wider ${
                      activeMatch.uncertainty_status === "valid"
                        ? "bg-teal/10 text-teal border border-teal/30"
                        : activeMatch.uncertainty_status === "flat_peak"
                        ? "bg-amber-500/10 text-amber-500 border border-amber-500/30"
                        : activeMatch.uncertainty_status === "multimodal"
                        ? "bg-rose-500/10 text-rose-500 border border-rose-500/30"
                        : "bg-slate-500/10 text-slate-400 border border-slate-500/30"
                    }`}>
                      {activeMatch.uncertainty_status}
                    </span>
                  )}
                </span>
              )}
              {activeMatch.ohrc_latlon ? (
                <span className="ml-3 text-ink-dim">
                  Lat/Lon: ({activeMatch.ohrc_latlon[0].toFixed(3)}°, {activeMatch.ohrc_latlon[1].toFixed(3)}°)
                </span>
              ) : (
                <span className="ml-3">
                  <GeoRefBadge />
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
  // The pane reports its measured natural dims so clicks land in the same
  // coordinate space as the dots (never a hardcoded 512).
  onCanvasClick: (e: React.MouseEvent<HTMLDivElement>, tileW: number, tileH: number) => void;
  innerRef: React.RefObject<HTMLDivElement>;
  points: MatchPoint[];
  coordKey: "ohrc_px" | "tmc_px";
  selectedIndex: number | null;
  hoveredIndex: number | null;
  onSelectIndex: (idx: number) => void;
  onHoverIndex: (idx: number | null) => void;
}) {
  // Natural raster dimensions: tiles are not guaranteed square (e.g. cropped
  // LRO swaths), so the displayed rect is contain-fit into the measured
  // square viewport and the click/dot frame is matched to it — never
  // object-cover, which clips edges and silently breaks the px-fraction
  // mapping. The viewport is measured (ResizeObserver) so panes shrink on
  // narrow screens instead of overflowing their grid column.
  const [nat, setNat] = useState({ w: TILE_PX, h: TILE_PX });
  const [box, setBox] = useState(PANE_MAX_PX);
  const boxRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    setNat({ w: TILE_PX, h: TILE_PX });
  }, [src]);
  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const measure = () => {
      const w = el.clientWidth;
      if (w > 0) setBox(w);
    };
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const S = Math.max(1, Math.min(box, PANE_MAX_PX));
  const scale = Math.min(S / Math.max(nat.w, 1), S / Math.max(nat.h, 1));
  const dispW = Math.max(1, nat.w * scale);
  const dispH = Math.max(1, nat.h * scale);
  const offX = (S - dispW) / 2;
  const offY = (S - dispH) / 2;

  return (
    <div className="flex w-full min-w-0 flex-col items-center gap-2.5">
      <div
        ref={boxRef}
        className="group relative w-full overflow-hidden rounded-xl border border-border bg-panel-raised shadow-2xl select-none"
        style={{ maxWidth: PANE_MAX_PX, aspectRatio: "1 / 1" }}
      >
        {/* Sensor raster tile: explicit contain-fit size, never cropped. */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          key={src}
          src={src}
          alt={label}
          draggable={false}
          onLoad={(e) => {
            const img = e.currentTarget as HTMLImageElement;
            if (img.naturalWidth > 0 && img.naturalHeight > 0) {
              setNat({ w: img.naturalWidth, h: img.naturalHeight });
            }
          }}
          onError={(e) => {
            (e.currentTarget as HTMLImageElement).style.display = "none";
          }}
          className="absolute select-none lunar-tile-contrast"
          style={{ left: offX, top: offY, width: dispW, height: dispH }}
        />

        {/* Sensor Label Tag overlay */}
        <div className="pointer-events-none absolute top-2 left-2 z-10 rounded bg-black/60 px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider text-ink-dim backdrop-blur-sm">
          {sensor.toUpperCase()}
        </div>

        {/* Coordinate frame: exactly the displayed image rect, so % fractions
            map 1:1 to tile px for both markers and click handling. */}
        <div
          ref={innerRef}
          onClick={(e) => onCanvasClick(e, nat.w, nat.h)}
          className="absolute cursor-crosshair"
          style={{ left: offX, top: offY, width: dispW, height: dispH }}
        >
        {/* All Verified Correspondence Markers */}
        {points.map((p, idx) => {
          const coords = p[coordKey];
          if (!coords || coords.length < 2) return null;
          // Fractions of the pane's own natural raster — the 512 hardcode
          // misplaced every dot on non-square crops (e.g. LRO swaths).
          const tileW = nat.w > 0 ? nat.w : TILE_PX;
          const tileH = nat.h > 0 ? nat.h : TILE_PX;
          const xFrac = coords[0] / tileW;
          const yFrac = coords[1] / tileH;
          const isSelected = idx === selectedIndex;
          const isHovered = idx === hoveredIndex;
          // Missing confidence renders as 0%, never NaN%.
          const confPct = (p.confidence ?? 0) * 100;

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
              {/* Uncertainty Ellipse (2-sigma spatial covariance) */}
              {p.sigma_major_px !== undefined && p.sigma_major_px !== null && (
                <svg
                  className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2 overflow-visible"
                  style={{
                    left: "50%",
                    top: "50%",
                    width: 0,
                    height: 0,
                  }}
                >
                  <ellipse
                    cx="0"
                    cy="0"
                    rx={Math.max(3.5, (p.sigma_major_px * (dispW / tileW)) * 2.5)}
                    ry={Math.max(2.5, ((p.sigma_minor_px ?? p.sigma_major_px) * (dispH / tileH)) * 2.5)}
                    transform={`rotate(${p.ellipse_angle_deg ?? 0})`}
                    fill={
                      p.uncertainty_status === "multimodal"
                        ? "rgba(244, 63, 94, 0.2)"
                        : p.uncertainty_status === "flat_peak"
                        ? "rgba(245, 158, 11, 0.2)"
                        : "rgba(45, 212, 191, 0.18)"
                    }
                    stroke={
                      p.uncertainty_status === "multimodal"
                        ? "#f43f5e"
                        : p.uncertainty_status === "flat_peak"
                        ? "#f59e0b"
                        : "#2dd4bf"
                    }
                    strokeWidth={isSelected ? 1.5 : 1}
                    strokeDasharray={p.uncertainty_status === "valid" ? undefined : "2 2"}
                  />
                </svg>
              )}

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
                    #{idx + 1} ({coords[0].toFixed(0)}, {coords[1].toFixed(0)}) · {confPct.toFixed(0)}%{p.sigma_major_px !== undefined && p.sigma_major_px !== null ? ` · σ=${p.sigma_major_px.toFixed(2)}px` : ""}
                </div>
              )}
            </div>
          );
        })}
        </div>
      </div>

      <span className="font-mono text-2xs text-ink-dim tracking-wide text-center">
        {label}
      </span>
    </div>
  );
}
