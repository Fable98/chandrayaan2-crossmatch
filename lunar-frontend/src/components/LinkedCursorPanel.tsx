"use client";

import { useRef, useState } from "react";
import type { MatchPoint } from "@/lib/types";
import { imageUrl } from "@/lib/api";

// The pipeline always produces 512x512 tiles for every sensor — see
// data-preprocessing Stage 4 ("Grid Standardization"). If that ever
// changes, this needs to read the real dimensions per triplet instead.
const TILE_PX = 512;

interface Props {
  tripletId: string;
  points: MatchPoint[];
}

interface Selection {
  ohrcPx: [number, number];
  match: MatchPoint | null;
  distancePx: number | null;
}

function nearestMatch(px: [number, number], points: MatchPoint[]): { match: MatchPoint; distance: number } | null {
  if (points.length === 0) return null;
  let best = points[0];
  let bestDist = Infinity;
  for (const p of points) {
    const dx = p.ohrc_px[0] - px[0];
    const dy = p.ohrc_px[1] - px[1];
    const d = Math.sqrt(dx * dx + dy * dy);
    if (d < bestDist) {
      bestDist = d;
      best = p;
    }
  }
  return { match: best, distance: bestDist };
}

export default function LinkedCursorPanel({ tripletId, points }: Props) {
  const [selection, setSelection] = useState<Selection | null>(null);
  const ohrcRef = useRef<HTMLDivElement>(null);

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const el = ohrcRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const xFrac = (e.clientX - rect.left) / rect.width;
    const yFrac = (e.clientY - rect.top) / rect.height;
    const px: [number, number] = [xFrac * TILE_PX, yFrac * TILE_PX];
    const nearest = nearestMatch(px, points);
    setSelection({
      ohrcPx: px,
      match: nearest?.match ?? null,
      distancePx: nearest?.distance ?? null,
    });
  };

  const CONFIDENCE_OK = 0.5;
  const NEARBY_PX = 40;

  const showMatch =
    selection?.match &&
    selection.distancePx !== null &&
    selection.distancePx < NEARBY_PX;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-border bg-panel/80 px-5 py-3">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-white">
          Linked cursor · <span className="font-normal text-ink-dim lowercase">click any teal dot on OHRC</span>
        </h2>
        <span className="rounded-full border border-teal/30 bg-teal/10 px-2.5 py-0.5 font-mono text-2xs text-teal font-semibold">
          {points.length} verified matches
        </span>
      </div>

      <div className="flex flex-1 items-center justify-center gap-8 p-6">
        <ImagePane
          label="OHRC · 0.25–0.32 m/px"
          innerRef={ohrcRef}
          src={imageUrl(`/images/ohrc/${tripletId}`)}
          onClick={handleClick}
          marker={selection ? { xFrac: selection.ohrcPx[0] / TILE_PX, yFrac: selection.ohrcPx[1] / TILE_PX, kind: "query" } : null}
          ghostMarkers={points.map((p) => ({
            xFrac: p.ohrc_px[0] / TILE_PX,
            yFrac: p.ohrc_px[1] / TILE_PX,
          }))}
        />

        <div className="flex flex-col items-center gap-1.5 text-ink-faint">
          <span className="font-mono text-2xs font-semibold text-teal">
            {showMatch ? `${(selection!.match!.confidence * 100).toFixed(0)}% conf` : "—"}
          </span>
          <div className="h-px w-10 bg-teal/40" />
        </div>

        <ImagePane
          label="TMC-2 · ~4–5 m/px"
          src={imageUrl(`/images/tmc/${tripletId}`)}
          marker={
            showMatch
              ? {
                  xFrac: selection!.match!.tmc_px[0] / TILE_PX,
                  yFrac: selection!.match!.tmc_px[1] / TILE_PX,
                  kind: "match",
                }
              : null
          }
        />
      </div>

      <div className="border-t border-border bg-panel/60 px-5 py-2.5 text-2xs font-mono text-ink-dim">
        {selection === null && (
          <span>
            <span className="text-teal font-semibold">• Teal dots</span> mark verified LoFTR + RANSAC correspondences — click one to project coordinates onto TMC-2.
          </span>
        )}
        {selection !== null && !showMatch && (
          <span className="text-alert">
            No match within {NEARBY_PX}px of that pixel — try clicking closer to an active teal dot.
          </span>
        )}
        {showMatch && (
          <span>
            <span className="text-white">ohrc_px</span>=({selection!.match!.ohrc_px[0].toFixed(1)}, {selection!.match!.ohrc_px[1].toFixed(1)}) → <span className="text-teal font-bold">tmc_px</span>=({selection!.match!.tmc_px[0].toFixed(1)}, {selection!.match!.tmc_px[1].toFixed(1)})
          </span>
        )}
      </div>
    </div>
  );
}

function ImagePane({
  label,
  src,
  onClick,
  marker,
  ghostMarkers,
  innerRef,
}: {
  label: string;
  src: string;
  onClick?: (e: React.MouseEvent<HTMLDivElement>) => void;
  marker: { xFrac: number; yFrac: number; kind: "query" | "match" } | null;
  ghostMarkers?: { xFrac: number; yFrac: number }[];
  innerRef?: React.RefObject<HTMLDivElement>;
}) {
  return (
    <div className="flex flex-col items-center gap-2.5">
      <div
        ref={innerRef}
        onClick={onClick}
        className={`relative h-[340px] w-[340px] overflow-hidden rounded-xl border border-border bg-panel-raised shadow-xl ${
          onClick ? "cursor-crosshair" : ""
        }`}
      >
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
        {ghostMarkers?.map((g, i) => (
          <div
            key={i}
            className="pointer-events-none absolute h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-teal shadow-[0_0_6px_rgba(63,181,201,0.8)] ring-1 ring-teal/50"
            style={{ left: `${g.xFrac * 100}%`, top: `${g.yFrac * 100}%` }}
          />
        ))}
        {marker && (
          <div
            className={`absolute h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 ${
              marker.kind === "query"
                ? "border-white bg-white/40 shadow-[0_0_12px_rgba(255,255,255,0.9)]"
                : "border-teal bg-teal/40 shadow-[0_0_16px_rgba(63,181,201,1)]"
            }`}
            style={{
              left: `${marker.xFrac * 100}%`,
              top: `${marker.yFrac * 100}%`,
            }}
          />
        )}
      </div>
      <span className="font-mono text-2xs text-ink-dim tracking-wide">{label}</span>
    </div>
  );
}
