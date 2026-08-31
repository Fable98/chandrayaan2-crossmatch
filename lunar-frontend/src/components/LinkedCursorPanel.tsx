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
      <div className="flex items-center justify-between border-b border-border px-4 py-2">
        <h2 className="text-2xs uppercase tracking-wide text-ink-faint">
          Linked cursor · click a feature on OHRC
        </h2>
        <span className="text-2xs font-mono text-ink-faint">
          {points.length} matches (post-RANSAC)
        </span>
      </div>

      <div className="flex flex-1 items-center justify-center gap-6 p-6">
        <ImagePane
          label="OHRC · 0.25–0.32 m/px"
          innerRef={ohrcRef}
          src={imageUrl(`/images/ohrc/${tripletId}`)}
          onClick={handleClick}
          marker={selection ? { xFrac: selection.ohrcPx[0] / TILE_PX, yFrac: selection.ohrcPx[1] / TILE_PX, kind: "query" } : null}
        />

        <div className="flex flex-col items-center gap-1 text-ink-faint">
          <span className="font-mono text-2xs">
            {showMatch ? `${selection!.match!.confidence.toFixed(2)} conf` : "—"}
          </span>
          <div className="h-px w-8 bg-border-bright" />
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

      <div className="border-t border-border px-4 py-2 text-2xs font-mono text-ink-faint">
        {selection === null &&
          "Click anywhere on the OHRC tile to find the nearest verified LoFTR + RANSAC correspondence."}
        {selection !== null && !showMatch &&
          `No confident match within ${NEARBY_PX}px of that pixel — matches are sparse by design (large craters and ridges, not a dense mesh).`}
        {showMatch &&
          `ohrc_px=(${selection!.match!.ohrc_px[0].toFixed(1)}, ${selection!.match!.ohrc_px[1].toFixed(1)})  →  tmc_px=(${selection!.match!.tmc_px[0].toFixed(1)}, ${selection!.match!.tmc_px[1].toFixed(1)})`}
      </div>
    </div>
  );
}

function ImagePane({
  label,
  src,
  onClick,
  marker,
  innerRef,
}: {
  label: string;
  src: string;
  onClick?: (e: React.MouseEvent<HTMLDivElement>) => void;
  marker: { xFrac: number; yFrac: number; kind: "query" | "match" } | null;
  innerRef?: React.RefObject<HTMLDivElement>;
}) {
  return (
    <div className="flex flex-col items-center gap-2">
      <div
        ref={innerRef}
        onClick={onClick}
        className={`relative h-[340px] w-[340px] overflow-hidden border border-border-bright bg-panel ${
          onClick ? "cursor-crosshair" : ""
        }`}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt={label}
          className="h-full w-full select-none object-cover"
          draggable={false}
          onError={(e) => {
            (e.currentTarget as HTMLImageElement).style.display = "none";
          }}
        />
        {marker && (
          <div
            className={`absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 ${
              marker.kind === "query"
                ? "border-regolith bg-regolith/30"
                : "border-parallax bg-parallax/30"
            }`}
            style={{
              left: `${marker.xFrac * 100}%`,
              top: `${marker.yFrac * 100}%`,
            }}
          />
        )}
      </div>
      <span className="font-mono text-2xs text-ink-dim">{label}</span>
    </div>
  );
}
