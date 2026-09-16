"use client";

import { useEffect, useState } from "react";
import { MapContainer, Rectangle, ImageOverlay, useMap } from "react-leaflet";
import type { LatLngBoundsExpression } from "leaflet";
import type { TripletSummary, IIRSOverlay } from "@/lib/types";
import { toLeafletBounds, boundsCenter } from "@/lib/geo";
import { imageUrl } from "@/lib/api";

interface Props {
  triplet: TripletSummary;
  iirsOverlay: IIRSOverlay | null;
}

function FitOnChange({ boundsKey, bounds }: { boundsKey: string; bounds: LatLngBoundsExpression }) {
  const map = useMap();
  useEffect(() => {
    map.fitBounds(bounds, { padding: [40, 40] });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [boundsKey]);
  return null;
}

type Layer = "iirs" | "dem";

/**
 * Browsers cannot render TIFF in <img>/Leaflet ImageOverlay — a raw .tif URL
 * would show a silently empty layer. Rewrite to the web preview the backend
 * generates alongside every raster (.png), and report whether a rewrite
 * happened so the UI can badge the fallback honestly.
 */
function toWebOverlayUrl(url: string | null | undefined): { url: string; wasTiff: boolean } {
  const raw = imageUrl(url ?? "");
  const wasTiff = /\.tiff?($|\?)/i.test(raw);
  const web = wasTiff ? raw.replace(/\.tiff?($|\?)/i, ".png$1") : raw;
  return { url: web, wasTiff };
}

export default function MapPanel({ triplet, iirsOverlay }: Props) {
  const [activeLayers, setActiveLayers] = useState<Set<Layer>>(new Set());

  const bounds = toLeafletBounds(triplet.bounds);
  const demBounds = triplet.dem_available ? toLeafletBounds(triplet.bounds) : null;
  const center = boundsCenter(triplet.bounds);

  const iirsWeb = toWebOverlayUrl(iirsOverlay?.image_url);
  const demWeb = toWebOverlayUrl(triplet.dem_url);
  const tiffFallback = (activeLayers.has("iirs") && iirsWeb.wasTiff) || (activeLayers.has("dem") && demWeb.wasTiff);

  const toggle = (layer: Layer) => {
    setActiveLayers((prev) => {
      const next = new Set(prev);
      if (next.has(layer)) next.delete(layer);
      else next.add(layer);
      return next;
    });
  };

  return (
    <div className="relative z-0 h-[460px] w-full overflow-hidden rounded-xl bg-[#dfe4ec] dark:bg-[#090b0e]">
      <MapContainer
        center={center}
        zoom={13}
        className="h-full w-full"
        attributionControl={false}
      >
        {/* No basemap tile layer: there's no verified public lunar tile
            server wired up here, and a fabricated URL would silently 404
            or (worse) render an Earth/Mars basemap under lunar data. The
            void background plus the footprint rectangle and sensor
            overlays are enough context for the demo. */}
        <Rectangle
          bounds={bounds}
          pathOptions={{ color: "#3fb5c9", weight: 2, fillOpacity: 0.05 }}
        />
        {activeLayers.has("iirs") && iirsOverlay && (
          <ImageOverlay
            url={iirsWeb.url}
            bounds={bounds}
            opacity={iirsOverlay.opacity_hint ?? 0.6}
          />
        )}
        {activeLayers.has("dem") && demBounds && triplet.dem_url && (
          <ImageOverlay url={demWeb.url} bounds={demBounds} opacity={0.7} />
        )}
        <FitOnChange boundsKey={triplet.id} bounds={bounds} />
      </MapContainer>

      {activeLayers.size === 0 && (
        <div className="pointer-events-none absolute left-1/2 top-1/2 z-10 -translate-x-1/2 -translate-y-1/2 rounded-xl border border-border bg-panel/90 px-5 py-3 text-center shadow-2xl backdrop-blur-md">
          <p className="text-xs font-medium text-ink">
            The <span className="text-teal font-semibold">teal box</span> is the shared OHRC/TMC/IIRS footprint.
          </p>
          <p className="mt-1 text-2xs font-mono text-ink-faint">
            Toggle a payload layer on the right to project raster data.
          </p>
        </div>
      )}

      <div className="absolute right-4 top-4 z-10 flex flex-col gap-2 rounded-xl border border-border bg-panel/90 p-3 shadow-2xl backdrop-blur-md">
        <span className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">
          Payload Layers
        </span>
        {tiffFallback && (
          <span
            title="The source raster is a GeoTIFF, which browsers cannot render — showing the backend-generated PNG preview instead."
            className="rounded-md border border-amber-300/60 bg-amber-50 px-2 py-1 font-mono text-[10px] font-bold text-amber-700 dark:bg-amber-950/40 dark:text-amber-300"
          >
            TIFF → PNG preview
          </span>
        )}
        <div className="flex flex-col gap-1.5">
          <LayerToggle
            label="IIRS Mineralogy"
            checked={activeLayers.has("iirs")}
            disabled={!iirsOverlay}
            onChange={() => toggle("iirs")}
          />
          <LayerToggle
            label="DEM Elevation"
            checked={activeLayers.has("dem")}
            disabled={!triplet.dem_available}
            onChange={() => toggle("dem")}
          />
        </div>
      </div>
    </div>
  );
}

function LayerToggle({
  label,
  checked,
  disabled,
  onChange,
}: {
  label: string;
  checked: boolean;
  disabled?: boolean;
  onChange: () => void;
}) {
  return (
    <button
      onClick={onChange}
      disabled={disabled}
      className={`flex items-center justify-between gap-3 rounded-full px-3 py-1.5 text-xs font-medium transition-all duration-200 ${
        disabled
          ? "cursor-not-allowed border border-slate-300/70 dark:border-white/5 text-ink-faint/50"
          : checked
          ? "bg-teal text-black font-semibold shadow-[0_0_15px_rgba(63,181,201,0.4)] hover:bg-[#52cde3]"
          : "border border-slate-300/70 dark:border-white/10 bg-white dark:bg-white/5 text-ink-dim hover:border-slate-400 dark:hover:border-white/20 hover:text-slate-900 dark:hover:text-white"
      }`}
    >
      <span>{label}</span>
      <span
        className={`h-2 w-2 rounded-full ${
          checked ? "bg-black" : disabled ? "bg-ink-faint/40" : "bg-ink-faint"
        }`}
      />
    </button>
  );
}
