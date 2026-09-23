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
    <div className="relative z-0 h-[460px] w-full overflow-hidden retro-inset bg-[#D5CFC2] dark:bg-[#090b0e]">
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
          pathOptions={{ color: "#1F4743", weight: 2, fillOpacity: 0.08 }}
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
        <div className="pointer-events-none absolute left-1/2 top-1/2 z-10 -translate-x-1/2 -translate-y-1/2 retro-outset bg-panel p-3 text-center font-mono">
          <p className="text-xs font-bold text-ink">
            The <span className="text-[#1F4743] dark:text-[#52938B]">teal box</span> is the shared OHRC/TMC/IIRS footprint.
          </p>
          <p className="mt-0.5 text-2xs text-[#717874] dark:text-[#8C9893]">
            Toggle a payload layer on the right to project raster data.
          </p>
        </div>
      )}

      <div className="absolute right-3 top-3 z-10 flex flex-col gap-2 retro-outset bg-panel p-2.5 font-mono">
        <span className="text-[10px] uppercase font-bold tracking-wider text-[#1F4743] dark:text-[#52938B]">
          Payload Layers
        </span>
        {tiffFallback && (
          <span
            title="The source raster is a GeoTIFF, which browsers cannot render — showing the backend-generated PNG preview instead."
            className="retro-inset bg-amber-50 px-2 py-0.5 text-[9px] font-bold text-amber-800 dark:bg-amber-950/40 dark:text-amber-300"
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
      className={`flex items-center justify-between gap-3 px-2.5 py-1 text-xs font-mono font-bold transition ${
        disabled
          ? "cursor-not-allowed opacity-40 retro-inset text-[#888]"
          : checked
          ? "retro-inset bg-[#143532] text-white"
          : "retro-button text-ink"
      }`}
    >
      <span>{label}</span>
      <span
        className={`h-2 w-2 border border-black/40 ${
          checked ? "bg-emerald-400 shadow-[0_0_6px_#34D399]" : disabled ? "bg-neutral-500/40" : "bg-neutral-400"
        }`}
      />
    </button>
  );
}
