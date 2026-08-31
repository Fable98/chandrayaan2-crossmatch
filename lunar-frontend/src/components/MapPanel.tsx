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

export default function MapPanel({ triplet, iirsOverlay }: Props) {
  const [activeLayers, setActiveLayers] = useState<Set<Layer>>(new Set());

  const bounds = toLeafletBounds(triplet.bounds);
  const demBounds = triplet.dem_available ? toLeafletBounds(triplet.bounds) : null;
  const center = boundsCenter(triplet.bounds);

  const toggle = (layer: Layer) => {
    setActiveLayers((prev) => {
      const next = new Set(prev);
      if (next.has(layer)) next.delete(layer);
      else next.add(layer);
      return next;
    });
  };

  return (
    <div className="relative h-full w-full">
      <MapContainer
        center={center}
        zoom={13}
        className="h-full w-full"
      >
        {/* No basemap tile layer: there's no verified public lunar tile
            server wired up here, and a fabricated URL would silently 404
            or (worse) render an Earth/Mars basemap under lunar data. The
            void background plus the footprint rectangle and sensor
            overlays are enough context for the demo. */}
        <Rectangle
          bounds={bounds}
          pathOptions={{ color: "#E8A33D", weight: 1.5, fillOpacity: 0.03 }}
        />
        {activeLayers.has("iirs") && iirsOverlay && (
          <ImageOverlay
            url={imageUrl(iirsOverlay.image_url)}
            bounds={bounds}
            opacity={iirsOverlay.opacity_hint ?? 0.6}
          />
        )}
        {activeLayers.has("dem") && demBounds && triplet.dem_url && (
          <ImageOverlay url={imageUrl(triplet.dem_url)} bounds={demBounds} opacity={0.7} />
        )}
        <FitOnChange boundsKey={triplet.id} bounds={bounds} />
      </MapContainer>

      <div className="absolute right-3 top-3 z-[1000] flex flex-col gap-1.5 border border-border bg-panel/95 p-2 backdrop-blur-sm">
        <span className="text-2xs uppercase tracking-wide text-ink-faint px-1">
          Layers
        </span>
        <LayerToggle
          label="IIRS mineralogy"
          swatch="bg-parallax"
          checked={activeLayers.has("iirs")}
          disabled={!iirsOverlay}
          onChange={() => toggle("iirs")}
        />
        <LayerToggle
          label="DEM elevation"
          swatch="bg-regolith"
          checked={activeLayers.has("dem")}
          disabled={!triplet.dem_available}
          onChange={() => toggle("dem")}
        />
      </div>
    </div>
  );
}

function LayerToggle({
  label,
  swatch,
  checked,
  disabled,
  onChange,
}: {
  label: string;
  swatch: string;
  checked: boolean;
  disabled?: boolean;
  onChange: () => void;
}) {
  return (
    <label
      className={`flex items-center gap-2 px-1 py-0.5 text-xs ${
        disabled ? "text-ink-faint cursor-not-allowed" : "text-ink cursor-pointer"
      }`}
    >
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={onChange}
        className="sr-only"
      />
      <span
        className={`h-3 w-3 border ${
          checked ? `${swatch} border-transparent` : "border-border-bright bg-transparent"
        }`}
      />
      {label}
    </label>
  );
}
