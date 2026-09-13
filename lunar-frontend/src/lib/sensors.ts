/**
 * sensors.ts — Centralized sensor metadata and resolution contracts (Step 1).
 *
 * SENSOR_META defines nominal spec constants for Chandrayaan-2 and external
 * lunar payloads. Live per-region values from the backend contract
 * (TripletSummary.sensors) take precedence via sensorMeta().
 */

import type { TripletSummary, SensorKind } from "./types";

export interface SensorSpec {
  label: string;
  gsdM: number;
  unit: string;
}

/**
 * Spec constants for supported lunar sensors (nominal resolution in meters).
 * Distinct from live per-region metadata returned by the backend contract.
 * - OHRC: 0.25 m/px (ISRO High Resolution Camera nadir)
 * - TMC-2: 5.0 m/px (ISRO Terrain Mapping Camera nominal; 4–5m orbital range)
 * - IIRS: 80.0 m/px (ISRO Imaging Infrared Spectrometer nominal)
 * - DEM: 5.0 m/px (TMC stereo DEM resolution)
 * - LRO NAC: 0.914 m/px (NASA LRO Narrow Angle Camera nominal at 50km)
 */
export const SENSOR_META: Record<SensorKind, SensorSpec> = {
  ohrc: {
    label: "OHRC",
    gsdM: 0.25, // spec constant
    unit: "m/px",
  },
  tmc: {
    label: "TMC-2",
    gsdM: 5.0, // spec constant
    unit: "m/px",
  },
  iirs: {
    label: "IIRS",
    gsdM: 80.0, // spec constant
    unit: "m/px",
  },
  dem: {
    label: "DEM",
    gsdM: 5.0, // spec constant
    unit: "m/px",
  },
  lro_nac: {
    label: "NASA LRO NAC",
    gsdM: 0.914, // spec constant
    unit: "m/px",
  },
};

/** Normalizes diverse sensor naming (e.g. "OHRC", "TMC", "tmc-2", "LRO_NAC") to canonical SensorKind. */
export function normalizeSensorId(id: string): SensorKind {
  const s = id.toLowerCase().replace(/[-_]/g, "");
  if (s.includes("ohrc")) return "ohrc";
  if (s.includes("tmc")) return "tmc";
  if (s.includes("iirs")) return "iirs";
  if (s.includes("dem")) return "dem";
  if (s.includes("lro")) return "lro_nac";
  return "ohrc";
}

export interface SensorMetaEntry {
  label: string;
  gsdM: number;
  unit: string;
  isLive?: boolean;
}

/**
 * Returns live gsd_m from triplet.sensors when present,
 * falling back to the spec constant otherwise.
 */
export function sensorMeta(
  triplet: TripletSummary | null | undefined,
  id: string
): SensorMetaEntry {
  const key = normalizeSensorId(id);
  const spec = SENSOR_META[key];

  let liveGsd: number | undefined;

  if (triplet?.sensors && Array.isArray(triplet.sensors)) {
    const match = triplet.sensors.find((s) => {
      const name = (s.sensor || "").toLowerCase().replace(/[-_]/g, "");
      return name === key.replace(/[-_]/g, "") || normalizeSensorId(s.sensor || "") === key;
    });
    if (match && typeof match.gsd_m === "number" && Number.isFinite(match.gsd_m) && match.gsd_m > 0) {
      liveGsd = match.gsd_m;
    }
  }

  if (liveGsd === undefined && key === "lro_nac" && typeof triplet?.lro_nac_gsd_m === "number" && triplet.lro_nac_gsd_m > 0) {
    liveGsd = triplet.lro_nac_gsd_m;
  }

  if (liveGsd === undefined && triplet?.gsd && typeof triplet.gsd === "object") {
    const gsdObj = triplet.gsd as Record<string, number>;
    const direct = gsdObj[key] ?? gsdObj[id];
    if (typeof direct === "number" && direct > 0) {
      liveGsd = direct;
    }
  }

  return {
    label: spec.label,
    gsdM: liveGsd ?? spec.gsdM,
    unit: spec.unit,
    isLive: liveGsd !== undefined,
  };
}

/**
 * Formats scale gaps (e.g. 5.0/0.25 -> "20x") so tags like "21x" / "~3.6x" / "~320x"
 * are computed, not typed.
 */
export function scaleRatioLabel(
  gsdA: number,
  gsdB: number,
  options?: { approx?: boolean; decimals?: number }
): string {
  if (!gsdA || !gsdB || !Number.isFinite(gsdA) || !Number.isFinite(gsdB)) return "—";
  const ratio = Math.max(gsdA, gsdB) / Math.min(gsdA, gsdB);
  const isInteger = Math.abs(ratio - Math.round(ratio)) < 0.05;
  const numStr = isInteger
    ? `${Math.round(ratio)}`
    : (options?.decimals !== undefined
        ? ratio.toFixed(options.decimals)
        : ratio < 10
          ? (Math.floor(ratio * 10) / 10).toFixed(1)
          : `${Math.round(ratio)}`);
  const approx = options?.approx ?? !isInteger;
  return `${approx ? "~" : ""}${numStr}x`;
}

/** Sensor badge for archive vault cards (live GSD when a triplet is in scope). */
export function sensorBadge(id: "ohrc" | "tmc" | "iirs" | "lro" | "qa", triplet?: TripletSummary | null): string {
  switch (id) {
    case "ohrc": {
      const gsd = triplet ? sensorMeta(triplet, "ohrc").gsdM : SENSOR_META.ohrc.gsdM;
      return `${SENSOR_META.ohrc.label} ${gsd}m`;
    }
    case "tmc": {
      const gsd = triplet ? sensorMeta(triplet, "tmc").gsdM : SENSOR_META.tmc.gsdM;
      return `${SENSOR_META.tmc.label} ${gsd}m`;
    }
    case "iirs": {
      const gsd = triplet ? sensorMeta(triplet, "iirs").gsdM : SENSOR_META.iirs.gsdM;
      return `${SENSOR_META.iirs.label} ${gsd}m`;
    }
    case "lro": {
      const gsd = triplet ? sensorMeta(triplet, "lro_nac").gsdM : SENSOR_META.lro_nac.gsdM;
      return `LRO NAC ${gsd.toFixed(1)}m`;
    }
    case "qa":
      return "Co-Reg QA";
  }
}

/** Filter button label for vault: "OHRC (0.25m)", "TMC-2 (4m)", "IIRS (70m)", "LRO NAC (0.9m)". */
export function sensorFilterLabel(id: "ohrc" | "tmc" | "iirs" | "lro"): string {
  switch (id) {
    case "ohrc":
      return `${SENSOR_META.ohrc.label} (${SENSOR_META.ohrc.gsdM}m)`;
    case "tmc":
      return `${SENSOR_META.tmc.label} (4m)`;
    case "iirs":
      return `${SENSOR_META.iirs.label} (70m)`;
    case "lro":
      return `LRO NAC (${SENSOR_META.lro_nac.gsdM.toFixed(1)}m)`;
  }
}

/** Dossier resolution badges. */
export const DOSSIER_SENSOR_LABELS = {
  ohrc: `${SENSOR_META.ohrc.gsdM}–0.32 m/px Optical`,
  tmc: `~4–5 m/px Single View`,
  iirs: `~70–80 m/px 256-Band`,
  lro_nac: (productId?: string | null) => `~${SENSOR_META.lro_nac.gsdM.toFixed(1)} m/px (${productId ?? "External"})`,
};

/** LinkedCursorPanel reference and source labels. */
export function referenceSensorLabel(isLro: boolean): string {
  return isLro
    ? `NASA LRO NAC · ~${SENSOR_META.lro_nac.gsdM.toFixed(1)} m/px (Reference)`
    : `${SENSOR_META.tmc.label} · ~4–5 m/px (Reference)`;
}

export function sourceSensorLabel(): string {
  return `${SENSOR_META.ohrc.label} · ${SENSOR_META.ohrc.gsdM} m/px (Source)`;
}

/** Sensor card progress breakdown strings in Console.tsx (live GSD throughout). */
export function sensorCardGsd(triplet: TripletSummary | null | undefined, id: SensorKind): string {
  switch (id) {
    case "lro_nac":
      return `${sensorMeta(triplet, "lro_nac").gsdM.toFixed(1)} m/px`;
    default:
      return `${sensorMeta(triplet, id).gsdM} m/px`;
  }
}

/** Warped OHRC tag in Console.tsx. */
export function warpedOhrcTag(triplet: TripletSummary | null | undefined, isLro: boolean): string {
  const gsd = `${sensorMeta(triplet, "ohrc").gsdM}m`;
  return isLro ? `${gsd} Primary (to LRO)` : `${gsd} Primary (to TMC-2)`;
}

/** Checkerboard QA tag in Console.tsx (live GSD throughout). */
export function checkerboardTag(triplet: TripletSummary | null | undefined, isLro: boolean): string {
  const gsd = isLro
    ? `${sensorMeta(triplet, "lro_nac").gsdM.toFixed(1)}m`
    : `${sensorMeta(triplet, "tmc").gsdM}m`;
  return `Continuity Verification (${gsd} grid)`;
}

/**
 * RegistrationLauncher select options. Region-independent picker labels use
 * spec-nominal GSDs (live per-region values apply at render sites that have
 * a triplet in scope). Note: legacy labels read "4.0 m/px" (TMC) and
 * "69 m/px" (IIRS); spec nominals are 5.0 / 80.0 — the old strings tracked
 * single-region live values, which is exactly the drift this module ends.
 */
export const SENSOR_OPTIONS: { value: "OHRC" | "TMC" | "IIRS" | "LRO_NAC"; label: string }[] = [
  { value: "OHRC", label: `OHRC (${SENSOR_META.ohrc.gsdM} m/px Panchromatic Optical)` },
  { value: "TMC", label: `TMC-2 (${SENSOR_META.tmc.gsdM} m/px Single-View Optical)` },
  { value: "LRO_NAC", label: `NASA LRO NAC (0.5–1.2 m/px Lunar Reference)` },
  { value: "IIRS", label: `IIRS (${SENSOR_META.iirs.gsdM} m/px Infrared Hyperspectral)` },
];
