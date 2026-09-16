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
 * - IIRS: 70.0 m/px (canonical stack nominal, aligned with ML_model SENSOR_GSD_MAP)
 * - DEM: 5.0 m/px (TMC stereo DEM resolution)
 * - LRO NAC: 0.9 m/px (canonical stack nominal, aligned with ML_model SENSOR_GSD_MAP)
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
    gsdM: 70.0, // canonical stack constant (OHRC 0.25 / TMC-2 5.0 / IIRS 70 / NAC 0.9)
    unit: "m/px",
  },
  dem: {
    label: "DEM",
    gsdM: 5.0, // spec constant
    unit: "m/px",
  },
  lro_nac: {
    label: "NASA LRO NAC",
    gsdM: 0.9, // canonical stack constant (matches ML_model SENSOR_GSD_MAP)
    unit: "m/px",
  },
};

/**
 * Normalizes diverse sensor naming (e.g. "OHRC", "TMC", "tmc-2", "LRO_NAC")
 * to a canonical SensorKind. Unrecognized ids return "unknown" — NEVER a
 * silent "ohrc" default, which mislabelled every future sensor as OHRC.
 */
export function normalizeSensorId(id: string): SensorKind | "unknown" {
  const s = id.toLowerCase().replace(/[-_]/g, "");
  if (s.includes("ohrc")) return "ohrc";
  if (s.includes("tmc")) return "tmc";
  if (s.includes("iirs")) return "iirs";
  if (s.includes("dem")) return "dem";
  if (s.includes("lro")) return "lro_nac";
  return "unknown";
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
  // Unknown sensors carry no spec constant: surface the raw id with an
  // absent (NaN) GSD rather than a fabricated OHRC 0.25m claim. Callers
  // render NaN via the "—" guards in sensorCardGsd/scaleRatioLabel.
  if (key === "unknown") {
    const live = liveGsdFor(triplet, id);
    return {
      label: id ? id.toUpperCase() : "Unknown",
      gsdM: live ?? NaN,
      unit: "m/px",
      isLive: live !== undefined,
    };
  }
  const spec = SENSOR_META[key];

  const liveGsd = liveGsdFor(triplet, id);

  return {
    label: spec.label,
    gsdM: liveGsd ?? spec.gsdM,
    unit: spec.unit,
    isLive: liveGsd !== undefined,
  };
}

/** Live per-region gsd_m lookup shared by known and unknown sensor ids. */
function liveGsdFor(
  triplet: TripletSummary | null | undefined,
  id: string
): number | undefined {
  const key = id.toLowerCase().replace(/[-_]/g, "");
  if (triplet?.sensors && Array.isArray(triplet.sensors)) {
    const want = normalizeSensorId(id);
    const match = triplet.sensors.find((s) => {
      const name = (s.sensor || "").toLowerCase().replace(/[-_]/g, "");
      if (name === key) return true;
      // Canonical-kind comparison only when BOTH sides are known — two
      // distinct unknown ids must never equal each other via "unknown".
      const got = normalizeSensorId(s.sensor || "");
      return want !== "unknown" && got === want;
    });
    if (match && typeof match.gsd_m === "number" && Number.isFinite(match.gsd_m) && match.gsd_m > 0) {
      return match.gsd_m;
    }
  }

  if (key.includes("lro") && typeof triplet?.lro_nac_gsd_m === "number" && triplet.lro_nac_gsd_m > 0) {
    return triplet.lro_nac_gsd_m;
  }

  if (triplet?.gsd && typeof triplet.gsd === "object") {
    const gsdObj = triplet.gsd as Record<string, number>;
    const direct = gsdObj[key] ?? gsdObj[id];
    if (typeof direct === "number" && direct > 0) {
      return direct;
    }
  }

  return undefined;
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

/** Filter button labels for the vault — GSDs derived from SENSOR_META spec nominals, never re-typed. */
export function sensorFilterLabel(id: "ohrc" | "tmc" | "iirs" | "lro"): string {
  switch (id) {
    case "ohrc":
      return `${SENSOR_META.ohrc.label} (${SENSOR_META.ohrc.gsdM}m)`;
    case "tmc":
      return `${SENSOR_META.tmc.label} (${SENSOR_META.tmc.gsdM}m)`;
    case "iirs":
      return `${SENSOR_META.iirs.label} (${SENSOR_META.iirs.gsdM}m)`;
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

/** Sensor card progress breakdown strings in Console.tsx (live GSD throughout; "—" when unknown). */
export function sensorCardGsd(triplet: TripletSummary | null | undefined, id: SensorKind): string {
  const gsd = sensorMeta(triplet, id).gsdM;
  if (!Number.isFinite(gsd)) return "—";
  switch (id) {
    case "lro_nac":
      return `${gsd.toFixed(1)} m/px`;
    default:
      return `${gsd} m/px`;
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
 * "69 m/px" (IIRS); spec nominals are 5.0 / 70.0 — the old strings tracked
 * single-region live values, which is exactly the drift this module ends.
 */
export const SENSOR_OPTIONS: { value: "OHRC" | "TMC" | "IIRS" | "LRO_NAC"; label: string }[] = [
  { value: "OHRC", label: `OHRC (${SENSOR_META.ohrc.gsdM} m/px Panchromatic Optical)` },
  { value: "TMC", label: `TMC-2 (${SENSOR_META.tmc.gsdM} m/px Single-View Optical)` },
  { value: "LRO_NAC", label: `NASA LRO NAC (0.5–1.2 m/px Lunar Reference)` },
  { value: "IIRS", label: `IIRS (${SENSOR_META.iirs.gsdM} m/px Infrared Hyperspectral)` },
];
