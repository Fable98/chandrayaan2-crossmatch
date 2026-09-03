import type { TripletBounds } from "./types";

// The backend serves longitude in the standard lunar/PDS4 planetocentric
// 0-360 convention (e.g. west_lon: 336.48). Leaflet's default CRS expects
// -180..180. This was validated against region_005/region_006, which
// genuinely straddle the 180 meridian (180.0822 normalizes to -179.9178,
// which is why the guard below is necessary and not just defensive).
export function normalizeTo180(lon: number): number {
  return (((lon + 180) % 360) + 360) % 360 - 180;
}

export type LatLngTuple = [number, number];
export type LatLngBoundsTuple = [LatLngTuple, LatLngTuple];

export function toLeafletBounds(bounds: TripletBounds): LatLngBoundsTuple {
  let west = normalizeTo180(bounds.west_lon);
  let east = normalizeTo180(bounds.east_lon);

  // Antimeridian guard: if normalizing flipped the order (west ended up
  // greater than east) but the original 0-360 values were in the expected
  // order, the region crosses the 180 meridian. Extend east by 360 so the
  // box is continuous instead of Leaflet interpreting it as wrapping the
  // long way around the Moon.
  if (west > east && bounds.east_lon > bounds.west_lon) {
    east += 360;
  }

  return [
    [bounds.south_lat, west],
    [bounds.north_lat, east],
  ];
}

export function boundsCenter(bounds: TripletBounds): LatLngTuple {
  let west = normalizeTo180(bounds.west_lon);
  let east = normalizeTo180(bounds.east_lon);
  if (west > east && bounds.east_lon > bounds.west_lon) {
    east += 360;
  }
  return [(bounds.south_lat + bounds.north_lat) / 2, (west + east) / 2];
}

// Ground footprint size in km, using the Moon's mean radius. Used for the
// per-region metadata readout — same formula used to sanity-check
// region_003 against known OHRC swath geometry during backend QA.
const MOON_RADIUS_KM = 1737.4;

export function footprintSizeKm(bounds: TripletBounds): {
  widthKm: number;
  heightKm: number;
} {
  const dLon = Math.abs(bounds.east_lon - bounds.west_lon);
  const dLat = Math.abs(bounds.north_lat - bounds.south_lat);
  const midLatRad = ((bounds.north_lat + bounds.south_lat) / 2) * (Math.PI / 180);
  const widthKm =
    (dLon * Math.PI) / 180 * MOON_RADIUS_KM * Math.cos(midLatRad);
  const heightKm = (dLat * Math.PI) / 180 * MOON_RADIUS_KM;
  return { widthKm, heightKm };
}
