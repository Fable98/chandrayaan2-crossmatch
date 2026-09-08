// Mirrors backend/schemas.py exactly. Keep in sync with the backend —
// these are not independently designed, they're the frontend's view of
// the same contract the backend team already tested against real data.

export interface TripletBounds {
  west_lon: number;
  east_lon: number;
  south_lat: number;
  north_lat: number;
}

export interface SensorMeta {
  sensor: string;
  gsd_m: number;
  sun_elevation_deg?: number | null;
  sun_azimuth_deg?: number | null;
  incidence_angle_deg?: number | null;
}

export interface TripletSummary {
  id: string;
  bounds: TripletBounds;
  sensors?: SensorMeta[];
  ohrc_product_id?: string | null;
  tmc2_product_id?: string | null;
  iirs_product_id?: string | null;
  gsd?: Record<string, number>;
  sun_angle?: Record<string, number>;
  incidence_angle?: Record<string, number>;
  dem_available?: boolean;
  dem_url?: string | null;
}

export interface TripletListResponse {
  triplets: TripletSummary[];
}

export interface MatchPoint {
  ohrc_px: [number, number];
  tmc_px: [number, number];
  ohrc_latlon: [number, number];
  tmc_latlon: [number, number];
  confidence: number;
}

export interface MatchMetrics {
  num_inliers: number;
  num_raw_matches: number;
  inlier_ratio: number;
  rmse_px: number;
  mean_reprojection_error_px: number;
  median_reprojection_error_px: number;
  max_reprojection_error_px: number;
  sub_pixel_accurate: boolean;
  fraction_below_1px: number;
  source_coverage_ratio: number;
  destination_coverage_ratio: number;
  combined_coverage_score: number;
  source_occupied_cells: number;
  destination_occupied_cells?: number;
  total_cells?: number;
  uniformity_score?: number;
  triplet_consistency_px?: number | null;
  method?: string | null;
  orthorectified?: boolean;
}

export interface MatchesResponse {
  triplet_id: string;
  num_matches: number;
  homography: number[][] | null;
  matches: MatchPoint[];
  metrics?: MatchMetrics | null;
}

export interface IIRSOverlay {
  triplet_id: string;
  image_url: string;
  bounds: TripletBounds;
  opacity_hint: number;
}

export type SensorKind = "ohrc" | "tmc" | "iirs" | "dem";
