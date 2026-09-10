// Step 13 contract: backend-identical shapes are re-exported from the
// GENERATED ./backend-types.ts (run `npm run gen:api`; CI runs
// `npm run gen:api:check`). Only UI-strict shapes (pixel tuples, index
// signatures, null-latlon, frontend-only extras) are hand-written below —
// each carries a note naming the backend schema it tracks.

export type {
  TripletBounds,
  HealthResponse,
  IIRSOverlay,
  TokenResponse,
  UserResponse,
  RegisterResponse,
  JobStatusResponse,
} from "./backend-types";

import type { TripletBounds, SensorMeta as BackendSensorMeta } from "./backend-types";

// Tracks backend SensorMeta + frontend-only tile_id.
export interface SensorMeta extends BackendSensorMeta {
  tile_id?: string | null;
}

export interface TripletSummary {
  id: string;
  bounds: TripletBounds;
  sensors?: SensorMeta[];
  ohrc_product_id?: string | null;
  tmc2_product_id?: string | null;
  iirs_product_id?: string | null;
  lro_nac_product_id?: string | null;
  lro_nac_available?: boolean;
  lro_nac_gsd_m?: number;
  reference_type?: "internal_TMC2" | "external_LRO_NAC";
  gsd?: Record<string, number>;
  sun_angle?: Record<string, number>;
  incidence_angle?: Record<string, number>;
  dem_available?: boolean;
  dem_url?: string | null;
  fit_rmse_px?: number | null;
  validation_rmse_px?: number | null;
  spatial_coverage?: number | null;
  spatial_uniformity?: number | null;
  quality_tier?: string | null;
  [key: string]: unknown;
}

export interface TripletListResponse {
  triplets: TripletSummary[];
}

export interface MatchPoint {
  ohrc_px: [number, number];
  tmc_px: [number, number];
  // Step 13: geographic coordinates are present ONLY when product bounds
  // exist. Pixel-only points carry null + georeferenced=false (the backend's
  // 336+fx demo patch is deleted) and render the shared no-georef badge.
  ohrc_latlon: [number, number] | null;
  tmc_latlon: [number, number] | null;
  georeferenced?: boolean;
  confidence: number;
}

// Moon-globe tie points (GET /api/registration/moon-points/{job_id}).
export interface MoonPoint {
  latitude: number | null;
  longitude: number | null;
  altitude?: number;
  confidence?: number;
  pixel_x?: number;
  pixel_y?: number;
  georeferenced: boolean;
}

export interface MoonPointsResponse {
  job_id: string;
  points: MoonPoint[];
  transformation_matrix?: number[][] | null;
  rmse_pixels?: number;
  rmse_meters?: number;
  georeferenced: boolean;
  georef_note?: string | null;
}

export interface MatchMetrics {
  num_inliers?: number;
  num_raw_matches?: number;
  inlier_ratio?: number;
  rmse_px?: number;
  fit_rmse_px?: number | null;
  validation_rmse_px?: number | null;
  mean_reprojection_error_px?: number;
  median_reprojection_error_px?: number;
  max_reprojection_error_px?: number;
  sub_pixel_accurate?: boolean;
  fraction_below_1px?: number;
  source_coverage_ratio?: number;
  destination_coverage_ratio?: number;
  combined_coverage_score?: number;
  spatial_coverage?: number;
  source_occupied_cells?: number;
  destination_occupied_cells?: number;
  total_cells?: number;
  uniformity_score?: number;
  spatial_uniformity?: number;
  triplet_consistency_px?: number | null;
  method?: string | null;
  orthorectified?: boolean;
  absolute_rmse_m?: number | null;
  validation_status?: string | null;
  quality_tier?: string | null;
  ssim?: number | null;
  psnr?: number | null;
  nmi?: number | null;
  composite_quality_score?: number | null;
  outlier_method?: string | null;
  [key: string]: unknown;
}

export interface MatchesResponse {
  triplet_id: string;
  num_matches?: number;
  homography: number[][] | null;
  matches: MatchPoint[];
  metrics?: MatchMetrics | null;
  [key: string]: unknown;
}

export type SensorKind = "ohrc" | "tmc" | "iirs" | "dem" | "lro_nac";
