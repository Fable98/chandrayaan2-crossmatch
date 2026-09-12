/**
 * backend-types.ts — GENERATED CONTRACT. DO NOT EDIT BY HAND.
 * Source: FastAPI openapi.json (title="SIH26166 — Lunar Image Correspondence API" version="0.1.0").
 * Regenerate: `npm run gen:api` (Step 13 single-contract rule).
 * Frontend extensions live in ./types.ts as `extends` interfaces.
 */

export interface Body_register_images_register_post {
  "source_file": string;
  "reference_file": string;
  "dem_file"?: string | null;
  "source_sensor"?: string;
  "reference_sensor"?: string;
  "method"?: string;
}

export interface Body_upload_and_ingest_api_ingest_upload_post {
  "files": string[];
  "containment"?: number;
  "tile_size"?: number;
  "no_large_aoi"?: boolean;
  "no_invariants"?: boolean;
  "max_time_gap_days"?: number | null;
  "require_dates"?: boolean;
}

export interface BundleAdjustmentRequest {
  "constraints": {
  [key: string]: unknown;
}[];
  "robust_loss"?: string;
  "max_iterations"?: number;
}

export interface FootprintResponse {
  "triplet_id": string;
  "bounds": TripletBounds;
}

export interface HTTPValidationError {
  "detail"?: ValidationError[];
}

export interface HealthResponse {
  "status": string;
  "triplets_loaded": number;
}

export interface IIRSOverlay {
  "triplet_id": string;
  "image_url": string;
  "bounds": TripletBounds;
  "opacity_hint"?: number;
}

export interface IIRSRegistrationRequest {
  "iirs_image_path": string;
  "ohrc_image_path": string;
  "grid_size"?: number;
}

export type JobStatus = "pending" | "running" | "success" | "failed";

export interface JobStatusResponse {
  "job_id": string;
  "status": JobStatus;
  "progress_percent"?: number;
  "current_phase"?: string;
  "result"?: {
  [key: string]: unknown;
} | null;
  "error"?: string | null;
}

export interface MatchMetrics {
  "num_inliers": number;
  "num_raw_matches"?: number;
  "inlier_ratio"?: number;
  "rmse_px"?: number;
  "mean_reprojection_error_px"?: number;
  "median_reprojection_error_px"?: number;
  "max_reprojection_error_px"?: number;
  "sub_pixel_accurate"?: boolean;
  "fraction_below_1px"?: number;
  "source_coverage_ratio"?: number;
  "destination_coverage_ratio"?: number;
  "combined_coverage_score"?: number;
  "uniformity_score"?: number;
  "triplet_consistency_px"?: number | null;
  "fit_rmse_px"?: number | null;
  "validation_rmse_px"?: number | null;
  "validation_status"?: string | null;
  "method"?: string | null;
  "orthorectified"?: boolean;
  "terrain_correction"?: {
  [key: string]: unknown;
} | null;
  "ssim"?: number | null;
  "psnr"?: number | null;
  "nmi"?: number | null;
  "composite_quality_score"?: number | null;
  "outlier_method"?: string | null;
  "absolute_rmse_m"?: number | null;
  "absolute_rmse_m_provenance"?: string | null;
  "metric_notes"?: {
  [key: string]: unknown;
} | null;
}

export interface MatchPoint {
  "ohrc_px": unknown[];
  "tmc_px": unknown[];
  "ohrc_latlon": unknown[];
  "tmc_latlon": unknown[];
  "confidence": number;
}

export interface MatchesResponse {
  "triplet_id": string;
  "num_matches": number;
  "homography": number[][] | null;
  "metrics"?: MatchMetrics | null;
  "matches": MatchPoint[];
}

export interface MoonPoint {
  "latitude"?: number | null;
  "longitude"?: number | null;
  "altitude"?: number;
  "confidence"?: number;
  "pixel_x"?: number;
  "pixel_y"?: number;
  "georeferenced"?: boolean;
}

export interface MoonPointsResponse {
  "job_id": string;
  "points": MoonPoint[];
  "transformation_matrix"?: number[][] | null;
  "rmse_pixels"?: number;
  "rmse_meters"?: number;
  "georeferenced"?: boolean;
  "georef_note"?: string | null;
}

export interface RegisterResponse {
  "status": string;
  "message"?: string | null;
  "metrics"?: {
  [key: string]: unknown;
} | null;
  "homography"?: number[][] | null;
  "visual_url"?: string | null;
  "warped_url"?: string | null;
  "source_url"?: string | null;
  "reference_url"?: string | null;
  "matches_url"?: string | null;
  "raster_url"?: string | null;
  "quiver_url"?: string | null;
  "metadata"?: {
  [key: string]: unknown;
} | null;
  "job_id"?: string | null;
  "report_url"?: string | null;
}

export interface RegistrationRequest {
  "src_image_path": string;
  "ref_image_path": string;
  "sensor_type"?: string;
  "min_inliers"?: number;
  "run_bundle_adjustment"?: boolean;
}

export interface RegistrationResponse {
  "job_id": string;
  "status": JobStatus;
  "message": string;
}

export interface SensorMeta {
  "sensor": string;
  "gsd_m": number;
  "sun_elevation_deg"?: number | null;
  "sun_azimuth_deg"?: number | null;
  "incidence_angle_deg"?: number | null;
}

export interface TokenResponse {
  "access_token": string;
  "token_type"?: string;
  "user": UserResponse;
}

export interface TripletBounds {
  "west_lon": number;
  "east_lon": number;
  "south_lat": number;
  "north_lat": number;
}

export interface TripletListResponse {
  "triplets": TripletSummary[];
}

export interface TripletSummary {
  "id": string;
  "bounds": TripletBounds;
  "sensors": SensorMeta[];
  "ohrc_product_id"?: string | null;
  "tmc2_product_id"?: string | null;
  "iirs_product_id"?: string | null;
  "lro_nac_available"?: boolean;
  "lro_nac_product_id"?: string | null;
  "lro_nac_gsd_m"?: number | null;
  "dem_available"?: boolean;
  "dem_url"?: string | null;
}

export interface UserCreate {
  "name": string;
  "email": string;
  "password": string;
}

export interface UserLogin {
  "email": string;
  "password": string;
}

export interface UserResponse {
  "id": string;
  "name": string;
  "email": string;
  "created_at": string;
}

export interface ValidationError {
  "loc": string | number[];
  "msg": string;
  "type": string;
}

/** All backend routes (path -> methods), for contract tests. */
export const BACKEND_API_PATHS = [
  "/" /* GET,HEAD */,
  "/api/ingest/jobs" /* GET */,
  "/api/ingest/results/{job_id}" /* GET */,
  "/api/ingest/status/{job_id}" /* GET */,
  "/api/ingest/upload" /* POST */,
  "/api/registration/bundle-adjust" /* POST */,
  "/api/registration/iirs" /* POST */,
  "/api/registration/logs/{job_id}" /* GET */,
  "/api/registration/moon-points/{job_id}" /* GET */,
  "/api/registration/report/{job_id}" /* GET */,
  "/api/registration/start" /* POST */,
  "/api/registration/status/{job_id}" /* GET */,
  "/auth/login" /* POST */,
  "/auth/me" /* GET */,
  "/auth/register" /* POST */,
  "/health" /* GET,HEAD */,
  "/images/{sensor}/{identifier}" /* GET */,
  "/refresh" /* GET */,
  "/register" /* POST */,
  "/triplets" /* GET */,
  "/triplets/{triplet_id}" /* GET */,
  "/triplets/{triplet_id}/footprint" /* GET */,
  "/triplets/{triplet_id}/iirs-overlay" /* GET */,
  "/triplets/{triplet_id}/lro-candidates" /* GET */,
  "/triplets/{triplet_id}/matches" /* GET */,
] as const;
