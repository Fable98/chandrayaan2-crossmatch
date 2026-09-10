/**
 * API service for the ingest pipeline.
 *
 * Step 13 contract: single API base imported from ./api (no second base
 * URL), and every ingest call carries the operator's Bearer token.
 */

import { API_BASE as SINGLE_API_BASE } from './api';
import { getAuthHeaders } from './auth';

const API_BASE = `${SINGLE_API_BASE}/api/ingest`;

export { SINGLE_API_BASE as API_BASE_ROOT };

export interface IngestConfig {
  containment: number;
  tileSize: number;
  noLargeAoi: boolean;
  noInvariants: boolean;
  maxTimeGapDays: number | null;
  requireDates: boolean;
}

export interface JobStatus {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  stage: string;
  progress_pct: number;
  started_at: string | null;
  completed_at: string | null;
  log_lines: string[];
  error: string | null;
}

export interface JobResult {
  job_id: string;
  status: string;
  triplets: Record<string, any>[];
  summary: string;
  output_dir: string;
}

export const DEFAULT_CONFIG: IngestConfig = {
  containment: 0.8,
  tileSize: 512,
  noLargeAoi: false,
  noInvariants: false,
  maxTimeGapDays: null,
  requireDates: false,
};

/**
 * Upload zip files and start the ingest pipeline.
 */
export async function uploadZips(
  files: File[],
  config: IngestConfig = DEFAULT_CONFIG
): Promise<{ job_id: string; files_uploaded: number }> {
  const form = new FormData();

  for (const file of files) {
    form.append('files', file);
  }
  form.append('containment', String(config.containment));
  form.append('tile_size', String(config.tileSize));
  form.append('no_large_aoi', String(config.noLargeAoi));
  form.append('no_invariants', String(config.noInvariants));
  if (config.maxTimeGapDays !== null) {
    form.append('max_time_gap_days', String(config.maxTimeGapDays));
  }
  form.append('require_dates', String(config.requireDates));

  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    headers: { ...getAuthHeaders() },
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Upload failed');
  }

  return res.json();
}

/**
 * Poll the status of a running ingest job.
 */
export async function pollStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/status/${jobId}`, {
    headers: { ...getAuthHeaders() },
  });
  if (!res.ok) {
    throw new Error(`Status check failed: ${res.statusText}`);
  }
  return res.json();
}

/**
 * Get the final results of a completed ingest job.
 */
export async function getResults(jobId: string): Promise<JobResult> {
  const res = await fetch(`${API_BASE}/results/${jobId}`, {
    headers: { ...getAuthHeaders() },
  });
  if (!res.ok) {
    throw new Error(`Results fetch failed: ${res.statusText}`);
  }
  return res.json();
}
