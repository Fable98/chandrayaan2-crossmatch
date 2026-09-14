/**
 * TrafficLightBadge.tsx — Epic 3 objective-verification badge (Task 5).
 *
 * Renders the backend traffic-light verdict (metrics `traffic_light_color`,
 * `confidence_score`, `ssim_score`) wherever registration metrics are shown.
 * Null-safe: unknown/missing color renders a neutral "UNVERIFIED" chip, never
 * crashes, never invents a verdict.
 */

export type TrafficLightColor = "GREEN" | "YELLOW" | "RED";

export interface TrafficLightBadgeProps {
  color?: TrafficLightColor | string | null;
  confidence_score?: number | null;
  ssim_score?: number | null;
  held_out_rmse?: number | null;
}

const STYLE: Record<TrafficLightColor, string> = {
  GREEN: "border-emerald-200 bg-emerald-50 text-emerald-700",
  YELLOW: "border-amber-200 bg-amber-50 text-amber-700",
  RED: "border-rose-200 bg-rose-50 text-rose-700",
};

const LABEL: Record<TrafficLightColor, string> = {
  GREEN: "Photogrammetric Grade",
  YELLOW: "Acceptable / Review",
  RED: "Rejected / Low Confidence",
};

function fmtPct(v: number | null | undefined): string | null {
  if (v === null || v === undefined || !Number.isFinite(Number(v))) return null;
  return `${Math.round(Number(v))}%`;
}

function fmtSsim(v: number | null | undefined): string | null {
  if (v === null || v === undefined || !Number.isFinite(Number(v))) return null;
  return Number(v).toFixed(2);
}

function fmtRmse(v: number | null | undefined): string | null {
  if (v === null || v === undefined || !Number.isFinite(Number(v))) return null;
  return `${Number(v).toFixed(2)}px`;
}

export default function TrafficLightBadge({
  color,
  confidence_score,
  ssim_score,
  held_out_rmse,
}: TrafficLightBadgeProps) {
  const known =
    color === "GREEN" || color === "YELLOW" || color === "RED" ? color : null;
  if (!known) {
    return (
      <span
        className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-white px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-slate-400"
        title="No objective verification available for this run"
      >
        <span className="h-1.5 w-1.5 rounded-full bg-current" />
        Unverified
      </span>
    );
  }
  const conf = fmtPct(confidence_score);
  const ssim = fmtSsim(ssim_score);
  const held = fmtRmse(held_out_rmse);
  const tipParts = [`Traffic light: ${known}`, `Verdict: ${LABEL[known]}`];
  if (conf !== null) tipParts.push(`Confidence: ${conf}`);
  if (ssim !== null) tipParts.push(`SSIM: ${ssim}`);
  if (held !== null) tipParts.push(`Held-out RMSE: ${held}`);
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-md border px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider ${STYLE[known]}`}
      title={tipParts.join(" · ")}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {known}
      {conf !== null ? ` · ${conf}` : ""}
      <span className="font-medium normal-case tracking-normal opacity-80">
        {LABEL[known]}
      </span>
    </span>
  );
}
