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
  GREEN:
    "border-emerald-200 dark:border-emerald-900/50 bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-300",
  YELLOW:
    "border-amber-200 dark:border-amber-900/50 bg-amber-50 dark:bg-amber-950/30 text-amber-700 dark:text-amber-300",
  RED: "border-rose-200 dark:border-rose-900/50 bg-rose-50 dark:bg-rose-950/30 text-rose-700 dark:text-rose-300",
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
        className="inline-flex items-center gap-1 rounded-md border border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-400"
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
      <span className="relative flex h-2 w-2">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 bg-current" />
        <span className="relative inline-flex rounded-full h-2 w-2 bg-current" />
      </span>
      {known}
      {conf !== null ? ` · ${conf}` : ""}
      <span className="font-medium normal-case tracking-normal opacity-85">
        {LABEL[known]}
      </span>
    </span>
  );
}
