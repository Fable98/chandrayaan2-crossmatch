import React, { useState } from 'react';
import Link from 'next/link';
import { imageUrl } from '@/lib/api';
import TrafficLightBadge from '@/components/TrafficLightBadge';

interface TripletResult {
  [key: string]: any;
}

interface ResultsTableProps {
  triplets: TripletResult[];
  containment: number;
}

function fmtAngle(val: any): string {
  if (val === null || val === undefined) return 'N/A';
  return `${Number(val).toFixed(1)}°`;
}

function fmtGsd(val: any): string {
  if (val === null || val === undefined) return 'N/A';
  return `${Number(val).toFixed(2)} m`;
}

export default function ResultsTable({ triplets, containment }: ResultsTableProps) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  // user_triplets.json is a mixed manifest: older external_LRO_NAC rows have
  // no tmc2/iirs ids or overlap. Only true OHRC+TMC-2+IIRS triplets belong here.
  const rows = (triplets || []).filter(
    (t) =>
      t &&
      t.reference_type !== 'external_LRO_NAC' &&
      t.tmc2_product_id &&
      t.iirs_product_id
  );
  // Honest accounting: the filter above must never silently shrink the run.
  const excludedCount = (triplets || []).length - rows.length;

  const rowKey = (t: TripletResult, i: number): string =>
    String(t.region_id ?? t.triplet_id ?? t.id ?? `row-${i}`);

  if (rows.length === 0) {
    return (
      <div className="rounded-2xl border border-slate-200/80 dark:border-[#1b2029] bg-white dark:bg-[#0e1117] p-12 text-center text-slate-400 text-xs shadow-sm">
        No triplet results yet. Run the pipeline to see results here.
      </div>
    );
  }

  const threshold = containment * 100;

  return (
    <div className="rounded-2xl border border-slate-200/80 dark:border-[#1b2029] bg-white dark:bg-[#0e1117] shadow-sm overflow-hidden animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-100 dark:border-[#1b2029] bg-white dark:bg-[#0e1117] px-6 py-4">
        <div className="flex items-center gap-2.5">
          <span className="rounded-lg border border-indigo-100 dark:border-indigo-900/50 bg-indigo-50 dark:bg-indigo-950/30 px-2.5 py-0.5 text-[11px] font-bold uppercase tracking-wider text-[#4F46E5] dark:text-indigo-300">
            Discovered Triplets
          </span>
          <span className="text-sm font-bold text-slate-900 dark:text-slate-100">
            Automated Cross-Match Results
          </span>
        </div>
        <span className="rounded-full bg-slate-100 dark:bg-white/10 px-3 py-1 font-mono text-xs font-bold text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-[#1b2029]">
          {rows.length} triplet(s)
        </span>
      </div>
      {excludedCount > 0 && (
        <p className="border-b border-slate-100 dark:border-[#1b2029] bg-amber-50/60 dark:bg-amber-950/20 px-6 py-2 font-mono text-[11px] text-amber-700 dark:text-amber-300">
          +{excludedCount} LRO/external row{excludedCount === 1 ? "" : "s"} excluded — this table renders true OHRC+TMC-2+IIRS triplets only.
        </p>
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-slate-200/80 dark:border-[#1b2029] bg-slate-50 dark:bg-white/5 font-sans text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              <th className="py-3 px-4">#</th>
              <th className="py-3 px-4">OHRC Product</th>
              <th className="py-3 px-4">TMC-2 Product</th>
              <th className="py-3 px-4">IIRS Product</th>
              <th className="py-3 px-4">Overlap</th>
              <th className="py-3 px-4">Sun El (OHRC)</th>
              <th className="py-3 px-4 text-right">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-[#1b2029]">
            {rows.map((t, i) => {
              const overlapPct = t.overlap_triplet_pct ?? 0;
              const pass = overlapPct >= threshold;
              const isExpanded = expandedIdx === i;

              return (
                <React.Fragment key={rowKey(t, i)}>
                  <tr
                    onClick={() => setExpandedIdx(isExpanded ? null : i)}
                    className={`transition-colors cursor-pointer ${
                      isExpanded ? 'bg-indigo-50/40 dark:bg-indigo-950/40' : 'hover:bg-slate-50/70 dark:hover:bg-white/5'
                    }`}
                  >
                    <td className="py-3.5 px-4 font-mono font-medium text-slate-400">
                      {i + 1}
                    </td>
                    <td className="py-3.5 px-4">
                      <div
                        className="font-mono text-xs font-medium text-slate-800 dark:text-slate-200 max-w-[180px] truncate"
                        title={t.ohrc_product_id}
                      >
                        {t.ohrc_product_id || '---'}
                      </div>
                    </td>
                    <td className="py-3.5 px-4">
                      <div
                        className="font-mono text-xs font-medium text-slate-800 dark:text-slate-200 max-w-[180px] truncate"
                        title={t.tmc2_product_id}
                      >
                        {t.tmc2_product_id || '---'}
                      </div>
                    </td>
                    <td className="py-3.5 px-4">
                      <div
                        className="font-mono text-xs font-medium text-slate-800 dark:text-slate-200 max-w-[180px] truncate"
                        title={t.iirs_product_id}
                      >
                        {t.iirs_product_id || '---'}
                      </div>
                    </td>
                    <td className="py-3.5 px-4">
                      <span
                        className={`font-mono text-xs font-bold ${
                          pass ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'
                        }`}
                      >
                        {overlapPct.toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-3.5 px-4 font-mono text-slate-600 dark:text-slate-400">
                      {fmtAngle(t.ohrc_sun_elevation_deg)}
                    </td>
                    <td className="py-3.5 px-4 text-right">
                      <span
                        className={`inline-flex items-center gap-1 rounded-md border px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider ${
                          pass
                            ? 'border-emerald-200 dark:border-emerald-900/50 bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-300'
                            : 'border-rose-200 dark:border-rose-900/50 bg-rose-50 dark:bg-rose-950/30 text-rose-700 dark:text-rose-300'
                        }`}
                      >
                        <span className="h-1.5 w-1.5 rounded-full bg-current" />
                        {pass ? 'PASS' : 'FAIL'}
                      </span>
                    </td>
                  </tr>

                  {/* Expanded detail row */}
                  {isExpanded && (
                    <tr>
                      <td colSpan={7} className="p-0 border-b border-slate-200 dark:border-[#1b2029] bg-slate-50/60 dark:bg-white/5">
                        <div className="p-5 space-y-4">
                          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                            <div className="rounded-xl border border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 p-3 shadow-xs">
                              <span className="block text-[10px] font-bold uppercase tracking-wider text-slate-400">
                                OHRC Sun Elevation
                              </span>
                              <span className="font-mono text-xs font-bold text-slate-800 dark:text-slate-200">
                                {fmtAngle(t.ohrc_sun_elevation_deg)}
                              </span>
                            </div>

                            <div className="rounded-xl border border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 p-3 shadow-xs">
                              <span className="block text-[10px] font-bold uppercase tracking-wider text-slate-400">
                                TMC-2 Sun Elevation
                              </span>
                              <span className="font-mono text-xs font-bold text-slate-800 dark:text-slate-200">
                                {fmtAngle(t.tmc2_sun_elevation_deg)}
                              </span>
                            </div>

                            <div className="rounded-xl border border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 p-3 shadow-xs">
                              <span className="block text-[10px] font-bold uppercase tracking-wider text-slate-400">
                                IIRS Sun Elevation
                              </span>
                              <span className="font-mono text-xs font-bold text-slate-800 dark:text-slate-200">
                                {fmtAngle(t.iirs_sun_elevation_deg)}
                              </span>
                            </div>

                            <div className="rounded-xl border border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 p-3 shadow-xs">
                              <span className="block text-[10px] font-bold uppercase tracking-wider text-slate-400">
                                Triplet Overlap
                              </span>
                              <span className="font-mono text-xs font-bold text-[#4F46E5]">
                                {overlapPct.toFixed(2)}%
                              </span>
                            </div>

                            <div className="rounded-xl border border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 p-3 shadow-xs">
                              <span className="block text-[10px] font-bold uppercase tracking-wider text-slate-400">
                                OHRC Native GSD
                              </span>
                              <span className="font-mono text-xs font-bold text-slate-800 dark:text-slate-200">
                                {fmtGsd(t.ohrc_gsd_m)}
                              </span>
                            </div>

                            <div className="rounded-xl border border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 p-3 shadow-xs">
                              <span className="block text-[10px] font-bold uppercase tracking-wider text-slate-400">
                                TMC-2 Native GSD
                              </span>
                              <span className="font-mono text-xs font-bold text-slate-800 dark:text-slate-200">
                                {fmtGsd(t.tmc2_gsd_m)}
                              </span>
                            </div>

                            <div className="rounded-xl border border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 p-3 shadow-xs">
                              <span className="block text-[10px] font-bold uppercase tracking-wider text-slate-400">
                                IIRS Native GSD
                              </span>
                              <span className="font-mono text-xs font-bold text-slate-800 dark:text-slate-200">
                                {fmtGsd(t.iirs_gsd_m)}
                              </span>
                            </div>
                          </div>

                          {t.intersection_wkt && (
                            <div className="rounded-xl border border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 p-3 shadow-xs">
                              <span className="block text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1">
                                Intersection Geometry (WKT)
                              </span>
                              <div className="font-mono text-[10px] text-slate-600 dark:text-slate-400 break-all max-h-24 overflow-y-auto leading-relaxed">
                                {t.intersection_wkt}
                              </div>
                            </div>
                          )}

                          {/* Data-region parity bundle: 512 crops, linked-cursor
                              matches, cross-grid QA — same views as curated regions */}
                          {t.region_id && (
                            <div className="rounded-xl border border-indigo-100 dark:border-indigo-900 bg-indigo-50/40 dark:bg-indigo-950/40 p-3 shadow-xs space-y-3">
                              <div className="flex items-center justify-between flex-wrap gap-2">
                                <span className="text-[10px] font-bold uppercase tracking-wider text-[#4F46E5]">
                                  Region bundle · {t.region_id}
                                </span>
                                <span className="flex items-center gap-1.5 flex-wrap">
                                  <Link
                                    href={`/?view=console&region=${t.region_id}`}
                                    prefetch={false}
                                    className="rounded-lg bg-[#4F46E5] px-2.5 py-1 text-[10px] font-bold text-white hover:bg-[#4338CA] transition"
                                  >
                                    Open Dashboard →
                                  </Link>
                                  <Link
                                    href={`/?view=console&subview=linked-cursor&region=${t.region_id}`}
                                    prefetch={false}
                                    className="rounded-lg border border-indigo-200 dark:border-indigo-800 bg-white dark:bg-white/5 px-2.5 py-1 text-[10px] font-bold text-[#4F46E5] dark:text-indigo-300 hover:bg-indigo-50 dark:hover:bg-white/10 transition"
                                  >
                                    ⊙ Linked Cursor
                                  </Link>
                                  <Link
                                    href={`/?view=console&subview=map&region=${t.region_id}`}
                                    prefetch={false}
                                    className="rounded-lg border border-indigo-200 dark:border-indigo-800 bg-white dark:bg-white/5 px-2.5 py-1 text-[10px] font-bold text-[#4F46E5] dark:text-indigo-300 hover:bg-indigo-50 dark:hover:bg-white/10 transition"
                                  >
                                    ☵ Planetary Map
                                  </Link>
                                </span>
                              </div>

                              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                                {[
                                  { sensor: 'ohrc', label: 'OHRC Primary', gsd: '0.25 m/px' },
                                  { sensor: 'tmc', label: 'TMC-2 Stereo', gsd: '5.0 m/px' },
                                  { sensor: 'iirs', label: 'IIRS Hyperspectral', gsd: '80 m/px' },
                                ].map(({ sensor, label, gsd }) => (
                                  <a
                                    key={sensor}
                                    href={imageUrl(`/images/${sensor}/${t.region_id}`)}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="group flex flex-col rounded-xl border border-slate-200/80 dark:border-[#1b2029] overflow-hidden bg-slate-50 dark:bg-[#0e1117] transition-all duration-300 hover:shadow-lg hover:border-indigo-400/50 dark:hover:border-indigo-500/40"
                                  >
                                    <div className="relative h-28 overflow-hidden bg-black">
                                      {/* eslint-disable-next-line @next/next/no-img-element */}
                                      <img
                                        src={imageUrl(`/images/${sensor}/${t.region_id}`)}
                                        alt={`${t.region_id} ${sensor}`}
                                        className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105"
                                        loading="lazy"
                                        onError={(e) => {
                                          (e.currentTarget as HTMLImageElement).style.display = 'none';
                                        }}
                                      />
                                      <div className="absolute top-2 right-2 rounded-md bg-black/60 backdrop-blur-md px-1.5 py-0.5 font-mono text-[9px] text-white/80 border border-white/10 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                                        VIEW ↗
                                      </div>
                                    </div>
                                    <div className="p-2.5 bg-white dark:bg-[#0e1117] border-t border-slate-100 dark:border-[#1b2029] flex items-center justify-between">
                                      <span className="text-xs font-bold text-slate-900 dark:text-slate-100 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                                        {label}
                                      </span>
                                      <span className="text-[10px] font-mono text-slate-500 dark:text-slate-400">
                                        {gsd}
                                      </span>
                                    </div>
                                  </a>
                                ))}
                              </div>

                              <div className="flex items-center gap-2 flex-wrap font-mono text-[10px] text-slate-600 dark:text-slate-400 pt-1">
                                <span
                                  className={`inline-flex items-center gap-1 rounded-md border px-2.5 py-1 font-bold uppercase tracking-wider ${
                                    t.matching?.status === 'success'
                                      ? 'border-emerald-200 dark:border-emerald-900/50 bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-300'
                                      : 'border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 text-slate-500 dark:text-slate-400'
                                  }`}
                                >
                                  match: {t.matching?.status || 'pending'}
                                  {typeof t.matching?.inlier_count === 'number' &&
                                    ` · n=${t.matching.inlier_count}`}
                                  {typeof t.matching?.fit_rmse_px === 'number' &&
                                    ` · ${t.matching.fit_rmse_px.toFixed(2)}px`}
                                </span>
                                <span
                                  className={`inline-flex items-center gap-1 rounded-md border px-2.5 py-1 font-bold uppercase tracking-wider ${
                                    t.registration?.status === 'success'
                                      ? 'border-emerald-200 dark:border-emerald-900/50 bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-300'
                                      : 'border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 text-slate-500 dark:text-slate-400'
                                  }`}
                                >
                                  reg-qa: {t.registration?.status || 'pending'}
                                </span>
                                <TrafficLightBadge
                                  color={
                                    t.matching?.traffic_light_color ??
                                    t.registration?.traffic_light_color ??
                                    null
                                  }
                                  confidence_score={
                                    t.matching?.confidence_score ??
                                    t.registration?.confidence_score ??
                                    null
                                  }
                                  ssim_score={
                                    t.matching?.ssim_score ??
                                    t.registration?.ssim_score ??
                                    null
                                  }
                                  held_out_rmse={
                                    t.matching?.held_out_rmse ??
                                    t.registration?.held_out_rmse ??
                                    null
                                  }
                                />
                                <a
                                  className="inline-flex items-center gap-1 rounded-lg border border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 px-2.5 py-1 text-[10px] font-semibold text-slate-700 dark:text-slate-300 hover:border-indigo-300 dark:hover:border-indigo-700 hover:text-indigo-600 dark:hover:text-indigo-300 hover:scale-[1.02] active:scale-[0.98] transition-all shadow-xs"
                                  href={imageUrl(`/images/registered/${t.region_id}/checkerboard_qa.png`)}
                                  target="_blank"
                                  rel="noreferrer"
                                >
                                  <span>▦</span>
                                  <span>Checkerboard QA ↗</span>
                                </a>
                                <a
                                  className="inline-flex items-center gap-1 rounded-lg border border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 px-2.5 py-1 text-[10px] font-semibold text-slate-700 dark:text-slate-300 hover:border-indigo-300 dark:hover:border-indigo-700 hover:text-indigo-600 dark:hover:text-indigo-300 hover:scale-[1.02] active:scale-[0.98] transition-all shadow-xs"
                                  href={imageUrl(`/images/registered/${t.region_id}/blend_overlay.png`)}
                                  target="_blank"
                                  rel="noreferrer"
                                >
                                  <span>◈</span>
                                  <span>Blend Overlay ↗</span>
                                </a>
                                <a
                                  className="inline-flex items-center gap-1 rounded-lg border border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 px-2.5 py-1 text-[10px] font-semibold text-slate-700 dark:text-slate-300 hover:border-indigo-300 dark:hover:border-indigo-700 hover:text-indigo-600 dark:hover:text-indigo-300 hover:scale-[1.02] active:scale-[0.98] transition-all shadow-xs"
                                  href={imageUrl(`/images/registered/${t.region_id}/displacement_quiver.png`)}
                                  target="_blank"
                                  rel="noreferrer"
                                >
                                  <span>➔</span>
                                  <span>Vector Quiver ↗</span>
                                </a>
                              </div>
                            </div>
                          )}
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
