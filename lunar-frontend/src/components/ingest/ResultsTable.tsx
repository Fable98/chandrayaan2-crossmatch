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
      <div className="retro-outset p-8 text-center font-mono text-xs text-[#555C58] dark:text-[#8C9893]">
        NO TRIPLET RESULTS YET. RUN THE PIPELINE TO POPULATE RESULTS.
      </div>
    );
  }

  const threshold = containment * 100;

  return (
    <div className="retro-outset p-1 overflow-hidden animate-fade-in">
      {/* Header */}
      <div className="bg-[#1F4743] text-white px-2.5 py-1 flex items-center justify-between text-xs font-bold font-mono tracking-wider">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400" />
          <span>DISCOVERED TRIPLETS // AUTOMATED CROSS-MATCH RESULTS</span>
        </div>
        <span className="retro-inset px-2 py-0.5 bg-[#143532] text-white font-mono text-[10px] font-bold">
          {rows.length} TRIPLET(S)
        </span>
      </div>
      {excludedCount > 0 && (
        <p className="border-b border-[#8B8579] dark:border-[#2D3835] bg-[#EFE3C6] dark:bg-[#342813] px-3 py-1 font-mono text-[10px] text-amber-900 dark:text-amber-200">
          +{excludedCount} LRO/external row{excludedCount === 1 ? "" : "s"} excluded — this table renders true OHRC+TMC-2+IIRS triplets only.
        </p>
      )}

      <div className="overflow-x-auto bg-[#E7E2D6] dark:bg-[#1A201E]">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b-2 border-[#8B8579] dark:border-[#2D3835] bg-[#DFD9CD] dark:bg-[#161B19] font-mono text-[10px] font-bold uppercase tracking-wider text-[#1E2321] dark:text-[#E7E2D6]">
              <th className="py-2 px-3">#</th>
              <th className="py-2 px-3">OHRC Product</th>
              <th className="py-2 px-3">TMC-2 Product</th>
              <th className="py-2 px-3">IIRS Product</th>
              <th className="py-2 px-3">Overlap</th>
              <th className="py-2 px-3">Sun El (OHRC)</th>
              <th className="py-2 px-3 text-right">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#8B8579]/30 dark:divide-[#2D3835]">
            {rows.map((t, i) => {
              const overlapPct = t.overlap_triplet_pct ?? 0;
              const pass = overlapPct >= threshold;
              const isExpanded = expandedIdx === i;

              return (
                <React.Fragment key={rowKey(t, i)}>
                  <tr
                    onClick={() => setExpandedIdx(isExpanded ? null : i)}
                    className={`transition-colors cursor-pointer font-mono ${
                      isExpanded
                        ? 'bg-[#28557E] text-white font-bold'
                        : 'hover:bg-[#D5CFC1] dark:hover:bg-white/5 text-[#1E2321] dark:text-[#E7E2D6]'
                    }`}
                  >
                    <td className="py-2 px-3 font-mono text-[10px]">
                      {i + 1}
                    </td>
                    <td className="py-2 px-3">
                      <div
                        className="text-xs max-w-[180px] truncate"
                        title={t.ohrc_product_id}
                      >
                        {t.ohrc_product_id || '---'}
                      </div>
                    </td>
                    <td className="py-2 px-3">
                      <div
                        className="text-xs max-w-[180px] truncate"
                        title={t.tmc2_product_id}
                      >
                        {t.tmc2_product_id || '---'}
                      </div>
                    </td>
                    <td className="py-2 px-3">
                      <div
                        className="text-xs max-w-[180px] truncate"
                        title={t.iirs_product_id}
                      >
                        {t.iirs_product_id || '---'}
                      </div>
                    </td>
                    <td className="py-2 px-3">
                      <span
                        className={`text-xs font-bold ${
                          isExpanded
                            ? 'text-white'
                            : pass
                            ? 'text-emerald-700 dark:text-emerald-400'
                            : 'text-rose-700 dark:text-rose-400'
                        }`}
                      >
                        {overlapPct.toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-2 px-3 text-[11px]">
                      {fmtAngle(t.ohrc_sun_elevation_deg)}
                    </td>
                    <td className="py-2 px-3 text-right">
                      <span
                        className={`inline-flex items-center gap-1 retro-inset px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider ${
                          pass
                            ? 'bg-[#D3E8D7] dark:bg-[#193A24] text-[#134E26] dark:text-[#88D49E] border-[#7BB887]'
                            : 'bg-[#EED2D2] dark:bg-[#3D1A1A] text-[#7A1D1D] dark:text-[#E89898]'
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
                      <td colSpan={7} className="p-2 border-b-2 border-[#8B8579] dark:border-[#2D3835] bg-[#DFD9CD] dark:bg-[#141817]">
                        <div className="space-y-3 p-1">
                          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                            <div className="retro-inset p-2 bg-[#E7E2D6] dark:bg-[#1A201E]">
                              <span className="block text-[9px] font-mono font-bold uppercase tracking-wider text-[#555C58] dark:text-[#8C9893]">
                                OHRC Sun Elevation
                              </span>
                              <span className="font-mono text-xs font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                                {fmtAngle(t.ohrc_sun_elevation_deg)}
                              </span>
                            </div>

                            <div className="retro-inset p-2 bg-[#E7E2D6] dark:bg-[#1A201E]">
                              <span className="block text-[9px] font-mono font-bold uppercase tracking-wider text-[#555C58] dark:text-[#8C9893]">
                                TMC-2 Sun Elevation
                              </span>
                              <span className="font-mono text-xs font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                                {fmtAngle(t.tmc2_sun_elevation_deg)}
                              </span>
                            </div>

                            <div className="retro-inset p-2 bg-[#E7E2D6] dark:bg-[#1A201E]">
                              <span className="block text-[9px] font-mono font-bold uppercase tracking-wider text-[#555C58] dark:text-[#8C9893]">
                                IIRS Sun Elevation
                              </span>
                              <span className="font-mono text-xs font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                                {fmtAngle(t.iirs_sun_elevation_deg)}
                              </span>
                            </div>

                            <div className="retro-inset p-2 bg-[#E7E2D6] dark:bg-[#1A201E]">
                              <span className="block text-[9px] font-mono font-bold uppercase tracking-wider text-[#555C58] dark:text-[#8C9893]">
                                Triplet Overlap
                              </span>
                              <span className="font-mono text-xs font-bold text-[#28557E] dark:text-cyan-400">
                                {overlapPct.toFixed(2)}%
                              </span>
                            </div>

                            <div className="retro-inset p-2 bg-[#E7E2D6] dark:bg-[#1A201E]">
                              <span className="block text-[9px] font-mono font-bold uppercase tracking-wider text-[#555C58] dark:text-[#8C9893]">
                                OHRC Native GSD
                              </span>
                              <span className="font-mono text-xs font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                                {fmtGsd(t.ohrc_gsd_m)}
                              </span>
                            </div>

                            <div className="retro-inset p-2 bg-[#E7E2D6] dark:bg-[#1A201E]">
                              <span className="block text-[9px] font-mono font-bold uppercase tracking-wider text-[#555C58] dark:text-[#8C9893]">
                                TMC-2 Native GSD
                              </span>
                              <span className="font-mono text-xs font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                                {fmtGsd(t.tmc2_gsd_m)}
                              </span>
                            </div>

                            <div className="retro-inset p-2 bg-[#E7E2D6] dark:bg-[#1A201E]">
                              <span className="block text-[9px] font-mono font-bold uppercase tracking-wider text-[#555C58] dark:text-[#8C9893]">
                                IIRS Native GSD
                              </span>
                              <span className="font-mono text-xs font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                                {fmtGsd(t.iirs_gsd_m)}
                              </span>
                            </div>
                          </div>

                          {t.intersection_wkt && (
                            <div className="retro-inset p-2 bg-[#E7E2D6] dark:bg-[#1A201E]">
                              <span className="block text-[9px] font-mono font-bold uppercase tracking-wider text-[#555C58] dark:text-[#8C9893] mb-1">
                                Intersection Geometry (WKT)
                              </span>
                              <div className="font-mono text-[10px] text-[#222] dark:text-[#D5D0C3] break-all max-h-24 overflow-y-auto leading-relaxed select-text">
                                {t.intersection_wkt}
                              </div>
                            </div>
                          )}

                          {t.region_id && (
                            <div className="retro-outset p-2 bg-[#E7E2D6] dark:bg-[#1A201E] space-y-2">
                              <div className="flex items-center justify-between flex-wrap gap-2">
                                <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-[#1F4743] dark:text-emerald-400">
                                  REGION BUNDLE // {t.region_id}
                                </span>
                                <div className="flex items-center gap-1.5 flex-wrap">
                                  <Link
                                    href={`/?view=console&region=${t.region_id}`}
                                    prefetch={false}
                                    className="retro-button-primary px-2 py-0.5 text-[10px] font-mono font-bold"
                                  >
                                    OPEN DASHBOARD →
                                  </Link>
                                  <Link
                                    href={`/?view=console&subview=linked-cursor&region=${t.region_id}`}
                                    prefetch={false}
                                    className="retro-button px-2 py-0.5 text-[10px] font-mono font-bold text-[#1E2321] dark:text-[#E7E2D6]"
                                  >
                                    ⊙ LINKED CURSOR
                                  </Link>
                                  <Link
                                    href={`/?view=console&subview=map&region=${t.region_id}`}
                                    prefetch={false}
                                    className="retro-button px-2 py-0.5 text-[10px] font-mono font-bold text-[#1E2321] dark:text-[#E7E2D6]"
                                  >
                                    ☵ PLANETARY MAP
                                  </Link>
                                </div>
                              </div>

                              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
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
                                    className="retro-outset p-1 flex flex-col bg-[#ECE7DC] dark:bg-[#161B19] group hover:border-[#28557E]"
                                  >
                                    <div className="relative h-24 overflow-hidden bg-black retro-inset">
                                      {/* eslint-disable-next-line @next/next/no-img-element */}
                                      <img
                                        src={imageUrl(`/images/${sensor}/${t.region_id}`)}
                                        alt={`${t.region_id} ${sensor}`}
                                        className="h-full w-full object-cover group-hover:scale-105 transition-transform"
                                        loading="lazy"
                                        onError={(e) => {
                                          (e.currentTarget as HTMLImageElement).style.display = 'none';
                                        }}
                                      />
                                      <div className="absolute top-1 right-1 retro-inset px-1 py-0.2 font-mono text-[8px] bg-black text-white">
                                        VIEW ↗
                                      </div>
                                    </div>
                                    <div className="pt-1.5 flex items-center justify-between font-mono text-[10px]">
                                      <span className="font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                                        {label}
                                      </span>
                                      <span className="text-[#555C58] dark:text-[#8C9893]">
                                        {gsd}
                                      </span>
                                    </div>
                                  </a>
                                ))}
                              </div>

                              <div className="flex items-center gap-1.5 flex-wrap font-mono text-[9px] pt-1">
                                <span className="retro-inset px-2 py-0.5 font-bold uppercase bg-[#ECE7DC] dark:bg-[#161B19] text-[#1E2321] dark:text-[#E7E2D6]">
                                  match: {t.matching?.status || 'pending'}
                                  {typeof t.matching?.inlier_count === 'number' && ` · n=${t.matching.inlier_count}`}
                                  {typeof t.matching?.fit_rmse_px === 'number' && ` · ${t.matching.fit_rmse_px.toFixed(2)}px`}
                                </span>
                                <span className="retro-inset px-2 py-0.5 font-bold uppercase bg-[#ECE7DC] dark:bg-[#161B19] text-[#1E2321] dark:text-[#E7E2D6]">
                                  reg-qa: {t.registration?.status || 'pending'}
                                </span>
                                <TrafficLightBadge
                                  color={t.matching?.traffic_light_color ?? t.registration?.traffic_light_color ?? null}
                                  confidence_score={t.matching?.confidence_score ?? t.registration?.confidence_score ?? null}
                                  ssim_score={t.matching?.ssim_score ?? t.registration?.ssim_score ?? null}
                                  held_out_rmse={t.matching?.held_out_rmse ?? t.registration?.held_out_rmse ?? null}
                                />
                                <a
                                  className="retro-button px-2 py-0.5 font-semibold text-[#1E2321] dark:text-[#E7E2D6]"
                                  href={imageUrl(`/images/registered/${t.region_id}/checkerboard_qa.png`)}
                                  target="_blank"
                                  rel="noreferrer"
                                >
                                  <span>▦ Checkerboard QA ↗</span>
                                </a>
                                <a
                                  className="retro-button px-2 py-0.5 font-semibold text-[#1E2321] dark:text-[#E7E2D6]"
                                  href={imageUrl(`/images/registered/${t.region_id}/blend_overlay.png`)}
                                  target="_blank"
                                  rel="noreferrer"
                                >
                                  <span>◈ Blend Overlay ↗</span>
                                </a>
                                <a
                                  className="retro-button px-2 py-0.5 font-semibold text-[#1E2321] dark:text-[#E7E2D6]"
                                  href={imageUrl(`/images/registered/${t.region_id}/displacement_quiver.png`)}
                                  target="_blank"
                                  rel="noreferrer"
                                >
                                  <span>➔ Vector Quiver ↗</span>
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
