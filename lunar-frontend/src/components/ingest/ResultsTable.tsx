import React, { useState } from 'react';

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

  if (triplets.length === 0) {
    return (
      <div className="rounded-2xl border border-slate-200/80 bg-white p-12 text-center text-slate-400 text-xs shadow-sm">
        No triplet results yet. Run the pipeline to see results here.
      </div>
    );
  }

  const threshold = containment * 100;

  return (
    <div className="rounded-2xl border border-slate-200/80 bg-white shadow-sm overflow-hidden animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-100 bg-white px-6 py-4">
        <div className="flex items-center gap-2.5">
          <span className="rounded-lg border border-indigo-100 bg-indigo-50 px-2.5 py-0.5 text-[11px] font-bold uppercase tracking-wider text-[#4F46E5]">
            Discovered Triplets
          </span>
          <span className="text-sm font-bold text-slate-900">
            Automated Cross-Match Results
          </span>
        </div>
        <span className="rounded-full bg-slate-100 px-3 py-1 font-mono text-xs font-bold text-slate-600 border border-slate-200">
          {triplets.length} triplet(s)
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-slate-200/80 bg-slate-50 font-sans text-[10px] font-bold uppercase tracking-wider text-slate-500">
              <th className="py-3 px-4">#</th>
              <th className="py-3 px-4">OHRC Product</th>
              <th className="py-3 px-4">TMC-2 Product</th>
              <th className="py-3 px-4">IIRS Product</th>
              <th className="py-3 px-4">Overlap</th>
              <th className="py-3 px-4">Sun El (OHRC)</th>
              <th className="py-3 px-4 text-right">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {triplets.map((t, i) => {
              const overlapPct = t.overlap_triplet_pct ?? 0;
              const pass = overlapPct >= threshold;
              const isExpanded = expandedIdx === i;

              return (
                <React.Fragment key={i}>
                  <tr
                    onClick={() => setExpandedIdx(isExpanded ? null : i)}
                    className={`transition-colors cursor-pointer ${
                      isExpanded ? 'bg-indigo-50/40' : 'hover:bg-slate-50/70'
                    }`}
                  >
                    <td className="py-3.5 px-4 font-mono font-medium text-slate-400">
                      {i + 1}
                    </td>
                    <td className="py-3.5 px-4">
                      <div
                        className="font-mono text-xs font-medium text-slate-800 max-w-[180px] truncate"
                        title={t.ohrc_product_id}
                      >
                        {t.ohrc_product_id || '---'}
                      </div>
                    </td>
                    <td className="py-3.5 px-4">
                      <div
                        className="font-mono text-xs font-medium text-slate-800 max-w-[180px] truncate"
                        title={t.tmc2_product_id}
                      >
                        {t.tmc2_product_id || '---'}
                      </div>
                    </td>
                    <td className="py-3.5 px-4">
                      <div
                        className="font-mono text-xs font-medium text-slate-800 max-w-[180px] truncate"
                        title={t.iirs_product_id}
                      >
                        {t.iirs_product_id || '---'}
                      </div>
                    </td>
                    <td className="py-3.5 px-4">
                      <span
                        className={`font-mono text-xs font-bold ${
                          pass ? 'text-emerald-600' : 'text-rose-600'
                        }`}
                      >
                        {overlapPct.toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-3.5 px-4 font-mono text-slate-600">
                      {fmtAngle(t.ohrc_sun_elevation_deg)}
                    </td>
                    <td className="py-3.5 px-4 text-right">
                      <span
                        className={`inline-flex items-center gap-1 rounded-md border px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider ${
                          pass
                            ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                            : 'border-rose-200 bg-rose-50 text-rose-700'
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
                      <td colSpan={7} className="p-0 border-b border-slate-200 bg-slate-50/60">
                        <div className="p-5 space-y-4">
                          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                            <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-xs">
                              <span className="block text-[10px] font-bold uppercase tracking-wider text-slate-400">
                                OHRC Sun Elevation
                              </span>
                              <span className="font-mono text-xs font-bold text-slate-800">
                                {fmtAngle(t.ohrc_sun_elevation_deg)}
                              </span>
                            </div>

                            <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-xs">
                              <span className="block text-[10px] font-bold uppercase tracking-wider text-slate-400">
                                TMC-2 Sun Elevation
                              </span>
                              <span className="font-mono text-xs font-bold text-slate-800">
                                {fmtAngle(t.tmc2_sun_elevation_deg)}
                              </span>
                            </div>

                            <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-xs">
                              <span className="block text-[10px] font-bold uppercase tracking-wider text-slate-400">
                                IIRS Sun Elevation
                              </span>
                              <span className="font-mono text-xs font-bold text-slate-800">
                                {fmtAngle(t.iirs_sun_elevation_deg)}
                              </span>
                            </div>

                            <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-xs">
                              <span className="block text-[10px] font-bold uppercase tracking-wider text-slate-400">
                                Triplet Overlap
                              </span>
                              <span className="font-mono text-xs font-bold text-[#4F46E5]">
                                {overlapPct.toFixed(2)}%
                              </span>
                            </div>

                            <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-xs">
                              <span className="block text-[10px] font-bold uppercase tracking-wider text-slate-400">
                                OHRC GSD
                              </span>
                              <span className="font-mono text-xs font-bold text-slate-800">
                                {fmtGsd(t.ohrc_gsd_m)}
                              </span>
                            </div>

                            <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-xs">
                              <span className="block text-[10px] font-bold uppercase tracking-wider text-slate-400">
                                TMC-2 GSD
                              </span>
                              <span className="font-mono text-xs font-bold text-slate-800">
                                {fmtGsd(t.tmc2_gsd_m)}
                              </span>
                            </div>

                            <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-xs">
                              <span className="block text-[10px] font-bold uppercase tracking-wider text-slate-400">
                                IIRS GSD
                              </span>
                              <span className="font-mono text-xs font-bold text-slate-800">
                                {fmtGsd(t.iirs_gsd_m)}
                              </span>
                            </div>
                          </div>

                          {t.intersection_wkt && (
                            <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-xs">
                              <span className="block text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1">
                                Intersection Geometry (WKT)
                              </span>
                              <div className="font-mono text-[10px] text-slate-600 break-all max-h-24 overflow-y-auto leading-relaxed">
                                {t.intersection_wkt}
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
