import React, { useState } from 'react';

interface TripletResult {
  [key: string]: any;
}

interface ResultsTableProps {
  triplets: TripletResult[];
  containment: number;
}

const styles: Record<string, React.CSSProperties> = {
  wrapper: {
    borderRadius: 'var(--radius-lg)',
    background: 'var(--bg-glass)',
    backdropFilter: 'blur(20px)',
    border: '1px solid var(--border-subtle)',
    boxShadow: 'var(--shadow-md)',
    overflow: 'hidden',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '20px 24px',
    borderBottom: '1px solid var(--border-subtle)',
  },
  headerTitle: {
    fontSize: '1rem',
    fontWeight: 700,
    color: 'var(--text-primary)',
  },
  headerCount: {
    fontSize: '0.8rem',
    color: 'var(--text-muted)',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse' as const,
    fontSize: '0.8rem',
  },
  th: {
    padding: '12px 16px',
    textAlign: 'left' as const,
    fontWeight: 600,
    fontSize: '0.7rem',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.06em',
    color: 'var(--text-muted)',
    borderBottom: '1px solid var(--border-subtle)',
    background: 'rgba(0,0,0,0.15)',
    whiteSpace: 'nowrap' as const,
  },
  td: {
    padding: '12px 16px',
    borderBottom: '1px solid var(--border-subtle)',
    color: 'var(--text-secondary)',
    verticalAlign: 'top' as const,
  },
  row: {
    transition: 'background 150ms ease',
    cursor: 'pointer',
  },
  productId: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.7rem',
    color: 'var(--text-muted)',
    maxWidth: '180px',
    overflow: 'hidden' as const,
    textOverflow: 'ellipsis' as const,
    whiteSpace: 'nowrap' as const,
  },
  detailPanel: {
    padding: '16px 24px',
    background: 'rgba(0, 0, 0, 0.1)',
    borderBottom: '1px solid var(--border-subtle)',
  },
  detailGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
    gap: '12px',
  },
  detailItem: {
    fontSize: '0.75rem',
  },
  detailLabel: {
    color: 'var(--text-muted)',
    fontWeight: 600,
    marginBottom: '2px',
  },
  detailValue: {
    fontFamily: 'var(--font-mono)',
    color: 'var(--text-secondary)',
  },
  empty: {
    textAlign: 'center' as const,
    padding: '48px 24px',
    color: 'var(--text-muted)',
    fontSize: '0.9rem',
  },
};

function fmtAngle(val: any): string {
  if (val === null || val === undefined) return 'N/A';
  return `${Number(val).toFixed(1)}deg`;
}

function fmtGsd(val: any): string {
  if (val === null || val === undefined) return 'N/A';
  return `${Number(val).toFixed(2)} m`;
}

export default function ResultsTable({ triplets, containment }: ResultsTableProps) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  if (triplets.length === 0) {
    return (
      <div style={styles.wrapper}>
        <div style={styles.empty}>
          No triplet results yet. Run the pipeline to see results here.
        </div>
      </div>
    );
  }

  const threshold = containment * 100;

  return (
    <div style={styles.wrapper} className="animate-fade-in">
      <div style={styles.header}>
        <span style={styles.headerTitle}>Discovered Triplets</span>
        <span style={styles.headerCount}>{triplets.length} triplet(s)</span>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.th}>#</th>
              <th style={styles.th}>OHRC Product</th>
              <th style={styles.th}>TMC-2 Product</th>
              <th style={styles.th}>IIRS Product</th>
              <th style={styles.th}>Overlap</th>
              <th style={styles.th}>Sun El (OHRC)</th>
              <th style={styles.th}>Status</th>
            </tr>
          </thead>
          <tbody>
            {triplets.map((t, i) => {
              const overlapPct = t.overlap_triplet_pct ?? 0;
              const pass = overlapPct >= threshold;
              const isExpanded = expandedIdx === i;

              return (
                <React.Fragment key={i}>
                  <tr
                    style={{
                      ...styles.row,
                      background: isExpanded ? 'rgba(99,102,241,0.05)' : undefined,
                    }}
                    onClick={() => setExpandedIdx(isExpanded ? null : i)}
                    onMouseEnter={(e) => {
                      if (!isExpanded) {
                        (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.02)';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!isExpanded) {
                        (e.currentTarget as HTMLElement).style.background = '';
                      }
                    }}
                  >
                    <td style={styles.td}>{i + 1}</td>
                    <td style={styles.td}>
                      <div style={styles.productId} title={t.ohrc_product_id}>
                        {t.ohrc_product_id || '---'}
                      </div>
                    </td>
                    <td style={styles.td}>
                      <div style={styles.productId} title={t.tmc2_product_id}>
                        {t.tmc2_product_id || '---'}
                      </div>
                    </td>
                    <td style={styles.td}>
                      <div style={styles.productId} title={t.iirs_product_id}>
                        {t.iirs_product_id || '---'}
                      </div>
                    </td>
                    <td style={{
                      ...styles.td,
                      fontFamily: 'var(--font-mono)',
                      fontWeight: 600,
                      color: pass ? 'var(--accent-success)' : 'var(--accent-danger)',
                    }}>
                      {overlapPct.toFixed(1)}%
                    </td>
                    <td style={{
                      ...styles.td,
                      fontFamily: 'var(--font-mono)',
                    }}>
                      {fmtAngle(t.ohrc_sun_elevation_deg)}
                    </td>
                    <td style={styles.td}>
                      <span className={pass ? 'badge badge-pass' : 'badge badge-fail'}>
                        {pass ? 'PASS' : 'FAIL'}
                      </span>
                    </td>
                  </tr>

                  {/* Expanded detail row */}
                  {isExpanded && (
                    <tr>
                      <td colSpan={7} style={{ padding: 0 }}>
                        <div style={styles.detailPanel}>
                          <div style={styles.detailGrid}>
                            <div style={styles.detailItem}>
                              <div style={styles.detailLabel}>OHRC Sun Elevation</div>
                              <div style={styles.detailValue}>{fmtAngle(t.ohrc_sun_elevation_deg)}</div>
                            </div>
                            <div style={styles.detailItem}>
                              <div style={styles.detailLabel}>TMC-2 Sun Elevation</div>
                              <div style={styles.detailValue}>{fmtAngle(t.tmc2_sun_elevation_deg)}</div>
                            </div>
                            <div style={styles.detailItem}>
                              <div style={styles.detailLabel}>IIRS Sun Elevation</div>
                              <div style={styles.detailValue}>{fmtAngle(t.iirs_sun_elevation_deg)}</div>
                            </div>
                            <div style={styles.detailItem}>
                              <div style={styles.detailLabel}>OHRC GSD</div>
                              <div style={styles.detailValue}>{fmtGsd(t.ohrc_gsd_m)}</div>
                            </div>
                            <div style={styles.detailItem}>
                              <div style={styles.detailLabel}>TMC-2 GSD</div>
                              <div style={styles.detailValue}>{fmtGsd(t.tmc2_gsd_m)}</div>
                            </div>
                            <div style={styles.detailItem}>
                              <div style={styles.detailLabel}>IIRS GSD</div>
                              <div style={styles.detailValue}>{fmtGsd(t.iirs_gsd_m)}</div>
                            </div>
                            <div style={styles.detailItem}>
                              <div style={styles.detailLabel}>Triplet Overlap</div>
                              <div style={styles.detailValue}>{overlapPct.toFixed(2)}%</div>
                            </div>
                            {t.intersection_wkt && (
                              <div style={{ ...styles.detailItem, gridColumn: '1 / -1' }}>
                                <div style={styles.detailLabel}>Intersection WKT</div>
                                <div style={{
                                  ...styles.detailValue,
                                  fontSize: '0.65rem',
                                  wordBreak: 'break-all',
                                  maxHeight: '60px',
                                  overflow: 'auto',
                                }}>
                                  {t.intersection_wkt}
                                </div>
                              </div>
                            )}
                          </div>
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
