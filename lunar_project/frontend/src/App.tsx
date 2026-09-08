import React from 'react';
import IngestPage from './pages/IngestPage';

const headerStyles: Record<string, React.CSSProperties> = {
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '14px 28px',
    background: 'rgba(10, 14, 26, 0.85)',
    backdropFilter: 'blur(16px)',
    borderBottom: '1px solid var(--border-subtle)',
    position: 'sticky' as const,
    top: 0,
    zIndex: 100,
  },
  logoGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  logoIcon: {
    fontSize: '1.4rem',
  },
  logoText: {
    fontSize: '0.95rem',
    fontWeight: 700,
    color: 'var(--text-primary)',
    letterSpacing: '-0.01em',
  },
  logoBadge: {
    padding: '2px 8px',
    borderRadius: 'var(--radius-full)',
    background: 'var(--accent-glow)',
    color: 'var(--accent-secondary)',
    fontSize: '0.6rem',
    fontWeight: 700,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.08em',
  },
  nav: {
    display: 'flex',
    gap: '4px',
  },
  navLink: {
    padding: '6px 14px',
    borderRadius: 'var(--radius-sm)',
    fontSize: '0.8rem',
    fontWeight: 500,
    color: 'var(--text-secondary)',
    cursor: 'pointer',
    transition: 'all 150ms ease',
    border: 'none',
    background: 'none',
    fontFamily: 'var(--font-sans)',
  },
  navLinkActive: {
    color: 'var(--text-primary)',
    background: 'var(--bg-elevated)',
  },
};

export default function App() {
  return (
    <div>
      {/* Header */}
      <header style={headerStyles.header}>
        <div style={headerStyles.logoGroup}>
          <span style={headerStyles.logoIcon}>&#127761;</span>
          <span style={headerStyles.logoText}>Lunar Crossmatch</span>
          <span style={headerStyles.logoBadge}>Pipeline</span>
        </div>
        <nav style={headerStyles.nav}>
          <button
            style={{ ...headerStyles.navLink, ...headerStyles.navLinkActive }}
          >
            Ingest
          </button>
          <button style={headerStyles.navLink}>Regions</button>
          <button style={headerStyles.navLink}>Predict</button>
        </nav>
      </header>

      {/* Page content */}
      <IngestPage />
    </div>
  );
}
