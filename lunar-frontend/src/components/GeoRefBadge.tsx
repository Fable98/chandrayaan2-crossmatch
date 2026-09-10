"use client";

// Shared badge for pixel-only data (Step 13 contract). The backend deleted
// its lon = 336 + fx / lat = -4 + fy demo patch: points without product
// bounds now arrive with latitude/longitude null and georeferenced=false.
// Render this badge instead of fabricated coordinates.

export default function GeoRefBadge({ className = "" }: { className?: string }) {
  return (
    <span
      className={`inline-block rounded-md border border-amber-300 bg-amber-50 px-1.5 py-0.5 text-[9px] font-black uppercase tracking-wider text-amber-700 ${className}`}
      title="No product bounds available: pixel coordinates only, no geographic position claimed."
    >
      no-georef
    </span>
  );
}
