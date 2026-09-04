/** Only validated `region_*` tiles belong on the dashboard. */

const HIDDEN_IDS = new Set([
  "triplet_01",
  "triplet_1",
  "triplet01",
  "triplet_new",
  "tripletnew",
]);

export function isDashboardRegion(id: string | null | undefined): boolean {
  if (!id) return false;
  const raw = id.trim();
  const key = raw.toLowerCase().replace(/[\s-]+/g, "_");
  if (HIDDEN_IDS.has(key) || key.startsWith("triplet")) return false;
  return /^region_\d+$/i.test(raw);
}
