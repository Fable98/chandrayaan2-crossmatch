/**
 * gen-openapi.mjs — Step 13: generate the TypeScript client contract from the
 * live FastAPI OpenAPI schema. Single source of truth: backend openapi.json.
 *
 *   npm run gen:api          # regenerate src/lib/backend-types.ts
 *   npm run gen:api:check    # CI: fail if the committed file drifted
 *
 * Env:
 *   API_BASE_URL  backend origin (default http://localhost:8000)
 *   OPENAPI_FILE  use a saved openapi.json instead of fetching
 *
 * Zero dependencies (pure node).
 */

import { readFileSync, writeFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const OUT = join(ROOT, "src", "lib", "backend-types.ts");
const CHECK = process.argv.includes("--check");

function tsType(schema, defs, seen = new Set()) {
  if (!schema || typeof schema !== "object") return "unknown";
  if (schema.$ref) {
    const name = schema.$ref.split("/").pop();
    return sanitize(name);
  }
  if (schema.anyOf) return schema.anyOf.map((s) => tsType(s, defs, seen)).join(" | ");
  if (schema.oneOf) return schema.oneOf.map((s) => tsType(s, defs, seen)).join(" | ");
  if (schema.allOf) {
    const parts = schema.allOf.map((s) => tsType(s, defs, seen)).filter((t) => t !== "unknown");
    return parts.length ? parts.join(" & ") : "unknown";
  }
  const t = schema.type;
  if (Array.isArray(t)) return t.map((x) => tsType({ ...schema, type: x }, defs, seen)).join(" | ");
  switch (t) {
    case "string":
      if (schema.enum) return schema.enum.map((e) => JSON.stringify(e)).join(" | ");
      return "string";
    case "integer":
    case "number":
      if (schema.enum) return schema.enum.join(" | ");
      return "number";
    case "boolean":
      return "boolean";
    case "null":
      return "null";
    case "array":
      return `${tsType(schema.items, defs, seen)}[]`;
    case "object": {
      const props = schema.properties || {};
      const required = new Set(schema.required || []);
      const entries = Object.entries(props).map(([k, v]) => {
        const opt = required.has(k) ? "" : "?";
        return `  ${JSON.stringify(k)}${opt}: ${tsType(v, defs, seen)};`;
      });
      if (schema.additionalProperties === true || (schema.additionalProperties && typeof schema.additionalProperties === "object")) {
        const valT = schema.additionalProperties === true ? "unknown" : tsType(schema.additionalProperties, defs, seen);
        entries.push(`  [key: string]: ${valT};`);
      }
      if (!entries.length) return "Record<string, unknown>";
      return `{\n${entries.join("\n")}\n}`;
    }
    default:
      if (schema.enum) return schema.enum.map((e) => JSON.stringify(e)).join(" | ");
      return "unknown";
  }
}

function sanitize(name) {
  return String(name).replace(/[^A-Za-z0-9_]/g, "_");
}

async function loadSpec() {
  if (process.env.OPENAPI_FILE) {
    return JSON.parse(readFileSync(process.env.OPENAPI_FILE, "utf-8"));
  }
  const base = (process.env.API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");
  const res = await fetch(`${base}/openapi.json`);
  if (!res.ok) throw new Error(`GET ${base}/openapi.json -> ${res.status}`);
  return res.json();
}

async function main() {
  const spec = await loadSpec();
  const defs = spec.components?.schemas || {};
  const names = Object.keys(defs).sort();
  const lines = [
    "/**",
    " * backend-types.ts — GENERATED CONTRACT. DO NOT EDIT BY HAND.",
    ` * Source: FastAPI openapi.json (title=${JSON.stringify(spec.info?.title)} version=${JSON.stringify(spec.info?.version)}).`,
    " * Regenerate: `npm run gen:api` (Step 13 single-contract rule).",
    " * Frontend extensions live in ./types.ts as `extends` interfaces.",
    " */",
    "",
  ];
  for (const name of names) {
    const schema = defs[name];
    const ident = sanitize(name);
    if (schema.type === "object" || schema.properties || schema.allOf) {
      lines.push(`export interface ${ident} ${tsType(schema, defs)}`);
    } else if (schema.enum) {
      lines.push(`export type ${ident} = ${tsType(schema, defs)};`);
    } else {
      lines.push(`export type ${ident} = ${tsType(schema, defs)};`);
    }
    lines.push("");
  }
  const paths = Object.keys(spec.paths || {}).sort();
  lines.push("/** All backend routes (path -> methods), for contract tests. */");
  lines.push("export const BACKEND_API_PATHS = [");
  for (const p of paths) {
    const methods = Object.keys(spec.paths[p]).filter((m) => m !== "parameters").sort();
    lines.push(`  ${JSON.stringify(p)} /* ${methods.join(",").toUpperCase()} */,`);
  }
  lines.push("] as const;");
  lines.push("");
  const out = lines.join("\n");
  if (CHECK) {
    const current = readFileSync(OUT, "utf-8");
    if (current !== out) {
      console.error("backend-types.ts drifted from backend openapi.json. Run `npm run gen:api`.");
      process.exit(1);
    }
    console.log(`contract OK (${names.length} schemas, ${paths.length} paths).`);
    return;
  }
  writeFileSync(OUT, out);
  console.log(`wrote ${OUT} (${names.length} schemas, ${paths.length} paths).`);
}

main().catch((err) => {
  console.error(`gen-openapi failed: ${err.message}`);
  process.exit(2);
});
