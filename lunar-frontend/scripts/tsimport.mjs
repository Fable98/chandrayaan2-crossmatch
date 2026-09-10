/**
 * tsimport.mjs — minimal TS importer for smoke tests (zero new deps).
 *
 * Uses the repo's own typescript (node_modules) to transpile src/lib/*.ts
 * to CJS in a temp dir, rewrites extensionless relative imports, and imports
 * the entry module. Only supports the erasable-syntax lib subset on purpose:
 * if someone adds enums/namespaces/parameter-properties to the API layer,
 * this helper (and `npm run smoke`) fails loudly.
 */

import { mkdtempSync, writeFileSync, readdirSync } from "fs";
import { tmpdir } from "os";
import { join, dirname, basename } from "path";
import { fileURLToPath, pathToFileURL } from "url";
import { createRequire } from "module";

const HERE = dirname(fileURLToPath(import.meta.url));
const LIB = join(HERE, "..", "src", "lib");

export async function importLib(entry = "api.ts") {
  const require = createRequire(join(HERE, "..", "package.json"));
  const ts = require("typescript");
  const dir = mkdtempSync(join(tmpdir(), "lr-smoke-"));
  for (const name of readdirSync(LIB)) {
    if (!name.endsWith(".ts")) continue;
    const src = require("fs").readFileSync(join(LIB, name), "utf-8");
    const { outputText, diagnostics } = ts.transpileModule(src, {
      compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
      fileName: name,
    });
    const errors = (diagnostics || []).filter((d) => d.category === ts.DiagnosticCategory.Error);
    if (errors.length) {
      throw new Error(`transpile ${name}: ${errors.map((e) => e.messageText).join("; ")}`);
    }
    const fixed = outputText.replace(/require\(["'](\.[^"']*)["']\)/g, (m, p) => {
      const withExt = p.endsWith(".ts") || p.endsWith(".js") ? p : `${p}.js`;
      return `require(${JSON.stringify(withExt)})`;
    });
    writeFileSync(join(dir, basename(name, ".ts") + ".js"), fixed);
  }
  return import(pathToFileURL(join(dir, basename(entry, ".ts") + ".js")).href);
}
