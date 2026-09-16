import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

export async function GET(
  request: NextRequest,
  { params }: { params: { slug: string[] } }
) {
  const slugParts = params.slug || [];
  if (slugParts.length === 0) {
    return new NextResponse("Not Found", { status: 404 });
  }

  // Reject traversal segments up front — a crafted ".." must 404 before it
  // ever reaches the filesystem join below.
  if (slugParts.some((p) => p === ".." || p === "." || p.includes("\0"))) {
    return new NextResponse("Not Found", { status: 404 });
  }

  const relativePath = slugParts.join("/");
  const publicDir = path.join(process.cwd(), "public", "images");

  // Browsers cannot render TIFF: transparently serve the PNG preview for
  // .tif/.tiff requests, both for static files and the backend proxy below.
  // Only a trailing extension is rewritten ("a.tif" -> "a.png"); names like
  // "a.tif.png" pass through untouched. The 404 message keeps the original.
  const lookupPath = relativePath.replace(/\.tiff?$/i, ".png");

  // Resolve-then-contain: path.join alone normalizes "a/../../etc/passwd"
  // OUTSIDE publicDir, and the old existsSync gate would have served it.
  // Every candidate must resolve inside publicDir or it is skipped.
  const withinRoot = (p: string): string | null => {
    const resolved = path.resolve(publicDir, p);
    return resolved === publicDir || resolved.startsWith(publicDir + path.sep)
      ? resolved
      : null;
  };

  // Candidates to look up in public/images
  const rawCandidates = [
    lookupPath,
    `${lookupPath}.png`,
    lookupPath.replace(/\.png$/, ""),
  ];

  // Specific fallback mappings
  if (slugParts[0] === "iirs" && lookupPath.includes("iirs_overlay")) {
    rawCandidates.push(
      path.join("iirs", "iirs_overlay.png"),
      path.join("iirs", "iirs_512.png")
    );
  }

  for (const raw of rawCandidates) {
    const filePath = withinRoot(raw);
    if (!filePath) continue;
    let stat;
    try {
      stat = await fs.stat(filePath);
    } catch {
      continue;
    }
    if (!stat.isFile()) continue;
    const fileBuffer = await fs.readFile(filePath);
    const ext = path.extname(filePath).toLowerCase();
    const contentType =
      ext === ".json"
        ? "application/json"
        : ext === ".jpg" || ext === ".jpeg"
        ? "image/jpeg"
        : "image/png";

    return new NextResponse(fileBuffer, {
      status: 200,
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "public, max-age=86400, stale-while-revalidate=604800",
      },
    });
  }

  // Ingested regions (region_auto_*) live only under
  // data_preprocessing_pipeline/processed_triplets/ on the backend host —
  // no static copy exists in public/images. Proxy to FastAPI /images/*
  // so vault thumbnails render the same as curated regions instead of 404
  // black tiles. Static public/images files above still win when present.
  try {
    const backendBase = (
      process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"
    ).replace(/\/$/, "");
    // Preserve sub-path + extension exactly as requested (lookupPath already
    // maps .tif/.tiff to the .png preview above).
    const hasExt = /\.(png|jpe?g|json)$/i.test(lookupPath);
    const backendPath = `/images/${lookupPath}${hasExt ? "" : ".png"}`;
    const upstream = await fetch(`${backendBase}${backendPath}`, {
      // Thumbnails are idempotent; keep the vault snappy.
      signal: AbortSignal.timeout(10000),
    });
    if (upstream.ok) {
      const buf = Buffer.from(await upstream.arrayBuffer());
      const ct =
        upstream.headers.get("content-type") ??
        (/\.jpe?g$/i.test(backendPath) ? "image/jpeg" : "image/png");
      return new NextResponse(buf, {
        status: 200,
        headers: {
          "Content-Type": ct,
          "Cache-Control": "public, max-age=300, stale-while-revalidate=3600",
        },
      });
    }
    // Backend answered but has no such tile: honest 404.
    return new NextResponse(`Image '${relativePath}' not found`, { status: 404 });
  } catch {
    // Backend unreachable / timed out: 502 Bad Gateway, NOT 404 — a 404
    // claims the tile doesn't exist, a 502 says the backend is down.
    return new NextResponse("Image backend unavailable", { status: 502 });
  }
}
