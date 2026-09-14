import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET(
  request: NextRequest,
  { params }: { params: { slug: string[] } }
) {
  const slugParts = params.slug || [];
  if (slugParts.length === 0) {
    return new NextResponse("Not Found", { status: 404 });
  }

  const relativePath = slugParts.join("/");
  const publicDir = path.join(process.cwd(), "public", "images");

  // Candidates to look up in public/images
  const candidates = [
    path.join(publicDir, relativePath),
    path.join(publicDir, `${relativePath}.png`),
    path.join(publicDir, relativePath.replace(/\.png$/, "")),
  ];

  // Specific fallback mappings
  if (slugParts[0] === "iirs" && relativePath.includes("iirs_overlay")) {
    candidates.push(path.join(publicDir, "iirs", "iirs_overlay.png"));
    candidates.push(path.join(publicDir, "iirs", "iirs_512.png"));
  }

  for (const filePath of candidates) {
    if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
      const fileBuffer = fs.readFileSync(filePath);
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
  }

  // Ingested regions (region_auto_*) live only under
  // data_preprocessing_pipeline/processed_triplets/ on the backend host —
  // no static copy exists in public/images. Proxy to FastAPI /images/*
  // so vault thumbnails render the same as curated regions instead of 404
  // black tiles. Static public/images files above still win when present.
  try {
    const backendBase = (
      process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"
    ).replace(/\/$/, "");
    // Preserve sub-path + extension exactly as requested.
    const hasExt = /\.(png|jpe?g|json)$/i.test(relativePath);
    const backendPath = `/images/${relativePath}${hasExt ? "" : ".png"}`;
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
  } catch {
    // Fall through to 404 below — never leak backend errors as images.
  }

  return new NextResponse(`Image '${relativePath}' not found`, { status: 404 });
}
