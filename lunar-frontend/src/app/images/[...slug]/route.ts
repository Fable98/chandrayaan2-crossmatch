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

  return new NextResponse(`Image '${relativePath}' not found`, { status: 404 });
}
