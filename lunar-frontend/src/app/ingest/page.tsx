"use client";

import dynamic from "next/dynamic";

const IngestPage = dynamic(() => import("@/components/ingest/IngestPage"), {
  ssr: false,
  loading: () => (
    <div className="flex h-screen w-screen items-center justify-center bg-slate-900 text-white">
      <div className="flex items-center gap-3">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-[#4F46E5] border-t-transparent" />
        <span className="text-sm font-semibold tracking-wider uppercase text-slate-400">
          Loading Ingestion Workspace...
        </span>
      </div>
    </div>
  ),
});

export default function IngestRoute() {
  return <IngestPage />;
}
