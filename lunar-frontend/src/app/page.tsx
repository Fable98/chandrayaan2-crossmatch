"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import Console from "@/components/Console";
import ExploreMoonHero from "@/components/hero/ExploreMoonHero";

function MainContent() {
  const searchParams = useSearchParams();
  const router = useRouter();

  // Initialize view from URL search param (?view=console) or default to ExploreMoonHero
  const initialView = searchParams.get("view") === "console" ? "console" : "hero";
  const [view, setView] = useState<"hero" | "console">(initialView);

  useEffect(() => {
    const urlView = searchParams.get("view");
    if (urlView === "console" && view !== "console") {
      setView("console");
    } else if (urlView !== "console" && view === "console" && !urlView) {
      setView("hero");
    }
  }, [searchParams, view]);

  const handleLaunchConsole = () => {
    setView("console");
    router.push("/?view=console");
  };

  const handleBackToHero = () => {
    setView("hero");
    router.push("/");
  };

  if (view === "console") {
    return <Console onBackToHero={handleBackToHero} />;
  }

  return <ExploreMoonHero onOpenConsole={handleLaunchConsole} />;
}

export default function Home() {
  return (
    <Suspense fallback={<div className="h-screen w-screen bg-[#000000]" />}>
      <MainContent />
    </Suspense>
  );
}
