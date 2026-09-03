"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import Console from "@/components/Console";
import ExploreMoonHero from "@/components/hero/ExploreMoonHero";
import AboutPage from "@/components/AboutPage";

function MainContent() {
  const searchParams = useSearchParams();
  const router = useRouter();

  // Initialize view from URL search param (?view=console or ?view=about) or default to ExploreMoonHero
  const paramView = searchParams.get("view");
  const initialView =
    paramView === "console" ? "console" : paramView === "about" ? "about" : "hero";
  const [view, setView] = useState<"hero" | "console" | "about">(initialView);

  useEffect(() => {
    const urlView = searchParams.get("view");
    if (urlView === "console" && view !== "console") {
      setView("console");
    } else if (urlView === "about" && view !== "about") {
      setView("about");
    } else if (!urlView && view !== "hero") {
      setView("hero");
    }
  }, [searchParams, view]);

  const handleLaunchConsole = () => {
    setView("console");
    router.push("/?view=console");
  };

  const handleOpenAbout = () => {
    setView("about");
    router.push("/?view=about");
  };

  const handleBackToHero = () => {
    setView("hero");
    router.push("/");
  };

  if (view === "console") {
    return <Console onBackToHero={handleBackToHero} />;
  }

  if (view === "about") {
    return (
      <AboutPage
        onBackToHero={handleBackToHero}
        onOpenConsole={handleLaunchConsole}
      />
    );
  }

  return (
    <ExploreMoonHero
      onOpenConsole={handleLaunchConsole}
      onOpenAbout={handleOpenAbout}
    />
  );
}

export default function Home() {
  return (
    <Suspense fallback={<div className="h-screen w-screen bg-[#000000]" />}>
      <MainContent />
    </Suspense>
  );
}
