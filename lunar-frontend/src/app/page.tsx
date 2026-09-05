"use client";

import { useState, useEffect, useCallback, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import Console from "@/components/Console";
import ExploreMoonHero from "@/components/hero/ExploreMoonHero";
import AboutPage from "@/components/AboutPage";
import LoginPage from "@/components/LoginPage";
import { isAuthenticated, logout, getCurrentUser, type AuthUser } from "@/lib/auth";

type View = "hero" | "console" | "about" | "login";

function MainContent() {
  const searchParams = useSearchParams();
  const router = useRouter();

  const [authChecked, setAuthChecked] = useState(false);
  const [user, setUser] = useState<AuthUser | null>(null);

  // Check authentication on mount
  useEffect(() => {
    if (isAuthenticated()) {
      setUser(getCurrentUser());
    }
    setAuthChecked(true);
  }, []);

  // Determine initial view from URL, but gate behind auth
  const paramView = searchParams.get("view");
  const getViewFromParam = useCallback((): View => {
    if (!isAuthenticated()) return "login";
    if (paramView === "console") return "console";
    if (paramView === "about") return "about";
    return "hero";
  }, [paramView]);

  const [view, setView] = useState<View>("login");

  // Sync view with auth state and URL param
  useEffect(() => {
    if (authChecked) {
      setView(getViewFromParam());
    }
  }, [authChecked, getViewFromParam]);

  // Sync URL param changes into view
  useEffect(() => {
    if (!authChecked) return;
    const urlView = searchParams.get("view");

    if (!isAuthenticated()) {
      if (view !== "login") setView("login");
      return;
    }

    if (urlView === "console" && view !== "console") {
      setView("console");
    } else if (urlView === "about" && view !== "about") {
      setView("about");
    } else if (!urlView && view !== "hero" && view !== "login") {
      setView("hero");
    }
  }, [searchParams, view, authChecked]);

  const handleLoginSuccess = useCallback(() => {
    setUser(getCurrentUser());
    setView("hero");
    router.push("/");
  }, [router]);

  const handleLogout = useCallback(() => {
    logout();
    setUser(null);
    setView("login");
    router.push("/");
  }, [router]);

  const handleLaunchConsole = useCallback(() => {
    setView("console");
    router.push("/?view=console");
  }, [router]);

  const handleOpenAbout = useCallback(() => {
    setView("about");
    router.push("/?view=about");
  }, [router]);

  const handleBackToHero = useCallback(() => {
    setView("hero");
    router.push("/");
  }, [router]);

  // Show nothing until auth is checked to prevent flash
  if (!authChecked) {
    return <div className="h-screen w-screen bg-[#000000]" />;
  }

  // Not authenticated → show login page
  if (view === "login" || !isAuthenticated()) {
    return <LoginPage onLoginSuccess={handleLoginSuccess} />;
  }

  if (view === "console") {
    return <Console onBackToHero={handleBackToHero} onLogout={handleLogout} />;
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
      onLogout={handleLogout}
      userName={user?.name}
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
