"use client";

import { useRouter } from "next/navigation";
import AboutPage from "@/components/AboutPage";

export default function AboutRoute() {
  const router = useRouter();

  const handleBackToHero = () => {
    router.push("/");
  };

  const handleOpenConsole = () => {
    router.push("/?view=console");
  };

  return (
    <AboutPage
      onBackToHero={handleBackToHero}
      onOpenConsole={handleOpenConsole}
    />
  );
}
