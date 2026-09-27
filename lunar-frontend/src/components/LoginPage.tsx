"use client";

import { useState, useEffect, useCallback } from "react";
import { login, register } from "@/lib/auth";

interface Props {
  onLoginSuccess: () => void;
}

export default function LoginPage({ onLoginSuccess }: Props) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [showSuccess, setShowSuccess] = useState(false);

  useEffect(() => {
    setIsLoaded(true);
    const handleMouseMove = (e: MouseEvent) => {
      const x = (e.clientX / window.innerWidth - 0.5) * 2;
      const y = (e.clientY / window.innerHeight - 0.5) * 2;
      setMousePos({ x, y });
    };
    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  const resetForm = useCallback(() => {
    setName("");
    setEmail("");
    setPassword("");
    setConfirmPassword("");
    setError(null);
  }, []);

  const switchMode = useCallback(
    (newMode: "login" | "register") => {
      setMode(newMode);
      resetForm();
    },
    [resetForm]
  );

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    // Validation
    if (!email.trim() || !password.trim()) {
      setError("Please fill in all fields");
      return;
    }

    if (mode === "register") {
      if (!name.trim()) {
        setError("Please enter your name");
        return;
      }
      if (password.length < 6) {
        setError("Password must be at least 6 characters");
        return;
      }
      if (password !== confirmPassword) {
        setError("Passwords do not match");
        return;
      }
    }

    setLoading(true);

    try {
      if (mode === "register") {
        await register(name.trim(), email.trim(), password);
      } else {
        await login(email.trim(), password);
      }
      setShowSuccess(true);
      setTimeout(() => {
        onLoginSuccess();
      }, 700);
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Something went wrong";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const handleOperatorDemoBypass = async () => {
    setError(null);
    setLoading(true);
    const demoEmail = "pilot@isro.gov.in";
    const demoPass = "Chandrayaan2024!";
    const demoName = "ISRO Flight Operator";

    try {
      try {
        await login(demoEmail, demoPass);
      } catch {
        // If demo operator does not exist in local db, register it
        await register(demoName, demoEmail, demoPass);
      }
      setShowSuccess(true);
      setTimeout(() => {
        onLoginSuccess();
      }, 700);
    } catch {
      // In case network or backend error occurs, allow operator pilot demo access
      const expTimestamp = Math.floor(Date.now() / 1000) + 86400;
      const b64Payload = btoa(JSON.stringify({ exp: expTimestamp }));
      const sessionToken = "session-token." + b64Payload + ".sig";
      localStorage.setItem("astralynx_auth_token", sessionToken);
      localStorage.setItem(
        "astralynx_auth_user",
        JSON.stringify({
          id: "demo-pilot-001",
          name: "ISRO Flight Operator (Demo)",
          email: "pilot@isro.gov.in",
          created_at: new Date().toISOString(),
        })
      );
      setShowSuccess(true);
      setTimeout(() => {
        onLoginSuccess();
      }, 700);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="dark relative min-h-screen w-full select-none overflow-x-hidden overflow-y-auto bg-[#0A0D0C] font-mono text-[#E7E2D6] flex flex-col justify-between">
      {/* 1. Full-Bleed Video Background with Subtle Parallax & Vignettes */}
      <div
        className="fixed inset-0 z-0 h-[106%] w-[106%] -left-[3%] -top-[3%] transition-transform duration-700 ease-out pointer-events-none"
        style={{
          transform: `translate3d(${mousePos.x * -8}px, ${mousePos.y * -8}px, 0)`,
        }}
      >
        <video
          autoPlay
          loop
          muted
          playsInline
          className="h-full w-full object-cover object-center"
        >
          <source src="/bgsih.mp4" type="video/mp4" />
        </video>

        {/* High-Legibility Overlays */}
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-[#0A0D0C]/85 via-black/45 to-[#0A0D0C]/90" />
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-r from-black/80 via-transparent to-black/80" />
      </div>

      {/* 2. Twinkling Space Stars */}
      <div className="pointer-events-none fixed inset-0 z-[1] overflow-hidden">
        {[
          { top: "15%", left: "12%", delay: "0.2s" },
          { top: "28%", left: "84%", delay: "1.4s" },
          { top: "65%", left: "15%", delay: "0.8s" },
          { top: "78%", left: "78%", delay: "2.1s" },
          { top: "42%", left: "92%", delay: "1.7s" },
        ].map((star, i) => (
          <div
            key={i}
            className="absolute h-1 w-1 rounded-full bg-white shadow-[0_0_6px_#3fb5c9] animate-star-twinkle"
            style={{
              top: star.top,
              left: star.left,
              animationDelay: star.delay,
            }}
          />
        ))}
      </div>

      {/* 3. Top Retro OS Workstation Bar */}
      <header
        className={`relative z-30 p-2 sm:px-6 transition-all duration-700 ${
          isLoaded ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-4"
        }`}
      >
        <div className="retro-outset bg-[#E7E2D6] dark:bg-[#1A201E] p-1.5 flex items-center justify-between shadow-xl">
          <div className="flex items-center gap-3">
            <div className="bg-[#1F4743] text-white px-2 py-1 flex items-center gap-2 text-xs font-bold font-mono tracking-wider">
              <span className="w-4 h-4 bg-white/20 flex items-center justify-center text-[10px] text-white font-mono font-bold">
                C2
              </span>
              <span>CHANDRAYAAN-2</span>
            </div>
            <span className="retro-inset px-2 py-0.5 text-[10px] font-mono font-bold text-[#143532] dark:text-emerald-400 bg-[#E9E4D8] dark:bg-[#161B19] border-t-[#8B8579] border-l-[#8B8579] border-r-white border-b-white">
              ISRO SAC // SECURE GATEWAY
            </span>
          </div>

          <div className="flex items-center gap-2">
            <div className="retro-inset px-2 py-0.5 text-[10px] font-mono text-[#555C58] dark:text-[#8C9893] bg-white dark:bg-[#0A0D0C] flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <span>PORTAL SECURED</span>
            </div>
          </div>
        </div>
      </header>

      {/* 4. Main Content: Dashboard-Styled Modal Window */}
      <div
        className={`relative z-20 flex-1 flex flex-col items-center justify-center px-4 py-6 transition-all duration-700 delay-100 ${
          isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
        }`}
      >
        <div
          className={`w-full max-w-md transition-all duration-500 ${
            showSuccess ? "scale-95 opacity-0" : "scale-100 opacity-100"
          }`}
        >
          {/* Retro Window Dialog */}
          <div className="retro-outset bg-[#E7E2D6] dark:bg-[#1A201E] text-[#1E2321] dark:text-[#E7E2D6] p-1 shadow-2xl">
            {/* Retro Window Titlebar */}
            <div className="bg-[#1F4743] px-3 py-1.5 text-white flex items-center justify-between text-xs font-bold font-mono tracking-wider select-none border-b border-[#143532]">
              <div className="flex items-center gap-2">
                <span>🔐</span>
                <span>AUTHENTICATION // OPERATOR LOGIN</span>
              </div>
              <div className="flex items-center gap-1">
                <span className="window-ctrl-btn">_</span>
                <span className="window-ctrl-btn">□</span>
                <span className="window-ctrl-btn">✕</span>
              </div>
            </div>

            {/* Sub-Header Strip */}
            <div className="px-3 py-1 bg-[#DED8CB] dark:bg-[#141817] border-b border-[#8B8579] dark:border-[#2D3835] flex items-center justify-between text-[11px] font-mono">
              <span className="text-[#555C58] dark:text-[#8C9893]">
                {mode === "login" ? "SESSION: AUTH REQUIRED" : "NEW OPERATOR ONBOARDING"}
              </span>
              <span className="retro-inset px-2 py-0.2 text-[10px] bg-white dark:bg-[#0A0D0C] text-[#1E2321] dark:text-[#E7E2D6]">
                ENCR: AES-256
              </span>
            </div>

            {/* Window Interior */}
            <div className="p-5 sm:p-6 space-y-4">
              {/* Mode Toggle Buttons (Dashboard Tab Bar) */}
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => switchMode("login")}
                  className={`flex-1 py-1.5 text-xs font-mono font-bold tracking-wider uppercase transition-none ${
                    mode === "login"
                      ? "retro-button-primary active-pressed"
                      : "retro-button bg-[#E7E2D6] dark:bg-[#222927] text-[#1E2321] dark:text-[#E7E2D6] hover:bg-[#F0ECE1] dark:hover:bg-[#2D3835]"
                  }`}
                >
                  Sign In
                </button>
                <button
                  type="button"
                  onClick={() => switchMode("register")}
                  className={`flex-1 py-1.5 text-xs font-mono font-bold tracking-wider uppercase transition-none ${
                    mode === "register"
                      ? "retro-button-primary active-pressed"
                      : "retro-button bg-[#E7E2D6] dark:bg-[#222927] text-[#1E2321] dark:text-[#E7E2D6] hover:bg-[#F0ECE1] dark:hover:bg-[#2D3835]"
                  }`}
                >
                  Register
                </button>
              </div>

              {/* Error Banner */}
              {error && (
                <div className="retro-inset p-2 bg-rose-100 dark:bg-rose-950/80 border-t-rose-800 border-l-rose-800 border-r-white border-b-white text-xs font-mono text-rose-800 dark:text-rose-300 flex items-start gap-2">
                  <span className="font-bold">⚠</span>
                  <div className="flex-1">
                    <span>{error}</span>
                    {mode === "register" && error.toLowerCase().includes("already exists") && (
                      <button
                        type="button"
                        onClick={() => switchMode("login")}
                        className="block mt-1 font-bold text-[#28557E] dark:text-teal underline"
                      >
                        Switch to Sign In →
                      </button>
                    )}
                  </div>
                </div>
              )}

              {/* Form */}
              <form onSubmit={handleSubmit} className="space-y-3.5">
                {/* Name (Register only) */}
                {mode === "register" && (
                  <div>
                    <label
                      htmlFor="login-name"
                      className="block text-[11px] font-mono font-bold text-[#555C58] dark:text-[#8C9893] uppercase mb-1"
                    >
                      Operator Name
                    </label>
                    <div className="retro-inset flex items-center px-2.5 py-1.5 bg-white dark:bg-[#0A0D0C] border-t-[#8B8579] border-l-[#8B8579] border-r-white border-b-white">
                      <input
                        id="login-name"
                        type="text"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        placeholder="Operator Name"
                        autoComplete="name"
                        className="w-full bg-transparent text-xs font-mono text-[#1E2321] dark:text-[#E7E2D6] placeholder-[#8B8579] outline-none"
                      />
                    </div>
                  </div>
                )}

                {/* Email Field */}
                <div>
                  <label
                    htmlFor="login-email"
                    className="block text-[11px] font-mono font-bold text-[#555C58] dark:text-[#8C9893] uppercase mb-1"
                  >
                    Officer ID / Email
                  </label>
                  <div className="retro-inset flex items-center px-2.5 py-1.5 bg-white dark:bg-[#0A0D0C] border-t-[#8B8579] border-l-[#8B8579] border-r-white border-b-white">
                    <input
                      id="login-email"
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="officer@isro.gov.in"
                      autoComplete="email"
                      required
                      className="w-full bg-transparent text-xs font-mono text-[#1E2321] dark:text-[#E7E2D6] placeholder-[#8B8579] outline-none"
                    />
                  </div>
                </div>

                {/* Password Field */}
                <div>
                  <label
                    htmlFor="login-password"
                    className="block text-[11px] font-mono font-bold text-[#555C58] dark:text-[#8C9893] uppercase mb-1"
                  >
                    Security Passcode
                  </label>
                  <div className="retro-inset flex items-center px-2.5 py-1.5 bg-white dark:bg-[#0A0D0C] border-t-[#8B8579] border-l-[#8B8579] border-r-white border-b-white">
                    <input
                      id="login-password"
                      type="password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder="••••••••••••"
                      autoComplete={mode === "register" ? "new-password" : "current-password"}
                      required
                      className="w-full bg-transparent text-xs font-mono text-[#1E2321] dark:text-[#E7E2D6] placeholder-[#8B8579] outline-none"
                    />
                  </div>
                </div>

                {/* Confirm Password (Register only) */}
                {mode === "register" && (
                  <div>
                    <label
                      htmlFor="login-confirm-password"
                      className="block text-[11px] font-mono font-bold text-[#555C58] dark:text-[#8C9893] uppercase mb-1"
                    >
                      Confirm Passcode
                    </label>
                    <div className="retro-inset flex items-center px-2.5 py-1.5 bg-white dark:bg-[#0A0D0C] border-t-[#8B8579] border-l-[#8B8579] border-r-white border-b-white">
                      <input
                        id="login-confirm-password"
                        type="password"
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        placeholder="••••••••••••"
                        autoComplete="new-password"
                        className="w-full bg-transparent text-xs font-mono text-[#1E2321] dark:text-[#E7E2D6] placeholder-[#8B8579] outline-none"
                      />
                    </div>
                  </div>
                )}

                {/* Submit Action */}
                <div className="pt-2 space-y-2">
                  <button
                    type="submit"
                    disabled={loading}
                    className="retro-button-primary w-full py-2.5 text-xs font-mono font-bold tracking-wider uppercase flex items-center justify-center gap-2 shadow-md active-pressed disabled:opacity-50"
                  >
                    {loading ? (
                      <>
                        <span className="h-3 w-3 rounded-full border-2 border-white border-t-transparent animate-spin" />
                        <span>PROCESSING...</span>
                      </>
                    ) : (
                      <>
                        <span>{mode === "login" ? "AUTHENTICATE & ENTER" : "CREATE OPERATOR ACCOUNT"}</span>
                        <span>&gt;&gt;</span>
                      </>
                    )}
                  </button>

                  {/* One-Click Operator Pilot Access */}
                  <button
                    type="button"
                    onClick={handleOperatorDemoBypass}
                    disabled={loading}
                    className="retro-button w-full py-2 text-xs font-mono font-bold text-[#1E2321] dark:text-[#E7E2D6] hover:bg-[#D9D3C5] dark:hover:bg-white/10 flex items-center justify-center gap-1.5 disabled:opacity-50"
                  >
                    <span>⚡</span>
                    <span>Pilot Operator Fast-Pass</span>
                  </button>
                </div>
              </form>
            </div>

            {/* Retro Dialog Footer Status */}
            <div className="px-3 py-1.5 bg-[#DED8CB] dark:bg-[#141817] border-t border-[#8B8579] dark:border-[#2D3835] flex items-center justify-between text-[10px] font-mono text-[#555C58] dark:text-[#8C9893]">
              <span className="flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                <span>CHANDRAYAAN-2 ORBITAL CONSOLE</span>
              </span>
              <span>STN: ISRO-SAC-01</span>
            </div>
          </div>
        </div>
      </div>

      {/* 5. Bottom Status Bar */}
      <footer
        className={`relative z-20 px-4 sm:px-6 pb-3 transition-all duration-700 delay-200 ${
          isLoaded ? "opacity-100" : "opacity-0"
        }`}
      >
        <div className="retro-outset bg-[#E7E2D6]/95 dark:bg-[#1A201E]/95 backdrop-blur-md px-3 py-1.5 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs font-mono text-[#555C58] dark:text-[#8C9893]">
          <div className="flex items-center gap-2 text-[11px]">
            <span className="font-bold text-[#1E2321] dark:text-[#E7E2D6]">ISRO SAC</span>
            <span>·</span>
            <span>Planetary Science Data System</span>
          </div>

          <div className="retro-inset px-2.5 py-0.5 text-[10px] bg-white dark:bg-[#0A0D0C] text-[#1E2321] dark:text-[#E7E2D6] flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
            <span>SESSION GATEWAY: READY</span>
          </div>

          <div className="text-[11px] text-[#555C58] dark:text-[#8C9893]">
            SIH 2024 · Problem 26166
          </div>
        </div>
      </footer>
    </section>
  );
}
