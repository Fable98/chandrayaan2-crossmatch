"use client";

import { useState, useEffect, useCallback } from "react";
import Image from "next/image";
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
      }, 800);
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Something went wrong";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="relative h-screen w-screen select-none overflow-hidden bg-[#000000] font-sans text-white">
      {/* Full-Bleed Lunar Background with Parallax */}
      <div
        className="absolute inset-0 z-0 h-[106%] w-[106%] -left-[3%] -top-[3%] transition-transform duration-700 ease-out"
        style={{
          transform: `translate3d(${mousePos.x * -8}px, ${mousePos.y * -8}px, 0)`,
        }}
      >
        <Image
          src="/lunar_crescent_backdrop.jpg"
          alt="Lunar Surface"
          fill
          priority
          sizes="100vw"
          className="object-cover object-center"
        />
        {/* Heavy vignette to focus attention on the card */}
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-black/60 via-black/30 to-black/70" />
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-r from-black/50 via-transparent to-black/50" />
      </div>

      {/* Floating Particle / Glow Effects */}
      <div className="pointer-events-none absolute inset-0 z-[1]">
        {/* Teal nebula glow behind card area */}
        <div
          className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full opacity-[0.06]"
          style={{
            background:
              "radial-gradient(circle, rgba(63,181,201,0.5) 0%, transparent 70%)",
          }}
        />
      </div>

      {/* Top Bar */}
      <header
        className={`relative z-30 flex h-16 items-center justify-between border-b border-white/5 px-6 transition-all duration-1000 md:px-12 ${
          isLoaded ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-4"
        }`}
      >
        <div />
        <div />
      </header>

      {/* Main Content — Login Card */}
      <div
        className={`relative z-20 flex h-[calc(100vh-64px)] items-center justify-center px-4 transition-all duration-1000 delay-200 ${
          isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
        }`}
      >
        <div
          className={`relative w-full max-w-md transition-all duration-500 ${
            showSuccess ? "scale-95 opacity-0" : "scale-100 opacity-100"
          }`}
        >
          {/* Glassmorphic Card */}
          <div className="relative overflow-hidden rounded-2xl border border-white/[0.08] bg-[#0a0e14]/80 backdrop-blur-xl shadow-[0_8px_60px_rgba(0,0,0,0.6)]">
            {/* Top glow accent line */}
            <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-teal/50 to-transparent" />

            {/* Inner glow */}
            <div className="pointer-events-none absolute -top-20 left-1/2 -translate-x-1/2 w-80 h-40 rounded-full opacity-[0.07]"
              style={{
                background: "radial-gradient(ellipse, rgba(63,181,201,0.8) 0%, transparent 70%)",
              }}
            />

            <div className="relative px-8 py-10">
              {/* Header */}
              <div className="text-center mb-8">
                <h1 className="text-2xl font-bold tracking-tight bg-gradient-to-b from-white to-white/70 bg-clip-text text-transparent">
                  {mode === "login"
                    ? "Welcome Back"
                    : "Create Account"}
                </h1>
                <p className="mt-2 text-sm text-ink-dim">
                  {mode === "login"
                    ? "Sign in to access the Lunar Console"
                    : "Join the Chandrayaan-2 Crossmatch platform"}
                </p>
              </div>

              {/* Mode Toggle */}
              <div className="relative mb-8 flex rounded-xl bg-[#0d1118] border border-white/[0.06] p-1">
                {/* Sliding indicator */}
                <div
                  className="absolute top-1 bottom-1 w-[calc(50%-4px)] rounded-lg bg-gradient-to-b from-teal/20 to-teal/10 border border-teal/20 shadow-[0_0_12px_rgba(63,181,201,0.15)] transition-all duration-300 ease-out"
                  style={{
                    left: mode === "login" ? "4px" : "calc(50%)",
                  }}
                />
                <button
                  type="button"
                  onClick={() => switchMode("login")}
                  className={`relative z-10 flex-1 py-2 text-sm font-medium rounded-lg transition-colors duration-300 ${
                    mode === "login" ? "text-teal" : "text-ink-faint hover:text-ink-dim"
                  }`}
                >
                  Sign In
                </button>
                <button
                  type="button"
                  onClick={() => switchMode("register")}
                  className={`relative z-10 flex-1 py-2 text-sm font-medium rounded-lg transition-colors duration-300 ${
                    mode === "register" ? "text-teal" : "text-ink-faint hover:text-ink-dim"
                  }`}
                >
                  Register
                </button>
              </div>

              {/* Error Message */}
              {error && (
                <div className="mb-6 flex items-center gap-2 rounded-lg border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-400 animate-in fade-in">
                  <svg className="h-4 w-4 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                  <span>{error}</span>
                </div>
              )}

              {/* Form */}
              <form onSubmit={handleSubmit} className="space-y-5">
                {/* Name Field (Register only) */}
                <div
                  className={`transition-all duration-400 ease-out overflow-hidden ${
                    mode === "register"
                      ? "max-h-24 opacity-100"
                      : "max-h-0 opacity-0"
                  }`}
                >
                  <label
                    htmlFor="login-name"
                    className="block text-xs font-medium text-ink-dim mb-1.5 tracking-wide uppercase"
                  >
                    Full Name
                  </label>
                  <input
                    id="login-name"
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="Your name"
                    autoComplete="name"
                    className="w-full rounded-lg border border-white/[0.08] bg-[#0d1118] px-4 py-3 text-sm text-ink placeholder-ink-faint outline-none transition-all duration-200 focus:border-teal/40 focus:ring-1 focus:ring-teal/20 focus:bg-[#0f1520] hover:border-white/[0.12]"
                  />
                </div>

                {/* Email Field */}
                <div>
                  <label
                    htmlFor="login-email"
                    className="block text-xs font-medium text-ink-dim mb-1.5 tracking-wide uppercase"
                  >
                    Email Address
                  </label>
                  <input
                    id="login-email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="you@example.com"
                    autoComplete="email"
                    required
                    className="w-full rounded-lg border border-white/[0.08] bg-[#0d1118] px-4 py-3 text-sm text-ink placeholder-ink-faint outline-none transition-all duration-200 focus:border-teal/40 focus:ring-1 focus:ring-teal/20 focus:bg-[#0f1520] hover:border-white/[0.12]"
                  />
                </div>

                {/* Password Field */}
                <div>
                  <label
                    htmlFor="login-password"
                    className="block text-xs font-medium text-ink-dim mb-1.5 tracking-wide uppercase"
                  >
                    Password
                  </label>
                  <input
                    id="login-password"
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    autoComplete={mode === "register" ? "new-password" : "current-password"}
                    required
                    className="w-full rounded-lg border border-white/[0.08] bg-[#0d1118] px-4 py-3 text-sm text-ink placeholder-ink-faint outline-none transition-all duration-200 focus:border-teal/40 focus:ring-1 focus:ring-teal/20 focus:bg-[#0f1520] hover:border-white/[0.12]"
                  />
                </div>

                {/* Confirm Password (Register only) */}
                <div
                  className={`transition-all duration-400 ease-out overflow-hidden ${
                    mode === "register"
                      ? "max-h-24 opacity-100"
                      : "max-h-0 opacity-0"
                  }`}
                >
                  <label
                    htmlFor="login-confirm-password"
                    className="block text-xs font-medium text-ink-dim mb-1.5 tracking-wide uppercase"
                  >
                    Confirm Password
                  </label>
                  <input
                    id="login-confirm-password"
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="••••••••"
                    autoComplete="new-password"
                    className="w-full rounded-lg border border-white/[0.08] bg-[#0d1118] px-4 py-3 text-sm text-ink placeholder-ink-faint outline-none transition-all duration-200 focus:border-teal/40 focus:ring-1 focus:ring-teal/20 focus:bg-[#0f1520] hover:border-white/[0.12]"
                  />
                </div>

                {/* Submit Button */}
                <button
                  type="submit"
                  disabled={loading}
                  className="group relative w-full overflow-hidden rounded-lg bg-gradient-to-r from-teal to-teal-dark py-3.5 text-sm font-bold tracking-wide text-black shadow-[0_0_30px_rgba(63,181,201,0.3)] transition-all duration-200 hover:shadow-[0_0_40px_rgba(63,181,201,0.5)] hover:scale-[1.01] active:scale-[0.99] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:hover:shadow-[0_0_30px_rgba(63,181,201,0.3)]"
                >
                  {/* Button shimmer effect */}
                  <div className="absolute inset-0 -translate-x-full group-hover:translate-x-full transition-transform duration-700 bg-gradient-to-r from-transparent via-white/20 to-transparent" />

                  <span className="relative flex items-center justify-center gap-2">
                    {loading ? (
                      <>
                        <svg
                          className="h-4 w-4 animate-spin"
                          viewBox="0 0 24 24"
                          fill="none"
                        >
                          <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                          />
                          <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                          />
                        </svg>
                        <span>
                          {mode === "login"
                            ? "Signing in..."
                            : "Creating account..."}
                        </span>
                      </>
                    ) : (
                      <>
                        <span>
                          {mode === "login"
                            ? "Sign In"
                            : "Create Account"}
                        </span>
                        <span className="transition-transform duration-200 group-hover:translate-x-1">
                          →
                        </span>
                      </>
                    )}
                  </span>
                </button>
              </form>

              {/* Divider */}
              <div className="mt-8 flex items-center gap-3">
                <div className="flex-1 h-px bg-gradient-to-r from-transparent to-white/[0.06]" />
                <span className="text-2xs text-ink-faint uppercase tracking-widest">
                  {mode === "login" ? "New here?" : "Already a member?"}
                </span>
                <div className="flex-1 h-px bg-gradient-to-l from-transparent to-white/[0.06]" />
              </div>

              {/* Switch mode link */}
              <div className="mt-4 text-center">
                <button
                  type="button"
                  onClick={() =>
                    switchMode(mode === "login" ? "register" : "login")
                  }
                  className="text-sm text-teal/80 hover:text-teal transition-colors duration-200 underline decoration-teal/30 underline-offset-4 hover:decoration-teal/60"
                >
                  {mode === "login"
                    ? "Create a free account"
                    : "Sign in to your account"}
                </button>
              </div>
            </div>
          </div>

          {/* Security Badge */}
          <div className="mt-6 flex items-center justify-center gap-2 text-2xs text-ink-faint/60">
            <svg className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
              <path
                fillRule="evenodd"
                d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z"
                clipRule="evenodd"
              />
            </svg>
            <span>Secured with JWT · End-to-end encrypted</span>
          </div>
        </div>
      </div>

      {/* Bottom Credits */}
      <footer
        className={`absolute bottom-6 left-0 right-0 z-20 flex items-center justify-center px-6 text-xs text-ink-faint transition-all duration-1000 delay-700 md:bottom-8 md:px-12 ${
          isLoaded ? "opacity-100" : "opacity-0"
        }`}
      >
        <span className="font-mono text-2xs tracking-wide">
          created for{" "}
          <span className="text-teal font-semibold">ISRO · SIH26166</span>
          <span className="mx-2 text-white/20">·</span>
          <span className="text-ink-dim">Chandrayaan-2 Crossmatch</span>
        </span>
      </footer>

      {/* Success Overlay */}
      {showSuccess && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-in fade-in">
          <div className="flex flex-col items-center gap-4">
            <div className="h-16 w-16 rounded-full bg-teal/20 border border-teal/40 flex items-center justify-center shadow-[0_0_40px_rgba(63,181,201,0.3)]">
              <svg
                className="h-8 w-8 text-teal"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M5 13l4 4L19 7"
                />
              </svg>
            </div>
            <p className="text-lg font-semibold text-white">
              {mode === "login" ? "Welcome back!" : "Account created!"}
            </p>
            <p className="text-sm text-ink-dim">
              Launching console...
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
