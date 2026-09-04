import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        void: "#090b0e",
        panel: "#0e1117",
        "panel-raised": "#141820",
        border: "#1b2029",
        "border-bright": "#28303d",
        ink: "#f0f2f5",
        "ink-dim": "#9aa3af",
        "ink-faint": "#5c6574",
        teal: "#3fb5c9",
        "teal-dim": "#16343d",
        "teal-dark": "#2ea3b8",
        regolith: "#3fb5c9",
        "regolith-dim": "#16343d",
        parallax: "#3fb5c9",
        "parallax-dim": "#16343d",
        alert: "#d9634a",
        obsidian: "#08080a",
        "obsidian-panel": "#0d0d11",
        "obsidian-card": "#121217",
        "obsidian-border": "#23211d",
        "obsidian-border-bright": "#38342d",
        gold: "#d4af37",
        "gold-light": "#f3df9b",
        "gold-cream": "#e8d5b5",
        "gold-dim": "#2c2619",
        lunar: {
          midnight: "#091540",
          cobalt: "#1B2CC1",
          periwinkle: "#7692FF",
          sky: "#ABD2FA",
        },
      },
      fontFamily: {
        sans: ["var(--font-inter)", "var(--font-plex-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-plex-mono)", "ui-monospace", "monospace"],
        serif: ["var(--font-playfair)", "Georgia", "serif"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem", letterSpacing: "0.01em" }],
      },
    },
  },
  plugins: [],
};
export default config;
