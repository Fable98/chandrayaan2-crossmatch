import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        // Theme-aware surface tokens: driven by CSS variables so every
        // consumer (panels, borders, ink text) follows the `dark` class on
        // <html> without per-component dark: duplicates. Values live in
        // globals.css (:root = light, .dark = dark). Brand accents below
        // stay fixed in both modes.
        void: "rgb(var(--c-void) / <alpha-value>)",
        panel: "rgb(var(--c-panel) / <alpha-value>)",
        "panel-raised": "rgb(var(--c-panel-raised) / <alpha-value>)",
        border: "rgb(var(--c-border) / <alpha-value>)",
        "border-bright": "rgb(var(--c-border-bright) / <alpha-value>)",
        ink: "rgb(var(--c-ink) / <alpha-value>)",
        "ink-dim": "rgb(var(--c-ink-dim) / <alpha-value>)",
        "ink-faint": "rgb(var(--c-ink-faint) / <alpha-value>)",
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
