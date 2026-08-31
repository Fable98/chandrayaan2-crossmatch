import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        void: "#0A0E14",
        panel: "#111823",
        "panel-raised": "#161F2C",
        border: "#232D3B",
        "border-bright": "#33404F",
        ink: "#E3E6EA",
        "ink-dim": "#8892A0",
        "ink-faint": "#5A6472",
        regolith: "#E8A33D",
        "regolith-dim": "#7A5C2A",
        parallax: "#4FB3A6",
        "parallax-dim": "#2A5850",
        alert: "#D9634A",
      },
      fontFamily: {
        sans: ["var(--font-plex-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-plex-mono)", "ui-monospace", "monospace"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem", letterSpacing: "0.01em" }],
      },
    },
  },
  plugins: [],
};
export default config;
