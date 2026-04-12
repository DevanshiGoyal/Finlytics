import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        background: "#050816",
        surface: "#0b1224",
        card: "#111a31",
        border: "rgba(255,255,255,0.08)",
        muted: "#94a3b8",
        primary: "#3b82f6",
        accent: "#22d3ee",
        success: "#22c55e",
        warning: "#f59e0b",
        danger: "#ef4444"
      },
      boxShadow: {
        glass: "0 10px 35px rgba(2, 6, 23, 0.5)",
        glow: "0 0 0 1px rgba(59,130,246,0.35), 0 8px 28px rgba(56,189,248,0.2)"
      },
      borderRadius: {
        xl: "1rem",
        "2xl": "1.25rem"
      },
      backgroundImage: {
        "fin-gradient":
          "radial-gradient(1200px 700px at 15% -10%, rgba(59,130,246,0.25), transparent 55%), radial-gradient(900px 500px at 90% 0%, rgba(34,211,238,0.17), transparent 55%), linear-gradient(135deg, #050816 0%, #080f1f 45%, #0b1224 100%)"
      }
    }
  },
  plugins: []
};

export default config;
