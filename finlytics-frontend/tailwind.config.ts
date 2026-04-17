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
        "card-rich": "#121f39",
        border: "rgba(255,255,255,0.08)",
        muted: "#94a3b8",
        primary: "#3b82f6",
        accent: "#22d3ee",
        "accent-soft": "#67e8f9",
        success: "#22c55e",
        warning: "#f59e0b",
        danger: "#ef4444"
      },
      boxShadow: {
        glass: "0 10px 35px rgba(2, 6, 23, 0.5)",
        glow: "0 0 0 1px rgba(59,130,246,0.35), 0 8px 28px rgba(56,189,248,0.2)",
        "premium-soft": "0 14px 38px rgba(2, 8, 23, 0.5)",
        "premium-lg": "0 22px 54px rgba(2, 8, 23, 0.62)",
        "inner-soft": "inset 0 1px 0 rgba(255,255,255,0.08)"
      },
      borderRadius: {
        xl: "1rem",
        "2xl": "1.25rem"
      },
      backgroundImage: {
        "fin-gradient":
          "radial-gradient(1200px 700px at 15% -10%, rgba(59,130,246,0.3), transparent 55%), radial-gradient(1100px 640px at 96% 0%, rgba(34,211,238,0.2), transparent 58%), radial-gradient(780px 420px at 50% 100%, rgba(99,102,241,0.12), transparent 65%), linear-gradient(145deg, #030712 0%, #060d1f 38%, #0b1224 100%)"
      }
    }
  },
  plugins: []
};

export default config;
