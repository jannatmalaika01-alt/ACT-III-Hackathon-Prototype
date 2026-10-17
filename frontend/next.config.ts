import type { NextConfig } from "next";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

const config: NextConfig = {
  output: "standalone",
  devIndicators: false,
  // Approving in EVOLUS_MODE=rest now returns at once (the Evolus agent runs in the background). This longer proxy timeout is a safety
  // net for EVOLUS_BACKGROUND=false, where the request waits for the agent.
  experimental: { proxyTimeout: 300_000 },
  async rewrites() {
    // Browser calls /api/backend/* -> FastAPI. No CORS problems.
    return [{ source: "/api/backend/:path*", destination: `${BACKEND}/:path*` }];
  },
};

export default config;