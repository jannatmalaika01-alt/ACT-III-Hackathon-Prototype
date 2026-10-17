import type { NextConfig } from "next";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

const config: NextConfig = {
  output: "standalone",
  // Approving a case in EVOLUS_MODE=rest waits for the Evolus agent (about 2 minutes), longer than the proxy's default 30 s.
  experimental: { proxyTimeout: 300_000 },
  async rewrites() {
    // Browser calls /api/backend/* -> FastAPI. No CORS problems.
    return [{ source: "/api/backend/:path*", destination: `${BACKEND}/:path*` }];
  },
};

export default config;