import type { NextConfig } from "next";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

const config: NextConfig = {
  output: "standalone",
  async rewrites() {
    // Browser calls /api/backend/* -> FastAPI. No CORS problems.
    return [{ source: "/api/backend/:path*", destination: `${BACKEND}/:path*` }];
  },
};

export default config;