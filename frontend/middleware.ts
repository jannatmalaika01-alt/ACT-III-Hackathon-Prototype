import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export function middleware(request: NextRequest) {
  // Intercept requests directed to the backend proxy
  if (request.nextUrl.pathname.startsWith("/api/backend")) {
    const requestHeaders = new Headers(request.headers);
    // Bypass Ngrok's free-tier browser interstitial warning page
    requestHeaders.set("ngrok-skip-browser-warning", "true");
    if (!requestHeaders.has("accept")) {
      requestHeaders.set("Accept", "application/json");
    }

    return NextResponse.next({
      request: {
        headers: requestHeaders,
      },
    });
  }

  return NextResponse.next();
}

export const config = {
  matcher: "/api/backend/:path*",
};
