import type { NextConfig } from "next";

// Allow the launcher to retarget the API when :8000 is taken. STUDIOLITE_API_URL
// is read fresh each time the dev server starts (next reads this file on boot),
// so the launcher can hand us a different port without rebuilding.
const apiBase = process.env.STUDIOLITE_API_URL || "http://localhost:8000";

const nextConfig: NextConfig = {
  output: "standalone",
  // Next 15+ blocks HMR websockets from any origin not in this list, which
  // silently breaks live-reload when the dev server is opened via 127.0.0.1
  // instead of localhost. Allowing both keeps hot-reload working either way.
  allowedDevOrigins: ["localhost", "127.0.0.1"],
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${apiBase}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
