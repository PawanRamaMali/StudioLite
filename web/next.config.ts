import type { NextConfig } from "next";

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
        destination: "http://localhost:8000/api/:path*",
      },
    ];
  },
};

export default nextConfig;
