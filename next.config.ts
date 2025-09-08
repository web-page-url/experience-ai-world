import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  images: {
    domains: ['api.placeholder', 'images.unsplash.com', 'picsum.photos'],
    formats: ['image/webp', 'image/avif'],
  },
};

export default nextConfig;
