import { MetadataRoute } from 'next'

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: 'Experience AI World | AI & Technology Blog',
    short_name: 'Experience AI World',
    description: 'Stay ahead with the latest in AI & Technology. Daily insights, tutorials, reviews, and trends shaping the future of Artificial Intelligence.',
    start_url: '/',
    display: 'standalone',
    background_color: '#000000',
    theme_color: '#00d4ff',
    icons: [
      {
        src: '/icon-192x192.png',
        sizes: '192x192',
        type: 'image/png',
      },
      {
        src: '/icon-512x512.png',
        sizes: '512x512',
        type: 'image/png',
      },
    ],
  }
}
