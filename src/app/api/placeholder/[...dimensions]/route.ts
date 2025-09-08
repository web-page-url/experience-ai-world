import { NextRequest, NextResponse } from 'next/server';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ dimensions: string[] }> }
) {
  try {
    const { dimensions } = await params;
    const [width, height] = dimensions;

    // Use a more reliable placeholder service
    const placeholderUrl = `https://picsum.photos/${width}/${height}?random=${Math.random()}`;

    try {
      const response = await fetch(placeholderUrl, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (compatible; ExperienceAIWorld/1.0)',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const blob = await response.blob();
      return new NextResponse(blob, {
        headers: {
          'Content-Type': response.headers.get('content-type') || 'image/jpeg',
          'Cache-Control': 'public, max-age=3600',
        },
      });
    } catch (fetchError) {
      console.error('Failed to fetch from primary service:', fetchError);

      // Create a simple SVG placeholder as fallback
      const svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="#6366f1"/>
        <text x="50%" y="50%" text-anchor="middle" dy=".3em" fill="white" font-family="Arial, sans-serif" font-size="16">
          AI Blog
        </text>
      </svg>`;

      return new NextResponse(svg, {
        headers: {
          'Content-Type': 'image/svg+xml',
          'Cache-Control': 'public, max-age=86400',
        },
      });
    }
  } catch (error) {
    console.error('Error in placeholder API:', error);

    // Ultimate fallback - simple colored rectangle
    const svg = `<svg width="400" height="250" xmlns="http://www.w3.org/2000/svg">
      <rect width="100%" height="100%" fill="#e5e7eb"/>
      <text x="50%" y="50%" text-anchor="middle" dy=".3em" fill="#6b7280" font-family="Arial, sans-serif" font-size="14">
        Image
      </text>
    </svg>`;

    return new NextResponse(svg, {
      headers: {
        'Content-Type': 'image/svg+xml',
        'Cache-Control': 'public, max-age=86400',
      },
    });
  }
}
