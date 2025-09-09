import { NextRequest, NextResponse } from 'next/server';

// AI-themed image templates
const aiThemes = [
  {
    name: 'neural-network',
    gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    icon: '🧠',
    title: 'Neural Networks',
    description: 'Deep Learning AI'
  },
  {
    name: 'robotics',
    gradient: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    icon: '🤖',
    title: 'AI Robotics',
    description: 'Intelligent Automation'
  },
  {
    name: 'quantum-computing',
    gradient: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    icon: '⚛️',
    title: 'Quantum AI',
    description: 'Next-Gen Computing'
  },
  {
    name: 'space-exploration',
    gradient: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
    icon: '🚀',
    title: 'Space AI',
    description: 'Cosmic Intelligence'
  },
  {
    name: 'data-science',
    gradient: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
    icon: '📊',
    title: 'Data Science',
    description: 'AI Analytics'
  },
  {
    name: 'future-tech',
    gradient: 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)',
    icon: '🔮',
    title: 'Future AI',
    description: 'Tomorrow\'s Technology'
  },
  {
    name: 'machine-learning',
    gradient: 'linear-gradient(135deg, #d299c2 0%, #fef9d7 100%)',
    icon: '🎯',
    title: 'ML Models',
    description: 'Smart Algorithms'
  },
  {
    name: 'cybersecurity',
    gradient: 'linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)',
    icon: '🔒',
    title: 'AI Security',
    description: 'Intelligent Protection'
  }
];

function generateAIPattern(width: number, height: number) {
  const themeIndex = Math.floor(Math.random() * aiThemes.length);
  const theme = aiThemes[themeIndex];

  // Create an AI-themed SVG with geometric patterns
  const svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <pattern id="aiGrid" x="0" y="0" width="40" height="40" patternUnits="userSpaceOnUse">
        <circle cx="20" cy="20" r="1" fill="rgba(255,255,255,0.3)"/>
        <line x1="0" y1="20" x2="40" y2="20" stroke="rgba(255,255,255,0.2)" stroke-width="0.5"/>
        <line x1="20" y1="0" x2="20" y2="40" stroke="rgba(255,255,255,0.2)" stroke-width="0.5"/>
      </pattern>
      <radialGradient id="aiGlow" cx="30%" cy="30%" r="70%">
        <stop offset="0%" style="stop-color:rgba(255,255,255,0.8);stop-opacity:1" />
        <stop offset="100%" style="stop-color:rgba(255,255,255,0.1);stop-opacity:0" />
      </radialGradient>
    </defs>

    <!-- Background gradient -->
    <rect width="100%" height="100%" fill="url(#aiGrid)"/>
    <rect width="100%" height="100%" fill="${theme.gradient}"/>

    <!-- AI Circuit pattern -->
    <g opacity="0.3">
      ${Array.from({length: 15}, (_, i) => `
        <line x1="${Math.random() * width}" y1="0" x2="${Math.random() * width}" y2="${height}"
              stroke="rgba(255,255,255,0.4)" stroke-width="1" opacity="0.6"/>
        <line x1="0" y1="${Math.random() * height}" x2="${width}" y2="${Math.random() * height}"
              stroke="rgba(255,255,255,0.4)" stroke-width="1" opacity="0.6"/>
      `).join('')}
    </g>

    <!-- Central AI Icon -->
    <g transform="translate(${width/2 - 40}, ${height/2 - 40})">
      <circle cx="40" cy="40" r="35" fill="url(#aiGlow)" opacity="0.7"/>
      <text x="40" y="50" text-anchor="middle" font-size="48" opacity="0.9">${theme.icon}</text>
    </g>

    <!-- Floating data points -->
    ${Array.from({length: 20}, (_, i) => `
      <circle cx="${Math.random() * width}" cy="${Math.random() * height}" r="${2 + Math.random() * 3}"
              fill="rgba(255,255,255,0.8)" opacity="0.6"/>
    `).join('')}

    <!-- Title and description -->
    <text x="50%" y="${height - 60}" text-anchor="middle" fill="white" font-family="Arial, sans-serif"
          font-size="18" font-weight="bold" opacity="0.9">
      ${theme.title}
    </text>
    <text x="50%" y="${height - 35}" text-anchor="middle" fill="rgba(255,255,255,0.8)" font-family="Arial, sans-serif"
          font-size="14" opacity="0.8">
      ${theme.description}
    </text>

    <!-- AI Brain visualization -->
    <g transform="translate(${width - 100}, 20)">
      ${Array.from({length: 8}, (_, i) => `
        <circle cx="${Math.cos(i * Math.PI / 4) * 15 + 15}" cy="${Math.sin(i * Math.PI / 4) * 15 + 15}"
                r="2" fill="rgba(255,255,255,0.8)" opacity="0.7"/>
        ${i < 7 ? `<line x1="${Math.cos(i * Math.PI / 4) * 15 + 15}" y1="${Math.sin(i * Math.PI / 4) * 15 + 15}"
                        x2="${Math.cos((i+1) * Math.PI / 4) * 15 + 15}" y2="${Math.sin((i+1) * Math.PI / 4) * 15 + 15}"
                        stroke="rgba(255,255,255,0.6)" stroke-width="1" opacity="0.5"/>` : ''}
      `).join('')}
    </g>
  </svg>`;

  return svg;
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ dimensions: string[] }> }
) {
  try {
    const { dimensions } = await params;
    const [width, height] = dimensions;

    // Check if this is a blog post image request (has additional parameters)
    const url = new URL(request.url);
    const postId = url.searchParams.get('postId');

    // Generate AI-themed image
    const aiImageSvg = generateAIPattern(parseInt(width), parseInt(height));

    return new NextResponse(aiImageSvg, {
      headers: {
        'Content-Type': 'image/svg+xml',
        'Cache-Control': 'public, max-age=3600',
      },
    });

  } catch (error) {
    console.error('Error in placeholder API:', error);

    // Ultimate fallback - simple AI-themed SVG
    const fallbackSvg = `<svg width="800" height="400" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="aiGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style="stop-color:#6366f1;stop-opacity:1" />
          <stop offset="100%" style="stop-color:#8b5cf6;stop-opacity:1" />
        </linearGradient>
      </defs>
      <rect width="100%" height="100%" fill="url(#aiGradient)"/>
      <text x="50%" y="40%" text-anchor="middle" fill="white" font-family="Arial, sans-serif" font-size="48" font-weight="bold">
        🤖 AI World
      </text>
      <text x="50%" y="60%" text-anchor="middle" fill="rgba(255,255,255,0.8)" font-family="Arial, sans-serif" font-size="24">
        Experience AI Blog
      </text>
    </svg>`;

    return new NextResponse(fallbackSvg, {
      headers: {
        'Content-Type': 'image/svg+xml',
        'Cache-Control': 'public, max-age=86400',
      },
    });
  }
}
