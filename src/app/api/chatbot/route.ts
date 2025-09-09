import { GoogleGenerativeAI } from '@google/generative-ai';
import { NextRequest, NextResponse } from 'next/server';

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);

// Available blog posts for the chatbot to reference
const blogPosts = [
  {
    id: '1',
    title: "The Future of AI: From ChatGPT to AGI",
    excerpt: "Explore how artificial intelligence is evolving from conversational models to artificial general intelligence, and what it means for humanity.",
    category: "AI",
    tags: ["AI", "AGI", "Future", "Technology"],
    url: "/blog/1"
  },
  {
    id: '2',
    title: "Building Your First Machine Learning Model",
    excerpt: "A comprehensive guide for beginners to create, train, and deploy their first ML model using Python and TensorFlow.",
    category: "Tutorials",
    tags: ["Machine Learning", "Python", "TensorFlow", "Tutorial", "Beginners"],
    url: "/blog/2"
  },
  {
    id: '3',
    title: "AI Ethics: Navigating the Moral Landscape",
    excerpt: "Understanding the ethical implications of artificial intelligence development and deployment in modern society.",
    category: "Opinion",
    tags: ["AI Ethics", "Bias", "Regulation", "Society", "Future"],
    url: "/blog/3"
  },
  {
    id: '4',
    title: "Claude vs GPT-5: The Ultimate AI Showdown",
    excerpt: "An in-depth comparison of Anthropic's Claude and OpenAI's latest GPT model across various use cases and performance metrics.",
    category: "Reviews",
    tags: ["AI", "Claude", "GPT-5", "Comparison", "Reviews", "Models"],
    url: "/blog/4"
  },
  {
    id: '5',
    title: "OpenAI's GPT-5 Architecture Deep Dive",
    excerpt: "Analyzing the technical architecture behind OpenAI's latest language model and its implications for AI development.",
    category: "AI",
    tags: ["AI", "GPT-5", "Architecture", "Deep Learning", "OpenAI"],
    url: "/blog/5"
  },
  {
    id: '6',
    title: "Quantum Computing: The Future of AI Processing Power",
    excerpt: "Exploring how quantum computers will revolutionize artificial intelligence and solve problems beyond classical computing capabilities.",
    category: "Quantum Computing",
    tags: ["Quantum Computing", "AI", "Future Tech", "Innovation", "Science"],
    url: "/blog/6"
  },
  {
    id: '7',
    title: "AI in Space Exploration: Journey to the Stars",
    excerpt: "How artificial intelligence is transforming space exploration, from autonomous spacecraft to intelligent mission planning and discovery.",
    category: "Space Exploration",
    tags: ["Space Exploration", "AI", "NASA", "Robotics", "Future", "Science"],
    url: "/blog/7"
  },
  {
    id: '8',
    title: "Robotics Revolution: AI-Powered Automation Everywhere",
    excerpt: "From smart factories to intelligent homes, explore how AI-driven robotics is transforming industries and daily life across the globe.",
    category: "Robotics",
    tags: ["Robotics", "AI", "Automation", "Industry 4.0", "Future Tech", "Innovation"],
    url: "/blog/8"
  },
  {
    id: '9',
    title: "Future of Work: AI as Your Ultimate Career Coach",
    excerpt: "Discover how artificial intelligence is reshaping careers, from personalized learning paths to intelligent job matching and skill development.",
    category: "Career Development",
    tags: ["Career Development", "AI", "Future of Work", "Professional Growth", "Technology"],
    url: "/blog/9"
  },
  {
    id: '10',
    title: "Sustainable AI: Green Technology for a Better Planet",
    excerpt: "Exploring how artificial intelligence can drive environmental sustainability, from carbon footprint reduction to intelligent resource management.",
    category: "Sustainability",
    tags: ["Sustainability", "AI", "Climate Change", "Environment", "Green Tech", "Future"],
    url: "/blog/10"
  }
];

function findRelevantArticles(query: string): Array<{post: any, relevance: number}> {
  const queryLower = query.toLowerCase();

  return blogPosts.map(post => {
    let relevance = 0;

    // Check category match
    if (queryLower.includes(post.category.toLowerCase())) {
      relevance += 3;
    }

    // Check tag matches
    post.tags.forEach(tag => {
      if (queryLower.includes(tag.toLowerCase())) {
        relevance += 2;
      }
    });

    // Check title keywords
    const titleWords = post.title.toLowerCase().split(' ');
    titleWords.forEach(word => {
      if (word.length > 3 && queryLower.includes(word)) {
        relevance += 1.5;
      }
    });

    // Check excerpt keywords
    const excerptWords = post.excerpt.toLowerCase().split(' ');
    excerptWords.forEach(word => {
      if (word.length > 3 && queryLower.includes(word)) {
        relevance += 1;
      }
    });

    return { post, relevance };
  })
  .filter(item => item.relevance > 0)
  .sort((a, b) => b.relevance - a.relevance)
  .slice(0, 3);
}

export async function POST(request: NextRequest) {
  try {
    const { message } = await request.json();

    if (!message) {
      return NextResponse.json(
        { error: 'Message is required' },
        { status: 400 }
      );
    }

    // Find relevant articles based on user query
    const relevantArticles = findRelevantArticles(message);

    // Create context for the AI
    const context = `
You are Anubhav's AI Assistant for the Experience AI World blog. You help users find relevant articles and guide them to the right content.

Available articles:
${blogPosts.map(post => `- ${post.title} (${post.category}): ${post.excerpt}`).join('\n')}

User query: "${message}"

${relevantArticles.length > 0 ?
  `Most relevant articles based on the query:\n${relevantArticles.map(item =>
    `- ${item.post.title} (Relevance: ${item.relevance.toFixed(1)}): ${item.post.excerpt}`
  ).join('\n')}` :
  'No highly relevant articles found for this query.'
}

Instructions:
- Be friendly and helpful
- Always mention that everything is created by Anubhav
- Guide users to relevant articles when appropriate
- If no relevant articles are found, suggest browsing the blog or ask for clarification
- Keep responses concise but informative
- Use emojis to make responses more engaging
- If suggesting articles, provide direct links in the format: [Article Title](/blog/article-id)
- Encourage users to explore more content
- End responses with a question to continue the conversation
`;

    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

    const result = await model.generateContent(context);
    const response = result.response;
    const aiResponse = response.text();

    // Add article links to the response
    let enhancedResponse = aiResponse;
    relevantArticles.forEach(item => {
      const linkText = `[${item.post.title}](${item.post.url})`;
      enhancedResponse = enhancedResponse.replace(
        new RegExp(item.post.title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'),
        linkText
      );
    });

    return NextResponse.json({
      response: enhancedResponse,
      relevantArticles: relevantArticles.map(item => ({
        id: item.post.id,
        title: item.post.title,
        excerpt: item.post.excerpt,
        category: item.post.category,
        url: item.post.url,
        relevance: item.relevance
      }))
    });

  } catch (error) {
    console.error('Chatbot error:', error);

    const isNetworkError = error instanceof Error &&
      (error.message.includes('fetch') || error.message.includes('network'));

    return NextResponse.json(
      {
        error: isNetworkError
          ? 'Network error. Please check your connection and try again.'
          : 'Failed to process your message. Please try again.',
        isNetworkError,
        fallbackResponse: "🤖 Hi! I'm Anubhav's AI Assistant! I'm here to help you find the perfect articles on our AI blog. What topics interest you? You can ask about AI, machine learning, robotics, quantum computing, or any other tech topic! 🚀"
      },
      { status: isNetworkError ? 503 : 500 }
    );
  }
}
