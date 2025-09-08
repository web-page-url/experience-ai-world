import { Metadata } from 'next';
import { notFound } from 'next/navigation';
import BlogPost from '@/components/blog/BlogPost';

// Mock blog posts data - in a real app, this would come from a CMS or database
const blogPosts = [
  {
    id: '1',
    title: "The Future of AI: From ChatGPT to AGI",
    excerpt: "Explore how artificial intelligence is evolving from conversational models to artificial general intelligence, and what it means for humanity.",
    content: `
      <h2>The Evolution of AI</h2>
      <p>Artificial Intelligence has come a long way since its inception. From simple rule-based systems to today's sophisticated neural networks, the journey has been remarkable.</p>

      <h2>Current State of AI</h2>
      <p>Today, we're witnessing unprecedented advances in AI capabilities. Models like GPT-4 and Claude have demonstrated near-human level performance in various tasks.</p>

      <h2>The Path to AGI</h2>
      <p>Artificial General Intelligence (AGI) represents the next frontier. Unlike narrow AI systems designed for specific tasks, AGI would possess the ability to understand, learn, and apply intelligence across any domain.</p>

      <blockquote>
        "The development of AGI will be the most important technological event in human history." - Various AI researchers
      </blockquote>

      <h2>Challenges Ahead</h2>
      <p>While the potential benefits of AGI are enormous, so too are the challenges. Ensuring safe and beneficial AI development requires careful consideration of ethical, technical, and societal implications.</p>

      <h2>Conclusion</h2>
      <p>As we stand on the brink of this technological revolution, it's crucial that we approach AGI development with both ambition and caution.</p>
    `,
    coverImage: "/api/placeholder/800/400",
    category: "AI",
    author: {
      name: "Dr. Sarah Chen",
      avatar: "/api/placeholder/100/100",
      bio: "AI Researcher and Professor at Stanford University",
      social: {
        twitter: "@sarahchen_ai",
        linkedin: "sarahchen-ai"
      }
    },
    date: "2025-01-15",
    readTime: "8 min read",
    tags: ["AI", "AGI", "Future", "Technology"],
    likes: 142,
    comments: 23,
  },
  {
    id: '2',
    title: "Building Your First Machine Learning Model",
    excerpt: "A comprehensive guide for beginners to create, train, and deploy their first ML model using Python and TensorFlow.",
    content: `
      <h2>Getting Started with Machine Learning</h2>
      <p>Machine Learning is a powerful tool that can help solve complex problems. This guide will walk you through building your first ML model from scratch.</p>

      <h2>Prerequisites</h2>
      <p>Before we begin, make sure you have Python installed along with the following libraries:</p>
      <ul>
        <li>TensorFlow or PyTorch</li>
        <li>NumPy</li>
        <li>Pandas</li>
        <li>Scikit-learn</li>
        <li>Matplotlib</li>
      </ul>

      <h2>Understanding the Problem</h2>
      <p>The first step in any ML project is clearly defining the problem you're trying to solve. Are you predicting house prices, classifying images, or something else?</p>

      <h2>Data Preparation</h2>
      <p>Quality data is the foundation of successful ML models. Learn about data cleaning, feature engineering, and preprocessing techniques.</p>

      <h2>Model Selection</h2>
      <p>Choose the right algorithm for your problem. We'll cover linear regression, decision trees, neural networks, and more.</p>

      <h2>Training and Evaluation</h2>
      <p>Learn how to train your model, evaluate its performance, and iterate on improvements.</p>

      <h2>Deployment</h2>
      <p>Finally, we'll cover how to deploy your trained model to production and make predictions on new data.</p>
    `,
    coverImage: "/api/placeholder/800/400",
    category: "Tutorials",
    author: {
      name: "Mike Johnson",
      avatar: "/api/placeholder/100/100",
      bio: "Machine Learning Engineer at Google",
      social: {
        twitter: "@mike_ml",
        linkedin: "mikejohnson-ml"
      }
    },
    date: "2025-01-12",
    readTime: "12 min read",
    tags: ["Machine Learning", "Python", "TensorFlow", "Tutorial"],
    likes: 89,
    comments: 15,
  },
];

interface BlogPostPageProps {
  params: {
    id: string;
  };
}

export async function generateMetadata({ params }: BlogPostPageProps): Promise<Metadata> {
  const post = blogPosts.find(p => p.id === params.id);

  if (!post) {
    return {
      title: 'Post Not Found | Experience AI World',
    };
  }

  return {
    title: `${post.title} | Experience AI World`,
    description: post.excerpt,
    openGraph: {
      title: post.title,
      description: post.excerpt,
      type: 'article',
      images: [
        {
          url: post.coverImage,
          width: 800,
          height: 400,
          alt: post.title,
        },
      ],
    },
    twitter: {
      card: 'summary_large_image',
      title: post.title,
      description: post.excerpt,
      images: [post.coverImage],
    },
  };
}

export default function BlogPostPage({ params }: BlogPostPageProps) {
  const post = blogPosts.find(p => p.id === params.id);

  if (!post) {
    notFound();
  }

  return <BlogPost post={post} />;
}

// Generate static paths for all blog posts
export async function generateStaticParams() {
  return blogPosts.map((post) => ({
    id: post.id,
  }));
}
