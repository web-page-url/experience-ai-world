

import { Metadata } from 'next';
import Hero from '@/components/blog/Hero';
import FeaturedPosts from '@/components/blog/FeaturedPosts';
import Categories from '@/components/blog/Categories';
import Newsletter from '@/components/blog/Newsletter';
import RecentPosts from '@/components/blog/RecentPosts';

export const metadata: Metadata = {
  title: 'Experience AI World | Stay Ahead with AI & Technology',
  description: 'Daily insights, tutorials, reviews, and trends shaping the future of Artificial Intelligence. Join thousands of readers staying ahead in the AI world.',
  keywords: ['AI', 'artificial intelligence', 'technology', 'machine learning', 'tech tutorials', 'AI news', 'tech reviews'],
  openGraph: {
    title: 'Experience AI World | Stay Ahead with AI & Technology',
    description: 'Daily insights, tutorials, reviews, and trends shaping the future of Artificial Intelligence.',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Experience AI World | Stay Ahead with AI & Technology',
    description: 'Daily insights, tutorials, reviews, and trends shaping the future of Artificial Intelligence.',
  },
};

export default function Home() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <Hero />

      {/* Featured Posts */}
      <FeaturedPosts />

      {/* Categories Section */}
      <Categories />

      {/* Recent Posts */}
      <RecentPosts />

      {/* Newsletter Signup */}
      <Newsletter />
    </div>
  );
}
