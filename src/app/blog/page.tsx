import { Metadata } from 'next';
import BlogHeader from '@/components/blog/BlogHeader';
import BlogFilters from '@/components/blog/BlogFilters';
import BlogGrid from '@/components/blog/BlogGrid';
import BlogPagination from '@/components/blog/BlogPagination';

export const metadata: Metadata = {
  title: 'Blog | Experience AI World',
  description: 'Read the latest articles, tutorials, and insights on Artificial Intelligence and Technology.',
  keywords: ['AI blog', 'technology blog', 'AI tutorials', 'tech insights', 'artificial intelligence'],
  openGraph: {
    title: 'Blog | Experience AI World',
    description: 'Read the latest articles, tutorials, and insights on Artificial Intelligence and Technology.',
    type: 'website',
  },
};

export default function BlogPage() {
  return (
    <div className="min-h-screen">
      <BlogHeader />
      <BlogFilters />
      <BlogGrid />
      <BlogPagination />
    </div>
  );
}
