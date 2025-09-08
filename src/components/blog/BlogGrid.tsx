'use client';

import { motion } from 'framer-motion';
import { Clock, User, Calendar, ArrowRight, Heart, MessageSquare, Share2 } from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';
import { useState, useEffect } from 'react';
import { useBlog } from './BlogContext';

const fadeInUp = {
  initial: { opacity: 0, y: 60 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6 }
};

const staggerChildren = {
  animate: {
    transition: {
      staggerChildren: 0.1
    }
  }
};

export default function BlogGrid() {
  const { filteredPosts } = useBlog();
  const [visiblePosts, setVisiblePosts] = useState(9);
  const [likedPosts, setLikedPosts] = useState<Set<string>>(new Set());

  // Reset visible posts when filtered results change significantly
  useEffect(() => {
    // Simple logic: if we have fewer posts than currently visible, adjust down
    // if we have more posts available, show up to 9
    const newVisiblePosts = Math.min(9, filteredPosts.length);
    if (newVisiblePosts !== visiblePosts) {
      setVisiblePosts(newVisiblePosts);
    }
  }, [filteredPosts.length, visiblePosts]);

  const handleLoadMore = () => {
    setVisiblePosts(prev => prev + 6);
  };

  const handleLike = (postId: string) => {
    setLikedPosts(prev => {
      const newSet = new Set(prev);
      if (newSet.has(postId)) {
        newSet.delete(postId);
      } else {
        newSet.add(postId);
      }
      return newSet;
    });
  };

  const displayedPosts = filteredPosts.slice(0, Math.min(visiblePosts, filteredPosts.length));

  return (
    <section className="py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Results Counter */}
        {filteredPosts.length > 0 && (
          <div className="mb-8 text-center">
            <p className="text-foreground/60 text-sm">
              Showing {displayedPosts.length} of {filteredPosts.length} blog posts
            </p>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {displayedPosts.map((post, index) => {
            return (
            <motion.article
              key={post.id}
              variants={fadeInUp}
              className="card group cursor-pointer overflow-hidden cyber-border"
              onClick={(e) => {
                // Only navigate if not clicking on interactive elements
                if (!(e.target as HTMLElement).closest('a, button')) {
                  console.log('🎯 CARD CLICKED:', post.id, post.title);
                  window.location.href = `/blog/${post.id}`;
                }
              }}
            >
              {/* Cover Image */}
              <div className="relative h-48 mb-4 overflow-hidden rounded-xl">
                <Image
                  src={post.coverImage}
                  alt={post.title}
                  fill
                  className="object-cover transition-transform duration-300 group-hover:scale-105"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

                {/* Category Badge */}
                <div className="absolute top-3 left-3">
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                    post.category === 'AI' ? 'bg-blue-500/90 text-white' :
                    post.category === 'Tutorials' ? 'bg-green-500/90 text-white' :
                    post.category === 'Reviews' ? 'bg-purple-500/90 text-white' :
                    post.category === 'Ethics' ? 'bg-gray-600/90 text-white' :
                    'bg-accent-blue text-white'
                  }`}>
                    {post.category}
                  </span>
                </div>

                {/* Featured Badge */}
                {post.featured && (
                  <div className="absolute top-3 right-3">
                    <span className="px-2 py-1 bg-yellow-500/90 text-black text-xs font-medium rounded-full">
                      Featured
                    </span>
                  </div>
                )}
              </div>

              {/* Content */}
              <div className="space-y-4">
                <h3 className="text-lg font-bold group-hover:text-accent-blue transition-colors duration-200 line-clamp-2 cursor-pointer">
                  <Link
                    href={`/blog/${post.id}`}
                    className="block w-full h-full"
                    onClick={(e) => {
                      e.preventDefault();
                      console.log('🎯 CLICKED ARTICLE:', post.id, post.title);
                      console.log('🔗 NAVIGATING TO:', `/blog/${post.id}`);
                      // Force navigation with multiple methods
                      setTimeout(() => {
                        window.location.href = `/blog/${post.id}`;
                      }, 100);
                    }}
                  >
                    {post.title}
                  </Link>
                </h3>

                <p className="text-foreground/70 text-sm line-clamp-3">
                  {post.excerpt}
                </p>

                {/* Meta Information */}
                <div className="flex items-center justify-between text-xs text-foreground/60 pt-2 border-t border-glass-border">
                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1">
                      <User className="w-3 h-3" />
                      <span>{typeof post.author === 'object' ? post.author.name : post.author}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Calendar className="w-3 h-3" />
                      <span>{new Date(post.date).toLocaleDateString()}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    <span>{post.readTime}</span>
                  </div>
                </div>

                {/* Engagement Stats */}
                <div className="flex items-center justify-between pt-2">
                  <div className="flex items-center gap-4">
                    <button
                      onClick={(e) => {
                        e.preventDefault();
                        handleLike(post.id);
                      }}
                      className={`flex items-center gap-1 text-xs transition-colors ${
                        likedPosts.has(post.id)
                          ? 'text-red-500'
                          : 'text-foreground/60 hover:text-red-500'
                      }`}
                    >
                      <Heart
                        className={`w-4 h-4 ${likedPosts.has(post.id) ? 'fill-current' : ''}`}
                      />
                      <span>{likedPosts.has(post.id) ? post.likes + 1 : post.likes}</span>
                    </button>

                    <div className="flex items-center gap-1 text-xs text-foreground/60">
                      <MessageSquare className="w-4 h-4" />
                      <span>{post.comments}</span>
                    </div>
                  </div>

                  <button className="p-1 text-foreground/60 hover:text-accent-blue transition-colors">
                    <Share2 className="w-4 h-4" />
                  </button>
                </div>

                {/* Read More Link */}
                <Link
                  href={`/blog/${post.id}`}
                  className="inline-flex items-center gap-1 text-accent-blue hover:text-accent-purple transition-colors duration-200 text-sm font-medium group/link cursor-pointer"
                  onClick={(e) => {
                    e.preventDefault();
                    console.log('🎯 READ MORE CLICKED:', post.id, post.title);
                    console.log('🔗 NAVIGATING TO:', `/blog/${post.id}`);
                    setTimeout(() => {
                      window.location.href = `/blog/${post.id}`;
                    }, 100);
                  }}
                >
                  Read More
                  <ArrowRight className="w-3 h-3 group-hover/link:translate-x-1 transition-transform" />
                </Link>
              </div>
            </motion.article>
            );
          })}
        </div>

        {/* Load More Button */}
        {visiblePosts < filteredPosts.length && filteredPosts.length > 9 && (
          <div className="text-center mt-12">
            <button
              onClick={handleLoadMore}
              className="btn-secondary cyber-border"
            >
              <span className="flex items-center gap-2">
                Load More Posts ({filteredPosts.length - visiblePosts} remaining)
                <ArrowRight className="w-5 h-5" />
              </span>
            </button>
          </div>
        )}

        {/* No More Posts */}
        {visiblePosts >= filteredPosts.length && filteredPosts.length > 0 && visiblePosts >= 9 && (
          <div className="text-center mt-12 py-8">
            <p className="text-foreground/60">
              You've reached the end of {filteredPosts.length === 9 ? 'our' : 'the filtered'} blog posts.
              {filteredPosts.length < 9 ? ' Check back soon for more content!' : ''}
            </p>
          </div>
        )}

        {/* No Results */}
        {filteredPosts.length === 0 && (
          <div className="text-center mt-12 py-16">
            <p className="text-foreground/60 text-lg mb-4">
              No blog posts found matching your criteria.
            </p>
            <p className="text-foreground/50 text-sm">
              Try adjusting your search query or selecting a different category.
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
