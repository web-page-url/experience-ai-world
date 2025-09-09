'use client';

import { motion } from 'framer-motion';
import { Clock, User, Calendar, Heart, MessageSquare, Share2, Twitter, Linkedin, Link2, ArrowLeft } from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';
import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface BlogPostProps {
  post: {
    id: string;
    title: string;
    excerpt: string;
    content: string;
    coverImage: string;
    category: string;
    author: {
      name: string;
      avatar: string;
      bio: string;
      social?: {
        twitter?: string;
        linkedin?: string;
      };
    };
    date: string;
    readTime: string;
    tags: string[];
    likes: number;
    comments: number;
  };
  recommendedArticles?: Array<{
    id: string;
    title: string;
    excerpt: string;
    category: string;
    tags: string[];
    relevanceScore?: number;
  }>;
}

export default function BlogPost({ post, recommendedArticles = [] }: BlogPostProps) {
  const [liked, setLiked] = useState(false);
  const [likesCount, setLikesCount] = useState(post.likes);

  const handleLike = () => {
    setLiked(!liked);
    setLikesCount(prev => liked ? prev - 1 : prev + 1);
  };

  const handleShare = (platform: 'twitter' | 'linkedin' | 'copy') => {
    const url = window.location.href;
    const text = `Check out "${post.title}" by ${post.author.name}`;

    switch (platform) {
      case 'twitter':
        window.open(`https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`);
        break;
      case 'linkedin':
        window.open(`https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(url)}`);
        break;
      case 'copy':
        navigator.clipboard.writeText(url);
        // You could add a toast notification here
        break;
    }
  };

  const fadeInUp = {
    initial: { opacity: 0, y: 60 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6 }
  };

  return (
    <article className="min-h-screen">
      {/* Back to Blog Button */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 pt-8">
        <Link href="/blog">
          <motion.button
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center gap-2 text-foreground/60 hover:text-accent-blue transition-colors duration-200 mb-8"
            whileHover={{ x: -5 }}
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Blog
          </motion.button>
        </Link>
      </div>

      {/* Hero Section */}
      <motion.section
        initial="initial"
        animate="animate"
        variants={fadeInUp}
        className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 mb-12"
      >
        {/* Category Badge */}
        <div className="mb-6">
          <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-accent-blue/10 text-accent-blue">
            {post.category}
          </span>
        </div>

        {/* Title */}
        <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold leading-tight mb-6">
          {post.title}
        </h1>

        {/* Excerpt */}
        <p className="text-xl text-foreground/70 mb-8 leading-relaxed">
          {post.excerpt}
        </p>

        {/* Meta Information */}
        <div className="flex flex-wrap items-center gap-6 mb-8 text-sm text-foreground/60">
          <div className="flex items-center gap-2">
            <Image
              src={post.author.avatar}
              alt={post.author.name}
              width={32}
              height={32}
              className="rounded-full"
            />
            <span className="font-medium">{post.author.name}</span>
          </div>
          <div className="flex items-center gap-1">
            <Calendar className="w-4 h-4" />
            <span>{new Date(post.date).toLocaleDateString('en-US', {
              year: 'numeric',
              month: 'long',
              day: 'numeric'
            })}</span>
          </div>
          <div className="flex items-center gap-1">
            <Clock className="w-4 h-4" />
            <span>{post.readTime}</span>
          </div>
        </div>

        {/* Tags */}
        <div className="flex flex-wrap gap-2 mb-8">
          {post.tags.map((tag) => (
            <span
              key={tag}
              className="px-3 py-1 bg-background-secondary text-foreground/70 text-sm rounded-full"
            >
              #{tag}
            </span>
          ))}
        </div>

        {/* Cover Image */}
        <div className="relative h-64 md:h-96 lg:h-[500px] mb-8 overflow-hidden rounded-2xl">
          <Image
            src={post.coverImage}
            alt={post.title}
            fill
            className="object-cover"
            priority
          />
        </div>
      </motion.section>

      {/* Content */}
      <section className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 60 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="prose prose-lg max-w-none prose-headings:text-foreground prose-p:text-foreground prose-strong:text-foreground prose-code:text-accent-blue prose-a:text-accent-blue hover:prose-a:text-accent-purple prose-blockquote:border-accent-blue prose-blockquote:text-foreground/80"
        >
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              code({ className, children, ...props }) {
                const match = /language-(\w+)/.exec(className || '');
                const isInline = !match || !String(children).includes('\n');

                return !isInline ? (
                  <SyntaxHighlighter
                    style={oneDark as any}
                    language={match ? match[1] : 'text'}
                    PreTag="div"
                    className="rounded-lg my-4"
                  >
                    {String(children).replace(/\n$/, '')}
                  </SyntaxHighlighter>
                ) : (
                  <code className="bg-background-secondary px-1 py-0.5 rounded text-sm font-mono" {...props}>
                    {children}
                  </code>
                );
              },
              pre({ children, ...props }) {
                return (
                  <pre className="bg-background-secondary p-4 rounded-lg overflow-x-auto my-4" {...props}>
                    {children}
                  </pre>
                );
              },
              h1({ children, ...props }) {
                return (
                  <h1 className="text-3xl font-bold mb-6 text-foreground border-b border-glass-border pb-2" {...props}>
                    {children}
                  </h1>
                );
              },
              h2({ children, ...props }) {
                return (
                  <h2 className="text-2xl font-bold mb-4 mt-8 text-foreground" {...props}>
                    {children}
                  </h2>
                );
              },
              h3({ children, ...props }) {
                return (
                  <h3 className="text-xl font-semibold mb-3 mt-6 text-foreground" {...props}>
                    {children}
                  </h3>
                );
              },
              blockquote({ children, ...props }) {
                return (
                  <blockquote className="border-l-4 border-accent-blue pl-4 italic text-foreground/80 my-4" {...props}>
                    {children}
                  </blockquote>
                );
              },
              p({ children, ...props }) {
                return (
                  <p className="mb-4 text-foreground/90 leading-relaxed" {...props}>
                    {children}
                  </p>
                );
              },
              a({ children, href, ...props }) {
                return (
                  <a
                    href={href}
                    className="text-accent-blue hover:text-accent-purple underline decoration-2 underline-offset-2 transition-colors"
                    target="_blank"
                    rel="noopener noreferrer"
                    {...props}
                  >
                    {children}
                  </a>
                );
              }
            }}
          >
            {post.content || 'No content available'}
          </ReactMarkdown>
        </motion.div>

        {/* Engagement Section */}
        <motion.div
          initial={{ opacity: 0, y: 60 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.3 }}
          className="mt-12 p-6 bg-background-secondary rounded-2xl border border-glass-border"
        >
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-6">
              <button
                onClick={handleLike}
                className={`flex items-center gap-2 transition-colors ${
                  liked ? 'text-red-500' : 'text-foreground/60 hover:text-red-500'
                }`}
              >
                <Heart className={`w-5 h-5 ${liked ? 'fill-current' : ''}`} />
                <span>{likesCount}</span>
              </button>

              <div className="flex items-center gap-2 text-foreground/60">
                <MessageSquare className="w-5 h-5" />
                <span>{post.comments}</span>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-sm text-foreground/60 mr-2">Share:</span>
              <button
                onClick={() => handleShare('twitter')}
                className="p-2 text-foreground/60 hover:text-blue-400 transition-colors"
                aria-label="Share on Twitter"
              >
                <Twitter className="w-4 h-4" />
              </button>
              <button
                onClick={() => handleShare('linkedin')}
                className="p-2 text-foreground/60 hover:text-blue-600 transition-colors"
                aria-label="Share on LinkedIn"
              >
                <Linkedin className="w-4 h-4" />
              </button>
              <button
                onClick={() => handleShare('copy')}
                className="p-2 text-foreground/60 hover:text-accent-blue transition-colors"
                aria-label="Copy link"
              >
                <Link2 className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Recommended Articles */}
          {recommendedArticles.length > 0 && (
            <div className="border-t border-glass-border pt-6">
              <h4 className="font-semibold mb-4 flex items-center gap-2">
                📚 Recommended Articles
                <span className="text-xs bg-accent-blue/10 text-accent-blue px-2 py-1 rounded-full">
                  Created by Anubhav
                </span>
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {recommendedArticles.map((article) => (
                  <Link key={article.id} href={`/blog/${article.id}`}>
                    <div className="p-4 bg-background rounded-lg border border-glass-border hover:border-accent-blue/50 transition-colors cursor-pointer group">
                      <div className="flex items-start justify-between mb-2">
                        <span className="text-xs bg-accent-blue/10 text-accent-blue px-2 py-1 rounded-full">
                          {article.category}
                        </span>
                        {article.relevanceScore && article.relevanceScore > 5 && (
                          <span className="text-xs text-accent-purple">⭐ Top Match</span>
                        )}
                      </div>
                      <h5 className="font-medium mb-2 group-hover:text-accent-blue transition-colors line-clamp-2">
                        {article.title}
                      </h5>
                      <p className="text-sm text-foreground/60 line-clamp-2 mb-3">
                        {article.excerpt}
                      </p>
                      <div className="flex flex-wrap gap-1">
                        {article.tags.slice(0, 2).map((tag) => (
                          <span
                            key={tag}
                            className="text-xs bg-background-secondary text-foreground/50 px-2 py-0.5 rounded"
                          >
                            #{tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
              <div className="mt-4 text-center">
                <p className="text-sm text-foreground/50 italic">
                  ✨ All articles are created by Anubhav
                </p>
              </div>
            </div>
          )}
        </motion.div>
      </section>
    </article>
  );
}
