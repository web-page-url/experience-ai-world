'use client';

import { motion } from 'framer-motion';
import { Clock, User, ArrowRight, Calendar } from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';

const recentPosts = [
  {
    id: 5,
    title: "OpenAI's GPT-5 Architecture Deep Dive",
    excerpt: "Analyzing the technical architecture behind OpenAI's latest language model and its implications for AI development.",
    coverImage: "/api/placeholder/400/250",
    category: "AI",
    author: {
      name: "Dr. Michael Chen",
      avatar: "/api/placeholder/64/64",
      bio: "AI architect and researcher specializing in large language models and neural network optimization.",
      social: {
        twitter: "michaelchenai",
        linkedin: "michaelchen"
      }
    },
    date: "2025-01-14",
    readTime: "15 min read",
  },
  {
    id: 6,
    title: "Building Scalable ML Pipelines with Kubernetes",
    excerpt: "Learn how to deploy and manage machine learning workloads at scale using Kubernetes and cloud-native tools.",
    coverImage: "/api/placeholder/400/250",
    category: "Tutorials",
    author: {
      name: "Sarah Johnson",
      avatar: "/api/placeholder/64/64",
      bio: "DevOps engineer and ML infrastructure specialist with expertise in cloud-native ML deployments.",
      social: {
        twitter: "sarahjohnsondev",
        linkedin: "sarahjohnson"
      }
    },
    date: "2025-01-13",
    readTime: "12 min read",
  },
  {
    id: 7,
    title: "The Rise of Multimodal AI Models",
    excerpt: "Exploring how AI systems are evolving to process and understand multiple types of data simultaneously.",
    coverImage: "/api/placeholder/400/250",
    category: "Technology",
    author: {
      name: "Prof. David Kim",
      avatar: "/api/placeholder/64/64",
      bio: "Professor of Computer Science and AI researcher focusing on multimodal learning and computer vision.",
      social: {
        twitter: "davidkimai",
        linkedin: "davidkim"
      }
    },
    date: "2025-01-11",
    readTime: "9 min read",
  },
  {
    id: 8,
    title: "AI Safety Research: Current State and Future Directions",
    excerpt: "A comprehensive overview of the current landscape in AI safety research and alignment challenges.",
    coverImage: "/api/placeholder/400/250",
    category: "AI Ethics",
    author: {
      name: "Dr. Emily Watson",
      avatar: "/api/placeholder/64/64",
      bio: "AI safety researcher and ethicist working on alignment problems and responsible AI development.",
      social: {
        twitter: "emilywatsonai",
        linkedin: "emilywatson"
      }
    },
    date: "2025-01-09",
    readTime: "14 min read",
  },
  {
    id: 9,
    title: "Fine-tuning LLMs for Domain-Specific Tasks",
    excerpt: "Practical guide to adapting large language models for specialized use cases and industry applications.",
    coverImage: "/api/placeholder/400/250",
    category: "Tutorials",
    author: {
      name: "Alex Rodriguez",
      avatar: "/api/placeholder/64/64",
      bio: "Machine learning engineer specializing in model fine-tuning and deployment for enterprise applications.",
      social: {
        twitter: "alexrodriguezml",
        linkedin: "alexrodriguez"
      }
    },
    date: "2025-01-07",
    readTime: "11 min read",
  },
  {
    id: 10,
    title: "The Future of AI Hardware: Neuromorphic Computing",
    excerpt: "Exploring brain-inspired computing architectures and their potential to revolutionize AI performance.",
    coverImage: "/api/placeholder/400/250",
    category: "Technology",
    author: {
      name: "Dr. Lisa Park",
      avatar: "/api/placeholder/64/64",
      bio: "Computer architect and researcher in neuromorphic computing and brain-inspired hardware design.",
      social: {
        twitter: "lisaparkneuromorph",
        linkedin: "lisapark"
      }
    },
    date: "2025-01-06",
    readTime: "10 min read",
  },
];

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

export default function RecentPosts() {
  return (
    <section className="py-20 bg-background-secondary">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial="initial"
          whileInView="animate"
          viewport={{ once: true }}
          variants={staggerChildren}
          className="text-center mb-16"
        >
          <motion.div variants={fadeInUp} className="inline-flex items-center gap-2 px-4 py-2 bg-accent-pink/10 border border-accent-pink/20 rounded-full mb-6">
            <span className="text-sm font-medium text-accent-pink">Latest Content</span>
          </motion.div>

          <motion.h2
            variants={fadeInUp}
            className="text-4xl md:text-5xl font-bold mb-6"
          >
            Recent <span className="text-gradient">Articles</span>
          </motion.h2>

          <motion.p
            variants={fadeInUp}
            className="text-xl text-foreground/70 max-w-3xl mx-auto"
          >
            Stay updated with our latest insights, tutorials, and analysis on AI and technology
          </motion.p>
        </motion.div>

        <motion.div
          initial="initial"
          whileInView="animate"
          viewport={{ once: true }}
          variants={staggerChildren}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
        >
          {recentPosts.map((post) => (
            <motion.article
              key={post.id}
              variants={fadeInUp}
              className="card group cursor-pointer"
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
                  <span className="px-2 py-1 bg-accent-blue text-white text-xs font-medium rounded-full">
                    {post.category}
                  </span>
                </div>
              </div>

              {/* Content */}
              <div className="space-y-3">
                <h3 className="text-lg font-bold group-hover:text-accent-blue transition-colors duration-200 line-clamp-2">
                  {post.title}
                </h3>

                <p className="text-foreground/70 text-sm line-clamp-2">
                  {post.excerpt}
                </p>

                {/* Meta Information */}
                <div className="flex items-center justify-between text-xs text-foreground/60 pt-2 border-t border-glass-border">
                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1">
                      <User className="w-3 h-3" />
                      <span>{post.author.name}</span>
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

                {/* Read More Link */}
                <Link
                  href={`/blog/${post.id}`}
                  className="inline-flex items-center gap-1 text-accent-blue hover:text-accent-purple transition-colors duration-200 text-sm font-medium group/link"
                >
                  Read More
                  <ArrowRight className="w-3 h-3 group-hover/link:translate-x-1 transition-transform" />
                </Link>
              </div>
            </motion.article>
          ))}
        </motion.div>

        {/* Load More Button */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.8 }}
          className="text-center mt-12"
        >
          <motion.button
            className="btn-secondary"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <span className="flex items-center gap-2">
              Load More Posts
              <ArrowRight className="w-5 h-5" />
            </span>
          </motion.button>
        </motion.div>
      </div>
    </section>
  );
}
