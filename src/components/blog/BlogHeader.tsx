'use client';

import { motion } from 'framer-motion';
import { BookOpen, TrendingUp, Users } from 'lucide-react';

const fadeInUp = {
  initial: { opacity: 0, y: 60 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6 }
};

const staggerChildren = {
  animate: {
    transition: {
      staggerChildren: 0.2
    }
  }
};

export default function BlogHeader() {
  return (
    <section className="py-20 bg-gradient-to-br from-accent-blue/5 via-accent-purple/5 to-accent-pink/5">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <motion.div
          initial="initial"
          animate="animate"
          variants={staggerChildren}
          className="space-y-8"
        >
          {/* Badge */}
          <motion.div
            variants={fadeInUp}
            className="inline-flex items-center gap-2 px-4 py-2 bg-accent-purple/10 border border-accent-purple/20 rounded-full"
          >
            <BookOpen className="w-4 h-4 text-accent-purple" />
            <span className="text-sm font-medium text-accent-purple">
              AI & Technology Blog
            </span>
          </motion.div>

          {/* Main Headline */}
          <motion.h1
            variants={fadeInUp}
            className="text-5xl md:text-6xl font-bold leading-tight"
          >
            Latest <span className="cyber-text">Articles</span> &
            <br />
            <span className="cyber-text">Insights</span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            variants={fadeInUp}
            className="text-xl md:text-2xl text-foreground/80 max-w-3xl mx-auto leading-relaxed"
          >
            Discover cutting-edge AI research, practical tutorials, industry analysis,
            and expert opinions from leading voices in technology.
          </motion.p>

          {/* Stats */}
          <motion.div
            variants={fadeInUp}
            className="flex flex-wrap justify-center gap-8 text-center"
          >
            <div className="space-y-2">
              <div className="flex items-center justify-center gap-2">
                <TrendingUp className="w-5 h-5 text-accent-blue" />
                <span className="text-2xl font-bold">500+</span>
              </div>
              <p className="text-sm text-foreground/70">Articles Published</p>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-center gap-2">
                <Users className="w-5 h-5 text-accent-purple" />
                <span className="text-2xl font-bold">50K+</span>
              </div>
              <p className="text-sm text-foreground/70">Monthly Readers</p>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-center gap-2">
                <BookOpen className="w-5 h-5 text-accent-pink" />
                <span className="text-2xl font-bold">15</span>
              </div>
              <p className="text-sm text-foreground/70">Categories</p>
            </div>
          </motion.div>

          {/* Featured Topics */}
          <motion.div
            variants={fadeInUp}
            className="flex flex-wrap justify-center gap-3"
          >
            {['AI Ethics', 'Machine Learning', 'Deep Learning', 'Startups', 'Tutorials', 'Reviews'].map((topic) => (
              <motion.span
                key={topic}
                className="px-4 py-2 bg-background-secondary hover:bg-background-tertiary border border-glass-border rounded-full text-sm font-medium transition-colors duration-200 cursor-pointer"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {topic}
              </motion.span>
            ))}
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}
