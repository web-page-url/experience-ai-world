'use client';

import { motion } from 'framer-motion';
import {
  Brain,
  Cpu,
  Rocket,
  BookOpen,
  Star,
  Code,
  TrendingUp,
  Shield
} from 'lucide-react';
import Link from 'next/link';

const categories = [
  {
    id: 'ai',
    name: 'Artificial Intelligence',
    description: 'Latest breakthroughs, research, and applications in AI',
    icon: Brain,
    color: 'from-blue-500 to-cyan-500',
    postCount: 156,
    featured: true,
  },
  {
    id: 'technology',
    name: 'Technology',
    description: 'Cutting-edge tech trends and innovations',
    icon: Cpu,
    color: 'from-purple-500 to-pink-500',
    postCount: 89,
    featured: true,
  },
  {
    id: 'startups',
    name: 'Startups',
    description: 'AI-powered startups and entrepreneurship',
    icon: Rocket,
    color: 'from-green-500 to-emerald-500',
    postCount: 67,
    featured: true,
  },
  {
    id: 'tutorials',
    name: 'Tutorials',
    description: 'Step-by-step guides and learning resources',
    icon: BookOpen,
    color: 'from-orange-500 to-red-500',
    postCount: 134,
    featured: true,
  },
  {
    id: 'reviews',
    name: 'Reviews',
    description: 'AI tools, software, and platform comparisons',
    icon: Star,
    color: 'from-indigo-500 to-purple-500',
    postCount: 78,
    featured: false,
  },
  {
    id: 'coding',
    name: 'Coding',
    description: 'Programming tutorials and development insights',
    icon: Code,
    color: 'from-teal-500 to-blue-500',
    postCount: 92,
    featured: false,
  },
  {
    id: 'trends',
    name: 'Industry Trends',
    description: 'Market analysis and future predictions',
    icon: TrendingUp,
    color: 'from-pink-500 to-rose-500',
    postCount: 45,
    featured: false,
  },
  {
    id: 'ethics',
    name: 'AI Ethics',
    description: 'Responsible AI development and governance',
    icon: Shield,
    color: 'from-gray-600 to-gray-800',
    postCount: 34,
    featured: false,
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

export default function Categories() {
  const featuredCategories = categories.filter(cat => cat.featured);
  const otherCategories = categories.filter(cat => !cat.featured);

  return (
    <section className="py-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial="initial"
          whileInView="animate"
          viewport={{ once: true }}
          variants={staggerChildren}
          className="text-center mb-16"
        >
          <motion.div variants={fadeInUp} className="inline-flex items-center gap-2 px-4 py-2 bg-accent-cyan/10 border border-accent-cyan/20 rounded-full mb-6">
            <span className="text-sm font-medium text-accent-cyan">Explore Topics</span>
          </motion.div>

          <motion.h2
            variants={fadeInUp}
            className="text-4xl md:text-5xl font-bold mb-6"
          >
            Discover by <span className="text-gradient">Category</span>
          </motion.h2>

          <motion.p
            variants={fadeInUp}
            className="text-xl text-foreground/70 max-w-3xl mx-auto"
          >
            Find exactly what you're looking for with our organized content categories
          </motion.p>
        </motion.div>

        {/* Featured Categories */}
        <motion.div
          initial="initial"
          whileInView="animate"
          viewport={{ once: true }}
          variants={staggerChildren}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12"
        >
          {featuredCategories.map((category) => (
            <motion.div
              key={category.id}
              variants={fadeInUp}
              className="group"
            >
              <Link href={`/categories/${category.id}`}>
                <div className="card h-full group-hover:scale-105 transition-transform duration-300 cyber-border">
                  <div className="text-center space-y-4">
                    {/* Icon */}
                    <motion.div
                      className={`w-16 h-16 mx-auto rounded-2xl bg-gradient-to-r ${category.color} flex items-center justify-center shadow-lg group-hover:shadow-xl transition-shadow duration-300`}
                      whileHover={{ rotate: 360 }}
                      transition={{ duration: 0.6 }}
                    >
                      <category.icon className="w-8 h-8 text-white" />
                    </motion.div>

                    {/* Content */}
                    <div>
                      <h3 className="text-lg font-bold mb-2 group-hover:text-accent-blue transition-colors duration-200">
                        {category.name}
                      </h3>
                      <p className="text-foreground/70 text-sm mb-3">
                        {category.description}
                      </p>
                      <div className="text-xs text-foreground/50">
                        {category.postCount} articles
                      </div>
                    </div>
                  </div>
                </div>
              </Link>
            </motion.div>
          ))}
        </motion.div>

        {/* Other Categories */}
        <motion.div
          initial="initial"
          whileInView="animate"
          viewport={{ once: true }}
          variants={staggerChildren}
          className="grid grid-cols-2 md:grid-cols-4 gap-4"
        >
          {otherCategories.map((category) => (
            <motion.div
              key={category.id}
              variants={fadeInUp}
              className="group"
            >
              <Link href={`/categories/${category.id}`}>
                <div className="p-4 rounded-xl bg-background-secondary hover:bg-background-tertiary transition-colors duration-200 border border-glass-border group-hover:border-accent-blue/20">
                  <div className="flex items-center gap-3">
                    <motion.div
                      className={`w-10 h-10 rounded-lg bg-gradient-to-r ${category.color} flex items-center justify-center flex-shrink-0`}
                      whileHover={{ scale: 1.1 }}
                      transition={{ duration: 0.2 }}
                    >
                      <category.icon className="w-5 h-5 text-white" />
                    </motion.div>
                    <div className="min-w-0 flex-1">
                      <h4 className="font-medium text-sm truncate group-hover:text-accent-blue transition-colors duration-200">
                        {category.name}
                      </h4>
                      <p className="text-xs text-foreground/60 truncate">
                        {category.postCount} posts
                      </p>
                    </div>
                  </div>
                </div>
              </Link>
            </motion.div>
          ))}
        </motion.div>

        {/* Explore All Button */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.6 }}
          className="text-center mt-12"
        >
          <Link href="/categories">
            <motion.button
              className="btn-secondary cyber-border"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Explore All Categories
            </motion.button>
          </Link>
        </motion.div>
      </div>
    </section>
  );
}
