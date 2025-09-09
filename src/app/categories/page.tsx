'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { ArrowLeft, Brain, Cpu, Rocket, BookOpen, TrendingUp, Zap, Code, Globe, Database } from 'lucide-react';

const categories = [
  {
    id: 'artificial-intelligence',
    name: 'Artificial Intelligence',
    description: 'Latest breakthroughs, research, and applications in AI',
    articleCount: 156,
    icon: Brain,
    color: 'from-blue-500 to-purple-600',
    bgColor: 'bg-blue-50 dark:bg-blue-900/20',
    textColor: 'text-blue-600 dark:text-blue-400',
    borderColor: 'border-blue-200 dark:border-blue-800'
  },
  {
    id: 'technology',
    name: 'Technology',
    description: 'Cutting-edge tech trends and innovations',
    articleCount: 89,
    icon: Cpu,
    color: 'from-green-500 to-teal-600',
    bgColor: 'bg-green-50 dark:bg-green-900/20',
    textColor: 'text-green-600 dark:text-green-400',
    borderColor: 'border-green-200 dark:border-green-800'
  },
  {
    id: 'startups',
    name: 'Startups',
    description: 'AI-powered startups and entrepreneurship',
    articleCount: 67,
    icon: Rocket,
    color: 'from-orange-500 to-red-600',
    bgColor: 'bg-orange-50 dark:bg-orange-900/20',
    textColor: 'text-orange-600 dark:text-orange-400',
    borderColor: 'border-orange-200 dark:border-orange-800'
  },
  {
    id: 'tutorials',
    name: 'Tutorials',
    description: 'Step-by-step guides and learning resources',
    articleCount: 134,
    icon: BookOpen,
    color: 'from-purple-500 to-pink-600',
    bgColor: 'bg-purple-50 dark:bg-purple-900/20',
    textColor: 'text-purple-600 dark:text-purple-400',
    borderColor: 'border-purple-200 dark:border-purple-800'
  },
  {
    id: 'machine-learning',
    name: 'Machine Learning',
    description: 'Algorithms, models, and practical applications',
    articleCount: 92,
    icon: Database,
    color: 'from-indigo-500 to-blue-600',
    bgColor: 'bg-indigo-50 dark:bg-indigo-900/20',
    textColor: 'text-indigo-600 dark:text-indigo-400',
    borderColor: 'border-indigo-200 dark:border-indigo-800'
  },
  {
    id: 'quantum-computing',
    name: 'Quantum Computing',
    description: 'Next-generation computing and quantum algorithms',
    articleCount: 45,
    icon: Zap,
    color: 'from-cyan-500 to-blue-600',
    bgColor: 'bg-cyan-50 dark:bg-cyan-900/20',
    textColor: 'text-cyan-600 dark:text-cyan-400',
    borderColor: 'border-cyan-200 dark:border-cyan-800'
  },
  {
    id: 'robotics',
    name: 'Robotics',
    description: 'AI-driven robotics and automation technologies',
    articleCount: 78,
    icon: Code,
    color: 'from-emerald-500 to-green-600',
    bgColor: 'bg-emerald-50 dark:bg-emerald-900/20',
    textColor: 'text-emerald-600 dark:text-emerald-400',
    borderColor: 'border-emerald-200 dark:border-emerald-800'
  },
  {
    id: 'space-exploration',
    name: 'Space Exploration',
    description: 'AI applications in space technology and exploration',
    articleCount: 63,
    icon: Globe,
    color: 'from-violet-500 to-purple-600',
    bgColor: 'bg-violet-50 dark:bg-violet-900/20',
    textColor: 'text-violet-600 dark:text-violet-400',
    borderColor: 'border-violet-200 dark:border-violet-800'
  },
  {
    id: 'ai-ethics',
    name: 'AI Ethics',
    description: 'Responsible AI development and ethical considerations',
    articleCount: 41,
    icon: TrendingUp,
    color: 'from-rose-500 to-pink-600',
    bgColor: 'bg-rose-50 dark:bg-rose-900/20',
    textColor: 'text-rose-600 dark:text-rose-400',
    borderColor: 'border-rose-200 dark:border-rose-800'
  }
];

export default function CategoriesPage() {
  return (
    <div className="min-h-screen">
      {/* Back to Home Button */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-8">
        <Link href="/">
          <motion.button
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center gap-2 text-foreground/60 hover:text-accent-blue transition-colors duration-200 mb-8"
            whileHover={{ x: -5 }}
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Home
          </motion.button>
        </Link>
      </div>

      {/* Hero Section */}
      <motion.section
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12"
      >
        <div className="text-center mb-16">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-5xl md:text-6xl lg:text-7xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-orange-600 bg-clip-text text-transparent mb-6"
          >
            Discover by Category
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="text-xl md:text-2xl text-foreground/70 max-w-3xl mx-auto leading-relaxed"
          >
            Find exactly what you're looking for with our organized content categories.
            Explore cutting-edge topics in AI, technology, and innovation.
          </motion.p>
        </div>

        {/* Statistics */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-16"
        >
          {[
            { label: 'Total Articles', value: '736' },
            { label: 'Categories', value: categories.length.toString() },
            { label: 'Topics Covered', value: '50+' },
            { label: 'Daily Updates', value: '12' }
          ].map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.8 + index * 0.1 }}
              className="bg-background-secondary/50 backdrop-blur-sm rounded-2xl p-6 text-center border border-glass-border"
            >
              <div className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-accent-blue to-accent-purple bg-clip-text text-transparent mb-2">
                {stat.value}
              </div>
              <div className="text-sm text-foreground/60 uppercase tracking-wide">
                {stat.label}
              </div>
            </motion.div>
          ))}
        </motion.div>
      </motion.section>

      {/* Categories Grid */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-20">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
        >
          {categories.map((category, index) => {
            const IconComponent = category.icon;
            return (
              <motion.div
                key={category.id}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1 + index * 0.1 }}
                whileHover={{ y: -8, scale: 1.02 }}
                className="group"
              >
                <Link href={`/blog?category=${category.id}`}>
                  <div className={`relative overflow-hidden rounded-3xl ${category.bgColor} ${category.borderColor} border-2 p-8 h-full transition-all duration-300 group-hover:shadow-2xl group-hover:shadow-black/10`}>
                    {/* Background Pattern */}
                    <div className="absolute inset-0 opacity-5">
                      <div className="absolute inset-0 bg-gradient-to-br from-transparent via-transparent to-white/10"></div>
                    </div>

                    {/* Content */}
                    <div className="relative z-10">
                      {/* Icon */}
                      <motion.div
                        whileHover={{ rotate: 5, scale: 1.1 }}
                        className={`inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-r ${category.color} mb-6 shadow-lg`}
                      >
                        <IconComponent className="w-8 h-8 text-white" />
                      </motion.div>

                      {/* Title */}
                      <h3 className={`text-2xl font-bold mb-3 ${category.textColor} group-hover:scale-105 transition-transform duration-300`}>
                        {category.name}
                      </h3>

                      {/* Description */}
                      <p className="text-foreground/70 mb-6 leading-relaxed group-hover:text-foreground/80 transition-colors duration-300">
                        {category.description}
                      </p>

                      {/* Article Count */}
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className={`w-2 h-2 rounded-full bg-gradient-to-r ${category.color}`}></div>
                          <span className="text-sm font-medium text-foreground/60">
                            {category.articleCount} articles
                          </span>
                        </div>

                        {/* Arrow */}
                        <motion.div
                          whileHover={{ x: 4 }}
                          className="text-foreground/40 group-hover:text-accent-blue transition-colors duration-300"
                        >
                          <ArrowLeft className="w-5 h-5 rotate-180" />
                        </motion.div>
                      </div>
                    </div>

                    {/* Hover Effect */}
                    <div className={`absolute inset-0 bg-gradient-to-r ${category.color} opacity-0 group-hover:opacity-5 transition-opacity duration-300 rounded-3xl`}></div>
                  </div>
                </Link>
              </motion.div>
            );
          })}
        </motion.div>

        {/* Call to Action */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 2 }}
          className="text-center mt-16"
        >
          <div className="bg-gradient-to-r from-background-secondary/50 to-background-secondary/30 backdrop-blur-sm rounded-3xl p-8 border border-glass-border">
            <h3 className="text-2xl font-bold mb-4 bg-gradient-to-r from-accent-blue to-accent-purple bg-clip-text text-transparent">
              Can't Find What You're Looking For?
            </h3>
            <p className="text-foreground/70 mb-6 max-w-2xl mx-auto">
              Our AI chatbot can help you discover the perfect articles based on your interests.
              Ask about specific topics, technologies, or get personalized recommendations!
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/blog">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="px-8 py-3 bg-gradient-to-r from-accent-blue to-accent-purple text-white rounded-full font-medium shadow-lg hover:shadow-xl transition-all duration-300"
                >
                  Browse All Articles
                </motion.button>
              </Link>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="px-8 py-3 bg-background-secondary text-foreground rounded-full font-medium border border-glass-border hover:border-accent-blue/50 transition-all duration-300"
                onClick={() => {
                  // This would trigger the chatbot
                  const chatbotButton = document.querySelector('[aria-label="Open AI Chat Assistant"]') as HTMLElement;
                  if (chatbotButton) {
                    chatbotButton.click();
                  }
                }}
              >
                Ask AI Assistant 🤖
              </motion.button>
            </div>
          </div>
        </motion.div>
      </section>
    </div>
  );
}
