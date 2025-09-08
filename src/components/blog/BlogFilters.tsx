'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Search, Filter, X, SlidersHorizontal } from 'lucide-react';
import { useBlog } from './BlogContext';

const categories = [
  'All',
  'AI',
  'Technology',
  'Tutorials',
  'Reviews',
  'Ethics',
  'AI Ethics'
];

const sortOptions = [
  { value: 'newest', label: 'Newest First' },
  { value: 'oldest', label: 'Oldest First' },
  { value: 'popular', label: 'Most Popular' },
  { value: 'trending', label: 'Trending' }
];

export default function BlogFilters() {
  const { filters, setFilters, clearFilters } = useBlog();
  const [showFilters, setShowFilters] = useState(false);

  return (
    <section className="py-8 border-b border-glass-border bg-background/50 backdrop-blur-sm sticky top-16 z-30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col lg:flex-row gap-4 items-center">
          {/* Search Bar */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="relative flex-1 w-full lg:max-w-md"
          >
            <input
              type="text"
              value={filters.searchQuery}
              onChange={(e) => setFilters({ searchQuery: e.target.value })}
              placeholder="Search articles..."
              className="input w-full pl-10 pr-4"
            />
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-foreground/40" />
            {filters.searchQuery && (
              <button
                onClick={() => setFilters({ searchQuery: '' })}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 p-1 hover:bg-background-secondary rounded-full transition-colors"
              >
                <X className="w-3 h-3" />
              </button>
            )}
          </motion.div>

          {/* Desktop Category Filters */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="hidden md:flex items-center gap-2 overflow-x-auto"
          >
            {categories.map((category) => (
              <button
                key={category}
                onClick={() => setFilters({ selectedCategory: category })}
                className={`px-4 py-2 rounded-full text-sm font-medium whitespace-nowrap transition-all duration-200 ${
                  filters.selectedCategory === category
                    ? 'bg-accent-blue text-white shadow-lg'
                    : 'bg-background-secondary hover:bg-background-tertiary text-foreground/70 hover:text-foreground'
                }`}
              >
                {category}
              </button>
            ))}
          </motion.div>

          {/* Sort Dropdown */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="relative"
          >
            <select
              value={filters.selectedSort}
              onChange={(e) => setFilters({ selectedSort: e.target.value })}
              className="input appearance-none pr-8 cursor-pointer"
            >
              {sortOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            <SlidersHorizontal className="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-foreground/40 pointer-events-none" />
          </motion.div>

          {/* Mobile Filter Toggle */}
          <motion.button
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            onClick={() => setShowFilters(!showFilters)}
            className="md:hidden p-2 rounded-lg hover:bg-background-secondary transition-colors"
          >
            <Filter className="w-5 h-5" />
          </motion.button>

          {/* Clear Filters */}
          {(filters.searchQuery || filters.selectedCategory !== 'All' || filters.selectedSort !== 'newest') && (
            <motion.button
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              onClick={clearFilters}
              className="px-4 py-2 text-sm text-accent-blue hover:text-accent-purple transition-colors"
            >
              Clear Filters
            </motion.button>
          )}
        </div>

        {/* Mobile Category Filters */}
        <motion.div
          initial={false}
          animate={{
            height: showFilters ? 'auto' : 0,
            opacity: showFilters ? 1 : 0
          }}
          className="md:hidden overflow-hidden"
        >
          <div className="pt-4 border-t border-glass-border">
            <div className="flex flex-wrap gap-2">
              {categories.map((category) => (
                <button
                  key={category}
                  onClick={() => setFilters({ selectedCategory: category })}
                  className={`px-3 py-2 rounded-full text-sm font-medium transition-all duration-200 ${
                    filters.selectedCategory === category
                      ? 'bg-accent-blue text-white'
                      : 'bg-background-secondary hover:bg-background-tertiary text-foreground/70 hover:text-foreground'
                  }`}
                >
                  {category}
                </button>
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
