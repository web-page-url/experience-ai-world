'use client';

import { createContext, useContext, useState, useMemo, useEffect, ReactNode } from 'react';

export interface BlogPost {
  id: string;
  title: string;
  excerpt: string;
  content?: string;
  coverImage: string;
  category: string;
  author: {
    name: string;
    avatar?: string;
    bio?: string;
    social?: {
      twitter?: string;
      linkedin?: string;
    };
  } | string;
  date: string;
  readTime: string;
  likes: number;
  comments: number;
  tags?: string[];
  featured: boolean;
}

export interface BlogFilters {
  searchQuery: string;
  selectedCategory: string;
  selectedSort: string;
}

export interface BlogContextType {
  filters: BlogFilters;
  setFilters: (filters: Partial<BlogFilters>) => void;
  clearFilters: () => void;
  filteredPosts: BlogPost[];
  allPosts: BlogPost[];
}

const BlogContext = createContext<BlogContextType | undefined>(undefined);

// Mock blog posts data (moved from BlogGrid)
const blogPosts: BlogPost[] = [
  {
    id: '1',
    title: "The Future of AI: From ChatGPT to AGI",
    excerpt: "Explore how artificial intelligence is evolving from conversational models to artificial general intelligence, and what it means for humanity.",
    coverImage: "/api/placeholder/600/400",
    category: "AI",
    author: {
      name: "Dr. Sarah Chen",
      avatar: "/api/placeholder/64/64",
      bio: "AI researcher and professor specializing in machine learning and AGI development.",
      social: {
        twitter: "sarahchenai",
        linkedin: "sarahchen"
      }
    },
    date: "2025-01-15",
    readTime: "8 min read",
    likes: 142,
    comments: 23,
    tags: ["AI", "AGI", "Future", "Technology"],
    featured: true,
  },
  {
    id: '2',
    title: "Building Your First Machine Learning Model",
    excerpt: "A comprehensive guide for beginners to create, train, and deploy their first ML model using Python and TensorFlow.",
    coverImage: "/api/placeholder/600/400",
    category: "Tutorials",
    author: {
      name: "Mike Johnson",
      avatar: "/api/placeholder/64/64",
      bio: "Data scientist and ML engineer with 5+ years of experience in predictive modeling and AI solutions.",
      social: {
        twitter: "mikejohnsonml",
        linkedin: "mikejohnson"
      }
    },
    date: "2025-01-12",
    readTime: "12 min read",
    likes: 89,
    comments: 15,
    tags: ["Machine Learning", "Python", "TensorFlow", "Tutorial", "Beginners"],
    featured: false,
  },
  {
    id: '3',
    title: "AI Ethics: Navigating the Moral Landscape",
    excerpt: "Understanding the ethical implications of artificial intelligence development and deployment in modern society.",
    coverImage: "/api/placeholder/600/400",
    category: "Ethics",
    author: "Prof. Elena Rodriguez",
    date: "2025-01-10",
    readTime: "6 min read",
    likes: 67,
    comments: 31,
    featured: false,
  },
  {
    id: '4',
    title: "Claude vs GPT-5: The Ultimate AI Showdown",
    excerpt: "An in-depth comparison of Anthropic's Claude and OpenAI's latest GPT model across various use cases and performance metrics.",
    coverImage: "/api/placeholder/600/400",
    category: "Reviews",
    author: "Alex Thompson",
    date: "2025-01-08",
    readTime: "10 min read",
    likes: 203,
    comments: 45,
    featured: true,
  },
  {
    id: '5',
    title: "OpenAI's GPT-5 Architecture Deep Dive",
    excerpt: "Analyzing the technical architecture behind OpenAI's latest language model and its implications for AI development.",
    coverImage: "/api/placeholder/600/400",
    category: "AI",
    author: "Dr. Michael Chen",
    date: "2025-01-14",
    readTime: "15 min read",
    likes: 178,
    comments: 28,
    featured: false,
  },
  {
    id: '6',
    title: "Building Scalable ML Pipelines with Kubernetes",
    excerpt: "Learn how to deploy and manage machine learning workloads at scale using Kubernetes and cloud-native tools.",
    coverImage: "/api/placeholder/600/400",
    category: "Tutorials",
    author: "Sarah Johnson",
    date: "2025-01-13",
    readTime: "12 min read",
    likes: 95,
    comments: 19,
    featured: false,
  },
  {
    id: '7',
    title: "The Rise of Multimodal AI Models",
    excerpt: "Exploring how AI systems are evolving to process and understand multiple types of data simultaneously.",
    coverImage: "/api/placeholder/600/400",
    category: "Technology",
    author: "Prof. David Kim",
    date: "2025-01-11",
    readTime: "9 min read",
    likes: 134,
    comments: 22,
    featured: false,
  },
  {
    id: '8',
    title: "AI Safety Research: Current State and Future Directions",
    excerpt: "A comprehensive overview of the current landscape in AI safety research and alignment challenges.",
    coverImage: "/api/placeholder/600/400",
    category: "AI Ethics",
    author: "Dr. Emily Watson",
    date: "2025-01-09",
    readTime: "14 min read",
    likes: 76,
    comments: 33,
    featured: false,
  },
  {
    id: '9',
    title: "Fine-tuning LLMs for Domain-Specific Tasks",
    excerpt: "Practical guide to adapting large language models for specialized use cases and industry applications.",
    coverImage: "/api/placeholder/600/400",
    category: "Tutorials",
    author: "Alex Rodriguez",
    date: "2025-01-07",
    readTime: "11 min read",
    likes: 112,
    comments: 17,
    featured: false,
  },
];

export function BlogProvider({ children }: { children: ReactNode }) {
  const [filters, setFiltersState] = useState<BlogFilters>({
    searchQuery: '',
    selectedCategory: 'All',
    selectedSort: 'newest',
  });


  const setFilters = (newFilters: Partial<BlogFilters>) => {
    setFiltersState(prev => ({ ...prev, ...newFilters }));
  };

  const clearFilters = () => {
    setFiltersState({
      searchQuery: '',
      selectedCategory: 'All',
      selectedSort: 'newest',
    });
  };

  const filteredPosts = useMemo(() => {
    let filtered = [...blogPosts];

    // Apply search filter
    if (filters.searchQuery && filters.searchQuery.trim()) {
      const query = filters.searchQuery.toLowerCase().trim();
      filtered = filtered.filter(post => {
        const matchesTitle = post.title.toLowerCase().includes(query);
        const matchesExcerpt = post.excerpt.toLowerCase().includes(query);
        const matchesTags = post.tags?.some(tag => tag.toLowerCase().includes(query));
        const matchesAuthor = typeof post.author === 'object' && post.author.name.toLowerCase().includes(query);

        return matchesTitle || matchesExcerpt || matchesTags || matchesAuthor;
      });
    }

    // Apply category filter
    if (filters.selectedCategory && filters.selectedCategory !== 'All') {
      filtered = filtered.filter(post => post.category === filters.selectedCategory);
    }

    // Apply sorting
    switch (filters.selectedSort) {
      case 'newest':
        filtered.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
        break;
      case 'oldest':
        filtered.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
        break;
      case 'popular':
        filtered.sort((a, b) => b.likes - a.likes);
        break;
      case 'trending':
        // For trending, we'll sort by a combination of likes, comments, and recency
        filtered.sort((a, b) => {
          const scoreA = a.likes + a.comments * 2 + (new Date(a.date).getTime() / 1000000000);
          const scoreB = b.likes + b.comments * 2 + (new Date(b.date).getTime() / 1000000000);
          return scoreB - scoreA;
        });
        break;
    }

    return filtered;
  }, [filters]);

  const value: BlogContextType = {
    filters,
    setFilters,
    clearFilters,
    filteredPosts,
    allPosts: blogPosts,
  };

  return (
    <BlogContext.Provider value={value}>
      {children}
    </BlogContext.Provider>
  );
}

export function useBlog() {
  const context = useContext(BlogContext);
  if (context === undefined) {
    throw new Error('useBlog must be used within a BlogProvider');
  }
  return context;
}

export { BlogContext };
