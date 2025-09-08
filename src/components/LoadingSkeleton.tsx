'use client';

import React from 'react';
import { motion } from 'framer-motion';

interface LoadingSkeletonProps {
  className?: string;
  lines?: number;
}

export function LoadingSkeleton({ className = "", lines = 3 }: LoadingSkeletonProps) {
  return (
    <div className={`space-y-3 ${className}`}>
      {Array.from({ length: lines }).map((_, index) => (
        <motion.div
          key={index}
          className="h-4 bg-gray-200 rounded-lg"
          style={{
            width: `${Math.random() * 40 + 60}%`,
          }}
          animate={{
            opacity: [0.5, 1, 0.5],
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            delay: index * 0.1,
          }}
        />
      ))}
    </div>
  );
}

export function PageLoadingSkeleton() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-violet-50 via-purple-50 via-indigo-50 to-purple-100">
      <div className="max-w-6xl mx-auto px-6 py-20">
        {/* Hero Section Skeleton */}
        <div className="text-center mb-16">
          <motion.div
            className="h-6 w-48 bg-violet-200 rounded-full mx-auto mb-6"
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
          <motion.div
            className="h-12 w-96 bg-gray-300 rounded-xl mx-auto mb-6"
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1.5, repeat: Infinity, delay: 0.1 }}
          />
          <motion.div
            className="h-6 w-80 bg-gray-200 rounded-lg mx-auto"
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1.5, repeat: Infinity, delay: 0.2 }}
          />
        </div>

        {/* Content Grid Skeleton */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {Array.from({ length: 6 }).map((_, index) => (
            <motion.div
              key={index}
              className="bg-white/80 backdrop-blur-sm rounded-2xl p-8 shadow-xl border border-white/50"
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                delay: index * 0.1,
              }}
            >
              <div className="h-12 w-12 bg-violet-200 rounded-xl mb-4" />
              <div className="h-6 w-3/4 bg-gray-300 rounded-lg mb-3" />
              <div className="space-y-2">
                <div className="h-4 w-full bg-gray-200 rounded" />
                <div className="h-4 w-5/6 bg-gray-200 rounded" />
                <div className="h-4 w-4/6 bg-gray-200 rounded" />
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}

export function CardSkeleton({ count = 3 }: { count?: number }) {
  return (
    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
      {Array.from({ length: count }).map((_, index) => (
        <motion.div
          key={index}
          className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-lg border border-white/50"
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            delay: index * 0.1,
          }}
        >
          <div className="flex items-center gap-4 mb-4">
            <div className="h-12 w-12 bg-violet-200 rounded-xl" />
            <div className="flex-1">
              <div className="h-5 w-3/4 bg-gray-300 rounded mb-2" />
              <div className="h-4 w-1/2 bg-gray-200 rounded" />
            </div>
          </div>
          <div className="space-y-2">
            <div className="h-4 w-full bg-gray-200 rounded" />
            <div className="h-4 w-5/6 bg-gray-200 rounded" />
            <div className="h-4 w-4/6 bg-gray-200 rounded" />
          </div>
        </motion.div>
      ))}
    </div>
  );
}
