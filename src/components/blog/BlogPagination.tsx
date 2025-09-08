'use client';

import { motion } from 'framer-motion';
import { ChevronLeft, ChevronRight, MoreHorizontal } from 'lucide-react';
import { useState } from 'react';

interface BlogPaginationProps {
  currentPage?: number;
  totalPages?: number;
  onPageChange?: (page: number) => void;
}

export default function BlogPagination({
  currentPage = 1,
  totalPages = 10,
  onPageChange
}: BlogPaginationProps) {
  const [page, setPage] = useState(currentPage);

  const handlePageChange = (newPage: number) => {
    setPage(newPage);
    onPageChange?.(newPage);
    // Scroll to top of blog section
    const blogSection = document.getElementById('blog-grid');
    blogSection?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const getPageNumbers = () => {
    const pages = [];
    const showPages = 5; // Number of page buttons to show

    if (totalPages <= showPages) {
      // Show all pages if total is small
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      // Show pages with ellipsis
      if (page <= 3) {
        // Near the start
        pages.push(1, 2, 3, 4, '...', totalPages);
      } else if (page >= totalPages - 2) {
        // Near the end
        pages.push(1, '...', totalPages - 3, totalPages - 2, totalPages - 1, totalPages);
      } else {
        // In the middle
        pages.push(1, '...', page - 1, page, page + 1, '...', totalPages);
      }
    }

    return pages;
  };

  if (totalPages <= 1) return null;

  return (
    <section className="py-12 border-t border-glass-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          {/* Results Info */}
          <div className="text-sm text-foreground/60">
            Showing page {page} of {totalPages}
          </div>

          {/* Pagination Controls */}
          <div className="flex items-center gap-2">
            {/* Previous Button */}
            <motion.button
              onClick={() => handlePageChange(page - 1)}
              disabled={page === 1}
              className="p-2 rounded-lg border border-glass-border hover:bg-background-secondary disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              whileHover={{ scale: page > 1 ? 1.05 : 1 }}
              whileTap={{ scale: page > 1 ? 0.95 : 1 }}
              aria-label="Previous page"
            >
              <ChevronLeft className="w-5 h-5" />
            </motion.button>

            {/* Page Numbers */}
            <div className="flex items-center gap-1">
              {getPageNumbers().map((pageNum, index) => (
                <div key={index}>
                  {pageNum === '...' ? (
                    <div className="px-3 py-2">
                      <MoreHorizontal className="w-4 h-4 text-foreground/40" />
                    </div>
                  ) : (
                    <motion.button
                      onClick={() => handlePageChange(pageNum as number)}
                      className={`px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                        page === pageNum
                          ? 'bg-accent-blue text-white shadow-lg'
                          : 'hover:bg-background-secondary text-foreground/70 hover:text-foreground'
                      }`}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      aria-label={`Go to page ${pageNum}`}
                      aria-current={page === pageNum ? 'page' : undefined}
                    >
                      {pageNum}
                    </motion.button>
                  )}
                </div>
              ))}
            </div>

            {/* Next Button */}
            <motion.button
              onClick={() => handlePageChange(page + 1)}
              disabled={page === totalPages}
              className="p-2 rounded-lg border border-glass-border hover:bg-background-secondary disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              whileHover={{ scale: page < totalPages ? 1.05 : 1 }}
              whileTap={{ scale: page < totalPages ? 0.95 : 1 }}
              aria-label="Next page"
            >
              <ChevronRight className="w-5 h-5" />
            </motion.button>
          </div>

          {/* Jump to Page */}
          <div className="flex items-center gap-2 text-sm">
            <span className="text-foreground/60">Jump to:</span>
            <select
              value={page}
              onChange={(e) => handlePageChange(Number(e.target.value))}
              className="input text-sm w-16"
              aria-label="Jump to page"
            >
              {Array.from({ length: totalPages }, (_, i) => i + 1).map((pageNum) => (
                <option key={pageNum} value={pageNum}>
                  {pageNum}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Pagination Summary */}
        <div className="mt-6 text-center">
          <p className="text-sm text-foreground/60">
            Page {page} of {totalPages} • {(page - 1) * 9 + 1}-{Math.min(page * 9, 90)} of 90 articles
          </p>
        </div>
      </div>
    </section>
  );
}
