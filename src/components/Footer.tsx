'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import {
  Twitter,
  Linkedin,
  Github,
  Mail,
  Heart,
  ArrowUp
} from 'lucide-react';

const footerLinks = {
  content: [
    { name: 'Blog', href: '/blog' },
    { name: 'Tutorials', href: '/tutorials' },
    { name: 'Reviews', href: '/reviews' },
    { name: 'Newsletter', href: '/newsletter' },
  ],
  categories: [
    { name: 'AI', href: '/categories/ai' },
    { name: 'Technology', href: '/categories/technology' },
    { name: 'Startups', href: '/categories/startups' },
    { name: 'Tutorials', href: '/categories/tutorials' },
  ],
  company: [
    { name: 'About', href: '/about' },
    { name: 'Contact', href: '/contact' },
    { name: 'Privacy', href: '/privacy' },
    { name: 'Terms', href: '/terms' },
  ],
};

const socialLinks = [
  { name: 'Twitter', icon: Twitter, href: 'https://twitter.com/experienceaiworld', color: 'hover:text-blue-400' },
  { name: 'LinkedIn', icon: Linkedin, href: 'https://linkedin.com/company/experienceaiworld', color: 'hover:text-blue-600' },
  { name: 'GitHub', icon: Github, href: 'https://github.com/experienceaiworld', color: 'hover:text-gray-600 dark:hover:text-gray-400' },
  { name: 'Email', icon: Mail, href: 'mailto:hello@experienceaiworld.com', color: 'hover:text-red-400' },
];

export default function Footer() {
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <footer className="bg-background-secondary border-t border-glass-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Main Footer Content */}
        <div className="py-12">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-8">
            {/* Brand Section */}
            <div className="lg:col-span-2">
              <Link href="/" className="flex items-center space-x-2 group mb-4">
                <motion.div
                  className="w-10 h-10 bg-gradient-primary rounded-lg flex items-center justify-center cyber-border"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <span className="text-white font-bold text-xl">AI</span>
                </motion.div>
                <div>
                  <h3 className="text-lg font-bold cyber-text">
                    Experience AI World
                  </h3>
                  <p className="text-sm text-foreground/60">
                    Your gateway to the future of AI
                  </p>
                </div>
              </Link>

              <p className="text-foreground/80 mb-6 max-w-md">
                Stay ahead with the latest in AI & Technology. Daily insights, tutorials,
                reviews, and trends shaping the future of Artificial Intelligence.
              </p>

              {/* Social Links */}
              <div className="flex space-x-4">
                {socialLinks.map((social) => (
                  <motion.a
                    key={social.name}
                    href={social.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={`p-2 rounded-lg bg-background-tertiary text-foreground/60 transition-all duration-200 cyber-border ${social.color}`}
                    whileHover={{ scale: 1.1, y: -2 }}
                    whileTap={{ scale: 0.9 }}
                    aria-label={`Follow us on ${social.name}`}
                  >
                    <social.icon className="w-5 h-5" />
                  </motion.a>
                ))}
              </div>
            </div>

            {/* Content Links */}
            <div>
              <h4 className="font-semibold text-foreground mb-4">Content</h4>
              <ul className="space-y-3">
                {footerLinks.content.map((link) => (
                  <li key={link.name}>
                    <Link
                      href={link.href}
                      className="text-foreground/60 hover:text-accent-blue transition-colors duration-200"
                    >
                      {link.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>

            {/* Categories Links */}
            <div>
              <h4 className="font-semibold text-foreground mb-4">Categories</h4>
              <ul className="space-y-3">
                {footerLinks.categories.map((link) => (
                  <li key={link.name}>
                    <Link
                      href={link.href}
                      className="text-foreground/60 hover:text-accent-purple transition-colors duration-200"
                    >
                      {link.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>

            {/* Company Links */}
            <div>
              <h4 className="font-semibold text-foreground mb-4">Company</h4>
              <ul className="space-y-3">
                {footerLinks.company.map((link) => (
                  <li key={link.name}>
                    <Link
                      href={link.href}
                      className="text-foreground/60 hover:text-accent-cyan transition-colors duration-200"
                    >
                      {link.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Newsletter Signup */}
        <div className="py-8 border-t border-glass-border">
          <div className="max-w-md mx-auto text-center">
            <h3 className="text-lg font-semibold text-foreground mb-2">
              Get Weekly AI Insights
            </h3>
            <p className="text-foreground/60 mb-6">
              Join thousands of readers staying ahead in the AI world.
            </p>
            <div className="flex flex-col sm:flex-row gap-3">
              <input
                type="email"
                placeholder="Enter your email"
                className="input flex-1"
              />
              <button className="btn-primary px-6 py-3 whitespace-nowrap">
                Subscribe
              </button>
            </div>
            <p className="text-xs text-foreground/50 mt-3">
              No spam, unsubscribe at any time.
            </p>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="py-6 border-t border-glass-border">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <div className="flex items-center space-x-2 text-sm text-foreground/60">
              <span>© 2025 Experience AI World. All rights reserved.</span>
              <span className="hidden md:inline">•</span>
              <span className="flex items-center space-x-1">
                Made with <Heart className="w-4 h-4 text-red-500 fill-current" /> for the AI community
              </span>
            </div>

            <motion.button
              onClick={scrollToTop}
              className="flex items-center space-x-2 text-foreground/60 hover:text-accent-blue transition-colors duration-200 group"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              aria-label="Scroll to top"
            >
              <span className="text-sm">Back to top</span>
              <ArrowUp className="w-4 h-4 group-hover:-translate-y-1 transition-transform" />
            </motion.button>
          </div>
        </div>
      </div>
    </footer>
  );
}