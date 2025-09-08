'use client';

import { motion } from 'framer-motion';
import { ArrowRight, Sparkles, TrendingUp, BookOpen, Users } from 'lucide-react';
import Link from 'next/link';
import { useState, useEffect } from 'react';
import Hyperspeed from '../Hyperspeed';
import { hyperspeedPresets } from '../../lib/hyperspeed-presets';

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

export default function Hero() {
  const [isMobile, setIsMobile] = useState(false);
  const [isWebGLSupported, setIsWebGLSupported] = useState(true);

  useEffect(() => {

    // Check for mobile device - optimized for performance
    const checkMobile = () => {
      const isSmallScreen = window.innerWidth < 768;
      const isVerySlowDevice = navigator.hardwareConcurrency && navigator.hardwareConcurrency < 2;
      const isOldDevice = !window.requestAnimationFrame || !window.performance;

      // Only disable on very slow or very old devices
      setIsMobile(isSmallScreen && (isVerySlowDevice || isOldDevice));
    };

    // Check WebGL support with better mobile detection
    const checkWebGL = () => {
      try {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') as WebGLRenderingContext | null ||
                  canvas.getContext('experimental-webgl') as WebGLRenderingContext | null;
        if (gl) {
          // Test if WebGL is actually working
          const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
          const renderer = debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) as string : '';

          // Blacklist some problematic mobile GPUs
          const isProblematicGPU = renderer.toLowerCase().includes('swiftshader') ||
                                 renderer.toLowerCase().includes('software') ||
                                 renderer.toLowerCase().includes('llvmpipe');

          setIsWebGLSupported(!isProblematicGPU);
        } else {
          setIsWebGLSupported(false);
        }
      } catch (e) {
        setIsWebGLSupported(false);
      }
    };

    checkMobile();
    checkWebGL();

    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  const scrollToBlog = () => {
    const blogSection = document.getElementById('featured-posts');
    blogSection?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        {/* Hyperspeed Background Animation - Optimized for all devices */}
        {isWebGLSupported && (
          <div className="absolute inset-0 z-0">
            <Hyperspeed effectOptions={isMobile ? hyperspeedPresets.mobile : hyperspeedPresets.cyber} />
          </div>
        )}

        {/* Enhanced Fallback Background - Only when WebGL is not supported */}
        {!isWebGLSupported && (
          <div className="absolute inset-0 z-0 bg-gradient-to-br from-background via-accent-blue/5 to-accent-purple/5">
            <div className="absolute inset-0 opacity-20">
              <div className="absolute top-20 left-10 w-32 h-32 bg-gradient-to-br from-accent-blue/20 to-accent-purple/20 rounded-full blur-xl animate-pulse"></div>
              <div className="absolute bottom-40 right-10 w-40 h-40 bg-gradient-to-br from-accent-pink/20 to-accent-purple/20 rounded-full blur-xl animate-pulse delay-1000"></div>
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-24 h-24 bg-gradient-to-br from-accent-cyan/20 to-accent-green/20 rounded-full blur-lg animate-pulse delay-500"></div>
            </div>
          </div>
        )}

        {/* Overlay for better text readability */}
        <div className="absolute inset-0 bg-gradient-to-b from-background/20 via-background/60 to-background/90 z-5"></div>

        {/* Cyber Animated Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-accent-blue/10 via-accent-purple/10 to-accent-pink/10 z-10">
        {/* Cyber Floating particles */}
        {[...Array(30)].map((_, i) => (
          <motion.div
            key={i}
            className={`absolute w-1 h-1 rounded-full ${
              i % 3 === 0 ? 'bg-accent-blue/30' :
              i % 3 === 1 ? 'bg-accent-purple/30' :
              'bg-accent-pink/30'
            }`}
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
            animate={{
              y: [0, -50, 0],
              x: [0, Math.random() * 20 - 10, 0],
              opacity: [0, 0.8, 0],
              scale: [0, 1.5, 0]
            }}
            transition={{
              duration: 3 + Math.random() * 3,
              repeat: Infinity,
              delay: Math.random() * 2,
              ease: "easeInOut"
            }}
          />
        ))}

        {/* Cyber Geometric shapes */}
        <motion.div
          className="absolute top-20 left-20 w-32 h-32 border-2 border-accent-blue/30 rounded-full"
          animate={{
            rotate: 360,
            scale: [1, 1.3, 1],
            boxShadow: [
              "0 0 20px rgba(0, 212, 255, 0.2)",
              "0 0 40px rgba(0, 212, 255, 0.4)",
              "0 0 20px rgba(0, 212, 255, 0.2)"
            ]
          }}
          transition={{
            rotate: { duration: 15, repeat: Infinity, ease: "linear" },
            scale: { duration: 3, repeat: Infinity },
            boxShadow: { duration: 2, repeat: Infinity }
          }}
        />
        <motion.div
          className="absolute bottom-32 right-16 w-24 h-24 border-2 border-accent-purple/30 rotate-45"
          animate={{
            rotate: [45, 135, 225, 315, 45],
            scale: [1, 1.4, 1],
            boxShadow: [
              "0 0 15px rgba(147, 51, 234, 0.2)",
              "0 0 30px rgba(147, 51, 234, 0.4)",
              "0 0 15px rgba(147, 51, 234, 0.2)"
            ]
          }}
          transition={{
            duration: 6,
            repeat: Infinity,
            ease: "easeInOut",
            boxShadow: { duration: 2, repeat: Infinity, delay: 0.5 }
          }}
        />
        <motion.div
          className="absolute top-1/2 left-1/4 w-16 h-16 border-2 border-accent-pink/30"
          animate={{
            rotate: [0, 90, 180, 270, 360],
            scale: [1, 1.2, 1],
            boxShadow: [
              "0 0 10px rgba(255, 0, 128, 0.2)",
              "0 0 25px rgba(255, 0, 128, 0.4)",
              "0 0 10px rgba(255, 0, 128, 0.2)"
            ]
          }}
          transition={{
            rotate: { duration: 10, repeat: Infinity, ease: "linear" },
            scale: { duration: 2.5, repeat: Infinity },
            boxShadow: { duration: 1.5, repeat: Infinity, delay: 1 }
          }}
        />
      </div>

      <div className="relative z-20 max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <motion.div
          variants={staggerChildren}
          initial="initial"
          animate="animate"
          className="space-y-8"
        >
          {/* Badge */}
          <motion.div
            variants={fadeInUp}
            className="inline-flex items-center gap-2 px-4 py-2 bg-accent-blue/10 border border-accent-blue/20 rounded-full"
          >
            <Sparkles className="w-4 h-4 text-accent-blue" />
            <span className="text-sm font-medium text-accent-blue">
              Stay Ahead in AI & Technology
            </span>
          </motion.div>

          {/* Main Headline */}
          <motion.h1
            variants={fadeInUp}
            className="text-5xl md:text-7xl font-bold leading-tight"
          >
            <span className="cyber-text">Stay Ahead</span>
            <br />
            with the Latest in{' '}
            <span className="cyber-text">AI & Technology</span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            variants={fadeInUp}
            className="text-xl md:text-2xl text-foreground/80 max-w-3xl mx-auto leading-relaxed"
          >
            Daily insights, tutorials, reviews, and trends shaping the future of
            Artificial Intelligence. Join thousands of readers staying ahead in the AI world.
          </motion.p>

          {/* Feature Highlights */}
          <motion.div
            variants={fadeInUp}
            className="flex flex-wrap justify-center gap-6 text-sm text-foreground/60"
          >
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-accent-blue" />
              <span>Latest AI Trends</span>
            </div>
            <div className="flex items-center gap-2">
              <BookOpen className="w-4 h-4 text-accent-purple" />
              <span>In-Depth Tutorials</span>
            </div>
            <div className="flex items-center gap-2">
              <Users className="w-4 h-4 text-accent-pink" />
              <span>Expert Insights</span>
            </div>
          </motion.div>

          {/* CTA Buttons */}
          <motion.div
            variants={fadeInUp}
            className="flex flex-col sm:flex-row gap-4 justify-center items-center"
          >
            <motion.button
              onClick={scrollToBlog}
              className="btn-primary group cyber-border cursor-target"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <span className="flex items-center gap-2">
                Read Blog
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </span>
            </motion.button>

            <Link href="/newsletter">
              <motion.button
                className="btn-secondary group cyber-border cursor-target"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <span className="flex items-center gap-2">
                  <Sparkles className="w-5 h-5" />
                  Join Newsletter
                  <motion.div
                    className="w-2 h-2 bg-accent-pink rounded-full"
                    animate={{ opacity: [1, 0.3, 1] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  />
                </span>
              </motion.button>
            </Link>
          </motion.div>

          {/* Social Proof */}
          <motion.div
            variants={fadeInUp}
            className="pt-8 border-t border-glass-border max-w-md mx-auto"
          >
            <div className="flex items-center justify-center gap-4 text-sm text-foreground/60">
              <div className="flex -space-x-2">
                {[...Array(5)].map((_, i) => (
                  <div
                    key={i}
                    className="w-8 h-8 bg-gradient-primary rounded-full border-2 border-background flex items-center justify-center text-xs"
                  >
                    👤
                  </div>
                ))}
              </div>
              <span>Join 50K+ AI enthusiasts</span>
            </div>
          </motion.div>
        </motion.div>

        {/* Scroll Indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 2, duration: 0.6 }}
          className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
        >
          <motion.div
            animate={{ y: [0, 8, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="w-6 h-10 border-2 border-accent-blue/30 rounded-full flex justify-center"
          >
            <motion.div
              animate={{ y: [0, 16, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="w-1 h-3 bg-accent-blue rounded-full mt-2"
            />
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}
