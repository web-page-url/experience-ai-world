'use client';

import { motion } from 'framer-motion';
import { Mail, Sparkles, CheckCircle, ArrowRight } from 'lucide-react';
import { useState } from 'react';

const fadeInUp = {
  initial: { opacity: 0, y: 60 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6 }
};

const staggerChildren = {
  animate: {
    transition: {
      staggerChildren: 0.15
    }
  }
};

const benefits = [
  {
    icon: Sparkles,
    title: "Weekly AI Insights",
    description: "Get curated content delivered to your inbox"
  },
  {
    icon: CheckCircle,
    title: "Early Access",
    description: "Be the first to read our latest tutorials and reviews"
  },
  {
    icon: Mail,
    title: "Exclusive Content",
    description: "Access premium articles and industry reports"
  }
];

export default function Newsletter() {
  const [email, setEmail] = useState('');
  const [isSubscribed, setIsSubscribed] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));

    setIsSubscribed(true);
    setIsLoading(false);
    setEmail('');
  };

  return (
    <section className="py-20 bg-gradient-to-br from-accent-blue/5 via-accent-purple/5 to-accent-pink/5">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <motion.div
          initial="initial"
          whileInView="animate"
          viewport={{ once: true }}
          variants={staggerChildren}
          className="space-y-12"
        >
          {/* Header */}
          <motion.div variants={fadeInUp} className="space-y-6">
            <motion.div
              className="inline-flex items-center gap-2 px-4 py-2 bg-accent-purple/10 border border-accent-purple/20 rounded-full"
              whileHover={{ scale: 1.05 }}
            >
              <Mail className="w-4 h-4 text-accent-purple" />
              <span className="text-sm font-medium text-accent-purple">
                Stay Connected
              </span>
            </motion.div>

            <h2 className="text-4xl md:text-5xl font-bold">
              Get Weekly <span className="text-gradient">AI Insights</span>
            </h2>

            <p className="text-xl text-foreground/70 max-w-2xl mx-auto">
              Join thousands of readers staying ahead in the AI world. Get exclusive insights,
              tutorials, and industry analysis delivered to your inbox every week.
            </p>
          </motion.div>

          {/* Benefits */}
          <motion.div
            variants={fadeInUp}
            className="grid grid-cols-1 md:grid-cols-3 gap-6"
          >
            {benefits.map((benefit, index) => (
              <motion.div
                key={benefit.title}
                variants={fadeInUp}
                className="card text-center space-y-3"
                whileHover={{ y: -5 }}
                transition={{ duration: 0.2 }}
              >
                <motion.div
                  className="w-12 h-12 mx-auto bg-gradient-primary rounded-xl flex items-center justify-center"
                  whileHover={{ rotate: 360 }}
                  transition={{ duration: 0.6 }}
                >
                  <benefit.icon className="w-6 h-6 text-white" />
                </motion.div>
                <h3 className="font-semibold text-lg">{benefit.title}</h3>
                <p className="text-foreground/70 text-sm">{benefit.description}</p>
              </motion.div>
            ))}
          </motion.div>

          {/* Newsletter Form */}
          <motion.div
            variants={fadeInUp}
            className="max-w-md mx-auto"
          >
            {!isSubscribed ? (
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="relative">
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="Enter your email address"
                    className="input w-full pr-12"
                    required
                    disabled={isLoading}
                  />
                  <Mail className="absolute right-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-foreground/40" />
                </div>

                <motion.button
                  type="submit"
                  disabled={isLoading || !email.trim()}
                  className="btn-primary w-full group disabled:opacity-50 disabled:cursor-not-allowed cyber-border cursor-target"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  {isLoading ? (
                    <span className="flex items-center gap-2">
                      <motion.div
                        className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full"
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      />
                      Subscribing...
                    </span>
                  ) : (
                    <span className="flex items-center gap-2">
                      Join Newsletter
                      <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                    </span>
                  )}
                </motion.button>

                <p className="text-xs text-foreground/50">
                  No spam, unsubscribe at any time. We respect your privacy.
                </p>
              </form>
            ) : (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                className="card text-center space-y-4"
              >
                <motion.div
                  className="w-16 h-16 mx-auto bg-green-500 rounded-full flex items-center justify-center"
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
                >
                  <CheckCircle className="w-8 h-8 text-white" />
                </motion.div>

                <div>
                  <h3 className="text-xl font-bold text-green-600 dark:text-green-400 mb-2">
                    Welcome aboard! 🎉
                  </h3>
                  <p className="text-foreground/70">
                    Check your email for a confirmation link to complete your subscription.
                  </p>
                </div>
              </motion.div>
            )}
          </motion.div>

          {/* Social Proof */}
          <motion.div
            variants={fadeInUp}
            className="pt-8 border-t border-glass-border max-w-md mx-auto"
          >
            <div className="flex items-center justify-center gap-4 text-sm text-foreground/60">
              <div className="flex -space-x-2">
                {[...Array(5)].map((_, i) => (
                  <motion.div
                    key={i}
                    className="w-8 h-8 bg-gradient-primary rounded-full border-2 border-background flex items-center justify-center text-xs"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: i * 0.1 }}
                  >
                    👤
                  </motion.div>
                ))}
              </div>
              <span>Join 15K+ subscribers</span>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}
