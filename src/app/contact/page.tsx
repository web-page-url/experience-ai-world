'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Mail,
  Phone,
  MapPin,
  Clock,
  Send,
  Twitter,
  Linkedin,
  Github,
  MessageSquare,
  ArrowLeft
} from 'lucide-react';
import Link from 'next/link';

const contactMethods = [
  {
    icon: Mail,
    title: 'Email Us',
    description: 'Send us an email and we\'ll respond within 24 hours',
    contact: 'hello@experienceaiworld.com',
    color: 'from-blue-500 to-purple-600',
    bgColor: 'bg-blue-50 dark:bg-blue-900/20'
  },
  {
    icon: MessageSquare,
    title: 'Live Chat',
    description: 'Chat with our AI assistant for instant help',
    contact: 'Available 24/7',
    color: 'from-green-500 to-teal-600',
    bgColor: 'bg-green-50 dark:bg-green-900/20'
  },
  {
    icon: Twitter,
    title: 'Twitter',
    description: 'Follow us for updates and quick responses',
    contact: '@experienceaiworld',
    color: 'from-cyan-500 to-blue-600',
    bgColor: 'bg-cyan-50 dark:bg-cyan-900/20'
  },
  {
    icon: Linkedin,
    title: 'LinkedIn',
    description: 'Connect with us professionally',
    contact: 'Experience AI World',
    color: 'from-blue-600 to-indigo-600',
    bgColor: 'bg-blue-50 dark:bg-blue-900/20'
  }
];

const faqs = [
  {
    question: 'How can I contribute an article?',
    answer: 'We welcome guest contributions! Send your article idea to hello@experienceaiworld.com with a brief outline and your expertise area.'
  },
  {
    question: 'Can I republish your articles?',
    answer: 'Please contact us for reprint permissions. We typically allow republishing with proper attribution and backlinks.'
  },
  {
    question: 'How do I advertise on your blog?',
    answer: 'Email us at partnerships@experienceaiworld.com for advertising opportunities and sponsorship packages.'
  },
  {
    question: 'Do you offer consulting services?',
    answer: 'Yes! We provide AI consulting services. Contact us to discuss your project requirements.'
  }
];

export default function ContactPage() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    // Simulate form submission
    await new Promise(resolve => setTimeout(resolve, 2000));

    setIsSubmitting(false);
    setSubmitted(true);

    // Reset form after 3 seconds
    setTimeout(() => {
      setSubmitted(false);
      setFormData({ name: '', email: '', subject: '', message: '' });
    }, 3000);
  };

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
            Get In Touch
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="text-xl md:text-2xl text-foreground/70 max-w-3xl mx-auto leading-relaxed"
          >
            Have questions about AI? Want to collaborate? Need consulting services?
            We'd love to hear from you!
          </motion.p>
        </div>
      </motion.section>

      {/* Contact Methods */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mb-16">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        >
          {contactMethods.map((method, index) => {
            const IconComponent = method.icon;
            return (
              <motion.div
                key={method.title}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.8 + index * 0.1 }}
                whileHover={{ y: -8, scale: 1.02 }}
                className={`relative overflow-hidden rounded-2xl ${method.bgColor} border border-glass-border p-6 h-full transition-all duration-300 group cursor-pointer`}
              >
                <div className="relative z-10">
                  <div className={`inline-flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-r ${method.color} mb-4`}>
                    <IconComponent className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-lg font-bold mb-2">{method.title}</h3>
                  <p className="text-foreground/70 mb-3 text-sm leading-relaxed">
                    {method.description}
                  </p>
                  <p className={`text-sm font-medium bg-gradient-to-r ${method.color} bg-clip-text text-transparent`}>
                    {method.contact}
                  </p>
                </div>
                <div className={`absolute inset-0 bg-gradient-to-r ${method.color} opacity-0 group-hover:opacity-5 transition-opacity duration-300 rounded-2xl`}></div>
              </motion.div>
            );
          })}
        </motion.div>
      </section>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
          {/* Contact Form */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 1 }}
            className="lg:col-span-2"
          >
            <div className="bg-background-secondary/50 backdrop-blur-sm rounded-3xl p-8 border border-glass-border">
              <h2 className="text-3xl font-bold mb-6">Send us a Message</h2>

              {submitted ? (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="text-center py-12"
                >
                  <div className="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Send className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-xl font-bold text-green-600 mb-2">Message Sent!</h3>
                  <p className="text-foreground/70">Thank you for reaching out. We'll get back to you within 24 hours.</p>
                </motion.div>
              ) : (
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <label className="block text-sm font-medium mb-2">Name *</label>
                      <input
                        type="text"
                        name="name"
                        required
                        value={formData.name}
                        onChange={handleInputChange}
                      className="w-full px-4 py-3 bg-background-secondary dark:bg-slate-800/80 border border-glass-border rounded-xl text-foreground dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-accent-blue focus:border-accent-blue transition-colors"
                      placeholder="Your full name"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-2">Email *</label>
                      <input
                        type="email"
                        name="email"
                        required
                        value={formData.email}
                        onChange={handleInputChange}
                      className="w-full px-4 py-3 bg-background-secondary dark:bg-slate-800/80 border border-glass-border rounded-xl text-foreground dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-accent-blue focus:border-accent-blue transition-colors"
                      placeholder="your@email.com"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-2">Subject *</label>
                    <select
                      name="subject"
                      required
                      value={formData.subject}
                      onChange={handleInputChange}
                      className="w-full px-4 py-3 bg-background-secondary dark:bg-slate-800/80 border border-glass-border rounded-xl text-foreground dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-accent-blue focus:border-accent-blue transition-colors"
                    >
                      <option value="">Select a subject</option>
                      <option value="general">General Inquiry</option>
                      <option value="collaboration">Collaboration</option>
                      <option value="advertising">Advertising</option>
                      <option value="consulting">AI Consulting</option>
                      <option value="contribution">Guest Contribution</option>
                      <option value="technical">Technical Support</option>
                      <option value="other">Other</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-2">Message *</label>
                    <textarea
                      name="message"
                      required
                      rows={6}
                      value={formData.message}
                      onChange={handleInputChange}
                    className="w-full px-4 py-3 bg-background-secondary dark:bg-slate-800/80 border border-glass-border rounded-xl text-foreground dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-accent-blue focus:border-accent-blue transition-colors resize-none"
                    placeholder="Tell us how we can help you..."
                    />
                  </div>

                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    type="submit"
                    disabled={isSubmitting}
                    className="w-full bg-gradient-to-r from-accent-blue to-accent-purple text-white py-4 px-6 rounded-xl font-semibold shadow-lg hover:shadow-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {isSubmitting ? (
                      <>
                        <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                        Sending...
                      </>
                    ) : (
                      <>
                        <Send className="w-5 h-5" />
                        Send Message
                      </>
                    )}
                  </motion.button>
                </form>
              )}
            </div>
          </motion.div>

          {/* Sidebar */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 1.2 }}
            className="space-y-8"
          >
            {/* Business Hours */}
            <div className="bg-background-secondary/50 backdrop-blur-sm rounded-3xl p-6 border border-glass-border">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-blue-500/20 rounded-xl flex items-center justify-center">
                  <Clock className="w-5 h-5 text-blue-500" />
                </div>
                <h3 className="text-xl font-bold">Business Hours</h3>
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Monday - Friday</span>
                  <span>9:00 AM - 6:00 PM</span>
                </div>
                <div className="flex justify-between">
                  <span>Saturday</span>
                  <span>10:00 AM - 4:00 PM</span>
                </div>
                <div className="flex justify-between">
                  <span>Sunday</span>
                  <span>Closed</span>
                </div>
              </div>
              <p className="text-xs text-foreground/60 mt-4">
                Response time: Within 24 hours during business days
              </p>
            </div>

            {/* Office Location */}
            <div className="bg-background-secondary/50 backdrop-blur-sm rounded-3xl p-6 border border-glass-border">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-green-500/20 rounded-xl flex items-center justify-center">
                  <MapPin className="w-5 h-5 text-green-500" />
                </div>
                <h3 className="text-xl font-bold">Our Location</h3>
              </div>
              <div className="space-y-2 text-sm">
                <p>Experience AI World</p>
                <p>Global Headquarters</p>
                <p>Digital Innovation Hub</p>
                <p>Worldwide</p>
              </div>
              <p className="text-xs text-foreground/60 mt-4">
                We're a remote-first company working with clients globally
              </p>
            </div>

            {/* Quick Links */}
            <div className="bg-background-secondary/50 backdrop-blur-sm rounded-3xl p-6 border border-glass-border">
              <h3 className="text-xl font-bold mb-4">Quick Links</h3>
              <div className="space-y-3">
                <Link href="/blog" className="block text-sm hover:text-accent-blue transition-colors">
                  📝 Browse Articles
                </Link>
                <Link href="/categories" className="block text-sm hover:text-accent-blue transition-colors">
                  🏷️ Explore Categories
                </Link>
                <Link href="/about" className="block text-sm hover:text-accent-blue transition-colors">
                  👥 About Us
                </Link>
                <a href="mailto:hello@experienceaiworld.com" className="block text-sm hover:text-accent-blue transition-colors">
                  📧 Email Support
                </a>
              </div>
            </div>
          </motion.div>
        </div>
      </div>

      {/* FAQ Section */}
      <motion.section
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.4 }}
        className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16"
      >
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Frequently Asked Questions</h2>
          <p className="text-xl text-foreground/70">Common questions about working with us</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {faqs.map((faq, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.6 + index * 0.1 }}
              className="bg-background-secondary/50 backdrop-blur-sm rounded-2xl p-6 border border-glass-border"
            >
              <h3 className="text-lg font-bold mb-3 text-accent-blue">{faq.question}</h3>
              <p className="text-foreground/70 leading-relaxed">{faq.answer}</p>
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* Social Media Section */}
      <motion.section
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.8 }}
        className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-20"
      >
        <div className="bg-gradient-to-r from-background-secondary/50 to-background-secondary/30 backdrop-blur-sm rounded-3xl p-8 border border-glass-border text-center">
          <h2 className="text-2xl font-bold mb-4">Follow Us</h2>
          <p className="text-foreground/70 mb-6 max-w-2xl mx-auto">
            Stay connected for the latest AI insights, tutorials, and industry updates.
            Follow us on social media for real-time updates!
          </p>
          <div className="flex justify-center gap-6">
            <motion.a
              whileHover={{ scale: 1.1, y: -2 }}
              href="https://twitter.com/experienceaiworld"
              target="_blank"
              rel="noopener noreferrer"
              className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center text-white hover:bg-blue-600 transition-colors"
            >
              <Twitter className="w-5 h-5" />
            </motion.a>
            <motion.a
              whileHover={{ scale: 1.1, y: -2 }}
              href="https://linkedin.com/company/experienceaiworld"
              target="_blank"
              rel="noopener noreferrer"
              className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center text-white hover:bg-blue-700 transition-colors"
            >
              <Linkedin className="w-5 h-5" />
            </motion.a>
            <motion.a
              whileHover={{ scale: 1.1, y: -2 }}
              href="https://github.com/experienceaiworld"
              target="_blank"
              rel="noopener noreferrer"
              className="w-12 h-12 bg-gray-800 rounded-full flex items-center justify-center text-white hover:bg-gray-900 transition-colors"
            >
              <Github className="w-5 h-5" />
            </motion.a>
          </div>
        </div>
      </motion.section>
    </div>
  );
}
