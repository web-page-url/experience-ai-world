'use client';

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageCircle, X, Send, Bot, User } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface Message {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
}

interface RelevantArticle {
  id: string;
  title: string;
  excerpt: string;
  category: string;
  url: string;
  relevance: number;
}

export default function Chatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'bot',
      content: "🤖 Hi! I'm Anubhav's AI Assistant! I'm here to help you discover amazing articles on our AI blog. What topics interest you? You can ask about AI, machine learning, robotics, quantum computing, or any other tech topic! 🚀",
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsTyping(true);

    try {
      const response = await fetch('/api/chatbot', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: inputMessage }),
      });

      const data = await response.json();

      if (response.ok) {
        const botMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'bot',
          content: data.response,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);
      } else {
        const errorMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'bot',
          content: data.fallbackResponse || "🤖 Sorry, I'm having trouble connecting right now. Please try again later!",
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: "🤖 Oops! Something went wrong. Please try again later!",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <>
      {/* Chatbot Toggle Button */}
      <motion.button
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        onClick={() => setIsOpen(!isOpen)}
        className="fixed bottom-24 right-6 z-50 bg-gradient-to-r from-blue-900 to-blue-800 text-white p-4 rounded-full shadow-lg hover:shadow-xl transition-all duration-300 border border-orange-500/30 hover:border-orange-400/50"
        aria-label="Open AI Chat Assistant"
      >
        <AnimatePresence mode="wait">
          {isOpen ? (
            <motion.div
              key="close"
              initial={{ rotate: -90, opacity: 0 }}
              animate={{ rotate: 0, opacity: 1 }}
              exit={{ rotate: 90, opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <X className="w-6 h-6" />
            </motion.div>
          ) : (
            <motion.div
              key="chat"
              initial={{ rotate: 90, opacity: 0 }}
              animate={{ rotate: 0, opacity: 1 }}
              exit={{ rotate: -90, opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <MessageCircle className="w-6 h-6" />
            </motion.div>
          )}
        </AnimatePresence>
      </motion.button>

      {/* Chat Window */}
      <AnimatePresence mode="wait">
        {isOpen && (
          <motion.div
            key="chat-window"
            initial={{ opacity: 0, y: 100, scale: 0.8 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 100, scale: 0.8 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className="fixed bottom-24 right-6 z-40 w-80 sm:w-96 h-[500px] bg-slate-900/95 backdrop-blur-md border border-orange-500/30 rounded-2xl shadow-2xl overflow-hidden flex flex-col"
          >
            {/* Header */}
            <div className="bg-gradient-to-r from-blue-900 to-blue-800 text-white p-4 border-b border-orange-500/30 flex-shrink-0">
              <div className="flex items-center gap-3 min-w-0 w-full">
                <div className="w-10 h-10 bg-orange-500/20 rounded-full flex items-center justify-center border border-orange-400/30 flex-shrink-0">
                  <Bot className="w-5 h-5" />
                </div>
                <div className="min-w-0 flex-1 overflow-hidden">
                  <h3 className="font-bold text-base sm:text-lg truncate w-full">Anubhav's AI Assistant</h3>
                  <p className="text-xs sm:text-sm opacity-90 truncate w-full">Your AI Blog Guide 🤖</p>
                </div>
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex gap-3 ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  {message.type === 'bot' && (
                    <div className="w-8 h-8 bg-orange-500/20 rounded-full flex items-center justify-center flex-shrink-0 border border-orange-400/30">
                      <Bot className="w-4 h-4 text-orange-400" />
                    </div>
                  )}

                  <div
                    className={`max-w-[240px] sm:max-w-[280px] min-w-0 p-3 rounded-2xl break-words ${
                      message.type === 'user'
                        ? 'bg-orange-500 text-white shadow-lg'
                        : 'bg-slate-800/80 text-gray-100 border border-orange-500/20'
                    }`}
                  >
                    <div className="text-sm leading-relaxed break-words overflow-wrap-anywhere prose prose-sm max-w-none prose-invert">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          a: ({ href, children, ...props }) => (
                            <a
                              href={href}
                              target="_blank"
                              rel="noopener noreferrer"
                              className={`${
                                message.type === 'user'
                                  ? 'text-white underline hover:text-orange-100'
                                  : 'text-orange-400 underline hover:text-orange-300'
                              } transition-colors`}
                              {...props}
                            >
                              {children}
                            </a>
                          ),
                          strong: ({ children, ...props }) => (
                            <strong
                              className={`${
                                message.type === 'user'
                                  ? 'font-bold text-white'
                                  : 'font-bold text-orange-300'
                              }`}
                              {...props}
                            >
                              {children}
                            </strong>
                          ),
                          p: ({ children, ...props }) => (
                            <p className="mb-2 last:mb-0" {...props}>
                              {children}
                            </p>
                          ),
                          ul: ({ children, ...props }) => (
                            <ul className="list-disc list-inside mb-2 space-y-1" {...props}>
                              {children}
                            </ul>
                          ),
                          li: ({ children, ...props }) => (
                            <li className="text-sm" {...props}>
                              {children}
                            </li>
                          ),
                        }}
                      >
                        {message.content}
                      </ReactMarkdown>
                    </div>
                    <span className="text-xs opacity-60 mt-2 block">
                      {message.timestamp.toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit'
                      })}
                    </span>
                  </div>

                  {message.type === 'user' && (
                    <div className="w-8 h-8 bg-blue-600/20 rounded-full flex items-center justify-center flex-shrink-0 border border-blue-400/30">
                      <User className="w-4 h-4 text-blue-400" />
                    </div>
                  )}
                </motion.div>
              ))}

              {/* Typing Indicator */}
              {isTyping && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex gap-3 justify-start"
                >
                  <div className="w-8 h-8 bg-orange-500/20 rounded-full flex items-center justify-center border border-orange-400/30">
                    <Bot className="w-4 h-4 text-orange-400" />
                  </div>
                  <div className="bg-slate-800/80 p-3 rounded-2xl border border-orange-500/20">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-orange-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-orange-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2 h-2 bg-orange-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                  </div>
                </motion.div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="border-t border-orange-500/30 p-4 flex-shrink-0">
              <div className="flex gap-2">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask about AI articles..."
                  className="flex-1 min-w-0 px-4 py-2 bg-slate-800/80 border border-orange-500/30 rounded-full text-sm text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-orange-400 focus:border-orange-400/50"
                  disabled={isTyping}
                />
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={sendMessage}
                  disabled={!inputMessage.trim() || isTyping}
                  className="p-2 bg-orange-500 text-white rounded-full hover:bg-orange-400 transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-lg"
                >
                  <Send className="w-4 h-4" />
                </motion.button>
              </div>
              <p className="text-xs text-orange-400/70 mt-2 text-center">
                ✨ Everything created by Anubhav
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}