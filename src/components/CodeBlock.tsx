'use client';

import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check } from 'lucide-react';
import { useState } from 'react';
import { useTheme } from './ThemeProvider';

interface CodeBlockProps {
  children: string;
  className?: string;
  language?: string;
}

export default function CodeBlock({ children, className, ...props }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const { theme } = useTheme();

  const match = /language-(\w+)/.exec(className || '');
  const language = match ? match[1] : '';

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(children);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  return (
    <div className="relative group">
      <button
        onClick={handleCopy}
        className="absolute top-4 right-4 p-2 rounded-lg bg-background-secondary hover:bg-background-tertiary transition-colors opacity-0 group-hover:opacity-100 z-10"
        aria-label="Copy code"
      >
        {copied ? (
          <Check className="w-4 h-4 text-green-500" />
        ) : (
          <Copy className="w-4 h-4 text-foreground/60" />
        )}
      </button>

      <SyntaxHighlighter
        language={language}
        style={theme === 'dark' ? oneDark : oneLight}
        customStyle={{
          margin: 0,
          borderRadius: '0.75rem',
          padding: '1.5rem',
          fontSize: '0.875rem',
          lineHeight: '1.5',
        }}
        showLineNumbers={true}
        wrapLines={true}
        {...props}
      >
        {children.trim()}
      </SyntaxHighlighter>
    </div>
  );
}
