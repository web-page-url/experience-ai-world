'use client';

import { useEffect, useState } from 'react';

// Skip to content link for screen readers
export function SkipToContent() {
  return (
    <a
      href="#main-content"
      className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 bg-accent-blue text-white px-4 py-2 rounded-md z-50 focus:outline-none focus:ring-2 focus:ring-accent-blue focus:ring-offset-2"
    >
      Skip to main content
    </a>
  );
}

// High contrast mode toggle
export function HighContrastToggle() {
  const [highContrast, setHighContrast] = useState(false);

  useEffect(() => {
    const root = document.documentElement;
    if (highContrast) {
      root.classList.add('high-contrast');
    } else {
      root.classList.remove('high-contrast');
    }
  }, [highContrast]);

  return (
    <button
      onClick={() => setHighContrast(!highContrast)}
      className="p-2 rounded-lg hover:bg-background-secondary transition-colors"
      aria-label={`Switch to ${highContrast ? 'normal' : 'high contrast'} mode`}
    >
      <span className="sr-only">
        {highContrast ? 'Disable' : 'Enable'} high contrast mode
      </span>
      <div className="w-5 h-5 border-2 border-current rounded-sm flex items-center justify-center">
        <div className={`w-2 h-2 rounded-full transition-colors ${highContrast ? 'bg-current' : 'bg-transparent'}`} />
      </div>
    </button>
  );
}

// Font size adjustment
export function FontSizeAdjuster() {
  const [fontSize, setFontSize] = useState(100);

  useEffect(() => {
    document.documentElement.style.fontSize = `${fontSize}%`;
  }, [fontSize]);

  return (
    <div className="flex items-center gap-2">
      <button
        onClick={() => setFontSize(Math.max(75, fontSize - 10))}
        className="p-1 rounded hover:bg-background-secondary transition-colors"
        aria-label="Decrease font size"
        disabled={fontSize <= 75}
      >
        <span className="text-sm font-bold">A-</span>
      </button>
      <span className="text-xs text-foreground/60 min-w-[3rem] text-center">
        {fontSize}%
      </span>
      <button
        onClick={() => setFontSize(Math.min(150, fontSize + 10))}
        className="p-1 rounded hover:bg-background-secondary transition-colors"
        aria-label="Increase font size"
        disabled={fontSize >= 150}
      >
        <span className="text-sm font-bold">A+</span>
      </button>
    </div>
  );
}

// Screen reader announcements
export function ScreenReaderAnnouncement({ message, priority = 'polite' }: { message: string; priority?: 'polite' | 'assertive' }) {
  const [announcement, setAnnouncement] = useState('');

  useEffect(() => {
    if (message) {
      setAnnouncement(message);
      // Clear the announcement after it's been read
      const timer = setTimeout(() => setAnnouncement(''), 1000);
      return () => clearTimeout(timer);
    }
  }, [message]);

  return (
    <div
      aria-live={priority}
      aria-atomic="true"
      className="sr-only"
    >
      {announcement}
    </div>
  );
}

// Focus trap for modals
export function useFocusTrap(ref: React.RefObject<HTMLElement>, isActive: boolean) {
  useEffect(() => {
    if (!isActive || !ref.current) return;

    const focusableElements = ref.current.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const firstElement = focusableElements[0] as HTMLElement;
    const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

    const handleTabKey = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          lastElement.focus();
          e.preventDefault();
        }
      } else {
        if (document.activeElement === lastElement) {
          firstElement.focus();
          e.preventDefault();
        }
      }
    };

    const handleEscapeKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        // You can emit a custom event or call a callback here
        document.dispatchEvent(new CustomEvent('closeModal'));
      }
    };

    document.addEventListener('keydown', handleTabKey);
    document.addEventListener('keydown', handleEscapeKey);

    // Focus first element
    firstElement?.focus();

    return () => {
      document.removeEventListener('keydown', handleTabKey);
      document.removeEventListener('keydown', handleEscapeKey);
    };
  }, [isActive, ref]);
}

// Accessible disclosure/collapsible component
export function Disclosure({
  children,
  summary,
  defaultOpen = false
}: {
  children: React.ReactNode;
  summary: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div>
      <button
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        className="flex items-center justify-between w-full p-4 text-left hover:bg-background-secondary transition-colors rounded-lg"
      >
        {summary}
        <span
          className={`transform transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
          aria-hidden="true"
        >
          ▼
        </span>
      </button>

      <div
        className={`overflow-hidden transition-all duration-300 ${
          isOpen ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
        }`}
      >
        <div className="p-4">
          {children}
        </div>
      </div>
    </div>
  );
}

// ARIA live region for dynamic content updates
export function LiveRegion({ children, priority = 'polite' }: { children: React.ReactNode; priority?: 'polite' | 'assertive' }) {
  return (
    <div
      aria-live={priority}
      aria-atomic="true"
      className="sr-only"
    >
      {children}
    </div>
  );
}

// Keyboard navigation helper
export function KeyboardNavigation() {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Skip to main content with Ctrl/Cmd + Home
      if ((e.ctrlKey || e.metaKey) && e.key === 'Home') {
        e.preventDefault();
        const mainContent = document.getElementById('main-content');
        mainContent?.focus();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  return null;
}
