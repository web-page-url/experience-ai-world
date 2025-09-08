'use client';

import { useEffect } from 'react';
import { usePathname, useSearchParams } from 'next/navigation';

// Google Analytics
export function GoogleAnalytics({ GA_ID }: { GA_ID?: string }) {
  const pathname = usePathname();
  const searchParams = useSearchParams();

  useEffect(() => {
    if (!GA_ID) return;

    // Track page views
    const url = pathname + (searchParams.toString() ? `?${searchParams.toString()}` : '');

    // Send page view to Google Analytics
    if (typeof window !== 'undefined' && (window as any).gtag) {
      (window as any).gtag('config', GA_ID, {
        page_path: url,
      });
    }
  }, [pathname, searchParams, GA_ID]);

  // Google Analytics script
  if (!GA_ID) return null;

  return (
    <>
      <script
        async
        src={`https://www.googletagmanager.com/gtag/js?id=${GA_ID}`}
      />
      <script
        dangerouslySetInnerHTML={{
          __html: `
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());
            gtag('config', '${GA_ID}', {
              page_title: document.title,
              page_location: window.location.href,
            });
          `,
        }}
      />
    </>
  );
}

// Vercel Analytics
export function VercelAnalytics() {
  return (
    <script
      src="https://vercel.com/vitals"
      data-vitals
      defer
    />
  );
}

// Web Vitals tracking
export function WebVitals() {
  useEffect(() => {
    // Web Vitals reporting
    const reportWebVitals = (metric: any) => {
      // Send to analytics service
      if (typeof window !== 'undefined' && (window as any).gtag) {
        (window as any).gtag('event', metric.name, {
          event_category: 'Web Vitals',
          event_label: metric.id,
          value: Math.round(metric.value),
          custom_map: { metric_value: metric.value },
          non_interaction: true,
        });
      }

      // Log to console in development
      if (process.env.NODE_ENV === 'development') {
        console.log('Web Vitals:', metric);
      }
    };

    // Import and use web-vitals library
    import('web-vitals').then(({ onCLS, onFCP, onLCP, onTTFB }) => {
      onCLS(reportWebVitals);
      onFCP(reportWebVitals);
      onLCP(reportWebVitals);
      onTTFB(reportWebVitals);
      // Note: onFID has been deprecated in newer versions of web-vitals
      // First Input Delay is now tracked via onTTFB
    });
  }, []);

  return null;
}

// Error tracking
export function ErrorTracking() {
  useEffect(() => {
    const handleError = (event: ErrorEvent) => {
      // Send error to analytics service
      if (typeof window !== 'undefined' && (window as any).gtag) {
        (window as any).gtag('event', 'exception', {
          description: event.error?.message || event.message,
          fatal: false,
        });
      }

      // Log to console in development
      if (process.env.NODE_ENV === 'development') {
        console.error('Error tracked:', event.error || event.message);
      }
    };

    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      // Send unhandled promise rejection to analytics service
      if (typeof window !== 'undefined' && (window as any).gtag) {
        (window as any).gtag('event', 'exception', {
          description: `Unhandled promise rejection: ${event.reason}`,
          fatal: false,
        });
      }

      // Log to console in development
      if (process.env.NODE_ENV === 'development') {
        console.error('Unhandled promise rejection:', event.reason);
      }
    };

    window.addEventListener('error', handleError);
    window.addEventListener('unhandledrejection', handleUnhandledRejection);

    return () => {
      window.removeEventListener('error', handleError);
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
    };
  }, []);

  return null;
}

// Combined analytics component
export function Analytics() {
  return (
    <>
      <GoogleAnalytics GA_ID={process.env.NEXT_PUBLIC_GA_ID} />
      <VercelAnalytics />
      <WebVitals />
      <ErrorTracking />
    </>
  );
}

// Event tracking utility
export const trackEvent = (
  action: string,
  category: string,
  label?: string,
  value?: number
) => {
  if (typeof window !== 'undefined' && (window as any).gtag) {
    (window as any).gtag('event', action, {
      event_category: category,
      event_label: label,
      value: value,
    });
  }

  // Log to console in development
  if (process.env.NODE_ENV === 'development') {
    console.log('Event tracked:', { action, category, label, value });
  }
};

// Page view tracking
export const trackPageView = (pagePath: string) => {
  if (typeof window !== 'undefined' && (window as any).gtag) {
    (window as any).gtag('config', process.env.NEXT_PUBLIC_GA_ID, {
      page_path: pagePath,
    });
  }
};