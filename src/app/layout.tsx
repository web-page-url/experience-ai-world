import type { Metadata } from "next";
import { Inter, Sora, Poppins } from "next/font/google";
import "./globals.css";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import ScrollToTop from "@/components/ScrollToTop";
import ThemeProvider from "@/components/ThemeProvider";
import { WebsiteStructuredData, OrganizationStructuredData } from "@/components/StructuredData";
import { Analytics } from "@/components/Analytics";
import { SkipToContent, KeyboardNavigation } from "@/components/Accessibility";
import TargetCursor from "@/components/TargetCursor";
import Chatbot from "@/components/Chatbot";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  display: "swap",
});

const sora = Sora({
  variable: "--font-sora",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800"],
  display: "swap",
});

const poppins = Poppins({
  variable: "--font-poppins",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  display: "swap",
});

export const metadata: Metadata = {
  metadataBase: new URL('https://experience-ai-world.vercel.app'),
  title: {
    default: "Experience AI World | AI & Technology Blog",
    template: "%s | Experience AI World"
  },
  description: "Stay ahead with the latest in AI & Technology. Daily insights, tutorials, reviews, and trends shaping the future of Artificial Intelligence.",
  keywords: ["AI", "artificial intelligence", "technology", "machine learning", "deep learning", "tech tutorials", "AI news", "tech reviews"],
  authors: [{ name: "Experience AI World Team" }],
  creator: "Experience AI World",
  publisher: "Experience AI World",
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://experience-ai-world.vercel.app",
    title: "Experience AI World | AI & Technology Blog",
    description: "Stay ahead with the latest in AI & Technology. Daily insights, tutorials, reviews, and trends shaping the future of Artificial Intelligence.",
    siteName: "Experience AI World",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "Experience AI World - AI & Technology Blog",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Experience AI World | AI & Technology Blog",
    description: "Stay ahead with the latest in AI & Technology. Daily insights, tutorials, reviews, and trends shaping the future.",
    images: ["/og-image.png"],
    creator: "@experienceaiworld",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  verification: {
    google: "google-site-verification-code",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="icon" href="/favicon.ico" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
        <link rel="manifest" href="/manifest.json" />
        <meta name="theme-color" content="#000000" />
        <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
        <meta name="robots" content="index, follow" />
        <meta name="googlebot" content="index, follow, max-video-preview:-1, max-image-preview:large, max-snippet:-1" />
        <link rel="canonical" href="https://experience-ai-world.vercel.app" />
      </head>
      <body
        className={`${inter.variable} ${sora.variable} ${poppins.variable} font-inter antialiased`}
        suppressHydrationWarning={true}
      >
        <ThemeProvider>
          {/* Target Cursor Effect */}
          <TargetCursor />

          {/* Accessibility */}
          <SkipToContent />
          <KeyboardNavigation />

          {/* Navigation */}
          <Navigation />

          <main id="main-content" className="min-h-screen" role="main">
            {children}
          </main>

          {/* Footer */}
          <Footer />

          {/* Scroll to Top */}
          <ScrollToTop />

          {/* Structured Data */}
          <WebsiteStructuredData />
          <OrganizationStructuredData />

          {/* Analytics */}
          <Analytics />

          {/* AI Chatbot */}
          <Chatbot />
        </ThemeProvider>
      </body>
    </html>
  );
}
