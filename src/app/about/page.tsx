import { Metadata } from 'next';
import AboutClient from './AboutClient';

export const metadata: Metadata = {
  title: 'About | Experience AI World',
  description: 'Learn about Anubhav Chaudhary, the creator of Experience AI World. Discover the mission to share AI knowledge and resources with enthusiasts and learners.',
  keywords: ['Anubhav Chaudhary', 'AI blogger', 'technology writer', 'IIT Mandi', 'AI education', 'tech enthusiast'],
  openGraph: {
    title: 'About | Experience AI World',
    description: 'Meet Anubhav Chaudhary, passionate about sharing AI knowledge with the world.',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'About | Experience AI World',
    description: 'Meet Anubhav Chaudhary, passionate about sharing AI knowledge with the world.',
  },
};

export default function AboutPage() {
  return <AboutClient />;
}
