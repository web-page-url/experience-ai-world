import { Metadata } from 'next';
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
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative py-20 bg-gradient-to-br from-accent-blue/5 via-accent-purple/5 to-accent-pink/5">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="space-y-6">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-accent-blue/10 border border-accent-blue/20 rounded-full">
              <span className="text-sm font-medium text-accent-blue">About Me</span>
            </div>

            <h1 className="text-4xl md:text-5xl font-bold">
              Meet the Creator of{' '}
              <span className="text-gradient">Experience AI World</span>
            </h1>

            <p className="text-xl text-foreground/70 max-w-2xl mx-auto">
              Passionate about making AI knowledge accessible to everyone through engaging content and practical insights.
            </p>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <section className="py-20">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            {/* About Content */}
            <div className="space-y-8">
              <div className="space-y-4">
                <h2 className="text-3xl font-bold text-foreground">
                  Hi, I'm <span className="text-gradient">Anubhav Chaudhary</span>
                </h2>
                <p className="text-lg text-foreground/70 leading-relaxed">
                  This webpage was created by Anubhav Chaudhary, a passionate technologist dedicated to making AI knowledge accessible to everyone.
                </p>
              </div>

              {/* Education */}
              <div className="card space-y-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-accent-blue/10 rounded-lg flex items-center justify-center">
                    <span className="text-accent-blue">🎓</span>
                  </div>
                  <h3 className="text-xl font-semibold">Education</h3>
                </div>
                <div className="space-y-2">
                  <p className="font-medium">B.Tech in Computer Science</p>
                  <p className="text-foreground/70">Indian Institute of Technology (IIT) Mandi</p>
                  <p className="text-sm text-foreground/60">
                    Specialized in Computer Science with a focus on emerging technologies and AI systems.
                  </p>
                </div>
              </div>

              {/* Passion */}
              <div className="card space-y-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-accent-purple/10 rounded-lg flex items-center justify-center">
                    <span className="text-accent-purple">✨</span>
                  </div>
                  <h3 className="text-xl font-semibold">Passion</h3>
                </div>
                <p className="text-foreground/70 leading-relaxed">
                  I'm passionate about sharing my AI knowledge with interested individuals. Through this platform,
                  I aim to break down complex AI concepts into digestible, practical insights that help learners
                  at all levels understand and apply artificial intelligence in their projects and careers.
                </p>
              </div>
            </div>

            {/* Profile Image Placeholder */}
            <div className="relative">
              <div className="relative w-full max-w-md mx-auto">
                <div className="aspect-square rounded-2xl bg-gradient-to-br from-accent-blue/10 to-accent-purple/10 flex items-center justify-center">
                  <span className="text-6xl">👨‍💻</span>
                </div>
                {/* Decorative elements */}
                <div className="absolute -top-4 -right-4 w-8 h-8 bg-accent-blue rounded-full flex items-center justify-center">
                  <span className="text-white text-sm">✨</span>
                </div>
                <div className="absolute -bottom-4 -left-4 w-8 h-8 bg-accent-purple rounded-full flex items-center justify-center">
                  <span className="text-white text-sm">🎯</span>
                </div>
              </div>
            </div>
          </div>

          {/* Mission Section */}
          <div className="mt-20">
            <div className="text-center mb-12">
              <h2 className="text-3xl md:text-4xl font-bold mb-4">
                Our <span className="text-gradient">Mission</span>
              </h2>
              <p className="text-xl text-foreground/70 max-w-3xl mx-auto">
                Democratizing AI knowledge through accessible, practical, and engaging content
              </p>
            </div>

            <div className="card max-w-4xl mx-auto">
              <div className="space-y-6">
                <h3 className="text-2xl font-semibold text-center">Purpose</h3>
                <p className="text-lg text-foreground/70 leading-relaxed text-center max-w-3xl mx-auto">
                  This platform was developed to disseminate AI-related knowledge and resources to enthusiasts and learners in the field.
                  Our goal is to create a comprehensive hub where anyone, regardless of their background, can learn about artificial intelligence,
                  stay updated with the latest trends, and gain practical skills to apply AI in real-world scenarios.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
                  <div className="text-center space-y-3">
                    <div className="w-12 h-12 bg-accent-blue/10 rounded-lg flex items-center justify-center mx-auto">
                      <span className="text-accent-blue">🎯</span>
                    </div>
                    <h4 className="font-semibold">Practical Learning</h4>
                    <p className="text-sm text-foreground/70">Hands-on tutorials and real-world applications</p>
                  </div>

                  <div className="text-center space-y-3">
                    <div className="w-12 h-12 bg-accent-purple/10 rounded-lg flex items-center justify-center mx-auto">
                      <span className="text-accent-purple">✨</span>
                    </div>
                    <h4 className="font-semibold">Latest Insights</h4>
                    <p className="text-sm text-foreground/70">Stay updated with cutting-edge AI developments</p>
                  </div>

                  <div className="text-center space-y-3">
                    <div className="w-12 h-12 bg-accent-pink/10 rounded-lg flex items-center justify-center mx-auto">
                      <span className="text-accent-pink">👥</span>
                    </div>
                    <h4 className="font-semibold">Community Driven</h4>
                    <p className="text-sm text-foreground/70">Learn together with fellow AI enthusiasts</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Connect Section */}
          <div className="mt-20 text-center">
            <h2 className="text-3xl font-bold mb-8">Let's Connect</h2>
            <p className="text-lg text-foreground/70 mb-8 max-w-2xl mx-auto">
              Interested in AI discussions, collaborations, or have questions? I'd love to hear from you!
            </p>

            <div className="flex flex-wrap justify-center gap-4">
              <a href="mailto:anubhav@example.com" className="btn-secondary cursor-target">
                📧 Email Me
              </a>
              <a href="https://linkedin.com/in/anubhavchaudhary" target="_blank" className="btn-secondary cursor-target">
                💼 LinkedIn
              </a>
              <a href="https://twitter.com/anubhav_ai" target="_blank" className="btn-secondary cursor-target">
                🐦 Twitter
              </a>
              <a href="https://github.com/anubhavchaudhary" target="_blank" className="btn-secondary cursor-target">
                💻 GitHub
              </a>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
