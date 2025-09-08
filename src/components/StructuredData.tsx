import { Metadata } from 'next';

interface StructuredDataProps {
  type: 'website' | 'article' | 'organization' | 'breadcrumb';
  data: any;
}

export function StructuredData({ type, data }: StructuredDataProps) {
  const getStructuredData = () => {
    switch (type) {
      case 'website':
        return {
          '@context': 'https://schema.org',
          '@type': 'WebSite',
          name: 'Experience AI World',
          description: 'Stay ahead with the latest in AI & Technology. Daily insights, tutorials, reviews, and trends shaping the future of Artificial Intelligence.',
          url: 'https://experience-ai-world.vercel.app',
          potentialAction: {
            '@type': 'SearchAction',
            target: 'https://experience-ai-world.vercel.app/blog?search={search_term_string}',
            'query-input': 'required name=search_term_string',
          },
          publisher: {
            '@type': 'Organization',
            name: 'Experience AI World',
            logo: {
              '@type': 'ImageObject',
              url: 'https://experience-ai-world.vercel.app/logo.png',
            },
          },
        };

      case 'article':
        return {
          '@context': 'https://schema.org',
          '@type': 'Article',
          headline: data.title,
          description: data.excerpt,
          image: [data.coverImage],
          datePublished: data.date,
          dateModified: data.date,
          author: {
            '@type': 'Person',
            name: data.author.name,
          },
          publisher: {
            '@type': 'Organization',
            name: 'Experience AI World',
            logo: {
              '@type': 'ImageObject',
              url: 'https://experience-ai-world.vercel.app/logo.png',
            },
          },
          mainEntityOfPage: {
            '@type': 'WebPage',
            '@id': `https://experience-ai-world.vercel.app/blog/${data.id}`,
          },
          articleSection: data.category,
          keywords: data.tags?.join(', '),
        };

      case 'organization':
        return {
          '@context': 'https://schema.org',
          '@type': 'Organization',
          name: 'Experience AI World',
          url: 'https://experience-ai-world.vercel.app',
          logo: 'https://experience-ai-world.vercel.app/logo.png',
          description: 'Your premier destination for AI and technology insights, tutorials, and industry analysis.',
          sameAs: [
            'https://twitter.com/experienceaiworld',
            'https://linkedin.com/company/experienceaiworld',
            'https://github.com/experienceaiworld',
          ],
          contactPoint: {
            '@type': 'ContactPoint',
            contactType: 'customer service',
            email: 'hello@experienceaiworld.com',
          },
        };

      case 'breadcrumb':
        return {
          '@context': 'https://schema.org',
          '@type': 'BreadcrumbList',
          itemListElement: data.map((item: any, index: number) => ({
            '@type': 'ListItem',
            position: index + 1,
            name: item.name,
            item: item.url,
          })),
        };

      default:
        return {};
    }
  };

  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{
        __html: JSON.stringify(getStructuredData()),
      }}
    />
  );
}

// Pre-built structured data components
export function WebsiteStructuredData() {
  return (
    <StructuredData
      type="website"
      data={{}}
    />
  );
}

export function ArticleStructuredData({ post }: { post: any }) {
  return (
    <StructuredData
      type="article"
      data={post}
    />
  );
}

export function OrganizationStructuredData() {
  return (
    <StructuredData
      type="organization"
      data={{}}
    />
  );
}

export function BreadcrumbStructuredData({ items }: { items: Array<{ name: string; url: string }> }) {
  return (
    <StructuredData
      type="breadcrumb"
      data={items}
    />
  );
}