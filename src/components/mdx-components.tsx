import type { ComponentProps } from 'react';
import CodeBlock from './CodeBlock';
import Image from 'next/image';

// Define proper types for component props
type ComponentPropsWithChildren<T extends keyof React.JSX.IntrinsicElements> = ComponentProps<T> & {
  children?: React.ReactNode;
};

// Specific type for img component to ensure src is a string
type ImgProps = Omit<ComponentProps<'img'>, 'src'> & {
  src?: string;
  children?: React.ReactNode;
};

// Define the components object type
type ComponentsType = Record<string, React.ComponentType<any>>;

export const mdxComponents: ComponentsType = {
  // Code blocks with syntax highlighting
  pre: ({ children, ...props }: ComponentPropsWithChildren<'pre'>) => {
    // Extract language from className if it exists
    const className = props.className as string;
    if (className?.startsWith('language-')) {
      return <CodeBlock {...props}>{children as string}</CodeBlock>;
    }
    return <pre {...props}>{children}</pre>;
  },

  code: ({ children, className, ...props }: ComponentPropsWithChildren<'code'>) => {
    const match = /language-(\w+)/.exec(className || '');
    if (match) {
      return <CodeBlock language={match[1]} {...props}>{children as string}</CodeBlock>;
    }
    return (
      <code className="bg-background-tertiary px-1.5 py-0.5 rounded text-sm font-mono" {...props}>
        {children}
      </code>
    );
  },

  // Enhanced blockquote
  blockquote: ({ children, ...props }: ComponentPropsWithChildren<'blockquote'>) => (
    <blockquote className="border-l-4 border-accent-blue pl-6 py-4 my-8 bg-background-secondary rounded-r-lg italic text-foreground/90" {...props}>
      {children}
    </blockquote>
  ),

  // Callout components
  div: ({ className, children, ...props }: ComponentPropsWithChildren<'div'>) => {
    if (className?.includes('callout')) {
      const variant = className.includes('warning') ? 'warning' :
                     className.includes('success') ? 'success' : 'info';
      return (
        <div className={`callout callout-${variant}`} {...props}>
          {children}
        </div>
      );
    }
    return <div className={className} {...props}>{children}</div>;
  },

  // Enhanced images
  img: ({ src, alt, ...props }: ImgProps) => {
    if (!src) return null;

    // If it's an external URL, use regular img tag
    if (src.startsWith('http')) {
      return (
        <img
          src={src}
          alt={alt}
          className="rounded-xl shadow-lg max-w-full h-auto my-8"
          {...props}
        />
      );
    }

    // For local images, use Next.js Image component
    return (
      <div className="my-8">
        <Image
          src={src}
          alt={alt || ''}
          width={800}
          height={400}
          className="rounded-xl shadow-lg w-full h-auto"
        />
      </div>
    );
  },

  // Enhanced headings with anchor links
  h1: ({ children, ...props }: ComponentPropsWithChildren<'h1'>) => (
    <h1 className="text-4xl font-bold mt-12 mb-6 text-gradient" {...props}>
      {children}
    </h1>
  ),

  h2: ({ children, ...props }: ComponentPropsWithChildren<'h2'>) => (
    <h2 className="text-3xl font-bold mt-10 mb-5 text-gradient" {...props}>
      {children}
    </h2>
  ),

  h3: ({ children, ...props }: ComponentPropsWithChildren<'h3'>) => (
    <h3 className="text-2xl font-bold mt-8 mb-4" {...props}>
      {children}
    </h3>
  ),

  h4: ({ children, ...props }: ComponentPropsWithChildren<'h4'>) => (
    <h4 className="text-xl font-semibold mt-6 mb-3" {...props}>
      {children}
    </h4>
  ),

  h5: ({ children, ...props }: ComponentPropsWithChildren<'h5'>) => (
    <h5 className="text-lg font-semibold mt-5 mb-3" {...props}>
      {children}
    </h5>
  ),

  h6: ({ children, ...props }: ComponentPropsWithChildren<'h6'>) => (
    <h6 className="text-base font-semibold mt-4 mb-2" {...props}>
      {children}
    </h6>
  ),

  // Enhanced paragraphs
  p: ({ children, ...props }: ComponentPropsWithChildren<'p'>) => (
    <p className="text-lg leading-relaxed mb-6 text-foreground/90" {...props}>
      {children}
    </p>
  ),

  // Enhanced lists
  ul: ({ children, ...props }: ComponentPropsWithChildren<'ul'>) => (
    <ul className="list-disc list-inside space-y-2 my-6 text-foreground/90" {...props}>
      {children}
    </ul>
  ),

  ol: ({ children, ...props }: ComponentPropsWithChildren<'ol'>) => (
    <ol className="list-decimal list-inside space-y-2 my-6 text-foreground/90" {...props}>
      {children}
    </ol>
  ),

  li: ({ children, ...props }: ComponentPropsWithChildren<'li'>) => (
    <li className="leading-relaxed" {...props}>
      {children}
    </li>
  ),

  // Enhanced links
  a: ({ children, href, ...props }: ComponentPropsWithChildren<'a'>) => (
    <a
      href={href}
      className="text-accent-blue hover:text-accent-purple transition-colors duration-200 underline decoration-2 underline-offset-2 hover:decoration-accent-purple"
      {...props}
    >
      {children}
    </a>
  ),

  // Tables
  table: ({ children, ...props }: ComponentPropsWithChildren<'table'>) => (
    <div className="overflow-x-auto my-8">
      <table className="min-w-full border-collapse border border-glass-border rounded-lg overflow-hidden" {...props}>
        {children}
      </table>
    </div>
  ),

  thead: ({ children, ...props }: ComponentPropsWithChildren<'thead'>) => (
    <thead className="bg-background-secondary" {...props}>
      {children}
    </thead>
  ),

  tbody: ({ children, ...props }: ComponentPropsWithChildren<'tbody'>) => (
    <tbody {...props}>
      {children}
    </tbody>
  ),

  tr: ({ children, ...props }: ComponentPropsWithChildren<'tr'>) => (
    <tr className="border-b border-glass-border hover:bg-background-secondary/50 transition-colors" {...props}>
      {children}
    </tr>
  ),

  th: ({ children, ...props }: ComponentPropsWithChildren<'th'>) => (
    <th className="px-6 py-4 text-left font-semibold text-foreground" {...props}>
      {children}
    </th>
  ),

  td: ({ children, ...props }: ComponentPropsWithChildren<'td'>) => (
    <td className="px-6 py-4 text-foreground/90" {...props}>
      {children}
    </td>
  ),

  // Horizontal rule
  hr: ({ ...props }: ComponentPropsWithChildren<'hr'>) => (
    <hr className="border-none border-t-2 border-accent-blue/20 my-12" {...props} />
  ),
};

export default mdxComponents;
