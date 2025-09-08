'use client';

import { motion } from 'framer-motion';
import { Clock, User, Calendar, ArrowRight, Heart, MessageSquare, Share2 } from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';
import { useState } from 'react';

// Mock blog posts data
const blogPosts = [
  {
    id: 1,
    title: "The Future of AI: From ChatGPT to AGI",
    excerpt: "Explore how artificial intelligence is evolving from conversational models to artificial general intelligence, and what it means for humanity.",
    content: `
# The Future of AI: From ChatGPT to AGI

Artificial Intelligence has come a long way since its inception. From simple rule-based systems to today's sophisticated neural networks, the journey has been remarkable and continues to accelerate at an unprecedented pace.

## The Evolution of AI

The field of artificial intelligence has evolved through several distinct phases:

1. **Rule-Based Systems** (1950s-1980s)
2. **Machine Learning** (1990s-2010s)
3. **Deep Learning** (2010s-Present)
4. **Artificial General Intelligence** (Future)

Each phase has brought new capabilities and challenges, pushing the boundaries of what machines can achieve.

## Current State of AI

Today, we're witnessing unprecedented advances in AI capabilities. Models like GPT-4, Claude, and Gemini have demonstrated near-human level performance in various tasks including:

- Natural language understanding and generation
- Code generation and debugging
- Creative content creation
- Complex problem-solving

## The Path to AGI

Artificial General Intelligence (AGI) represents the next frontier. Unlike narrow AI systems designed for specific tasks, AGI would possess the ability to understand, learn, and apply intelligence across any domain.

### Key Characteristics of AGI:
- **General Problem Solving**: Ability to tackle any intellectual task
- **Transfer Learning**: Apply knowledge from one domain to another
- **Self-Improvement**: Capability to improve its own algorithms
- **Creativity**: Generate novel ideas and solutions

> "The development of AGI will be the most important technological event in human history." - Various AI researchers

## Challenges Ahead

While the potential benefits of AGI are enormous, so too are the challenges:

### Technical Challenges:
- **Scalability**: Training models that can handle any task
- **Efficiency**: Reducing computational requirements
- **Safety**: Ensuring reliable and predictable behavior

### Ethical Considerations:
- **Alignment**: Ensuring AI goals align with human values
- **Bias**: Eliminating unfair biases in training data
- **Control**: Maintaining human oversight and control

## Societal Impact

The development of AGI will have profound implications for society:

- **Job Displacement**: Many traditional jobs may be automated
- **Economic Growth**: New industries and opportunities will emerge
- **Healthcare**: Advanced diagnostics and personalized medicine
- **Education**: Adaptive learning systems for all

## Conclusion

As we stand on the brink of this technological revolution, it's crucial that we approach AGI development with both ambition and caution. The future of AI is not just about building more powerful systems—it's about ensuring those systems benefit all of humanity.

The journey from ChatGPT to AGI is not just a technological evolution; it's a fundamental shift in how we understand intelligence itself. By approaching this challenge thoughtfully and collaboratively, we can create a future where AI enhances human potential rather than replacing it.

---

*What are your thoughts on the future of AI? Share your perspective in the comments below.*
    `,
    coverImage: "/api/placeholder/600/400",
    category: "AI",
    author: {
      name: "Dr. Sarah Chen",
      avatar: "/api/placeholder/64/64",
      bio: "AI researcher and professor specializing in machine learning and AGI development.",
      social: {
        twitter: "sarahchenai",
        linkedin: "sarahchen"
      }
    },
    date: "2025-01-15",
    readTime: "8 min read",
    likes: 142,
    comments: 23,
    tags: ["AI", "AGI", "Future", "Technology"],
    featured: true,
  },
  {
    id: 2,
    title: "Building Your First Machine Learning Model",
    excerpt: "A comprehensive guide for beginners to create, train, and deploy their first ML model using Python and TensorFlow.",
    content: `
# Building Your First Machine Learning Model

Machine Learning has revolutionized the way we approach problem-solving across industries. This comprehensive guide will walk you through creating your first ML model from scratch.

## Prerequisites

Before we begin, ensure you have:
- Python 3.8+ installed
- Basic understanding of Python programming
- Familiarity with data structures (lists, dictionaries, pandas DataFrames)

## Setting Up Your Environment

First, let's create a virtual environment and install the necessary packages:

\`\`\`bash
# Create a virtual environment
python -m venv ml_env

# Activate the environment
# On Windows:
ml_env\\Scripts\\activate
# On macOS/Linux:
source ml_env/bin/activate

# Install required packages
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
\`\`\`

## Understanding the Problem

For this tutorial, we'll build a simple classification model that predicts whether a customer will churn based on their usage patterns. This is a common business problem that many companies face.

### Dataset Overview
Our dataset contains:
- Customer demographics
- Usage statistics
- Service information
- Churn status (our target variable)

## Data Preprocessing

Data preprocessing is crucial for building effective ML models:

\`\`\`python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
df = pd.read_csv('customer_data.csv')

# Handle missing values
df = df.dropna()

# Encode categorical variables
le = LabelEncoder()
categorical_columns = ['gender', 'contract_type', 'internet_service']
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Feature scaling
scaler = StandardScaler()
numerical_columns = ['monthly_charges', 'total_charges', 'tenure']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
\`\`\`

## Model Selection

For this binary classification problem, we'll use several algorithms:

1. **Logistic Regression** - Simple and interpretable
2. **Random Forest** - Ensemble method with good performance
3. **Support Vector Machine** - Effective for complex datasets

## Training the Model

\`\`\`python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split the data
X = df.drop('churn', axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, predictions)}")
print(classification_report(y_test, predictions))
\`\`\`

## Model Evaluation

Understanding your model's performance is crucial:

- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## Improving Your Model

Several techniques to enhance performance:

### Feature Engineering
- Create new features from existing data
- Use domain knowledge to identify important variables
- Consider interaction terms

### Hyperparameter Tuning
\`\`\`python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
\`\`\`

## Deployment Considerations

When deploying your ML model:

1. **Model Persistence**: Save trained models using joblib or pickle
2. **API Development**: Create REST APIs for model predictions
3. **Monitoring**: Track model performance in production
4. **Retraining**: Set up automated model retraining pipelines

## Conclusion

Building your first ML model is an exciting journey that combines programming, statistics, and domain knowledge. Remember:

- Start with simple models and gradually increase complexity
- Always validate your results with appropriate metrics
- Consider the business context when interpreting results
- Continuously iterate and improve your models

The key to success in machine learning is not just technical expertise, but also understanding the problem you're trying to solve and the business context in which your model will operate.

Happy learning! 🚀
    `,
    coverImage: "/api/placeholder/600/400",
    category: "Tutorials",
    author: {
      name: "Mike Johnson",
      avatar: "/api/placeholder/64/64",
      bio: "Data scientist and ML engineer with 5+ years of experience in predictive modeling and AI solutions.",
      social: {
        twitter: "mikejohnsonml",
        linkedin: "mikejohnson"
      }
    },
    date: "2025-01-12",
    readTime: "12 min read",
    likes: 89,
    comments: 15,
    tags: ["Machine Learning", "Python", "TensorFlow", "Tutorial", "Beginners"],
    featured: false,
  },
  {
    id: 3,
    title: "AI Ethics: Navigating the Moral Landscape",
    excerpt: "Understanding the ethical implications of artificial intelligence development and deployment in modern society.",
    coverImage: "/api/placeholder/600/400",
    category: "Ethics",
    author: "Prof. Elena Rodriguez",
    date: "2025-01-10",
    readTime: "6 min read",
    likes: 67,
    comments: 31,
    featured: false,
  },
  {
    id: 4,
    title: "Claude vs GPT-5: The Ultimate AI Showdown",
    excerpt: "An in-depth comparison of Anthropic's Claude and OpenAI's latest GPT model across various use cases and performance metrics.",
    coverImage: "/api/placeholder/600/400",
    category: "Reviews",
    author: "Alex Thompson",
    date: "2025-01-08",
    readTime: "10 min read",
    likes: 203,
    comments: 45,
    featured: true,
  },
  {
    id: 5,
    title: "OpenAI's GPT-5 Architecture Deep Dive",
    excerpt: "Analyzing the technical architecture behind OpenAI's latest language model and its implications for AI development.",
    coverImage: "/api/placeholder/600/400",
    category: "AI",
    author: "Dr. Michael Chen",
    date: "2025-01-14",
    readTime: "15 min read",
    likes: 178,
    comments: 28,
    featured: false,
  },
  {
    id: 6,
    title: "Building Scalable ML Pipelines with Kubernetes",
    excerpt: "Learn how to deploy and manage machine learning workloads at scale using Kubernetes and cloud-native tools.",
    coverImage: "/api/placeholder/600/400",
    category: "Tutorials",
    author: "Sarah Johnson",
    date: "2025-01-13",
    readTime: "12 min read",
    likes: 95,
    comments: 19,
    featured: false,
  },
  {
    id: 7,
    title: "The Rise of Multimodal AI Models",
    excerpt: "Exploring how AI systems are evolving to process and understand multiple types of data simultaneously.",
    coverImage: "/api/placeholder/600/400",
    category: "Technology",
    author: "Prof. David Kim",
    date: "2025-01-11",
    readTime: "9 min read",
    likes: 134,
    comments: 22,
    featured: false,
  },
  {
    id: 8,
    title: "AI Safety Research: Current State and Future Directions",
    excerpt: "A comprehensive overview of the current landscape in AI safety research and alignment challenges.",
    coverImage: "/api/placeholder/600/400",
    category: "AI Ethics",
    author: "Dr. Emily Watson",
    date: "2025-01-09",
    readTime: "14 min read",
    likes: 76,
    comments: 33,
    featured: false,
  },
  {
    id: 9,
    title: "Fine-tuning LLMs for Domain-Specific Tasks",
    excerpt: "Practical guide to adapting large language models for specialized use cases and industry applications.",
    coverImage: "/api/placeholder/600/400",
    category: "Tutorials",
    author: "Alex Rodriguez",
    date: "2025-01-07",
    readTime: "11 min read",
    likes: 112,
    comments: 17,
    featured: false,
  },
];

const fadeInUp = {
  initial: { opacity: 0, y: 60 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6 }
};

const staggerChildren = {
  animate: {
    transition: {
      staggerChildren: 0.1
    }
  }
};

export default function BlogGrid() {
  const [visiblePosts, setVisiblePosts] = useState(9);
  const [likedPosts, setLikedPosts] = useState<Set<number>>(new Set());

  const handleLoadMore = () => {
    setVisiblePosts(prev => prev + 6);
  };

  const handleLike = (postId: number) => {
    setLikedPosts(prev => {
      const newSet = new Set(prev);
      if (newSet.has(postId)) {
        newSet.delete(postId);
      } else {
        newSet.add(postId);
      }
      return newSet;
    });
  };

  const displayedPosts = blogPosts.slice(0, visiblePosts);

  return (
    <section className="py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial="initial"
          whileInView="animate"
          viewport={{ once: true }}
          variants={staggerChildren}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
        >
          {displayedPosts.map((post) => (
            <motion.article
              key={post.id}
              variants={fadeInUp}
              className="card group cursor-pointer overflow-hidden cyber-border"
            >
              {/* Cover Image */}
              <div className="relative h-48 mb-4 overflow-hidden rounded-xl">
                <Image
                  src={post.coverImage}
                  alt={post.title}
                  fill
                  className="object-cover transition-transform duration-300 group-hover:scale-105"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

                {/* Category Badge */}
                <div className="absolute top-3 left-3">
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                    post.category === 'AI' ? 'bg-blue-500/90 text-white' :
                    post.category === 'Tutorials' ? 'bg-green-500/90 text-white' :
                    post.category === 'Reviews' ? 'bg-purple-500/90 text-white' :
                    post.category === 'Ethics' ? 'bg-gray-600/90 text-white' :
                    'bg-accent-blue text-white'
                  }`}>
                    {post.category}
                  </span>
                </div>

                {/* Featured Badge */}
                {post.featured && (
                  <div className="absolute top-3 right-3">
                    <span className="px-2 py-1 bg-yellow-500/90 text-black text-xs font-medium rounded-full">
                      Featured
                    </span>
                  </div>
                )}
              </div>

              {/* Content */}
              <div className="space-y-4">
                <h3 className="text-lg font-bold group-hover:text-accent-blue transition-colors duration-200 line-clamp-2">
                  <Link href={`/blog/${post.id}`}>
                    {post.title}
                  </Link>
                </h3>

                <p className="text-foreground/70 text-sm line-clamp-3">
                  {post.excerpt}
                </p>

                {/* Meta Information */}
                <div className="flex items-center justify-between text-xs text-foreground/60 pt-2 border-t border-glass-border">
                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1">
                      <User className="w-3 h-3" />
                      <span>{post.author.name}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Calendar className="w-3 h-3" />
                      <span>{new Date(post.date).toLocaleDateString()}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    <span>{post.readTime}</span>
                  </div>
                </div>

                {/* Engagement Stats */}
                <div className="flex items-center justify-between pt-2">
                  <div className="flex items-center gap-4">
                    <button
                      onClick={(e) => {
                        e.preventDefault();
                        handleLike(post.id);
                      }}
                      className={`flex items-center gap-1 text-xs transition-colors ${
                        likedPosts.has(post.id)
                          ? 'text-red-500'
                          : 'text-foreground/60 hover:text-red-500'
                      }`}
                    >
                      <Heart
                        className={`w-4 h-4 ${likedPosts.has(post.id) ? 'fill-current' : ''}`}
                      />
                      <span>{likedPosts.has(post.id) ? post.likes + 1 : post.likes}</span>
                    </button>

                    <div className="flex items-center gap-1 text-xs text-foreground/60">
                      <MessageSquare className="w-4 h-4" />
                      <span>{post.comments}</span>
                    </div>
                  </div>

                  <button className="p-1 text-foreground/60 hover:text-accent-blue transition-colors">
                    <Share2 className="w-4 h-4" />
                  </button>
                </div>

                {/* Read More Link */}
                <Link
                  href={`/blog/${post.id}`}
                  className="inline-flex items-center gap-1 text-accent-blue hover:text-accent-purple transition-colors duration-200 text-sm font-medium group/link"
                >
                  Read More
                  <ArrowRight className="w-3 h-3 group-hover/link:translate-x-1 transition-transform" />
                </Link>
              </div>
            </motion.article>
          ))}
        </motion.div>

        {/* Load More Button */}
        {visiblePosts < blogPosts.length && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mt-12"
          >
            <motion.button
              onClick={handleLoadMore}
              className="btn-secondary cyber-border"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <span className="flex items-center gap-2">
                Load More Posts
                <ArrowRight className="w-5 h-5" />
              </span>
            </motion.button>
          </motion.div>
        )}

        {/* No More Posts */}
        {visiblePosts >= blogPosts.length && blogPosts.length > 9 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center mt-12 py-8"
          >
            <p className="text-foreground/60">
              You've reached the end of our blog posts. Check back soon for more content!
            </p>
          </motion.div>
        )}
      </div>
    </section>
  );
}
