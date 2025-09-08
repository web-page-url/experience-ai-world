'use client';

import { motion } from 'framer-motion';
import { Clock, User, ArrowRight, Calendar } from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';

const featuredPosts = [
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
    featured: true,
  },
  {
    id: 3,
    title: "AI Ethics: Navigating the Moral Landscape",
    excerpt: "Understanding the ethical implications of artificial intelligence development and deployment in modern society.",
    content: `
# AI Ethics: Navigating the Moral Landscape

As artificial intelligence becomes increasingly integrated into our daily lives, the ethical implications of its development and deployment have become a critical concern for researchers, policymakers, and society at large.

## The Ethical Framework

AI ethics encompasses several key principles:

### 1. **Fairness and Bias**
AI systems can perpetuate or amplify existing biases in training data, leading to discriminatory outcomes.

### 2. **Transparency and Explainability**
Understanding how AI systems make decisions is crucial for accountability and trust.

### 3. **Privacy and Data Protection**
AI systems often require vast amounts of data, raising concerns about privacy and consent.

### 4. **Safety and Robustness**
Ensuring AI systems operate safely and reliably, even in unexpected situations.

## Current Challenges

### Algorithmic Bias
Machine learning models trained on biased datasets can produce unfair outcomes:

- **Hiring algorithms** that favor certain demographics
- **Credit scoring systems** that disadvantage minority groups
- **Facial recognition** systems with higher error rates for certain ethnicities

### Autonomous Weapons
The development of AI-powered military systems raises profound ethical questions about:
- The delegation of life-and-death decisions to machines
- The risk of accidental escalation
- The erosion of human dignity in warfare

## Building Ethical AI

### Technical Solutions
- **Bias detection and mitigation** techniques
- **Explainable AI** (XAI) frameworks
- **Privacy-preserving machine learning** methods
- **Adversarial robustness** training

### Governance and Regulation
- **Industry standards** for ethical AI development
- **Government regulations** and oversight mechanisms
- **International cooperation** on AI governance
- **Ethical review boards** for AI projects

## The Human Element

Ultimately, the ethical development of AI depends on human judgment and values. We must:

1. **Foster interdisciplinary collaboration** between technologists, ethicists, and policymakers
2. **Promote diversity** in AI development teams to bring different perspectives
3. **Educate stakeholders** about AI capabilities and limitations
4. **Develop ethical frameworks** that evolve with technological advances

## Looking Forward

As AI continues to advance, ethical considerations will become increasingly important. The choices we make today about how to develop and deploy AI will shape the future of our society.

The goal is not to slow down AI progress, but to ensure that progress serves humanity's best interests and creates a future where technology enhances human flourishing rather than diminishing it.

---

*What ethical considerations do you think are most important for AI development?*
    `,
    coverImage: "/api/placeholder/600/400",
    category: "Opinion",
    author: {
      name: "Prof. Elena Rodriguez",
      avatar: "/api/placeholder/64/64",
      bio: "Professor of AI Ethics and Philosophy at Stanford University, specializing in technology policy and human-AI interaction.",
      social: {
        twitter: "elenarodriguezet",
        linkedin: "elenarodriguez"
      }
    },
    date: "2025-01-10",
    readTime: "6 min read",
    likes: 67,
    comments: 31,
    tags: ["AI Ethics", "Bias", "Regulation", "Society", "Future"],
    featured: true,
  },
  {
    id: 4,
    title: "Claude vs GPT-5: The Ultimate AI Showdown",
    excerpt: "An in-depth comparison of Anthropic's Claude and OpenAI's latest GPT model across various use cases and performance metrics.",
    content: `
# Claude vs GPT-5: The Ultimate AI Showdown

The AI landscape continues to evolve rapidly, with major players pushing the boundaries of what's possible. This comprehensive comparison pits Anthropic's Claude against OpenAI's latest GPT-5 model across multiple dimensions.

## Model Specifications

### Claude (Latest Version)
- **Company**: Anthropic
- **Architecture**: Transformer-based with safety-first design
- **Context Window**: 200K tokens
- **Training Data**: Up to 2023
- **Key Features**: Constitutional AI, safety-focused, multi-modal capabilities

### GPT-5
- **Company**: OpenAI
- **Architecture**: Advanced transformer with multi-modal support
- **Context Window**: 128K tokens (expandable)
- **Training Data**: Up to 2024
- **Key Features**: Multimodal, real-time knowledge, advanced reasoning

## Performance Comparison

### Natural Language Understanding
Both models excel in language tasks, but show different strengths:

**Claude's Strengths:**
- More nuanced understanding of context
- Better at maintaining consistent personas
- Superior in creative writing tasks
- More reliable in handling sensitive topics

**GPT-5's Strengths:**
- More comprehensive knowledge base
- Better at complex mathematical reasoning
- Superior code generation capabilities
- More flexible in handling diverse topics

### Coding Capabilities

\`\`\`python
# Example: Complex algorithm implementation

def fibonacci_optimized(n):
    """
    Calculate nth Fibonacci number using matrix exponentiation
    Time complexity: O(log n)
    """
    if n <= 1:
        return n

    def multiply_matrices(a, b):
        return [
            [a[0][0]*b[0][0] + a[0][1]*b[1][0], a[0][0]*b[0][1] + a[0][1]*b[1][1]],
            [a[1][0]*b[0][0] + a[1][1]*b[1][0], a[1][0]*b[0][1] + a[1][1]*b[1][1]]
        ]

    def matrix_power(matrix, power):
        result = [[1, 0], [0, 1]]  # Identity matrix
        while power > 0:
            if power % 2 == 1:
                result = multiply_matrices(result, matrix)
            matrix = multiply_matrices(matrix, matrix)
            power //= 2
        return result

    transformation_matrix = [[1, 1], [1, 0]]
    powered_matrix = matrix_power(transformation_matrix, n - 1)
    return powered_matrix[0][0]

# Test the function
print(fibonacci_optimized(10))  # Output: 55
\`\`\`

**Coding Results:**
- **GPT-5**: 95% accuracy, faster code generation, better optimization
- **Claude**: 92% accuracy, more readable code, better documentation

### Creative Tasks

When given the prompt: "Write a short story about a time-traveling librarian who accidentally changes history by recommending the wrong book."

**Claude's Approach:**
- Focuses on character development and emotional depth
- Creates more nuanced, literary prose
- Emphasizes themes of responsibility and human connection

**GPT-5's Approach:**
- More adventurous and plot-driven
- Includes more technical details about time travel mechanics
- Often incorporates humor or unexpected twists

## Safety and Ethics

### Claude's Safety Features
- Constitutional AI framework
- Built-in safety mechanisms
- More conservative in handling controversial topics
- Better at detecting and refusing harmful requests

### GPT-5's Safety Features
- Advanced content filtering
- Real-time safety monitoring
- More flexible in handling edge cases
- Better at providing helpful responses to complex queries

## Pricing and Accessibility

### Claude
- **Free Tier**: Limited usage
- **Pro Tier**: $20/month
- **Team/Business**: Custom pricing
- **API Access**: Available

### GPT-5
- **Free Tier**: Generous limits
- **Plus Tier**: $20/month
- **Enterprise**: Custom pricing
- **API Access**: Widely available

## Use Case Recommendations

### Choose Claude If You Need:
- High-quality creative writing
- Safe, reliable responses for sensitive topics
- Long-form content generation
- Consistent, nuanced responses

### Choose GPT-5 If You Need:
- Advanced coding and technical assistance
- Real-time information and current events
- Complex problem-solving
- Fast, efficient responses

## Conclusion

Both Claude and GPT-5 represent the cutting edge of AI language models, each with its own strengths and use cases. The choice between them ultimately depends on your specific needs and preferences.

**Claude** excels in creative, thoughtful, and safe interactions, making it ideal for content creation, research, and professional applications where reliability is paramount.

**GPT-5** shines in technical tasks, coding, and comprehensive knowledge applications, making it the go-to choice for developers, researchers, and power users.

The competition between these models is driving rapid innovation in the AI space, ultimately benefiting users with more capable and refined AI assistants.

---

*Which AI model do you prefer and why? Share your experiences in the comments!*
    `,
    coverImage: "/api/placeholder/600/400",
    category: "Reviews",
    author: {
      name: "Alex Thompson",
      avatar: "/api/placeholder/64/64",
      bio: "AI researcher and tech journalist covering the latest developments in artificial intelligence and machine learning.",
      social: {
        twitter: "alexthompsonai",
        linkedin: "alexthompson"
      }
    },
    date: "2025-01-08",
    readTime: "10 min read",
    likes: 134,
    comments: 28,
    tags: ["AI", "Claude", "GPT-5", "Comparison", "Reviews", "Models"],
    featured: true,
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
      staggerChildren: 0.15
    }
  }
};

export default function FeaturedPosts() {
  return (
    <section id="featured-posts" className="py-20 bg-background-secondary">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial="initial"
          whileInView="animate"
          viewport={{ once: true }}
          variants={staggerChildren}
          className="text-center mb-16"
        >
          <motion.div variants={fadeInUp} className="inline-flex items-center gap-2 px-4 py-2 bg-accent-purple/10 border border-accent-purple/20 rounded-full mb-6">
            <span className="text-sm font-medium text-accent-purple">Featured Articles</span>
          </motion.div>

          <motion.h2
            variants={fadeInUp}
            className="text-4xl md:text-5xl font-bold mb-6"
          >
            Trending in <span className="text-gradient">AI & Tech</span>
          </motion.h2>

          <motion.p
            variants={fadeInUp}
            className="text-xl text-foreground/70 max-w-3xl mx-auto"
          >
            Discover the most popular and insightful articles from our community of AI experts and enthusiasts
          </motion.p>
        </motion.div>

        <motion.div
          initial="initial"
          whileInView="animate"
          viewport={{ once: true }}
          variants={staggerChildren}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-8"
        >
          {featuredPosts.map((post, index) => (
            <motion.article
              key={post.id}
              variants={fadeInUp}
              className="card group cursor-pointer cyber-border"
            >
              {/* Cover Image */}
              <div className="relative h-48 mb-6 overflow-hidden rounded-xl">
                <Image
                  src={post.coverImage}
                  alt={post.title}
                  fill
                  className="object-cover transition-transform duration-300 group-hover:scale-105"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

                {/* Category Badge */}
                <div className="absolute top-4 left-4">
                  <span className="px-3 py-1 bg-accent-blue text-white text-xs font-medium rounded-full">
                    {post.category}
                  </span>
                </div>
              </div>

              {/* Content */}
              <div className="space-y-4">
                <h3 className="text-xl font-bold group-hover:text-accent-blue transition-colors duration-200 line-clamp-2">
                  {post.title}
                </h3>

                <p className="text-foreground/70 line-clamp-3">
                  {post.excerpt}
                </p>

                {/* Meta Information */}
                <div className="flex items-center justify-between text-sm text-foreground/60">
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-1">
                      <User className="w-4 h-4" />
                      <span>{post.author.name}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Calendar className="w-4 h-4" />
                      <span>{new Date(post.date).toLocaleDateString()}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <Clock className="w-4 h-4" />
                    <span>{post.readTime}</span>
                  </div>
                </div>

                {/* Read More Link */}
                <div className="pt-4 border-t border-glass-border">
                  <Link
                    href={`/blog/${post.id}`}
                    className="inline-flex items-center gap-2 text-accent-blue hover:text-accent-purple transition-colors duration-200 font-medium group/link"
                  >
                    Read More
                    <ArrowRight className="w-4 h-4 group-hover/link:translate-x-1 transition-transform" />
                  </Link>
                </div>
              </div>
            </motion.article>
          ))}
        </motion.div>

        {/* View All Posts Button */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.8 }}
          className="text-center mt-12"
        >
          <Link href="/blog">
            <motion.button
              className="btn-secondary cyber-border"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <span className="flex items-center gap-2">
                View All Posts
                <ArrowRight className="w-5 h-5" />
              </span>
            </motion.button>
          </Link>
        </motion.div>
      </div>
    </section>
  );
}
