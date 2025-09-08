# Study Sparks Demo Instructions

## 🎮 Quick Start Demo

### 1. Start the Application
```bash
npm run dev
```
Visit: `http://localhost:3000`

### 2. Explore Features

#### 🖱️ Scroll To Top Icon
- Scroll down the page
- Notice the violet circular button appears in bottom-right
- Click it to smoothly scroll back to top

#### 🤖 AI Chatbot (StudyBot)
- Look for the violet chat bubble icon in bottom-right
- Click it to open the chat window
- Try these sample questions:

**Platform Questions:**
- "What subjects do you teach?"
- "How does Study Sparks work?"
- "Tell me about your coding courses"

**Educational Questions:**
- "How can I learn Python for beginners?"
- "What are some fun math activities?"
- "How does AI help with learning?"

**Support Questions:**
- "How do I enroll in a course?"
- "What are your class timings?"
- "Do you offer trial classes?"

### 3. Test Visual Features

#### 🎨 Violet Theme
- Notice the consistent violet/purple color scheme
- Hover over buttons and cards for smooth animations
- Check mobile responsiveness by resizing browser

#### ⚡ Animations
- Smooth page transitions
- Floating elements in background
- Interactive hover states
- Typing indicators in chat

### 4. API Integration

#### Without API Key:
- Chatbot will show: "Sorry, the AI assistant is not configured yet..."

#### With API Key:
- Replace `'your_gemini_api_key_here'` in `src/components/Chatbot.tsx`
- Or create `.env.local` with `NEXT_PUBLIC_GEMINI_API_KEY=your_key`
- Chatbot will respond intelligently to educational queries

## 🔧 Development Features

### Components Added:
- ✅ Hero Section (Violet Theme)
- ✅ Features Section (Violet Cards)
- ✅ About Section
- ✅ Subjects Section (Violet Cards)
- ✅ Live Classes Section
- ✅ Community Section
- ✅ Testimonials Section
- ✅ Demo Form Modal
- ✅ CTA Section
- ✅ FAQ Section
- ✅ Footer Section
- ✅ **Scroll To Top Icon** ⭐
- ✅ **AI Chatbot (StudyBot)** ⭐

### Tech Stack:
- Next.js 15.5.2
- React 19
- TypeScript
- TailwindCSS v4
- Framer Motion
- Google Generative AI (Gemini)
- Lucide Icons

## 🚀 Deployment Ready

The application is fully responsive and ready for production deployment on Vercel, Netlify, or any hosting platform.

## 📱 Mobile Optimization

- Touch-friendly interfaces
- Responsive design for all screen sizes
- Optimized performance
- Fast loading times

## 🎯 Perfect for:
- Educational platforms
- Learning management systems
- Online tutoring services
- Student engagement portals
- EdTech startups

Enjoy exploring Study Sparks! 🎓✨
