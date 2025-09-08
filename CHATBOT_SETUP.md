# Study Sparks AI Chatbot Setup

## 🚀 Features Added

### 1. Scroll To Top Icon
- Appears when user scrolls down 300px
- Smooth scroll animation to top
- Matches violet theme with hover effects

### 2. AI Chatbot (StudyBot)
- Floating button on bottom-right
- Chat window with violet theme
- Uses Google Gemini API for AI responses
- Typing indicators and smooth animations
- Educational focus for Study Sparks platform

## 🔧 Setup Instructions

### 1. Get Google Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key

### 2. Configure API Key

**Important:** For Next.js client-side components, environment variables must be prefixed with `NEXT_PUBLIC_` to be accessible in the browser.

Create a `.env.local` file in your project root with:

```bash
NEXT_PUBLIC_GEMINI_API_KEY=your_actual_gemini_api_key_here
```

**Do NOT use:**
```bash
GEMINI_API_KEY=your_actual_gemini_api_key_here  # ❌ Won't work in client components
```

The chatbot component is already configured to use `process.env.NEXT_PUBLIC_GEMINI_API_KEY`.

### 3. Test the Chatbot
1. Start the development server: `npm run dev`
2. Visit `http://localhost:3000`
3. Click the chat bubble icon (bottom-right)
4. Start chatting with StudyBot!

## 🤖 Chatbot Capabilities

StudyBot can help users with:
- Questions about Study Sparks platform
- Information about subjects (Coding, Math, Science, English, AI)
- Learning tips and advice
- Platform features explanation
- General educational queries

## 🔒 Security Notes

- Never commit API keys to version control
- Use environment variables for sensitive data
- The chatbot includes proper error handling for missing API keys

## 🚀 Environment Variables Setup

Create a `.env.local` file in your project root:

```bash
# Required for Gemini AI Chatbot
NEXT_PUBLIC_GEMINI_API_KEY=your_actual_gemini_api_key_here

# Optional: For existing AI comparison feature
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

**Important Notes:**
- The `NEXT_PUBLIC_` prefix is required for client-side access
- Restart your development server after adding environment variables
- Never commit `.env.local` to version control

## 📱 Mobile Responsiveness

- Chatbot works perfectly on mobile devices
- Touch-friendly interface
- Responsive chat window
- Optimized for all screen sizes

## 🎨 Theme Integration

- Matches the violet theme perfectly
- Smooth animations and transitions
- Professional educational appearance
- Consistent with Study Sparks branding

## 🛠️ Troubleshooting

**Chatbot not responding:**
- **Environment Variable Issue:** Make sure your `.env.local` file has `NEXT_PUBLIC_GEMINI_API_KEY` (not just `GEMINI_API_KEY`)
- **Restart Server:** After adding environment variables, restart `npm run dev`
- **API Key Validity:** Verify your Gemini API key is active and has proper permissions
- **Network Connection:** Check your internet connection
- **Browser Console:** Open developer tools and check for any JavaScript errors

**Common Issues:**
- **"API key not configured"**: Add `NEXT_PUBLIC_GEMINI_API_KEY` to `.env.local`
- **"NEXT_PUBLIC_" prefix missing**: Client components require this prefix
- **Server not restarted**: Always restart after environment variable changes

**Build errors:**
- Ensure `@google/generative-ai` is installed: `npm install @google/generative-ai`
- Check Node.js version compatibility (Node 18+ recommended)
- Clear `.next` cache if needed: `rm -rf .next`
- Check for TypeScript errors in the Chatbot component

**Debug Steps:**
1. Check `.env.local` file exists and has correct variable name
2. Restart development server completely
3. Open browser console (F12) and check for errors
4. Test Gemini API key directly at [Google AI Studio](https://makersuite.google.com/app/apikey)
