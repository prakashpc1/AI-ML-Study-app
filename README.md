
# AI and ML Study Hub

This is a modern, interactive web application designed to help students learn Artificial Intelligence (AI) and Machine Learning (ML) concepts. It features a sleek, dark-themed UI, AI-powered learning tools, and a comprehensive set of topics tailored for engineering students. The app is designed to be **offline-first**, storing all student data locally on the device.

## Key Features

- **Comprehensive Topics:** A wide range of subjects from AI basics and core algorithms to neural networks and model evaluation.
- **Interactive UI:** A sophisticated, professional user interface with a fixed sidebar for easy navigation.
- **AI-Powered Tools (Online):**
    - **Smart Summarization:** Generate concise, AI-powered summaries of complex topics.
    - **Code Generation:** Get runnable Python code snippets for various algorithms.
    - **Dynamic Quizzes:** Test your knowledge with multiple-choice quizzes generated on the fly by an AI.
- **AI Assistant:** A chat-based AI tutor to answer questions and provide real-world examples.
- **Offline-First Student Profile:** A one-time student credential form stores profile data locally. No online account or login required.
- **Local Data Storage:** Utilizes IndexedDB for offline access to topics, bookmarks, and chat history.
- **Responsive Design:** Fully functional across desktop, tablet, and mobile devices.

## Tech Stack

- **Frontend:** React, TypeScript, React Router
- **Styling:** Tailwind CSS
- **AI Integration:** Google Gemini API
- **Offline Storage:** IndexedDB (via `idb` library), LocalStorage
- **Speech Recognition:** Web Speech API
- **Charting:** Recharts

## Getting Started

To run this project locally, follow these steps:

1.  **Prerequisites:** You need to have Node.js and a package manager (like npm or yarn) installed.

2.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

3.  **Install dependencies:**
    ```bash
    npm install
    ```

4.  **Configure Environment Variables (Optional, for AI features):**
    The core application works entirely offline. To enable the AI-powered features (Assistant, Summarizer, etc.), you need an API key from Google AI Studio. Create a `.env` file in the root of your project and add your key:

    ```.env
    # Google Gemini API Key
    API_KEY="YOUR_GEMINI_API_KEY_HERE"
    ```

    - You can get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    - If you do not provide a key, the app will still run, but all AI-powered tools will be disabled.

5.  **Run the development server:**
    ```bash
    npm run dev
    ```

    The application should now be running on your local server.