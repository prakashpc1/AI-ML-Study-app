# AI and ML Study Hub

This is a modern, interactive web application designed to help students learn Artificial Intelligence (AI) and Machine Learning (ML) concepts. It features a sleek, dark-themed UI, AI-powered learning tools, and a comprehensive set of topics tailored for engineering students.

## Key Features

- **Comprehensive Topics:** A wide range of subjects from AI basics and core algorithms to neural networks and model evaluation.
- **Interactive UI:** A sophisticated, professional user interface inspired by modern design principles, with a fixed sidebar for easy navigation.
- **AI-Powered Tools:**
    - **Smart Summarization:** Generate concise, AI-powered summaries of complex topics.
    - **Code Generation:** Get runnable Python code snippets for various algorithms.
    - **Dynamic Quizzes:** Test your knowledge with multiple-choice quizzes generated on the fly by an AI.
- **AI Assistant:** A chat-based AI tutor to answer questions and provide real-world examples.
- **Phone Authentication:** Secure user sign-in using phone number and OTP verification, powered by Firebase.
- **Offline First:** Utilizes IndexedDB for offline access to topics and chat history (sync with Firebase when online).
- **Bookmarking:** Save topics to review later, with data synced to the cloud via Firestore.
- **Responsive Design:** Fully functional across desktop, tablet, and mobile devices.

## Tech Stack

- **Frontend:** React, TypeScript, React Router
- **Styling:** Tailwind CSS
- **AI Integration:** Google Gemini API
- **Backend & Database:** Firebase (Authentication, Firestore)
- **Offline Storage:** IndexedDB (via `idb` library)
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

4.  **Configure Environment Variables:**
    This project requires API keys from Firebase and Google AI Studio to function correctly. Create a `.env` file in the root of your project and add your credentials:

    ```.env
    # Firebase Configuration
    FIREBASE_API_KEY="AIzaSy..."
    FIREBASE_AUTH_DOMAIN="your-project-id.firebaseapp.com"
    FIREBASE_PROJECT_ID="your-project-id"
    FIREBASE_STORAGE_BUCKET="your-project-id.appspot.com"
    FIREBASE_MESSAGING_SENDER_ID="1234567890"
    FIREBASE_APP_ID="1:1234567890:web:abcdef123456"

    # Google Gemini API Key
    API_KEY="YOUR_GEMINI_API_KEY_HERE"
    ```

    - You can get your Firebase credentials from your project's settings in the [Firebase Console](https://console.firebase.google.com/).
    - You can get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

5.  **Run the development server:**
    ```bash
    npm run dev
    ```

    The application should now be running on your local server.
