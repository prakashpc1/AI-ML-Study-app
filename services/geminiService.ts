
import { GoogleGenAI, Type } from "@google/genai";
import { ChatMessage, QuizQuestion } from '../types';

// FIX: Retrieve API key from environment variables or use a placeholder.
const API_KEY = process.env.API_KEY || "YOUR_GEMINI_API_KEY_HERE";

let ai: GoogleGenAI | null = null;
export let isGeminiInitialized = false;

// FIX: Update initialization logic to use a named parameter.
if (API_KEY && API_KEY !== "YOUR_GEMINI_API_KEY_HERE") {
    try {
        ai = new GoogleGenAI({ apiKey: API_KEY });
        isGeminiInitialized = true;
    } catch (e) {
        console.error("Gemini AI initialization error:", e);
        isGeminiInitialized = false;
    }
} else {
    console.warn("Gemini API key is not configured. Please set the API_KEY environment variable or replace the placeholder in services/geminiService.ts. AI features will be disabled.");
}

const model = "gemini-2.5-flash";

const ensureAi = (): GoogleGenAI => {
    if (!ai) {
        throw new Error("Gemini AI is not initialized. Please check your configuration in services/geminiService.ts.");
    }
    return ai;
}

export const getChatResponse = async (history: ChatMessage[], newMessage: string): Promise<string> => {
    if (!isGeminiInitialized) {
        return "The AI Assistant is not configured. An API key is required.";
    }
    try {
        const aiInstance = ensureAi();
        const chat = aiInstance.chats.create({
            model: model,
            config: {
                systemInstruction: "You are an expert AI and Machine Learning tutor for engineering students. Your role is to teach, guide, and assist students with clear explanations and real-world examples. When asked a question, provide a concise answer and then give a practical, relatable example.",
            },
            history: history.map(msg => ({
                role: msg.sender === 'user' ? 'user' : 'model',
                parts: [{ text: msg.text }]
            }))
        });

        const response = await chat.sendMessage({ message: newMessage });
        return response.text;

    } catch (error) {
        console.error("Error in getChatResponse:", error);
        return "An error occurred while communicating with the AI. Please check your API key and network connection.";
    }
};


export const getSummary = async (textToSummarize: string): Promise<string> => {
    if (!isGeminiInitialized) {
        throw new Error("The AI Summarizer is not configured. An API key is required.");
    }
    try {
        const aiInstance = ensureAi();
        const prompt = `Please summarize the following text for a student who is new to the topic. Structure the summary in 2-3 clear, concise bullet points focusing on the most important concepts. Here is the text:\n\n---\n\n${textToSummarize}`;
        
        const response = await aiInstance.models.generateContent({
            model: model,
            contents: prompt,
        });

        return response.text;
    } catch (error) {
        console.error("Error in getSummary:", error);
        throw new Error("Failed to generate summary.");
    }
};

export const generateCodeSnippet = async (topicTitle: string): Promise<string> => {
    if (!isGeminiInitialized) {
        throw new Error("AI is not configured.");
    }
    try {
        const aiInstance = ensureAi();
        const prompt = `Generate a simple, runnable Python code snippet demonstrating the concept of "${topicTitle}". The code should use common libraries like scikit-learn, numpy, or pandas. Include comments to explain each step. Do not include the introductory backticks for the code block.`;

        const response = await aiInstance.models.generateContent({
            model,
            contents: prompt,
        });

        return response.text;
    } catch (error) {
        console.error("Error generating code snippet:", error);
        throw new Error("Failed to generate code snippet.");
    }
};


export const generateQuiz = async (topicContent: string): Promise<QuizQuestion[]> => {
    if (!isGeminiInitialized) {
        throw new Error("AI is not configured.");
    }
    try {
        const aiInstance = ensureAi();
        const prompt = `Based on the following content, generate a 3-question multiple-choice quiz to test a student's understanding. For each question, provide 4 options and indicate the correct answer index (0-3).

        Content:
        ---
        ${topicContent}
        ---`;

        const response = await aiInstance.models.generateContent({
            model,
            contents: prompt,
            config: {
                responseMimeType: "application/json",
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        quiz: {
                            type: Type.ARRAY,
                            items: {
                                type: Type.OBJECT,
                                properties: {
                                    question: { type: Type.STRING },
                                    options: {
                                        type: Type.ARRAY,
                                        items: { type: Type.STRING }
                                    },
                                    answer: { 
                                        type: Type.INTEGER,
                                        description: "The index of the correct option in the options array."
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
        
        const jsonString = response.text;
        const result = JSON.parse(jsonString);
        return result.quiz;

    } catch (error) {
        console.error("Error generating quiz:", error);
        throw new Error("Failed to generate quiz.");
    }
};