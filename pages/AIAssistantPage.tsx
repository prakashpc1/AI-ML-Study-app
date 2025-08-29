
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { getChatResponse, isGeminiInitialized } from '../services/geminiService';
import { getChatHistory, saveChatMessage } from '../services/dbService';
import { useStudentProfile } from '../hooks/useStudentProfile';
import { ChatMessage as ChatMessageType } from '../types';
import ChatMessage from '../components/ChatMessage';
import LoadingSpinner from '../components/LoadingSpinner';
import { Send, Mic, Bot } from '../components/icons/Icons';
import { useSpeechRecognition } from '../hooks/useSpeechRecognition';
import PageHeader from '../components/PageHeader';

const AIAssistantPage: React.FC = () => {
    const { studentProfile } = useStudentProfile();
    const [messages, setMessages] = useState<ChatMessageType[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const { transcript, listening, startListening, stopListening, browserSupportsSpeechRecognition } = useSpeechRecognition();

    useEffect(() => {
        const loadHistory = async () => {
            const history = await getChatHistory();
            setMessages(history);
        };
        loadHistory();
    }, []);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);
    
    useEffect(() => {
        if(transcript) {
            setInput(transcript);
        }
    }, [transcript]);

    const handleSend = useCallback(async (messageText: string) => {
        if (!messageText.trim()) return;

        const userMessage: ChatMessageType = {
            id: `user-${Date.now()}`,
            text: messageText,
            sender: 'user',
            timestamp: Date.now(),
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            await saveChatMessage(userMessage);
            const aiResponseText = await getChatResponse(messages, messageText);
            
            const aiMessage: ChatMessageType = {
                id: `ai-${Date.now()}`,
                text: aiResponseText,
                sender: 'ai',
                timestamp: Date.now(),
            };
            setMessages(prev => [...prev, aiMessage]);
            await saveChatMessage(aiMessage);

        } catch (error) {
            console.error("Error getting AI response:", error);
            const errorMessage: ChatMessageType = {
                id: `ai-error-${Date.now()}`,
                text: "I'm having trouble connecting right now. Please try again later.",
                sender: 'ai',
                timestamp: Date.now(),
            };
            setMessages(prev => [...prev, errorMessage]);
            await saveChatMessage(errorMessage);
        } finally {
            setIsLoading(false);
        }
    }, [messages]);
    
    const handleFormSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        handleSend(input);
    };

    const handleMicClick = () => {
        if (listening) {
            stopListening();
            if(transcript){
                handleSend(transcript);
            }
        } else {
            startListening();
        }
    };
    
    return (
        <div className="flex flex-col h-full p-4 sm:p-6 lg:p-8">
            <PageHeader title="AI Assistant" />
            <div className="flex-grow overflow-y-auto mb-4 p-4 bg-white/30 dark:bg-gray-800/30 rounded-2xl shadow-inner backdrop-blur-sm border border-white/20 dark:border-gray-700/30">
                 
                    <>
                        {messages.map(msg => <ChatMessage key={msg.id} message={msg} />)}
                        {isLoading && <ChatMessage message={{ id: 'loading', text: '', sender: 'ai', timestamp: 0 }} />}
                        {isLoading && <div className="flex justify-start pl-12"><LoadingSpinner /></div>}
                        <div ref={messagesEndRef} />
                    </>
                
            </div>
            <form onSubmit={handleFormSubmit} className="flex items-center gap-2">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder={!isGeminiInitialized ? "Please configure Gemini API Key" : (listening ? "Listening..." : "Ask anything about AI & ML...")}
                    className="flex-grow p-3 border border-gray-300/50 dark:border-gray-600/50 rounded-full bg-white/50 dark:bg-gray-700/50 focus:outline-none focus:ring-2 focus:ring-orange-500 backdrop-blur-sm"
                    disabled={isLoading || !isGeminiInitialized}
                />
                {browserSupportsSpeechRecognition && (
                    <button
                        type="button"
                        onClick={handleMicClick}
                        className={`p-3 rounded-full transition-colors ${listening ? 'bg-red-500 text-white' : 'bg-gray-200/70 dark:bg-gray-600/70 hover:bg-gray-300/70 dark:hover:bg-gray-500/70'}`}
                        disabled={isLoading || !isGeminiInitialized}
                    >
                        <Mic className="w-6 h-6" />
                    </button>
                )}
                <button
                    type="submit"
                    className="bg-green-500 text-white p-3 rounded-full hover:bg-green-600 disabled:bg-gray-400 transition-transform transform hover:scale-110"
                    disabled={isLoading || !input.trim() || !isGeminiInitialized}
                >
                    <Send className="w-6 h-6" />
                </button>
            </form>
        </div>
    );
};

export default AIAssistantPage;