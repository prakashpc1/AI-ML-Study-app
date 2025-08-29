
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useStudentProfile } from '../hooks/useStudentProfile';
import { ChevronRight, Book, Bot } from '../components/icons/Icons';
import { TOPIC_CATEGORIES } from '../constants';

const HomePage: React.FC = () => {
    const { studentProfile } = useStudentProfile();
    const navigate = useNavigate();
    const displayName = studentProfile?.fullName?.split(' ')[0] || 'Student';

    const totalTopics = TOPIC_CATEGORIES.reduce((acc, category) => acc + category.topics.length, 0);

    return (
        <div className="p-4 sm:p-6 lg:p-12 animate-fade-in text-white">
            
            <div className="relative bg-gray-900 p-8 rounded-2xl mb-12 overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-r from-purple-600/20 to-blue-600/20 opacity-50"></div>
                 <div 
                    className="absolute -top-10 -right-10 w-40 h-40 bg-pink-500/20 rounded-full filter blur-3xl"
                ></div>
                <div 
                    className="absolute -bottom-12 -left-12 w-48 h-48 bg-teal-500/20 rounded-full filter blur-3xl"
                ></div>
                <div className="relative z-10">
                    <h1 className="text-4xl md:text-5xl font-bold">Welcome Back, {displayName}!</h1>
                    <p className="mt-2 text-lg text-gray-300">Ready to unlock the secrets of AI? Your learning adventure continues here.</p>
                    <div className="mt-6 flex gap-4">
                        <span className="bg-white/10 px-3 py-1 text-sm rounded-full">New Content Added</span>
                        <span className="bg-white/10 px-3 py-1 text-sm rounded-full">Popular This Week</span>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
                <div className="lg:col-span-3 bg-gray-900 p-8 rounded-2xl border border-gray-800">
                    <div className="flex items-center gap-4 mb-4">
                        <div className="w-12 h-12 bg-blue-500/20 text-blue-400 flex items-center justify-center rounded-lg">
                            <Book className="w-6 h-6" />
                        </div>
                        <h2 className="text-2xl font-bold">Explore Topics</h2>
                    </div>
                    <p className="text-gray-400 mb-6">Discover AI & ML concepts from basics to advanced.</p>
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-gray-400 text-sm">Available Topics</p>
                            <p className="text-2xl font-semibold">{totalTopics}+</p>
                        </div>
                        <button
                            onClick={() => navigate('/topics')}
                            className="bg-white text-black font-semibold py-2 px-4 rounded-lg flex items-center gap-2 transition-transform transform hover:scale-105"
                        >
                            <span>Start Learning</span>
                            <ChevronRight className="w-5 h-5" />
                        </button>
                    </div>
                </div>

                <div className="lg:col-span-2 bg-gray-900 p-8 rounded-2xl border border-gray-800">
                     <div className="flex items-center gap-4 mb-4">
                        <div className="w-12 h-12 bg-green-500/20 text-green-400 flex items-center justify-center rounded-lg">
                            <Bot className="w-6 h-6" />
                        </div>
                        <h2 className="text-2xl font-bold">AI Assistant</h2>
                    </div>
                    <p className="text-gray-400 mb-6">Get instant answers to your AI/ML questions.</p>
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-gray-400 text-sm">Available 24/7</p>
                            <div className="flex items-center gap-2">
                                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                                <p className="font-semibold text-green-400">Online</p>
                            </div>
                        </div>
                         <button
                            onClick={() => navigate('/ai-assistant')}
                            className="bg-gray-800 text-white font-semibold py-2 px-4 rounded-lg flex items-center gap-2 transition-colors hover:bg-gray-700"
                        >
                            <span>Ask AI</span>
                             <ChevronRight className="w-5 h-5" />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default HomePage;