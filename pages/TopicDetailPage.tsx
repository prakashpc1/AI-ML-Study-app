
import React, { useState, useEffect, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { getTopicById, getBookmarkStatus, toggleBookmark } from '../services/dbService';
import { useStudentProfile } from '../hooks/useStudentProfile';
import { Topic, QuizQuestion } from '../types';
import { ChevronLeft, Bookmark as BookmarkIcon, Tag as TagIcon, Sparkles, Code, ClipboardCheck } from '../components/icons/Icons';
import LoadingSpinner from '../components/LoadingSpinner';
import { extractKeywords } from '../services/offlineAiService';
import { getSummary, generateQuiz, generateCodeSnippet, isGeminiInitialized } from '../services/geminiService';

const TopicDetailPage: React.FC = () => {
    const { topicId } = useParams<{ topicId: string }>();
    const navigate = useNavigate();
    const { studentProfile } = useStudentProfile();
    const [topic, setTopic] = useState<Topic | null>(null);
    const [isBookmarked, setIsBookmarked] = useState(false);
    
    // AI Features State
    const [summary, setSummary] = useState('');
    const [isSummaryLoading, setIsSummaryLoading] = useState(false);
    const [codeSnippet, setCodeSnippet] = useState('');
    const [isCodeLoading, setIsCodeLoading] = useState(false);
    const [quiz, setQuiz] = useState<QuizQuestion[] | null>(null);
    const [isQuizLoading, setIsQuizLoading] = useState(false);
    const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
    const [userAnswers, setUserAnswers] = useState<number[]>([]);
    const [showScore, setShowScore] = useState(false);

    useEffect(() => {
        if (topicId) {
            const fetchTopicAndBookmarkStatus = async () => {
                const fetchedTopic = await getTopicById(topicId);
                setTopic(fetchedTopic);
                if (fetchedTopic) {
                    const status = await getBookmarkStatus(fetchedTopic.id);
                    setIsBookmarked(status);
                } else {
                    setIsBookmarked(false);
                }
            };
            fetchTopicAndBookmarkStatus();
        }
    }, [topicId, studentProfile]);
    
    const keywords = useMemo(() => {
        if (topic) {
            return extractKeywords(topic.content);
        }
        return [];
    }, [topic]);

    const handleBookmarkToggle = async () => {
        if (!studentProfile || !topic) return;

        try {
            const newStatus = await toggleBookmark(topic);
            setIsBookmarked(newStatus);
        } catch (error) {
            console.error("Error toggling bookmark:", error);
        }
    };
    
    // --- AI Feature Handlers ---
    const handleGenerateSummary = async () => {
        if (!topic) return;
        setIsSummaryLoading(true);
        try {
            const result = await getSummary(topic.content);
            setSummary(result);
        } catch (error) {
            setSummary("Sorry, I couldn't generate a summary right now.");
        } finally {
            setIsSummaryLoading(false);
        }
    };

    const handleGenerateCode = async () => {
        if (!topic) return;
        setIsCodeLoading(true);
        try {
            const result = await generateCodeSnippet(topic.title);
            setCodeSnippet(result);
        } catch (error) {
            setCodeSnippet("Sorry, I couldn't generate a code snippet right now.");
        } finally {
            setIsCodeLoading(false);
        }
    };
    
    const handleGenerateQuiz = async () => {
        if (!topic) return;
        setIsQuizLoading(true);
        resetQuiz();
        try {
            const result = await generateQuiz(topic.content);
            setQuiz(result);
        } catch (error) {
            console.error("Quiz generation failed:", error);
            setQuiz(null); // Or set an error state
        } finally {
            setIsQuizLoading(false);
        }
    };
    
    const handleAnswerSelect = (optionIndex: number) => {
        const newAnswers = [...userAnswers];
        newAnswers[currentQuestionIndex] = optionIndex;
        setUserAnswers(newAnswers);

        setTimeout(() => {
            if (quiz && currentQuestionIndex < quiz.length - 1) {
                setCurrentQuestionIndex(prev => prev + 1);
            } else {
                setShowScore(true);
            }
        }, 500);
    };

    const calculateScore = () => {
        if (!quiz) return 0;
        return quiz.reduce((score, question, index) => {
            return score + (question.answer === userAnswers[index] ? 1 : 0);
        }, 0);
    };

    const resetQuiz = () => {
        setQuiz(null);
        setCurrentQuestionIndex(0);
        setUserAnswers([]);
        setShowScore(false);
    };


    if (!topic) {
        return <div className="flex justify-center items-center h-screen"><LoadingSpinner /></div>;
    }

    return (
        <div className="animate-fade-in pb-12">
            <header className="sticky top-0 z-40 bg-gray-900/60 backdrop-blur-lg border-b border-gray-700/50 shadow-sm">
                <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-between h-16">
                    <button onClick={() => navigate(-1)} className="p-2 rounded-full hover:bg-gray-700/50">
                        <ChevronLeft className="h-6 w-6 text-white" />
                    </button>
                    <h1 className="text-lg font-bold text-center truncate px-4 text-white">{topic.title}</h1>
                    <button
                        onClick={handleBookmarkToggle}
                        className="p-2 rounded-full hover:bg-gray-700/50"
                        title="Toggle bookmark"
                    >
                        <BookmarkIcon className={`h-6 w-6 transition-colors ${isBookmarked ? 'fill-yellow-400 text-yellow-500' : 'text-gray-400'}`} />
                    </button>
                </div>
            </header>

            <div className="p-4 sm:p-6 lg:p-8 max-w-5xl mx-auto">
                {/* Main content card */}
                <div className="bg-gray-800/50 p-6 rounded-xl shadow-lg border border-gray-700/50 backdrop-blur-sm">
                    <p className={`text-sm ${topic.color || 'text-orange-400'} font-semibold mb-2`}>{topic.category}</p>
                    <h2 className="text-4xl font-extrabold mb-4 text-white">{topic.title}</h2>
                    <p className="text-gray-400 mb-6 italic">{topic.description}</p>
                    <div className="prose prose-invert max-w-none text-gray-300 space-y-4">
                        {topic.content.split('\n').filter(p => p.trim() !== '').map((paragraph, index) => {
                             if (paragraph.startsWith('```')) {
                                return (
                                    <pre key={index} className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
                                        <code>{paragraph.replace(/```python|```/g, '')}</code>
                                    </pre>
                                );
                            }
                            if (paragraph.startsWith('### ')) {
                                return <h3 key={index} className="text-xl font-bold mt-6 mb-2">{paragraph.replace('### ', '')}</h3>
                            }
                            return <p key={index}>{paragraph}</p>
                        })}
                    </div>
                </div>

                {/* Keywords Card */}
                {keywords.length > 0 && (
                     <div className="mt-8 bg-gray-800/50 p-6 rounded-xl shadow-lg border border-gray-700/50 backdrop-blur-sm">
                        <h3 className="text-2xl font-bold mb-4 flex items-center gap-2 text-white">
                            <TagIcon className="h-6 w-6 text-purple-400" />
                            <span>Key Concepts</span>
                        </h3>
                        <div className="flex flex-wrap gap-2">
                            {keywords.map((keyword, index) => (
                                <span key={index} className="bg-purple-900 text-purple-300 text-sm font-medium px-3 py-1 rounded-full">
                                    {keyword}
                                </span>
                            ))}
                        </div>
                    </div>
                )}
                
                {/* AI-Powered Tools Section */}
                <div className="mt-8 bg-gray-800/50 p-6 rounded-xl shadow-lg border border-gray-700/50 backdrop-blur-sm">
                    <h3 className="text-2xl font-bold mb-6 text-white text-center">AI-Powered Tools</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        {/* Summary */}
                        <div className="flex flex-col items-center text-center">
                             <Sparkles className="h-8 w-8 text-yellow-400 mb-2"/>
                             <h4 className="font-bold text-white mb-2">Smart Summary</h4>
                             <button onClick={handleGenerateSummary} disabled={isSummaryLoading || !isGeminiInitialized} className="bg-yellow-500 hover:bg-yellow-600 text-white font-semibold py-2 px-4 rounded-lg w-full disabled:bg-gray-600">
                                {isSummaryLoading ? 'Generating...' : 'Summarize'}
                             </button>
                             {isSummaryLoading && <LoadingSpinner />}
                             {summary && !isSummaryLoading && <p className="text-sm text-gray-300 mt-4 text-left p-4 bg-gray-900/50 rounded-lg">{summary}</p>}
                        </div>
                        {/* Code Snippet */}
                        <div className="flex flex-col items-center text-center">
                             <Code className="h-8 w-8 text-green-400 mb-2"/>
                             <h4 className="font-bold text-white mb-2">Code Example</h4>
                             <button onClick={handleGenerateCode} disabled={isCodeLoading || !isGeminiInitialized} className="bg-green-500 hover:bg-green-600 text-white font-semibold py-2 px-4 rounded-lg w-full disabled:bg-gray-600">
                                {isCodeLoading ? 'Generating...' : 'Get Code'}
                             </button>
                             {isCodeLoading && <LoadingSpinner />}
                             {codeSnippet && !isCodeLoading && <pre className="text-sm text-left mt-4 p-4 bg-gray-900/50 rounded-lg w-full overflow-x-auto"><code className="text-green-300">{codeSnippet}</code></pre>}
                        </div>
                         {/* Quiz */}
                        <div className="flex flex-col items-center text-center">
                             <ClipboardCheck className="h-8 w-8 text-blue-400 mb-2"/>
                             <h4 className="font-bold text-white mb-2">Test Your Knowledge</h4>
                             <button onClick={handleGenerateQuiz} disabled={isQuizLoading || !isGeminiInitialized} className="bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded-lg w-full disabled:bg-gray-600">
                                {isQuizLoading ? 'Generating...' : 'Start Quiz'}
                             </button>
                        </div>
                    </div>
                     {!isGeminiInitialized && <p className="text-center text-yellow-500 text-sm mt-6">Please configure your Gemini API key to use AI-powered tools.</p>}
                </div>
                
                {/* Quiz UI */}
                {isQuizLoading && <div className="mt-8 flex justify-center"><LoadingSpinner /></div>}
                {quiz && !isQuizLoading && (
                    <div className="mt-8 bg-gray-800/50 p-6 rounded-xl shadow-lg border border-gray-700/50 backdrop-blur-sm text-white">
                        {!showScore ? (
                             <div>
                                <p className="text-sm text-gray-400 mb-2">Question {currentQuestionIndex + 1} of {quiz.length}</p>
                                <h4 className="text-lg font-semibold mb-6">{quiz[currentQuestionIndex].question}</h4>
                                <div className="space-y-3">
                                    {quiz[currentQuestionIndex].options.map((option, index) => {
                                        const isSelected = userAnswers[currentQuestionIndex] === index;
                                        return (
                                             <button
                                                key={index}
                                                onClick={() => handleAnswerSelect(index)}
                                                className={`w-full text-left p-3 rounded-lg border-2 transition-colors ${isSelected ? 'bg-purple-600 border-purple-400' : 'bg-gray-700 border-gray-600 hover:border-purple-500'}`}
                                             >
                                                {option}
                                             </button>
                                        );
                                    })}
                                </div>
                             </div>
                        ) : (
                             <div className="text-center">
                                <h3 className="text-2xl font-bold mb-4">Quiz Complete!</h3>
                                <p className="text-lg mb-6">Your score: <span className="font-bold text-green-400">{calculateScore()}</span> / {quiz.length}</p>
                                <button onClick={resetQuiz} className="bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-6 rounded-lg">
                                    Try Again
                                </button>
                             </div>
                        )}
                    </div>
                )}


                {/* Chart card */}
                {topic.chartData && (
                    <div className="mt-8 bg-gray-800/50 p-6 rounded-xl shadow-lg border border-gray-700/50 backdrop-blur-sm">
                        <h3 className="text-2xl font-bold mb-4 text-white">Interactive Visualization</h3>
                        <div style={{ width: '100%', height: 300 }}>
                            <ResponsiveContainer>
                                <LineChart data={topic.chartData}>
                                    <CartesianGrid strokeDasharray="3 3" className="stroke-gray-700" />
                                    <XAxis dataKey="name" className="text-xs fill-gray-400" />
                                    <YAxis className="text-xs fill-gray-400" />
                                    <Tooltip contentStyle={{
                                        backgroundColor: 'rgba(31, 41, 55, 0.8)',
                                        border: '1px solid rgba(255,255,255,0.2)',
                                        borderRadius: '0.75rem',
                                        backdropFilter: 'blur(4px)',
                                        color: '#fff'
                                    }} />
                                    <Legend />
                                    <Line type="monotone" dataKey="value" stroke="#f97316" strokeWidth={2} activeDot={{ r: 8 }} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default TopicDetailPage;