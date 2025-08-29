import React, { useEffect, useState } from 'react';
import { HashRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider, useTheme } from './hooks/useTheme';
import { StudentProfileProvider, useStudentProfile } from './hooks/useStudentProfile';
import HomePage from './pages/HomePage';
import TopicsPage from './pages/TopicsPage';
import TopicDetailPage from './pages/TopicDetailPage';
import AIAssistantPage from './pages/AIAssistantPage';
import BookmarksPage from './pages/BookmarksPage';
import SettingsPage from './pages/SettingsPage';
import SideNav from './components/SideNav';
import StudentGate from './components/StudentGate';
import LoadingSpinner from './components/LoadingSpinner';

const AppContent: React.FC = () => {
    const { theme } = useTheme();

    useEffect(() => {
        if (theme === 'dark') {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
    }, [theme]);

    return (
        <div className="flex h-screen bg-gray-100 dark:bg-black text-gray-900 dark:text-gray-100 transition-colors duration-300">
            <SideNav />
            <main className="flex-grow overflow-y-auto pl-64">
                <Routes>
                    <Route path="/" element={<HomePage />} />
                    <Route path="/topics" element={<TopicsPage />} />
                    <Route path="/topics/:topicId" element={<TopicDetailPage />} />
                    <Route path="/ai-assistant" element={<AIAssistantPage />} />
                    <Route path="/bookmarks" element={<BookmarksPage />} />
                    <Route path="/settings" element={<SettingsPage />} />
                </Routes>
            </main>
        </div>
    );
};

const AppInitializer: React.FC = () => {
    const { studentProfile, loading, setStudentProfile } = useStudentProfile();
    
    if (loading) {
        return (
            <div className="flex h-screen w-screen items-center justify-center bg-black">
                <LoadingSpinner />
            </div>
        );
    }
    
    if (!studentProfile) {
        return <StudentGate onProfileReady={(profile) => setStudentProfile(profile)} />;
    }

    return <AppContent />;
};

const App: React.FC = () => {
    return (
        <ThemeProvider>
            <StudentProfileProvider>
                <HashRouter>
                    <AppInitializer />
                </HashRouter>
            </StudentProfileProvider>
        </ThemeProvider>
    );
};

export default App;