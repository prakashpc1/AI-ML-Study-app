
import React, { useState, useEffect } from 'react';
import { getBookmarks } from '../services/dbService';
import { Topic } from '../types';
import TopicCard from '../components/TopicCard';
import { Bookmark } from '../components/icons/Icons';
import LoadingSpinner from '../components/LoadingSpinner';
import PageHeader from '../components/PageHeader';

const BookmarksPage: React.FC = () => {
    const [bookmarkedTopics, setBookmarkedTopics] = useState<Topic[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchBookmarks = async () => {
            setLoading(true);
            const bookmarks = await getBookmarks();
            setBookmarkedTopics(bookmarks);
            setLoading(false);
        };
        fetchBookmarks();
    }, []);
    
    if (loading) {
        return <div className="flex justify-center pt-20"><LoadingSpinner /></div>;
    }

    return (
        <div className="p-4 sm:p-6 lg:p-8 animate-fade-in">
            <div className="max-w-4xl mx-auto">
                <PageHeader title="Your Bookmarks" />
                {bookmarkedTopics.length > 0 ? (
                    <div className="space-y-4">
                        {bookmarkedTopics.map(topic => (
                            <TopicCard key={topic.id} topic={topic} />
                        ))}
                    </div>
                ) : (
                    <div className="text-center py-12 px-6 bg-white/50 dark:bg-gray-800/50 rounded-xl shadow-lg border border-white/30 dark:border-gray-700/50 backdrop-blur-sm">
                        <Bookmark className="h-16 w-16 mx-auto text-gray-400 mb-4" />
                        <h2 className="text-2xl font-bold">No Bookmarks Yet</h2>
                        <p className="text-gray-500 dark:text-gray-400 mt-2">
                            Tap the bookmark icon on a topic page to save it for later.
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default BookmarksPage;